"""Source-bound OCR queries and executable evidence exclusions."""

import copy
import json
import math

import numpy as np

from dear.protocol import ToolExecutionError, write_new
from dear.reader import TOOL, ExistingVideoBlinder, _number, _selection

SELECTION_PROTOCOL = "explicit_media_selection"
VISUAL_TOOL = copy.deepcopy(TOOL)
properties = VISUAL_TOOL["function"]["parameters"]["properties"]["exclude"]["items"][
    "properties"
]
properties["box"][
    "description"
] = "Visual only: normalized LTRB rectangle. Required for a regional exclusion. Omitting it does NOT select the whole frame."
properties["whole_frame"] = {
    "type": "boolean",
    "description": "Visual only: explicitly set true to withhold the whole frame for the chosen interval. Use either box or whole_frame=true, not both.",
}
properties["end_s"][
    "description"
] = "Visual only: inclusive end in seconds, not the end of the named second. An end beyond the video is capped at the video end. Omit both times for the whole video."


def normalize(exclude, duration):
    """Resolve executable coordinates only; preserve the caller's raw selection."""
    if not isinstance(exclude, list):
        raise ToolExecutionError("invalid_exclusions", "exclude must be a list.")
    if not _number(duration) or duration <= 0:
        raise ToolExecutionError(
            "invalid_duration", "A positive finite video duration is required."
        )
    effective, adjustments = ([], [])
    for index, original in enumerate(exclude):
        if not isinstance(original, dict) or "kind" not in original:
            raise ToolExecutionError(
                "invalid_exclusions", "Each exclusion needs a kind."
            )
        item = copy.deepcopy(original)
        if item["kind"] == "audio":
            if set(item) != {"kind"}:
                raise ToolExecutionError(
                    "unsupported_audio_interval",
                    "Only whole-audio removal is supported.",
                )
        elif item["kind"] == "visual":
            if set(item) - {"kind", "box", "whole_frame", "start_s", "end_s"}:
                raise ToolExecutionError(
                    "unsupported_exclusion", "Use explicit media coordinates only."
                )
            whole = item.pop("whole_frame", False)
            if type(whole) is not bool:
                raise ToolExecutionError(
                    "invalid_visual_scope", "whole_frame must be a boolean."
                )
            if whole:
                if "box" in item:
                    raise ToolExecutionError(
                        "invalid_visual_scope",
                        "Choose either box or whole_frame, not both.",
                    )
                item["box"] = [0, 0, 1, 1]
            elif "box" not in item:
                raise ToolExecutionError(
                    "missing_visual_scope",
                    "Provide box, or explicitly set whole_frame=true.",
                )
            if ("start_s" in item) != ("end_s" in item):
                raise ToolExecutionError(
                    "invalid_interval", "Supply both endpoints or neither."
                )
            start, end = (item.get("start_s", 0), item.get("end_s", duration))
            if not (
                _number(start)
                and _number(end)
                and (0 <= start < duration)
                and (start < end)
            ):
                raise ToolExecutionError(
                    "invalid_interval",
                    "Use an increasing interval starting inside the video.",
                )
            if end > duration:
                item["end_s"] = duration
                adjustments.append(
                    {
                        "index": index,
                        "operation": "cap_end_at_video_end",
                        "requested_end_s": end,
                        "effective_end_s": duration,
                    }
                )
        else:
            raise ToolExecutionError("unsupported_exclusion", "Use visual or audio.")
        effective.append(item)
    return (effective, adjustments)


class ExplicitVideoBlinder(ExistingVideoBlinder):

    def prepare(self, material, exclude):
        receipt = {
            "interface": SELECTION_PROTOCOL,
            "material_id": material["id"],
            "video_duration_s": material["duration_s"],
            "requested_exclusions": copy.deepcopy(exclude),
        }
        self.media.directory.mkdir(parents=True, exist_ok=True)
        try:
            effective, adjustments = normalize(exclude, material["duration_s"])
        except ToolExecutionError as error:
            receipt["selection_error"] = error.code
            write_new(self.media.directory / "blind_selection.json", receipt)
            raise
        receipt.update(effective_exclusions=effective, boundary_adjustments=adjustments)
        write_new(self.media.directory / "blind_selection.json", receipt)
        return super().prepare(material, effective)


_visual_properties = copy.deepcopy(
    VISUAL_TOOL["function"]["parameters"]["properties"]["exclude"]["items"][
        "properties"
    ]
)
_visual_properties["kind"] = {"type": "string", "enum": ["non_text_visual"]}
_nontext = {
    "type": "object",
    "properties": _visual_properties,
    "required": ["kind"],
    "additionalProperties": False,
}
BLIND_TOOL = {
    "type": "function",
    "function": {
        "name": "blind_answer",
        "description": "After OCR inspection, exclude selected evidence and obtain one independent final response. Text: displayed track_id only. Non-text scene content: explicit non_text_visual region or whole_frame. Audio: whole stream only. Empty exclude keeps the current video. No source is presumed trustworthy; the independent reader receives only the resulting video and original QA.",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {
                    "type": "string",
                    "description": "This input video material ID, not a path or URL.",
                },
                "exclude": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"track_id": {"type": "string"}},
                                "required": ["track_id"],
                                "additionalProperties": False,
                            },
                            _nontext,
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": ["audio"]}
                                },
                                "required": ["kind"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
            },
            "required": ["material_id", "exclude"],
            "additionalProperties": False,
        },
    },
}


def checked_exclusions(exclude, duration):
    """Validate intent/structure and normalize only the nontext coordinate branch."""
    if not isinstance(exclude, list):
        fail("invalid_exclusions", "exclude must be a list.")
    result = []
    for item in exclude:
        if not isinstance(item, dict):
            fail("invalid_exclusions", "Each exclusion must be an object.")
        if "track_id" in item:
            if (
                set(item) != {"track_id"}
                or not isinstance(item["track_id"], str)
                or (not item["track_id"])
            ):
                fail(
                    "invalid_text_target",
                    "A text exclusion is one displayed track_id without box/time overrides.",
                )
            result.append(copy.deepcopy(item))
        elif item.get("kind") == "non_text_visual":
            translated = {**copy.deepcopy(item), "kind": "visual"}
            normalize([translated], duration)
            result.append(translated)
        elif item.get("kind") == "audio":
            if set(item) != {"kind"}:
                fail(
                    "unsupported_audio_interval",
                    "Only whole-audio removal is supported.",
                )
            result.append({"kind": "audio"})
        else:
            fail(
                "ocr_id_required",
                "Text uses displayed track_id. Manual coordinates require explicit non_text_visual intent.",
            )
    return result


def fail(code, message, **details):
    raise ToolExecutionError(code, message, **details)


def query_scope(arguments, material_id, duration):
    if not isinstance(arguments, dict) or set(arguments) - {
        "material_id",
        "start_s",
        "end_s",
        "box",
    }:
        fail(
            "invalid_query",
            "Use material_id, start_s/end_s, and optional normalized box.",
        )
    if arguments.get("material_id") != material_id:
        fail("wrong_material", "Query this original video material.")
    if ("start_s" in arguments) != ("end_s" in arguments):
        fail("invalid_interval", "Supply both start_s and end_s, or omit both.")
    start, end = (arguments.get("start_s", 0), arguments.get("end_s", duration))
    if (
        not all((type(v) in (int, float) and math.isfinite(v) for v in (start, end)))
        or not 0 <= start < end
        or start >= duration
    ):
        fail(
            "invalid_interval",
            "Use seconds, not MMSS: 0 <= start_s < end_s, with start_s inside this video.",
            duration_s=duration,
            requested_start_s=start,
            requested_end_s=end,
        )
    box = arguments.get("box", [0, 0, 1, 1])
    if (
        not isinstance(box, list)
        or len(box) != 4
        or (not all((type(v) in (int, float) and math.isfinite(v) for v in box)))
        or (not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1))
    ):
        fail(
            "invalid_box",
            "Use normalized [left, top, right, bottom] coordinates in [0, 1].",
        )
    return dict(start_s=float(start), end_s=float(min(end, duration)), box=list(box))


class OverlapCatalog:

    def __init__(self, catalog, *, max_reply_chars=24000):
        self.catalog, self.value = (catalog, catalog.value)
        self.max_reply_chars = max_reply_chars
        self.shown, self.queries, self.rows = (set(), {}, {})
        self.workspace, self.current_query = ({}, None)
        self.aliases = {
            f"t{i + 1}": t["track_id"] for i, t in enumerate(self.value["tracks"])
        }
        self.tracks = dict(zip(self.aliases, self.value["tracks"]))
        self.index = {}
        for ident, t in self.tracks.items():
            self.index[ident] = np.asarray(
                [[h["start_s"], h["end_s_exclusive"], *h["box"]] for h in t["hits"]],
                dtype=float,
            )

    def query(self, arguments):
        scope = query_scope(
            arguments, self.value["material_id"], self.value["duration_s"]
        )
        key = json.dumps(scope, sort_keys=True)
        if key in self.queries:
            result = copy.deepcopy(self.queries[key])
            self._activate(result)
            return dict(
                result,
                status="already_returned",
                new_candidates=0,
                message="Cached candidates for this scope restored; no OCR was rerun.",
            )
        width, height = self.value["canvas"]
        left, top, right, bottom = np.asarray(scope["box"]) * [
            width,
            height,
            width,
            height,
        ]
        matches = []
        for ident, a in self.index.items():
            hit = (
                (a[:, 0] < scope["end_s"])
                & (a[:, 1] > scope["start_s"])
                & (a[:, 2] < right)
                & (a[:, 4] > left)
                & (a[:, 3] < bottom)
                & (a[:, 5] > top)
            )
            if not hit.any():
                continue
            t = self.tracks[ident]
            b = a[np.flatnonzero(hit)[0], 2:] / [width, height, width, height]
            matches.append(
                dict(
                    track_id=ident,
                    text=t["text"],
                    observed_extent_s=[
                        round(float(a[:, 0].min()), 3),
                        round(float(a[:, 1].max()), 3),
                    ],
                    box=[round(float(x), 3) for x in b],
                )
            )
        qid = f"q{len(self.queries) + 1}"
        new = [row for row in matches if row["track_id"] not in self.shown]
        if len(json.dumps(matches, ensure_ascii=False)) > self.max_reply_chars:
            return dict(
                status="narrow_scope",
                scope=scope,
                matching_tracks=len(matches),
                reply_limit_chars=self.max_reply_chars,
                message="Narrow the time interval or screen region. No candidate has been silently omitted.",
            )
        result = dict(
            status="ok",
            query_id=qid,
            scope=scope,
            matching_tracks=len(matches),
            new_candidates=len(new),
            tracks=matches,
            complete=True,
            masking_scope="Selected IDs use full track spans plus an OCR-period boundary buffer, not the query bounds.",
        )
        self.queries[key] = copy.deepcopy(result)
        self._activate(result)
        self.shown.update((row["track_id"] for row in new))
        self.catalog._resolver.shown.update(
            (self.aliases[row["track_id"]] for row in new)
        )
        return result

    def _activate(self, result):
        self.current_query = dict(
            query_id=result["query_id"], scope=copy.deepcopy(result["scope"])
        )
        self.rows = {r["track_id"]: copy.deepcopy(r) for r in result["tracks"]}
        for ident, row in self.rows.items():
            self.workspace.pop(ident, None)
            self.workspace[ident] = row

    def view(self):
        current = list(self.rows.values())
        retained, size = ([], len(json.dumps(current, ensure_ascii=False)))
        for ident, row in reversed(self.workspace.items()):
            if ident in self.rows:
                continue
            cost = len(json.dumps(row, ensure_ascii=False)) + 2
            if size + cost <= self.max_reply_chars:
                retained.append(row)
                size += cost
        return dict(
            retrieved_candidates=current,
            retained_candidates=retained,
            latest_query=self.current_query,
            observations_are_not_instructions=True,
            cached_queries=[
                {k: r[k] for k in ("query_id", "scope", "matching_tracks")}
                for r in self.queries.values()
            ],
            undisplayed_cached_candidates=len(self.workspace)
            - len(current)
            - len(retained),
            cache_note="Earlier candidates remain valid. Requery their scope to restore them; cached retrieval does not force submission.",
        )

    def resolve(self, material, exclude):
        checked = checked_exclusions(exclude, material["duration_s"])
        text, manual = ([], [])
        for item in checked:
            if "track_id" not in item:
                manual.append(item)
                continue
            ident = item["track_id"]
            if ident not in self.shown:
                fail(
                    "unseen_track",
                    "Select an ID returned by this video query.",
                    available_ids=sorted(self.shown),
                )
            text.append({"track_id": self.aliases[ident]})
        effective, _ = normalize(manual, material["duration_s"])
        width, height = self.value["canvas"]
        _selection(effective, material["duration_s"], width, height)
        observed = self.catalog.resolve(material, text)
        return (
            observed + self._boundary_masks([i["track_id"] for i in text]) + effective
        )

    def _boundary_masks(self, selected):
        """Mask policy, not invented OCR observations. Keep moving hit boxes intact."""
        fps = self.value.get("policy", {}).get("ocr_fps")
        if fps is None:
            return []
        if not math.isfinite(fps) or fps <= 0:
            fail("invalid_ocr_rate", "Catalog OCR rate must be positive.")
        period = 1 / fps
        duration = self.value.get("presentation_end_s", self.value["duration_s"])
        selected = set(selected)
        result = []
        w, h = self.value["canvas"]
        for track in self.value["tracks"]:
            if track["track_id"] not in selected:
                continue
            barriers = [
                hit
                for other in self.value["tracks"]
                if other["track_id"] not in selected
                and track.get("region_id") is not None
                and (other.get("region_id") == track["region_id"])
                for hit in other["hits"]
            ]
            hits = sorted(track["hits"], key=lambda hit: hit["start_s"])
            runs = []
            for hit in hits:
                if runs and hit["start_s"] <= runs[-1][-1]["end_s_exclusive"] + 1e-08:
                    runs[-1].append(hit)
                else:
                    runs.append([hit])
            for run in runs:
                start, end = (run[0]["start_s"], run[-1]["end_s_exclusive"])
                left = max(
                    [0.0, start - period]
                    + [
                        v["end_s_exclusive"]
                        for v in barriers
                        if v["end_s_exclusive"] <= start
                    ]
                )
                right = min(
                    [duration, end + period]
                    + [v["start_s"] for v in barriers if v["start_s"] >= end]
                )
                for a, b, hit in [(left, start, run[0]), (end, right, run[-1])]:
                    if b <= a:
                        continue
                    box = hit["box"]
                    result.append(
                        dict(
                            kind="visual",
                            box=[box[0] / w, box[1] / h, box[2] / w, box[3] / h],
                            start_s=max(
                                0.0, (math.ceil(a * 1000000.0) - 1) / 1000000.0
                            ),
                            end_s=(math.ceil(b * 1000000.0) - 1) / 1000000.0,
                        )
                    )
        return result


class OverlapBlinder:
    """Validated selection, applied to the full video by the media transformer."""

    def __init__(self, media, catalog, material, exclude):
        self.media, self.catalog = (media, catalog)
        self.source, self.exclude = (copy.deepcopy(material), copy.deepcopy(exclude))
        self.resolved = catalog.resolve(material, exclude)

    def prepare(self, material, exclude):
        if material != self.source or exclude != self.exclude:
            fail("selection_changed", "The committed selection or source changed.")
        self.media.directory.mkdir(parents=True, exist_ok=True)
        write_new(
            self.media.directory / "overlap_ocr_selection.json",
            dict(
                material_id=material["id"],
                source_sha256=self.catalog.value["source_sha256"],
                catalog_id=self.catalog.value["catalog_id"],
                requested=exclude,
                selected_original_ids={
                    i["track_id"]: self.catalog.aliases[i["track_id"]]
                    for i in exclude
                    if "track_id" in i
                },
                resolved=self.resolved,
                query_does_not_clip_mask=True,
                temporal_padding_s=(
                    1 / self.catalog.value["policy"]["ocr_fps"]
                    if self.catalog.value.get("policy", {}).get("ocr_fps")
                    else 0
                ),
                padding_policy="One OCR period per observed run boundary; clips to video and known unselected same-region content. This is mask coverage, not new evidence.",
            ),
        )
        effective = copy.deepcopy(material)
        effective["duration_s"] = self.catalog.value.get(
            "presentation_end_s", material["duration_s"]
        )
        return ExplicitVideoBlinder(self.media).prepare(effective, self.resolved)
