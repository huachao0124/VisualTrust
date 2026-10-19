"""Fixed-grid OCR with optical-flow tracking between observations."""

import copy
import hashlib
import json
import math
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from dear.protocol import ToolExecutionError


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(code, message):
    raise ToolExecutionError(code, message)


class Catalog:

    def __init__(self, value):
        self.value = copy.deepcopy(value)
        self.known = {t["track_id"]: t for t in value["tracks"]}
        if len(self.known) != len(value["tracks"]):
            fail("duplicate_id", "Duplicate track IDs.")
        self.shown = set()

    def resolve(self, material, exclude):
        if (
            material["id"] != self.value["material_id"]
            or material["duration_s"] != self.value["duration_s"]
        ):
            fail("wrong_material", "Track IDs belong to a different video material.")
        if sha_file(material["receipt"]["video"]) != self.value["source_sha256"]:
            fail("source_changed", "Catalog source bytes changed.")
        if not isinstance(exclude, list):
            fail("invalid_exclusions", "Expected a list.")
        result = []
        w, h = self.value["canvas"]
        for item in exclude:
            if not isinstance(item, dict):
                fail("invalid_exclusions", "Expected exclusion objects.")
            if "track_id" not in item:
                result.append(copy.deepcopy(item))
                continue
            if set(item) != {"track_id"} or item["track_id"] not in self.shown:
                fail(
                    "unseen_track",
                    "Choose a displayed track_id alone, without coordinate overrides.",
                )
            for hit in self.known[item["track_id"]]["hits"]:
                start = max(0, (math.ceil(hit["start_s"] * 1000000.0) - 1) / 1000000.0)
                end = (math.ceil(hit["end_s_exclusive"] * 1000000.0) - 1) / 1000000.0
                if end <= hit["start_s"]:
                    fail(
                        "unsupported_frame_rate",
                        "Frame interval below renderer precision.",
                    )
                box = [
                    hit["box"][0] / w,
                    hit["box"][1] / h,
                    hit["box"][2] / w,
                    hit["box"][3] / h,
                ]
                if (
                    result
                    and result[-1].get("box") == box
                    and (abs(result[-1]["end_s"] + 1e-06 - hit["start_s"]) < 1e-06)
                ):
                    result[-1]["end_s"] = end
                else:
                    result.append(
                        {"kind": "visual", "box": box, "start_s": start, "end_s": end}
                    )
        return result


VERSION = "fixed_grid_ocr"


@dataclass(frozen=True)
class Settings:
    ocr_fps: float = 2.0
    max_gap_s: float = 1.5
    min_match: float = 0.58
    min_appearance: float = 0.72
    flow_error_px: float = 1.5
    change_confirmations: int = 2
    cut_delta: float = 0.35

    def __post_init__(self):
        if not (
            math.isfinite(self.ocr_fps)
            and self.ocr_fps > 0
            and math.isfinite(self.max_gap_s)
            and (self.max_gap_s >= 0)
            and math.isfinite(self.flow_error_px)
            and (self.flow_error_px > 0)
        ):
            raise ValueError(
                "Positive FPS/flow error and finite nonnegative gap required"
            )
        if not all(
            (0 < x <= 1 for x in (self.min_match, self.min_appearance, self.cut_delta))
        ):
            raise ValueError("Similarity thresholds must lie in (0, 1]")
        if type(self.change_confirmations) is not int or self.change_confirmations < 2:
            raise ValueError("Content changes require at least two OCR anchors")


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_similarity(a, b):
    a, b = (text_key(a), text_key(b))
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def crop(gray, box):
    h, w = gray.shape
    l, t, r, b = np.rint(np.clip(box, [0, 0, 0, 0], [w, h, w, h])).astype(int)
    return gray[t:b, l:r]


def appearance(a, b):
    """Containment-aware patch comparison: a partially visible word may match."""
    if not a.size or not b.size or min(a.std(), b.std()) < 1:
        return 0.0

    def scale(x):
        return cv2.resize(x, (max(2, round(x.shape[1] * 24 / x.shape[0])), 24))

    a, b = (scale(a), scale(b))
    if a.shape[1] < b.shape[1]:
        a, b = (b, a)
    return float(np.clip(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max(), 0, 1))


def geometry(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    wh_a, wh_b = (a[2:] - a[:2], b[2:] - b[:2])
    distance = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
    scale = max(np.linalg.norm(wh_a), np.linalg.norm(wh_b), 1)
    area_ratio = min(np.prod(wh_a), np.prod(wh_b)) / max(np.prod(wh_a), np.prod(wh_b))
    return float(math.exp(-2 * distance / scale) * math.sqrt(area_ratio))


def flow_box(previous, current, box, tolerance):
    """Forward/backward LK consistency and robust translation, no extrapolation."""
    mask = np.zeros(previous.shape, np.uint8)
    h, w = mask.shape
    l, t, r, b = np.rint(np.clip(box, [0, 0, 0, 0], [w, h, w, h])).astype(int)
    mask[t:b, l:r] = 255
    points = cv2.goodFeaturesToTrack(previous, 60, 0.02, 3, mask=mask)
    if points is None or len(points) < 4:
        return None
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if moved is None:
        return None
    back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
    if back is None:
        return None
    good = ok.ravel().astype(bool) & reverse_ok.ravel().astype(bool)
    good &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) <= tolerance
    delta = (moved - points).reshape(-1, 2)[good]
    if len(delta) < 4:
        return None
    translation = np.median(delta, axis=0)
    if (np.linalg.norm(delta - translation, axis=1) <= tolerance).mean() < 0.6:
        return None
    return np.asarray(box) + np.tile(translation, 2)


@dataclass
class Region:
    id: str
    box: np.ndarray
    template: np.ndarray
    last_seen: float
    visible: bool = True
    segments: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    pending_text: str = ""
    pending_time: float = 0.0
    pending_count: int = 0

    @property
    def text(self):
        return self.votes.most_common(1)[0][0] if self.votes else ""


class TextTracker:

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.regions = []
        self.active = []
        self.previous = None
        self.time = None
        self.frame_index = -1
        self.pending_hits = []
        self.closed = False

    def _segment(self, region):
        segment = dict(region_id=region.id, hits=[], observations=[])
        region.segments.append(segment)
        return segment

    def _observe(self, region, detection, gray, time_s, visual_match):
        text = detection["text"]
        old_text = region.text
        box = np.asarray(detection["box"], dtype=float)
        patch = crop(gray, box)
        key, old_key = (text_key(text), text_key(old_text))
        narrower = (
            patch.shape[1] / patch.shape[0]
            < 0.9 * region.template.shape[1] / region.template.shape[0]
        )
        partial = (
            narrower
            and key
            and (key in old_key)
            and (visual_match >= self.settings.min_appearance)
        )
        changed = bool(
            old_key and key != old_key and (not partial) and (visual_match < 0.995)
        )
        if changed:
            if key == region.pending_text:
                region.pending_count += 1
            else:
                region.pending_text, region.pending_time, region.pending_count = (
                    key,
                    time_s,
                    1,
                )
            if region.pending_count >= self.settings.change_confirmations:
                before = region.segments[-1]
                after = self._segment(region)
                for field_name, clock in [
                    ("hits", "start_s"),
                    ("observations", "time_s"),
                ]:
                    after[field_name] = [
                        v for v in before[field_name] if v[clock] >= region.pending_time
                    ]
                    before[field_name] = [
                        v for v in before[field_name] if v[clock] < region.pending_time
                    ]
                region.votes.clear()
                for o in after["observations"]:
                    if not o["partial"]:
                        region.votes[o["text"]] += o["confidence"]
                region.pending_count = 0
        else:
            region.pending_count = 0
            region.pending_text = ""
        observation = dict(
            time_s=time_s,
            text=text,
            confidence=float(detection.get("confidence", 1)),
            box=box.tolist(),
            partial=bool(partial),
        )
        region.segments[-1]["observations"].append(observation)
        if not partial and (not region.pending_count):
            region.votes[text] += observation["confidence"]
            region.template = patch.copy()
        region.box, region.last_seen, region.visible = (box, time_s, True)

    def update(self, rgb, time_s, detections=None, *, scene_cut=False):
        """detections=None: flow-only frame; []: OCR ran and returned no regions."""
        if (
            self.closed
            or not math.isfinite(time_s)
            or time_s < 0
            or (self.time is not None and time_s < self.time)
        ):
            raise ValueError("Feed nondecreasing source timestamps before finish()")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB frame")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.previous is not None and gray.shape != self.previous.shape:
            raise ValueError("Canvas changed")
        if detections is not None:
            h, w = gray.shape
            for d in detections:
                box = d["box"]
                if (
                    not isinstance(d["text"], str)
                    or not d["text"].strip()
                    or len(box) != 4
                    or (not all((math.isfinite(x) for x in box)))
                ):
                    raise ValueError("Malformed OCR detection")
                if (
                    not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)
                    or not crop(gray, box).size
                ):
                    raise ValueError("OCR box outside canvas or below pixel resolution")
                if not 0 <= float(d.get("confidence", 1)) <= 1:
                    raise ValueError("OCR confidence outside [0, 1]")
        if self.time is None or time_s > self.time:
            for hit in self.pending_hits:
                hit["end_s_exclusive"] = time_s
            self.pending_hits = []
        self.frame_index += 1
        if self.previous is not None:
            delta = (
                np.abs(
                    cv2.resize(gray, (64, 36)).astype(float)
                    - cv2.resize(self.previous, (64, 36))
                ).mean()
                / 255
            )
            scene_cut = scene_cut or delta > self.settings.cut_delta
        if scene_cut:
            self.active.clear()
        self.active = [
            r for r in self.active if time_s - r.last_seen <= self.settings.max_gap_s
        ]
        predictions = {}
        for region in self.active:
            region.visible = False
            if self.previous is not None and region.last_seen == self.time:
                predicted = flow_box(
                    self.previous, gray, region.box, self.settings.flow_error_px
                )
                if (
                    predicted is not None
                    and appearance(region.template, crop(gray, predicted))
                    >= self.settings.min_appearance
                ):
                    predictions[region.id] = predicted
        if detections is not None:
            scores = np.zeros((len(self.active), len(detections)))
            visuals = np.zeros_like(scores)
            patches = [crop(gray, d["box"]) for d in detections]
            for i, region in enumerate(self.active):
                expected = predictions.get(region.id, region.box)
                for j, d in enumerate(detections):
                    spatial = geometry(expected, d["box"])
                    visual = appearance(region.template, patches[j])
                    textual = text_similarity(region.text, d["text"])
                    if spatial >= 0.15 and textual >= 0.35 and (visual >= 0.3):
                        scores[i, j] = 0.5 * textual + 0.3 * visual + 0.2 * spatial
                    visuals[i, j] = visual
            allowed = np.where(scores >= self.settings.min_match, scores, 0.0)
            pairs = zip(*linear_sum_assignment(-allowed)) if scores.size else []
            matched = set()
            for i, j in pairs:
                if allowed[i, j] == 0:
                    continue
                self._observe(
                    self.active[i], detections[j], gray, time_s, visuals[i, j]
                )
                matched.add(j)
            for j, d in enumerate(detections):
                if j in matched:
                    continue
                region = Region(
                    f"region_{len(self.regions):04d}",
                    np.asarray(d["box"], dtype=float),
                    crop(gray, d["box"]).copy(),
                    time_s,
                )
                self._segment(region)
                self._observe(region, d, gray, time_s, 1.0)
                self.regions.append(region)
                self.active.append(region)
        else:
            for region in self.active:
                if region.id in predictions:
                    region.box, region.last_seen, region.visible = (
                        predictions[region.id],
                        time_s,
                        True,
                    )
        for region in self.active:
            if not region.visible:
                continue
            h, w = gray.shape
            box = np.clip(region.box, [0, 0, 0, 0], [w, h, w, h])
            if box[2] <= box[0] or box[3] <= box[1]:
                region.visible = False
                continue
            hit = dict(
                frame_index=self.frame_index,
                start_s=time_s,
                end_s_exclusive=None,
                box=box.tolist(),
                localization="ocr" if detections is not None else "flow",
            )
            region.segments[-1]["hits"].append(hit)
            self.pending_hits.append(hit)
        self.previous, self.time = (gray, time_s)

    def finish(self, duration_s):
        if (
            self.closed
            or self.time is None
            or (not math.isfinite(duration_s))
            or (duration_s <= self.time)
        ):
            raise ValueError("Finish once, with the source end after the last frame")
        for hit in self.pending_hits:
            hit["end_s_exclusive"] = duration_s
        self.closed = True
        for region in self.regions:
            region.segments[-1]["unconfirmed_text_change"] = bool(region.pending_count)
        return [s for r in self.regions for s in r.segments if s["hits"]]


class TrackedCatalog:

    def __init__(self, value):
        self._resolver = Catalog(value)
        self.value = self._resolver.value

    def resolve(self, material, exclude):
        return self._resolver.resolve(material, exclude)


def catalog_from_tracker(tracker, duration_s, material_id, source_sha256):
    segments = tracker.finish(duration_s)
    h, w = tracker.previous.shape
    binding = dict(
        material_id=material_id,
        source_sha256=source_sha256,
        duration_s=duration_s,
        canvas=[w, h],
        policy=dict(version=VERSION, **asdict(tracker.settings)),
    )
    tracks = []
    for segment in segments:
        votes = Counter()
        variants = list(dict.fromkeys((o["text"] for o in segment["observations"])))
        for o in segment["observations"]:
            if not o["partial"]:
                votes[o["text"]] += o["confidence"]
        text = max(variants, key=lambda s: (votes[s], len(s)))
        tracks.append(
            dict(
                **segment,
                text=text,
                variants=variants,
                track_id="text_" + digest(dict(binding=binding, segment=segment))[:24],
            )
        )
    value = dict(
        **binding,
        tracks=tracks,
        decoded_frames=tracker.frame_index + 1,
        coverage_certified=False,
    )
    value["catalog_id"] = digest(value)
    return TrackedCatalog(value)


def scan_frames(
    frames, read_ocr, *, material_id, source_sha256, duration_s, settings=None
):
    """frames yields (source seconds, RGB); read_ocr is existing worker.read.

    OCR runs only on the fixed grid. Tracking failures and scene cuts wait
    for the next scheduled observation.
    """
    tracker = TextTracker(settings)
    ocr_times = []
    next_ocr = 0.0
    for time_s, rgb in frames:
        due = time_s >= next_ocr
        detections = None
        if due:
            result = read_ocr(rgb)
            if "error" in result:
                raise RuntimeError(result["error"])
            detections = result["regions"]
            ocr_times.append(time_s)
            next_ocr = (
                math.floor(time_s * tracker.settings.ocr_fps) + 1
            ) / tracker.settings.ocr_fps
        tracker.update(rgb, time_s, detections)
    catalog = catalog_from_tracker(tracker, duration_s, material_id, source_sha256)
    catalog.value["ocr_times_s"] = ocr_times
    catalog.value["catalog_id"] = digest(
        {k: v for k, v in catalog.value.items() if k != "catalog_id"}
    )
    return catalog


def scan_video(material, read_ocr, *, settings=None):
    """Optional PyAV adapter. Model/video/OCR calls are supplied, never created."""
    import av

    path = material["receipt"]["video"]
    before = sha_file(path)
    with av.open(path) as container:
        if (
            len(container.streams.video) != 1
            or int(container.streams.video[0].metadata.get("rotate", 0)) % 360
        ):
            raise ValueError("Use a single video stream with unrotated OCR coordinates")
        frames = (
            (float(f.time), f.to_ndarray(format="rgb24"))
            for f in container.decode(video=0)
        )
        catalog = scan_frames(
            frames,
            read_ocr,
            material_id=material["id"],
            source_sha256=before,
            duration_s=material["duration_s"],
            settings=settings,
        )
    if sha_file(path) != before:
        raise ValueError("Video changed during tracking")
    return catalog


class TrackedCatalogProvider:
    """Lazy adapter for the agent; inject an existing OCR worker's read method."""

    def __init__(self, read_ocr, *, settings=None):
        self.read_ocr, self.settings = (read_ocr, settings)
        self.catalog = None

    def __call__(self, material):
        if self.catalog is None:
            self.catalog = scan_video(material, self.read_ocr, settings=self.settings)
        elif (
            material["id"] != self.catalog.value["material_id"]
            or material["duration_s"] != self.catalog.value["duration_s"]
            or sha_file(material["receipt"]["video"])
            != self.catalog.value["source_sha256"]
        ):
            raise ValueError("Provider is bound to one source material")
        return self.catalog
