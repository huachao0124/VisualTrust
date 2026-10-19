"""Terminal reader with original QA and selected media only."""

import copy
import json
import math

from dear.media import ExistingVideoBlinder
from dear.protocol import ANSWER, ToolExecutionError, answer_prompt, text_block

TOOL = {
    "type": "function",
    "function": {
        "name": "blind_answer",
        "description": "Finish by excluding the evidence you specify, then obtain one independent answer from the remaining video and original QA. No answer, rationale, transcript or prior report is sent to the reader. Exclusions may target any visual content, not only text, or the whole audio track. No modality is assumed trustworthy. Empty exclude means no new blinding. This tool ends the task; do not synthesize afterward.",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {
                    "type": "string",
                    "description": "One existing video material ID; never a path or URL.",
                },
                "exclude": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["visual", "audio"]},
                            "box": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "Visual only: normalized LTRB rectangle; omitted means the whole frame.",
                            },
                            "start_s": {
                                "type": "number",
                                "description": "Visual only: start in the selected video's clock.",
                            },
                            "end_s": {
                                "type": "number",
                                "description": "Visual only: inclusive end. Omit both times for the whole video.",
                            },
                        },
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                    "description": "Visual regions/time intervals or the whole audio track. Partial audio intervals are not supported by this adapter.",
                },
            },
            "required": ["material_id", "exclude"],
            "additionalProperties": False,
        },
    },
}


class BlindAnswerFinished(BaseException):
    """Host catches outside its tool loop; never append this as a tool result."""

    def __init__(self, result):
        self.result = result
        super().__init__(result["status"])


def reader_messages(qa, video_url, availability):
    """The shared public input for Direct and terminal independent readings."""
    return [
        {
            "role": "user",
            "content": [
                text_block(ANSWER),
                text_block(answer_prompt(qa["question"], qa["options"])),
                text_block(
                    json.dumps(
                        {
                            "evidence_availability": availability,
                            "scope": "Withheld regions or streams are unavailable evidence, not evidence that an event or object was absent.",
                        }
                    )
                ),
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _selection(exclude, duration, width, height):
    """Validate geometry, not source credibility, semantics or answer support."""
    if not isinstance(exclude, list):
        raise ToolExecutionError("invalid_exclusions", "exclude must be a list.")
    tracks, audio = ([], False)
    for item in exclude:
        if not isinstance(item, dict) or "kind" not in item:
            raise ToolExecutionError(
                "invalid_exclusions", "Each exclusion needs a kind."
            )
        if item["kind"] == "audio":
            if set(item) != {"kind"}:
                raise ToolExecutionError(
                    "unsupported_audio_interval",
                    "This adapter removes the whole audio track only; no partial success.",
                )
            audio = True
            continue
        if item["kind"] != "visual" or set(item) - {"kind", "box", "start_s", "end_s"}:
            raise ToolExecutionError(
                "unsupported_exclusion",
                "Use supported media coordinates, not an asserted answer or claim text.",
            )
        if ("start_s" in item) != ("end_s" in item):
            raise ToolExecutionError(
                "invalid_interval", "Supply both interval endpoints, or neither."
            )
        start, end = (item.get("start_s", 0), item.get("end_s", duration))
        if not (_number(start) and _number(end) and (0 <= start < end <= duration)):
            raise ToolExecutionError(
                "invalid_interval",
                "Interval must lie inside the selected video; no clamping.",
            )
        box = item.get("box", [0, 0, 1, 1])
        if (
            not isinstance(box, list)
            or len(box) != 4
            or (not all((_number(x) for x in box)))
        ):
            raise ToolExecutionError(
                "invalid_box", "Use four finite normalized LTRB coordinates."
            )
        l, t, r, b = box
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            raise ToolExecutionError(
                "invalid_box", "Rectangle must lie inside the selected video."
            )
        track = {
            "start_s": start,
            "end_s": end,
            "box": [
                math.floor(l * width),
                math.floor(t * height),
                math.ceil(r * width),
                math.ceil(b * height),
            ],
        }
        if track not in tracks:
            tracks.append(track)
    return (tracks, audio)


class BlindAnswer:
    """Callable terminal tool. Original QA is host-bound, not model-supplied."""

    def __init__(self, workspace, *, ask, media, record, blinder=None):
        self.workspace, self.ask, self.media = (workspace, ask, media)
        self.record = record
        self.blinder = blinder or ExistingVideoBlinder(media)
        self.qa = copy.deepcopy({k: workspace.job[k] for k in ("question", "options")})
        self.finished = False

    def __call__(self, arguments):
        if self.finished:
            raise ToolExecutionError(
                "terminal_already_finished",
                "No second answer or reconciliation after blind_answer.",
            )
        if not isinstance(arguments, dict) or set(arguments) != {
            "material_id",
            "exclude",
        }:
            raise ToolExecutionError(
                "invalid_arguments",
                "Use material_id and exclude only; no rationale, answer or modified question.",
            )
        if self.qa != {k: self.workspace.job[k] for k in self.qa}:
            raise ToolExecutionError("question_changed", "Original QA changed.")
        material = self.workspace.material(arguments["material_id"])
        if material["kind"] != "video":
            raise ToolExecutionError(
                "unsupported_media",
                "This adapter reads one existing video, not a new frame packet.",
            )
        self.finished = True
        result = {
            "tool": "blind_answer",
            "terminal": True,
            "status": "preparation_error",
            "response": None,
            "final_started": False,
        }
        try:
            receipt, flags = self.blinder.prepare(
                copy.deepcopy(material), copy.deepcopy(arguments["exclude"])
            )
            delivered = material["id"]
            if receipt["video_url"] != material["receipt"]["video_url"]:
                delivered = self.workspace._register(
                    "video",
                    receipt=receipt,
                    parents=[material["id"]],
                    duration_s=material["duration_s"],
                    transformed=True,
                )
            result.update(
                source_material_id=material["id"], final_material_ids=[delivered]
            )
            availability = {
                k: flags[k]
                for k in ("visual_withheld", "audio_withheld", "prior_intervention")
            }
            if not all((type(v) is bool for v in availability.values())):
                raise ToolExecutionError(
                    "invalid_blinding_receipt",
                    "Only boolean availability flags may enter the final read.",
                )
            messages = reader_messages(self.qa, receipt["video_url"], availability)
            self.record(
                {
                    "event": "independent_request",
                    "kind": "independent_answer",
                    "tool": "blind_answer",
                    "material_ids": [delivered],
                    "messages": copy.deepcopy(messages),
                }
            )
            result["status"] = "technical_error"
            result["final_started"] = True
            final, meta = self.ask(
                messages,
                tools=[],
                kind="independent_answer",
                material_ids=[delivered],
            )
            if meta.get("finish_reason") != "stop" or final.get("tool_calls"):
                result["status"] = "incomplete_response"
            elif (
                not isinstance(final.get("content"), str)
                or not final["content"].strip()
            ):
                result["status"] = "empty_response"
            else:
                result.update(
                    status="completed",
                    response=final["content"],
                    reasoning=final.get("reasoning_content", ""),
                )
        except Exception as error:
            result["error"] = getattr(error, "code", type(error).__name__)
        try:
            self.record(
                {"event": "blind_answer_terminal", "result": copy.deepcopy(result)}
            )
        except Exception as error:
            result["terminal_record_error"] = type(error).__name__
        raise BlindAnswerFinished(result)
