"""Independent consistency check over the video and question."""

import copy

from dear.protocol import ToolExecutionError, action, fail

ROLE = "You are a video evidence counterexaminer. Investigate whether accounts in the supplied video disagree about the fact asked in the question. Your deliverable is the strongest observable conflict witnesses, not a response to the question or a story reconciling the accounts. For each witness, locate both accounts and state their incompatible claims about the same subject and event. Distinguish a witnessed disagreement from missing information; leave unresolved correspondence explicit. Submit through report_conflict_witnesses."
_ACCOUNT = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "enum": ["scene", "on_screen_text", "speech"]},
        "reference": {
            "type": "string",
            "description": "Video time and the relevant subject or screen region.",
        },
        "content": {
            "type": "string",
            "description": "The observed event or exact source statement.",
        },
    },
    "required": ["source", "reference", "content"],
    "additionalProperties": False,
}
TOOL = {
    "type": "function",
    "function": {
        "name": "report_conflict_witnesses",
        "description": "Report witnessed factual disagreements. Empty witnesses means none found, not that every account is verified. Submission ends the investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "witnesses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "The question-relevant fact in dispute.",
                            },
                            "account_a": copy.deepcopy(_ACCOUNT),
                            "account_b": copy.deepcopy(_ACCOUNT),
                            "disagreement": {
                                "type": "string",
                                "description": "What the two accounts assert incompatibly; an information gap alone is not a disagreement.",
                            },
                        },
                        "required": ["fact", "account_a", "account_b", "disagreement"],
                        "additionalProperties": False,
                    },
                },
                "unresolved": {
                    "type": "string",
                    "description": "Remaining visibility, subject, time or interpretation uncertainty; empty if none.",
                },
            },
            "required": ["witnesses", "unresolved"],
            "additionalProperties": False,
        },
    },
}


def messages(question, video_url):
    return [
        dict(role="system", content=ROLE),
        dict(
            role="user",
            content=[
                dict(
                    type="text",
                    text="Question (scope only; do not answer):\n" + question,
                ),
                dict(type="video_url", video_url={"url": video_url}),
            ],
        ),
    ]


def validate(args):
    if (
        not isinstance(args, dict)
        or set(args) != {"witnesses", "unresolved"}
        or (not isinstance(args["witnesses"], list))
        or (not isinstance(args["unresolved"], str))
    ):
        fail("invalid_witness_report", "Submit witnesses and unresolved.")
    for w in args["witnesses"]:
        if (
            not isinstance(w, dict)
            or set(w) != {"fact", "account_a", "account_b", "disagreement"}
            or any((not isinstance(w[k], str) for k in ("fact", "disagreement")))
        ):
            fail(
                "invalid_witness_report",
                "Each witness needs a fact, two accounts and their disagreement.",
            )
        for k in ("account_a", "account_b"):
            a = w[k]
            if (
                not isinstance(a, dict)
                or set(a) != {"source", "reference", "content"}
                or a["source"] not in ("scene", "on_screen_text", "speech")
                or (not isinstance(a["reference"], str))
                or (not isinstance(a["content"], str))
            ):
                fail(
                    "invalid_witness_report",
                    "Each account needs its source, reference and content.",
                )
    return copy.deepcopy(args)


def investigate(job, ask, *, material_id):
    request = messages(job["question"], job["video_url"])
    for attempt in range(2):
        response, meta = ask(
            copy.deepcopy(request),
            tools=[copy.deepcopy(TOOL)],
            kind="conflict_witness",
            material_ids=[material_id],
        )
        try:
            _, args = action(response, meta, {"report_conflict_witnesses"})
            return validate(args)
        except ToolExecutionError:
            if (
                attempt
                or meta.get("finish_reason") != "stop"
                or response.get("tool_calls")
            ):
                raise
            request.append(
                dict(
                    role="user",
                    content="No tool call was received. Submit the public report through report_conflict_witnesses.",
                )
            )
    raise AssertionError("unreachable")
