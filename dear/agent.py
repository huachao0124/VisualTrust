"""DEAR: consistency check, evidence selection, and isolated reading."""

import copy
import json

from dear.detector import validate
from dear.protocol import (
    SELECTOR_ROLE,
    ToolExecutionError,
    action,
    fail,
    signature,
    text_block,
)
from dear.reader import BlindAnswerFinished
from dear.selection import BLIND_TOOL, OverlapBlinder, OverlapCatalog, query_scope

VERSION = "dear_fixed_grid"
QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "list_text_tracks",
        "description": "Query cached OCR by overlap. Return complete text and short IDs for tracks intersecting the requested time AND screen region. Querying does not mask or crop anything.",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "start_s": {
                    "type": "number",
                    "description": "Start in seconds; provide with end_s. Omit both for all times.",
                },
                "end_s": {
                    "type": "number",
                    "description": "End in seconds, capped at video end.",
                },
                "box": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                    "description": "Normalized [left,top,right,bottom]. Any positive overlap matches, not full containment. Omit for full frame.",
                },
            },
            "required": ["material_id"],
            "additionalProperties": False,
        },
    },
}
QUERY_STEP = "Current task: call list_text_tracks with a time interval, a screen region, or both, to locate evidence relevant to the conflict report. Submit a tool call, not a QA response."
SELECT_STEP = "Current task: use the returned IDs to call blind_answer with your exclusions. If a different location needs checking, call list_text_tracks with a different scope. An empty exclusion list is allowed. Select text by ID; use explicit non_text_visual regions or whole_frame, or the whole audio track, for other content. Queries only locate candidates; chosen IDs mask their tracked spans with an OCR-period boundary buffer. Submit a tool call, not a QA response."


class DEARAgent:

    def __init__(self, terminal, provider, report, *, max_reply_chars=24000):
        self.terminal, self.provider = (terminal, provider)
        self.max_reply_chars = max_reply_chars
        self.stage, self.finished = ("not_started", False)
        self.material_id = terminal.workspace.original
        self.qa = copy.deepcopy(terminal.qa)
        self.material = terminal.workspace.material(self.material_id)
        self.evidence = validate(report)
        self.evidence_hash = signature(self.evidence)
        self.conflict = bool(self.evidence["witnesses"])

    def record(self, event):
        self.terminal.record(copy.deepcopy(event))

    def guard(self):
        t = self.terminal
        if t.qa != self.qa or self.qa != {k: t.workspace.job[k] for k in self.qa}:
            fail("question_changed", "Original QA changed.")
        if t.workspace.material(self.material_id) != self.material:
            fail("material_changed", "Original video material changed.")
        if self.evidence is not None and signature(self.evidence) != self.evidence_hash:
            fail("evidence_changed", "Committed observations changed.")

    def prefix(self, role, *, include_record=False):
        self.guard()
        messages = self.terminal.workspace.messages(self.material_id, role)
        if include_record:
            messages[-1]["content"].append(
                text_block(
                    "Conflict report:\n" + json.dumps(self.evidence, ensure_ascii=False)
                )
            )
        return messages

    def direct_answer(self):
        """No-conflict path: the reader receives the original video, without a report, OCR or new media."""
        self.guard()
        self.stage = "direct_answer"
        self.record(
            dict(
                event="automatic_tool_dispatch",
                tool="answer",
                reason="no_conflict",
                context="direct",
            )
        )
        try:
            self.terminal(dict(material_id=self.material_id, exclude=[]))
        except BlindAnswerFinished as ended:
            return {**ended.result, "answer_context": "direct"}
        raise RuntimeError("Expected terminal answer signal")

    def select(self):
        catalog, feedback = (None, None)
        stalled = set()
        while True:
            self.stage = (
                "ocr_query"
                if catalog is None or not catalog.queries
                else "evidence_selection"
            )
            queried = catalog is not None and bool(catalog.queries)
            step = SELECT_STEP if queried else QUERY_STEP
            messages = self.prefix(SELECTOR_ROLE, include_record=True)
            messages[-1]["content"].append(text_block(step))
            if queried:
                messages[-1]["content"].append(
                    text_block(json.dumps(catalog.view(), ensure_ascii=False))
                )
            if feedback:
                messages[-1]["content"].append(
                    text_block(json.dumps(feedback, ensure_ascii=False))
                )
            blind = copy.deepcopy(BLIND_TOOL)
            if queried:
                text_schema = blind["function"]["parameters"]["properties"]["exclude"][
                    "items"
                ]["anyOf"][0]
                text_schema["properties"]["track_id"]["pattern"] = "^t[1-9][0-9]*$"
            tools = [QUERY_TOOL, blind] if queried else [QUERY_TOOL]
            self.guard()
            self.record(
                dict(
                    event="scoped_action_request",
                    stage=self.stage,
                    messages=messages,
                    tools=tools,
                )
            )
            message, meta = self.terminal.ask(
                copy.deepcopy(messages),
                tools=copy.deepcopy(tools),
                kind="scoped_ocr_action",
                material_ids=[self.material_id],
            )
            self.guard()
            try:
                call, args = action(
                    message, meta, {t["function"]["name"] for t in tools}
                )
                name = call["function"]["name"]
                if name == "list_text_tracks":
                    query_scope(args, self.material_id, self.material["duration_s"])
                    if catalog is None:
                        catalog = OverlapCatalog(
                            self.provider(copy.deepcopy(self.material)),
                            max_reply_chars=self.max_reply_chars,
                        )
                    response = catalog.query(args)
                    self.record(
                        dict(event="scoped_ocr_query", arguments=args, result=response)
                    )
                    feedback = {k: v for k, v in response.items() if k != "tracks"}
                    if response.get("new_candidates", 0):
                        stalled.clear()
                    else:
                        key = signature(
                            dict(action=name, arguments=args, status=response["status"])
                        )
                        if key in stalled:
                            return dict(
                                status="pending_protocol",
                                response=None,
                                stage=self.stage,
                                error="no_progress",
                                retryable=True,
                                terminal=False,
                            )
                        stalled.add(key)
                    continue
                if (
                    set(args) != {"material_id", "exclude"}
                    or args["material_id"] != self.material_id
                ):
                    raise ToolExecutionError(
                        "invalid_selection", "Use this material_id and exclude only."
                    )
                prepared = OverlapBlinder(
                    self.terminal.media, catalog, self.material, args["exclude"]
                )
            except (ToolExecutionError, json.JSONDecodeError) as error:
                feedback = dict(
                    error=getattr(error, "code", "invalid_json"),
                    message=str(error),
                    next_step=step,
                    details=getattr(error, "details", {}),
                )
                self.record(
                    dict(
                        event="scoped_action_rejected",
                        feedback=feedback,
                        message=message,
                        meta=meta,
                    )
                )
                key = signature(
                    dict(
                        error=feedback["error"],
                        tool_calls=[
                            c.get("function") for c in message.get("tool_calls") or []
                        ],
                    )
                )
                if key in stalled:
                    return dict(
                        status="pending_protocol",
                        response=None,
                        stage=self.stage,
                        error=feedback["error"],
                        retryable=True,
                        terminal=False,
                    )
                stalled.add(key)
                continue
            self.record(
                dict(event="evidence_selection", arguments=args, interface=VERSION)
            )
            self.stage = "independent_answer"
            terminal = self.terminal
            before = terminal.blinder
            terminal.blinder = prepared
            try:
                terminal(args)
            except BlindAnswerFinished as ended:
                return {
                    **ended.result,
                    "answer_context": "independent",
                    "selection_interface": VERSION,
                }
            finally:
                terminal.blinder = before

    def run(self):
        if self.finished or self.terminal.finished or self.stage != "not_started":
            raise ToolExecutionError(
                "terminal_already_finished", "Run this continuation once."
            )
        self.record(
            dict(
                event="conflict_report",
                report=self.evidence,
                sha256=self.evidence_hash,
                conflict=self.conflict,
            )
        )
        try:
            result = self.select() if self.conflict else self.direct_answer()
        except Exception as error:
            result = dict(
                status="pending_execution",
                response=None,
                stage=self.stage,
                error=getattr(error, "code", type(error).__name__),
                retryable=True,
                terminal=False,
            )
        self.finished = self.terminal.finished = True
        result.update(
            coordinator_protocol=VERSION,
            conflict=self.conflict,
            evidence_sha256=self.evidence_hash,
        )
        self.record(dict(event="scoped_agent_terminal", result=result))
        return result
