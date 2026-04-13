from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr

from endless_context.agent import AnchorState, ConversationSnapshot
from endless_context.channel import GradioChannel, _format_error_payload_data


@dataclass(frozen=True)
class _Entry:
    id: int
    kind: str
    payload: dict[str, Any]
    meta: dict[str, Any]


def _build_snapshot(view_mode: str, anchor_name: str | None) -> ConversationSnapshot:
    entries = [
        _Entry(1, "message", {"role": "user", "content": "hi"}, {}),
        _Entry(2, "anchor", {"name": "handoff:phase-1", "state": {"phase": "Phase 1"}}, {}),
        _Entry(3, "message", {"role": "assistant", "content": "ok"}, {}),
    ]
    anchors = [
        AnchorState(
            entry_id=2,
            name="handoff:phase-1",
            label="Phase 1",
            summary="checkpoint",
            facts=["f1"],
            created_at="2026-02-10T00:00:00",
        )
    ]
    if view_mode == "full":
        active_anchor = None
        context_entries = entries
    elif view_mode == "from-anchor" and anchor_name == "handoff:phase-1":
        active_anchor = anchors[0]
        context_entries = [entries[2]]
    elif view_mode == "latest":
        active_anchor = anchors[0]
        context_entries = [entries[2]]
    else:
        active_anchor = None
        context_entries = entries
    return ConversationSnapshot(
        tape_name="t1",
        entries=entries,
        anchors=anchors,
        active_anchor=active_anchor,
        context_entries=context_entries,
        estimated_tokens=42,
    )


class FakeAgent:
    def __init__(self) -> None:
        self.snapshot_calls: list[tuple[str, str | None, str]] = []
        self.handoff_calls: list[dict[str, Any]] = []
        self.context_events: list[dict[str, Any]] = []

    def snapshot(
        self, session_id: str, *, view_mode: str = "latest", anchor_name: str | None = None
    ) -> ConversationSnapshot:
        self.snapshot_calls.append((session_id, view_mode, anchor_name))
        return _build_snapshot(view_mode, anchor_name)

    def handoff(
        self, session_id: str, name: str, *, phase: str = "", summary: str = "", facts: list[str] | None = None
    ) -> str:
        self.handoff_calls.append(
            {"session_id": session_id, "name": name, "phase": phase, "summary": summary, "facts": list(facts or [])}
        )
        return "handoff:phase-1"

    def append_context_selection_event(self, session_id: str, **payload: Any) -> None:
        self.context_events.append({"session_id": session_id, **payload})


def _build_channel() -> GradioChannel:
    framework = SimpleNamespace(workspace=Path("."))
    channel = GradioChannel(lambda message: None, framework)  # type: ignore[arg-type]
    channel._gr = gr
    channel._agent = FakeAgent()  # type: ignore[assignment]
    return channel


def test_build_view_from_anchor_fallbacks_to_latest_anchor():
    channel = _build_channel()

    _, _, anchor_update, _, _, _ = channel._build_view("gradio:test", "from-anchor", "handoff:missing")

    assert anchor_update["interactive"] is True
    assert anchor_update["value"] == "handoff:phase-1"
    assert channel._agent.snapshot_calls[0] == ("gradio:test", "from-anchor", "handoff:missing")  # type: ignore[attr-defined]
    assert channel._agent.snapshot_calls[1] == ("gradio:test", "from-anchor", "handoff:phase-1")  # type: ignore[attr-defined]


def test_create_handoff_success_resets_fields():
    channel = _build_channel()

    result = channel._create_handoff("gradio:test", "phase-1", "Phase 1", "Checkpoint", "fact-a\nfact-b\n")

    assert result[0] == ""
    assert result[4] == "latest"
    assert "Handoff created: handoff:phase-1" in result[-1]
    assert channel._agent.handoff_calls[-1]["facts"] == ["fact-a", "fact-b"]  # type: ignore[attr-defined]


def test_render_log_html_marks_context_and_active_anchor():
    channel = _build_channel()
    snapshot = _build_snapshot("latest", None)

    content = channel._build_view("gradio:test", "latest", None)[1]

    assert "in-context" in content
    assert "active-anchor" in content
    assert snapshot.active_anchor is not None


def test_send_with_blank_message_does_not_queue_turn():
    channel = _build_channel()

    _, history, status, pending = channel._send_stage1("   ", [])

    assert history == []
    assert status == ""
    assert pending == ""


def test_ui_contains_bottom_message_input_textbox():
    channel = _build_channel()
    demo = channel._build_demo(gr)
    components = demo.config.get("components", [])
    message_boxes = [
        component
        for component in components
        if component.get("type") == "textbox" and component.get("props", {}).get("label") == "Message"
    ]

    assert message_boxes


def test_format_error_payload_data_matches_error_payload_as_dict():
    assert "[provider] timeout" in _format_error_payload_data(
        {"kind": "provider", "message": "timeout", "details": {"retry": 1}}
    )
    assert _format_error_payload_data({"message": "plain"}) == "plain"


def test_context_indicator_contains_status_and_progress():
    channel = _build_channel()
    channel._agent = FakeAgent()  # type: ignore[assignment]
    snapshot = ConversationSnapshot(
        tape_name="t1",
        entries=[],
        anchors=[],
        active_anchor=None,
        context_entries=[],
        estimated_tokens=3200,
    )
    from endless_context.channel import _render_context

    content = _render_context(snapshot, "full")

    assert "HIGH" in content
    assert "ctx-high" in content
    assert "ctx-fill" in content


class _FakeSelectEvent:
    def __init__(self, index):
        self.index = index


def test_switch_view_returns_full_mode():
    channel = _build_channel()

    mode, *_ = channel._switch_view("gradio:test", "full")

    assert mode == "full"


def test_select_anchor_from_table_switches_to_from_anchor():
    channel = _build_channel()

    rows = [["", "Phase 1", "handoff:phase-1", "checkpoint"]]
    mode, _, _, anchor_update, *_ = channel._select_anchor_from_table(
        "gradio:test", rows, False, _FakeSelectEvent((0, 2))
    )

    assert mode == "from-anchor"
    assert anchor_update["value"] == "handoff:phase-1"
