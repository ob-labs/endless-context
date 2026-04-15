from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
from bub.channels.message import ChannelMessage

from endless_context.agent import AnchorState, ConversationSnapshot
from endless_context.channel import GradioChannel


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
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
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


class StatefulAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self._sessions: dict[str, dict[str, Any]] = {}

    def set_conversation(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        *,
        anchor_name: str = "session/start",
        label: str = "session/start",
        summary: str = "-",
    ) -> None:
        entries: list[_Entry] = []
        anchors: list[AnchorState] = []
        next_id = 1

        if anchor_name:
            entries.append(_Entry(next_id, "anchor", {"name": anchor_name, "state": {"phase": label}}, {}))
            anchors.append(
                AnchorState(
                    entry_id=next_id,
                    name=anchor_name,
                    label=label,
                    summary=summary,
                    facts=[],
                    created_at="2026-02-10T00:00:00",
                )
            )
            next_id += 1

        for message in messages:
            entries.append(_Entry(next_id, "message", dict(message), {}))
            next_id += 1

        context_entries = entries if not anchors else [entry for entry in entries if entry.kind == "message"]
        self._sessions[session_id] = {
            "entries": entries,
            "anchors": anchors,
            "messages": [dict(message) for message in messages],
            "context_entries": context_entries,
        }

    def snapshot(
        self, session_id: str, *, view_mode: str = "latest", anchor_name: str | None = None
    ) -> ConversationSnapshot:
        self.snapshot_calls.append((session_id, view_mode, anchor_name))
        session = self._sessions.get(session_id)
        if session is None:
            return _build_snapshot(view_mode, anchor_name)

        anchors = list(session["anchors"])
        entries = list(session["entries"])
        if view_mode == "full":
            active_anchor = None
            context_entries = entries
        elif anchors:
            active_anchor = anchors[-1]
            context_entries = list(session["context_entries"])
        else:
            active_anchor = None
            context_entries = list(session["context_entries"])
        return ConversationSnapshot(
            tape_name=f"tape:{session_id}",
            entries=entries,
            anchors=anchors,
            active_anchor=active_anchor,
            context_entries=context_entries,
            messages=list(session["messages"]),
            estimated_tokens=64,
        )

    def handoff(
        self, session_id: str, name: str, *, phase: str = "", summary: str = "", facts: list[str] | None = None
    ) -> str:
        normalized = super().handoff(session_id, name, phase=phase, summary=summary, facts=facts)
        messages = self._sessions.get(session_id, {}).get("messages", [])
        label = phase or normalized
        self.set_conversation(session_id, list(messages), anchor_name=normalized, label=label, summary=summary or "-")
        return normalized


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


def test_render_log_html_uses_raw_payload_block():
    channel = _build_channel()

    content = channel._build_view("gradio:test", "latest", None)[1]

    assert "Raw payload" in content


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


def test_dispatch_and_wait_updates_conversation_and_tape_view():
    async def _run() -> None:
        framework = SimpleNamespace(workspace=Path("."))
        session_id = "gradio:test-flow"
        agent = StatefulAgent()
        agent.set_conversation(session_id, [])
        observed: dict[str, Any] = {}

        async def on_receive(message: ChannelMessage) -> None:
            observed["message"] = message
            agent.set_conversation(
                session_id,
                [
                    {"role": "user", "content": message.content},
                    {"role": "assistant", "content": "pong"},
                ],
                anchor_name="session/start",
                label="Session",
            )
            await channel.send(
                ChannelMessage(
                    session_id=message.session_id,
                    channel="gradio",
                    chat_id="test-flow",
                    content="pong",
                )
            )

        channel = GradioChannel(on_receive, framework)  # type: ignore[arg-type]
        channel._gr = gr
        channel._agent = agent  # type: ignore[assignment]
        channel._loop = asyncio.get_running_loop()

        reply = await asyncio.to_thread(channel._dispatch_and_wait, session_id, "hello", "from-anchor", "handoff:x")
        chat, log, _, anchors, footer, _ = await asyncio.to_thread(channel._build_view, session_id, "latest", None)

        assert reply == "pong"
        assert observed["message"].view_mode == "from-anchor"
        assert observed["message"].anchor_name == "handoff:x"
        assert chat == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "pong"},
        ]
        assert anchors == [["✓", "Session", "session/start", "-"]]
        assert "pong" in log
        assert "From latest anchor" in footer

    asyncio.run(_run())


def test_send_stage2_returns_updated_chat_and_records_context_event():
    channel = _build_channel()
    agent = StatefulAgent()
    session_id = "gradio:test-stage2"
    agent.set_conversation(session_id, [])
    channel._agent = agent  # type: ignore[assignment]

    def fake_dispatch(session_id: str, content: str, view_mode: str, anchor_name: str | None) -> str:
        assert session_id == "gradio:test-stage2"
        assert view_mode == "latest"
        assert anchor_name is None
        agent.set_conversation(
            session_id,
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "pong"},
            ],
            anchor_name="session/start",
            label="Session",
        )
        return "pong"

    channel._dispatch_and_wait = fake_dispatch  # type: ignore[method-assign]

    outputs = list(channel._send_stage2(session_id, "hello", "latest", None, False))
    final = outputs[-1]

    assert final[0] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "pong"},
    ]
    assert final[6] == ""
    assert final[7] == ""
    assert channel._agent.context_events[-1]["view_mode"] == "latest"  # type: ignore[attr-defined]
    assert "pong" in final[1]


def test_create_handoff_updates_anchor_view():
    channel = _build_channel()
    agent = StatefulAgent()
    session_id = "gradio:test-handoff"
    agent.set_conversation(
        session_id,
        [{"role": "user", "content": "hello"}],
        anchor_name="session/start",
        label="Session",
    )
    channel._agent = agent  # type: ignore[assignment]

    result = channel._create_handoff(
        session_id,
        "impl-details",
        "Implementation",
        "Checkpoint",
        "fact-a\nfact-b",
    )

    assert result[4] == "latest"
    assert result[-1] == "Handoff created: handoff:phase-1"
    assert result[8] == [["✓", "Implementation", "handoff:phase-1", "Checkpoint"]]
    assert "From latest anchor" in result[9]
