from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bub.framework import BubFramework

from endless_context.agent import BubAgent, estimate_tokens
from endless_context.tape_store import cached_store


@dataclass
class _Entry:
    id: int
    kind: str
    payload: dict[str, Any]
    meta: dict[str, Any]
    date: str = "2026-02-10T00:00:00Z"


class _FakeTape:
    def __init__(self, name: str, entries: list[_Entry]) -> None:
        self.name = name
        self._entries = entries

    @property
    def query_async(self):
        entries = self._entries

        class _Query:
            def __init__(self, selected_entries: list[_Entry]) -> None:
                self._selected_entries = list(selected_entries)

            def last_anchor(inner_self):  # noqa: ANN001
                last_anchor_id = 0
                for item in inner_self._selected_entries:
                    if item.kind == "anchor":
                        last_anchor_id = item.id
                if last_anchor_id == 0:
                    return _Query([])
                return _Query([item for item in inner_self._selected_entries if item.id > last_anchor_id])

            def after_anchor(inner_self, name: str):  # noqa: ANN001
                anchor_id = 0
                for item in inner_self._selected_entries:
                    if item.kind == "anchor" and item.payload.get("name") == name:
                        anchor_id = item.id
                if anchor_id == 0:
                    return _Query([])
                return _Query([item for item in inner_self._selected_entries if item.id > anchor_id])

            async def all(inner_self):  # noqa: ANN001
                return list(inner_self._selected_entries)

        return _Query(entries)


class _FakeTapeService:
    def __init__(self, entries: list[_Entry] | None = None) -> None:
        self.entries = list(entries or [])
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.handoff_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.reset_calls: list[tuple[str, bool]] = []

    def session_tape(self, session_id: str, workspace) -> _FakeTape:  # noqa: ANN001
        return _FakeTape(f"tape:{session_id}", self.entries)

    async def ensure_bootstrap_anchor(self, tape_name: str) -> None:
        has_anchor = any(item.kind == "anchor" for item in self.entries)
        if not has_anchor:
            self.entries.append(
                _Entry(len(self.entries) + 1, "anchor", {"name": "session/start", "state": {"owner": "human"}}, {})
            )

    async def append_event(self, tape_name: str, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))

    async def handoff(self, tape_name: str, *, name: str, state: dict[str, Any] | None = None):
        self.handoff_calls.append((name, state))
        self.entries.append(_Entry(len(self.entries) + 1, "anchor", {"name": name, "state": state or {}}, {}))
        return []

    async def reset(self, tape_name: str, archive: bool = False) -> str:
        self.reset_calls.append((tape_name, archive))
        self.entries.clear()
        return "ok"


class _FakeRuntimeAgent:
    def __init__(self, tape_service: _FakeTapeService) -> None:
        self.tapes = tape_service


def _build_agent(
    entries: list[_Entry] | None = None,
) -> tuple[BubAgent, _FakeTapeService]:
    tape_service = _FakeTapeService(entries)
    agent = BubAgent.__new__(BubAgent)
    agent._framework = SimpleNamespace(workspace=".", get_tape_store=lambda: None)
    agent._runtime_agent = _FakeRuntimeAgent(tape_service)
    agent._workspace = "."
    return agent, tape_service


def test_snapshot_latest_uses_last_anchor() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "message", {"role": "user", "content": "a"}, {}),
            _Entry(2, "anchor", {"name": "handoff:first", "state": {"phase": "First"}}, {}),
            _Entry(3, "message", {"role": "assistant", "content": "b"}, {}),
            _Entry(4, "anchor", {"name": "handoff:second", "state": {"phase": "Second"}}, {}),
            _Entry(5, "message", {"role": "user", "content": "c"}, {}),
        ]
    )

    snapshot = agent.snapshot("gradio:test", view_mode="latest")

    assert snapshot.active_anchor is not None
    assert snapshot.active_anchor.name == "handoff:second"
    assert [entry.id for entry in snapshot.context_entries] == [5]


def test_snapshot_from_missing_anchor_uses_latest_anchor() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "message", {"role": "user", "content": "a"}, {}),
            _Entry(2, "anchor", {"name": "handoff:one", "state": {"phase": "One"}}, {}),
            _Entry(3, "message", {"role": "assistant", "content": "b"}, {}),
        ]
    )

    snapshot = agent.snapshot("gradio:test", view_mode="from-anchor", anchor_name="handoff:not-found")

    assert snapshot.active_anchor is not None
    assert snapshot.active_anchor.name == "handoff:one"
    assert [entry.id for entry in snapshot.context_entries] == [3]


def test_snapshot_from_anchor_without_any_anchor_keeps_existing_history() -> None:
    agent, tape_service = _build_agent(
        [
            _Entry(1, "message", {"role": "user", "content": "a"}, {}),
        ]
    )

    snapshot = agent.snapshot("gradio:test", view_mode="from-anchor", anchor_name="handoff:not-found")

    assert snapshot.active_anchor is None
    assert [entry.id for entry in snapshot.context_entries] == [1]
    assert snapshot.messages == [{"role": "user", "content": "a"}]
    assert not any(entry.kind == "anchor" for entry in tape_service.entries)


def test_handoff_normalizes_name_and_appends() -> None:
    agent, tape_service = _build_agent()

    anchor_name = agent.handoff(
        "gradio:test",
        "Implementation Details",
        phase="Implementation",
        summary="Checkpoint",
        facts=["A", "B"],
    )

    assert anchor_name == "handoff:implementation-details"
    assert tape_service.handoff_calls[-1][0] == "handoff:implementation-details"


def test_append_context_selection_event_records_payload() -> None:
    agent, tape_service = _build_agent()

    agent.append_context_selection_event(
        "gradio:test",
        view_mode="latest",
        anchor_name="handoff:phase-1",
        context_entry_count=2,
        estimated_tokens=123,
    )

    assert tape_service.events[-1][0] == "gradio.context_selection"


def test_reset_archives_tape() -> None:
    agent, tape_service = _build_agent()

    agent.reset("gradio:test")

    assert tape_service.reset_calls == [("tape:gradio:test", True)]


def test_estimate_tokens_prefers_usage_event() -> None:
    entries = [
        _Entry(1, "message", {"role": "user", "content": "hello"}, {}),
        _Entry(
            2,
            "event",
            {
                "name": "run",
                "data": {
                    "usage": {
                        "total_tokens": 321,
                    }
                },
            },
            {},
        ),
    ]

    assert estimate_tokens(entries) == 321


def test_snapshot_messages_strip_gradio_context_prefix() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "anchor", {"name": "session/start", "state": {"owner": "human"}}, {}),
            _Entry(
                2,
                "message",
                {
                    "role": "user",
                    "content": (
                        "surface=gradio|channel=$gradio|chat_id=abc\n---Date: 2026-04-13T08:20:23Z---\nhello again"
                    ),
                },
                {"run_id": "r1"},
            ),
            _Entry(3, "message", {"role": "assistant", "content": "Hello!"}, {"run_id": "r1"}),
        ],
    )

    snapshot = agent.snapshot("gradio:test", view_mode="latest")

    assert snapshot.messages == [
        {"role": "user", "content": "hello again"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_snapshot_messages_keep_user_message_for_multi_step_run() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "event", {"name": "loop.step.start", "data": {"step": 1}}, {}),
            _Entry(2, "event", {"name": "loop.step.start", "data": {"step": 2}}, {}),
            _Entry(3, "system", {"content": "system"}, {"run_id": "r1"}),
            _Entry(4, "message", {"role": "user", "content": "hello"}, {"run_id": "r1"}),
            _Entry(5, "message", {"role": "assistant", "content": "done"}, {"run_id": "r1"}),
        ],
    )

    snapshot = agent.snapshot("gradio:test", view_mode="latest")

    assert snapshot.active_anchor is None
    assert snapshot.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]


def test_snapshot_latest_messages_follow_latest_anchor() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "anchor", {"name": "session/start", "state": {"owner": "human"}}, {}),
            _Entry(2, "message", {"role": "user", "content": "first question"}, {}),
            _Entry(3, "message", {"role": "assistant", "content": "first answer"}, {}),
            _Entry(4, "anchor", {"name": "handoff:impl", "state": {"phase": "Implementation"}}, {}),
            _Entry(5, "message", {"role": "user", "content": "second question"}, {}),
            _Entry(6, "message", {"role": "assistant", "content": "second answer"}, {}),
        ]
    )

    snapshot = agent.snapshot("gradio:test", view_mode="latest")

    assert snapshot.messages == [
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]


def test_snapshot_from_anchor_messages_follow_selected_anchor() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "anchor", {"name": "session/start", "state": {"owner": "human"}}, {}),
            _Entry(2, "message", {"role": "user", "content": "first question"}, {}),
            _Entry(3, "message", {"role": "assistant", "content": "first answer"}, {}),
            _Entry(4, "anchor", {"name": "handoff:impl", "state": {"phase": "Implementation"}}, {}),
            _Entry(5, "message", {"role": "user", "content": "second question"}, {}),
            _Entry(6, "message", {"role": "assistant", "content": "second answer"}, {}),
            _Entry(7, "anchor", {"name": "handoff:qa", "state": {"phase": "QA"}}, {}),
            _Entry(8, "message", {"role": "user", "content": "third question"}, {}),
            _Entry(9, "message", {"role": "assistant", "content": "third answer"}, {}),
        ]
    )

    snapshot = agent.snapshot("gradio:test", view_mode="from-anchor", anchor_name="handoff:impl")

    assert snapshot.active_anchor is not None
    assert snapshot.active_anchor.name == "handoff:impl"
    assert snapshot.messages == [
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
    ]


def test_handoff_moves_latest_messages_to_new_anchor_boundary() -> None:
    agent, _ = _build_agent(
        [
            _Entry(1, "anchor", {"name": "session/start", "state": {"owner": "human"}}, {}),
            _Entry(2, "message", {"role": "user", "content": "first question"}, {}),
            _Entry(3, "message", {"role": "assistant", "content": "first answer"}, {}),
        ]
    )

    agent.handoff("gradio:test", "impl")

    snapshot = agent.snapshot("gradio:test", view_mode="latest")

    assert snapshot.active_anchor is not None
    assert snapshot.active_anchor.name == "handoff:impl"
    assert snapshot.messages == []


def test_snapshot_latest_creates_bootstrap_anchor_with_real_store(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "tapes.db"
    monkeypatch.setenv("BUB_TAPESTORE_SQLALCHEMY_URL", f"sqlite+pysqlite:///{db_path}")
    cached_store.cache_clear()

    framework = BubFramework()
    framework.load_hooks()
    agent = BubAgent(framework)

    snapshot = agent.snapshot("gradio:real-bootstrap", view_mode="latest")

    assert [anchor.name for anchor in snapshot.anchors] == ["session/start"]
    assert snapshot.active_anchor is not None
    assert snapshot.active_anchor.name == "session/start"

    cached_store.cache_clear()


def test_handoff_persists_anchor_with_real_store(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "tapes.db"
    monkeypatch.setenv("BUB_TAPESTORE_SQLALCHEMY_URL", f"sqlite+pysqlite:///{db_path}")
    cached_store.cache_clear()

    framework = BubFramework()
    framework.load_hooks()
    agent = BubAgent(framework)
    agent.snapshot("gradio:real-handoff", view_mode="latest")

    normalized = agent.handoff(
        "gradio:real-handoff",
        "impl-details",
        phase="Implementation",
        summary="Checkpoint",
        facts=["fact-a"],
    )
    snapshot = agent.snapshot("gradio:real-handoff", view_mode="latest")

    assert normalized == "handoff:impl-details"
    assert [anchor.name for anchor in snapshot.anchors] == ["session/start", "handoff:impl-details"]
    assert snapshot.active_anchor is not None
    assert snapshot.active_anchor.name == "handoff:impl-details"
    assert snapshot.active_anchor.label == "Implementation"
    assert snapshot.active_anchor.summary == "Checkpoint"

    cached_store.cache_clear()
