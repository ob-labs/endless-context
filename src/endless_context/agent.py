from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from bub.builtin.agent import Agent as BubRuntimeAgent
from bub.framework import BubFramework
from republic.tape import Tape, TapeEntry

DEFAULT_SYSTEM_PROMPT = (
    "You are a tape-first assistant. Keep answers concise, grounded in recorded facts, "
    "and maintain continuity with handoff anchors."
)
AUTO_BOOTSTRAP_ANCHOR = "session/start"
AUTO_BOOTSTRAP_STATE = {
    "owner": "human",
}

ViewMode = Literal["full", "latest", "from-anchor"]


def _run_async(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _strip_bub_context_prefix(content: str) -> str:
    first_line, separator, remainder = content.partition("\n")
    if not separator or "channel=$" not in first_line:
        return content

    date_marker = "---\n"
    date_offset = remainder.find(date_marker)
    if date_offset < 0:
        return content

    stripped = remainder[date_offset + len(date_marker) :].strip()
    return stripped or content


@dataclass(frozen=True)
class AnchorState:
    entry_id: int
    name: str
    label: str
    summary: str
    facts: list[str]
    created_at: str | None


@dataclass(frozen=True)
class ConversationSnapshot:
    tape_name: str
    entries: list[TapeEntry]
    anchors: list[AnchorState]
    active_anchor: AnchorState | None
    context_entries: list[TapeEntry]
    estimated_tokens: int

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def context_entry_count(self) -> int:
        return len(self.context_entries)

    @property
    def messages(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        run_step: dict[str, int] = {}
        current_step = 1
        for entry in self.entries:
            kind = getattr(entry, "kind", "")
            payload = getattr(entry, "payload", {})
            meta = getattr(entry, "meta", {})
            if not isinstance(payload, dict):
                continue
            if not isinstance(meta, dict):
                meta = {}

            if kind == "event" and payload.get("name") == "loop.step.start":
                data = payload.get("data")
                if isinstance(data, dict):
                    step = data.get("step")
                    if isinstance(step, int) and step > 0:
                        current_step = step
                continue

            if kind == "system":
                run_id = meta.get("run_id")
                if isinstance(run_id, str) and run_id:
                    run_step[run_id] = current_step
                continue

            if kind == "message":
                role = payload.get("role")
                content = payload.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    if role == "user":
                        run_id = meta.get("run_id")
                        if isinstance(run_id, str) and run_step.get(run_id, 1) > 1:
                            continue
                        content = _strip_bub_context_prefix(content)
                    result.append({"role": role, "content": content})
                continue

            if kind != "event" or payload.get("name") != "command":
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            raw = data.get("raw")
            output = data.get("output")
            if isinstance(raw, str) and raw.strip():
                user_msg = {"role": "user", "content": raw}
                if not result or result[-1] != user_msg:
                    result.append(user_msg)
            if isinstance(output, str) and output.strip():
                result.append({"role": "assistant", "content": output})
        return result


class BubAgent:
    """Snapshot/handoff adapter built on top of Bub's native Agent and tape service."""

    def __init__(self, framework: BubFramework) -> None:
        self._framework = framework
        self._runtime_agent = BubRuntimeAgent(framework)
        self._workspace = framework.workspace.resolve()

    def snapshot(
        self,
        session_id: str,
        *,
        view_mode: ViewMode = "latest",
        anchor_name: str | None = None,
    ) -> ConversationSnapshot:
        tape = self._session_tape(session_id)
        resolved_mode, resolved_anchor_name, entries, anchors = self._resolve_view(
            tape=tape,
            view_mode=view_mode,
            anchor_name=anchor_name,
            ensure_anchor=view_mode != "full",
        )
        active_anchor, context_entries = select_context_entries(
            entries,
            anchors,
            resolved_mode,
            resolved_anchor_name,
        )
        return ConversationSnapshot(
            tape_name=tape.name,
            entries=entries,
            anchors=anchors,
            active_anchor=active_anchor,
            context_entries=context_entries,
            estimated_tokens=estimate_tokens(context_entries),
        )

    def append_context_selection_event(
        self,
        session_id: str,
        *,
        view_mode: ViewMode,
        anchor_name: str | None,
        context_entry_count: int,
        estimated_tokens: int,
    ) -> None:
        tape = self._session_tape(session_id)
        _run_async(
            self._runtime_agent.tapes.append_event(
                tape.name,
                "gradio.context_selection",
                {
                    "view_mode": view_mode,
                    "anchor_name": anchor_name,
                    "context_entry_count": context_entry_count,
                    "estimated_tokens": estimated_tokens,
                },
            )
        )

    def handoff(
        self,
        session_id: str,
        name: str,
        *,
        phase: str = "",
        summary: str = "",
        facts: list[str] | None = None,
    ) -> str:
        normalized = self._normalize_anchor_name(name)
        state: dict[str, Any] = {}
        if phase.strip():
            state["phase"] = phase.strip()
        if summary.strip():
            state["summary"] = summary.strip()
        if facts:
            clean_facts = [item.strip() for item in facts if item.strip()]
            if clean_facts:
                state["facts"] = clean_facts
        tape = self._session_tape(session_id)
        _run_async(self._runtime_agent.tapes.handoff(tape.name, name=normalized, state=state or None))
        return normalized

    def reset(self, session_id: str) -> None:
        tape = self._session_tape(session_id)
        _run_async(
            self._runtime_agent.tapes.append_event(
                tape.name,
                "gradio.tape_archived",
                {
                    "old_tape": tape.name,
                    "reason": "user_reset",
                },
            )
        )
        _run_async(self._runtime_agent.tapes.reset(tape.name, archive=True))

    def _session_tape(self, session_id: str) -> Tape:
        return self._runtime_agent.tapes.session_tape(session_id, self._workspace)

    def _read_entries(self, tape: Tape) -> list[TapeEntry]:
        direct_entries = self._read_entries_from_store(tape.name)
        if direct_entries is not None:
            return direct_entries
        entries = _run_async(tape.query_async.all())
        return list(entries)

    def _read_entries_from_store(self, tape_name: str) -> list[TapeEntry] | None:
        store = self._framework.get_tape_store()
        if store is None:
            return None

        read = getattr(store, "read", None)
        if callable(read):
            entries = read(tape_name)
            if entries is not None:
                return list(entries)

        if store.__class__.__name__ != "SQLAlchemyTapeStore":
            return None

        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return None

        try:
            from bub_tapestore_sqlalchemy.models import TapeEntryRecord, TapeRecord
            from sqlalchemy import select
        except Exception:
            return None

        with session_factory() as session:
            tape_record = session.scalar(select(TapeRecord).where(TapeRecord.name == tape_name))
            if tape_record is None:
                return []
            records = session.scalars(
                select(TapeEntryRecord)
                .where(TapeEntryRecord.tape_id == tape_record.id)
                .order_by(TapeEntryRecord.entry_id)
            ).all()

        return [
            TapeEntry(
                id=record.entry_id,
                kind=record.kind,
                payload=dict(record.payload) if isinstance(record.payload, dict) else {},
                meta=dict(record.meta) if isinstance(record.meta, dict) else {},
                date=record.entry_date,
            )
            for record in records
        ]

    def _create_bootstrap_anchor(self, tape: Tape) -> tuple[list[TapeEntry], list[AnchorState], AnchorState | None]:
        _run_async(self._runtime_agent.tapes.ensure_bootstrap_anchor(tape.name))
        entries = self._read_entries(tape)
        anchors = extract_anchors(entries)
        created = find_anchor_by_name(anchors, AUTO_BOOTSTRAP_ANCHOR)
        if created is None and anchors:
            created = anchors[-1]
        return entries, anchors, created

    def _resolve_view(
        self,
        *,
        tape: Tape,
        view_mode: ViewMode,
        anchor_name: str | None,
        ensure_anchor: bool,
    ) -> tuple[ViewMode, str | None, list[TapeEntry], list[AnchorState]]:
        entries = self._read_entries(tape)
        anchors = extract_anchors(entries)

        if view_mode == "full":
            return "full", None, entries, anchors

        if view_mode == "latest":
            if not anchors and ensure_anchor:
                entries, anchors, _ = self._create_bootstrap_anchor(tape)
            resolved_anchor_name = anchors[-1].name if anchors else None
            return "latest", resolved_anchor_name, entries, anchors

        target = find_anchor_by_name(anchors, anchor_name) if anchor_name else None
        if target is None and anchors:
            target = anchors[-1]
        if target is None and ensure_anchor:
            entries, anchors, target = self._create_bootstrap_anchor(tape)
        resolved_anchor_name = target.name if target else None
        return "from-anchor", resolved_anchor_name, entries, anchors

    @staticmethod
    def _normalize_anchor_name(name: str) -> str:
        raw = name.strip()
        if not raw:
            raise ValueError("anchor name cannot be empty")
        if raw.startswith("handoff:") or raw.startswith("phase:") or raw.startswith("session/"):
            return raw
        safe = raw.lower().replace(" ", "-")
        return f"handoff:{safe}"


def extract_anchors(entries: list[TapeEntry]) -> list[AnchorState]:
    anchors: list[AnchorState] = []
    for entry in entries:
        if getattr(entry, "kind", "") != "anchor":
            continue
        payload = getattr(entry, "payload", {})
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, str):
            continue
        state = payload.get("state")
        if not isinstance(state, dict):
            state = {}
        summary = str(state.get("summary", "")).strip()
        facts_raw = state.get("facts")
        facts: list[str] = []
        if isinstance(facts_raw, list):
            facts = [str(item).strip() for item in facts_raw if str(item).strip()]
        phase = str(state.get("phase", "")).strip()
        label = phase or name
        meta = getattr(entry, "meta", {})
        created_at = None
        if isinstance(meta, dict):
            raw_created_at = meta.get("created_at")
            if isinstance(raw_created_at, str):
                created_at = raw_created_at
        anchors.append(
            AnchorState(
                entry_id=int(getattr(entry, "id", 0)),
                name=name,
                label=label,
                summary=summary,
                facts=facts,
                created_at=created_at,
            )
        )
    return anchors


def find_anchor_by_name(anchors: list[AnchorState], anchor_name: str | None) -> AnchorState | None:
    if not anchor_name:
        return None
    for anchor in anchors:
        if anchor.name == anchor_name:
            return anchor
    return None


def select_context_entries(
    entries: list[TapeEntry],
    anchors: list[AnchorState],
    view_mode: ViewMode,
    anchor_name: str | None,
) -> tuple[AnchorState | None, list[TapeEntry]]:
    if view_mode == "full":
        return None, list(entries)
    if not anchors:
        return None, list(entries)
    if view_mode == "latest":
        active_anchor = anchors[-1]
    else:
        active_anchor = find_anchor_by_name(anchors, anchor_name) or anchors[-1]
    return active_anchor, entries_after_id(entries, active_anchor.entry_id)


def entries_after_id(entries: list[TapeEntry], entry_id: int) -> list[TapeEntry]:
    return [entry for entry in entries if int(getattr(entry, "id", 0)) > entry_id]


def _extract_usage_tokens(entry: TapeEntry) -> int | None:
    if getattr(entry, "kind", "") != "event":
        return None
    payload = getattr(entry, "payload", {})
    if not isinstance(payload, dict):
        return None
    if payload.get("name") != "run":
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("input_tokens", "prompt_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _fallback_token_estimate_by_chars(entries: list[TapeEntry]) -> int:
    total_chars = 0
    for entry in entries:
        payload = getattr(entry, "payload", {})
        if not isinstance(payload, dict):
            continue
        content = payload.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        else:
            total_chars += len(str(payload))
    return max(1, total_chars // 4) if total_chars else 0


def estimate_tokens(entries: list[TapeEntry]) -> int:
    for entry in reversed(entries):
        usage_tokens = _extract_usage_tokens(entry)
        if usage_tokens is not None:
            return usage_tokens
    return _fallback_token_estimate_by_chars(entries)
