from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Literal

from bub.builtin.agent import Agent as BubRuntimeAgent
from bub.builtin.context import default_tape_context
from bub.framework import BubFramework
from republic.tape import Tape, TapeEntry
from republic.tape.context import LAST_ANCHOR, TapeContext

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

    if not remainder.startswith("---Date:"):
        return content

    _, date_separator, body = remainder.partition("\n")
    if not date_separator:
        return content

    stripped = body.strip()
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
    messages: list[dict[str, str]] = field(default_factory=list)
    estimated_tokens: int = 0

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def context_entry_count(self) -> int:
        return len(self.context_entries)


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
        active_anchor = resolve_active_anchor(anchors, resolved_mode, resolved_anchor_name)
        if resolved_mode == "full" or not anchors:
            context_entries = entries
        else:
            context_entries = self._read_context_entries(tape, resolved_mode, resolved_anchor_name)
        return ConversationSnapshot(
            tape_name=tape.name,
            entries=entries,
            anchors=anchors,
            active_anchor=active_anchor,
            context_entries=context_entries,
            messages=extract_conversation_messages(context_entries),
            estimated_tokens=estimate_tokens(context_entries),
        )

    async def run(
        self,
        session_id: str,
        *,
        prompt: str | list[dict],
        state: dict[str, object],
        allowed_tools: set[str] | None = None,
    ) -> str:
        tape = self._session_tape(session_id)
        requested_view_mode = _coerce_view_mode(state.get("_gradio_view_mode"))
        requested_anchor_name = _coerce_anchor_name(state.get("_gradio_anchor_name"))
        resolved_mode, resolved_anchor_name, _, anchors = await self._resolve_view_async(
            tape=tape,
            view_mode=requested_view_mode,
            anchor_name=requested_anchor_name,
            ensure_anchor=requested_view_mode != "full",
        )
        runtime_context = build_runtime_tape_context(
            view_mode=resolved_mode,
            anchor_name=resolved_anchor_name,
            state=dict(state),
            has_anchor=bool(anchors),
        )
        tape.context = dc_replace(
            tape.context,
            anchor=runtime_context.anchor,
            select=runtime_context.select,
            state=runtime_context.state,
        )

        merge_back = not session_id.startswith("temp/")
        async with self._runtime_agent.tapes.fork_tape(tape.name, merge_back=merge_back):
            await self._runtime_agent.tapes.ensure_bootstrap_anchor(tape.name)
            if isinstance(prompt, str) and prompt.strip().startswith(","):
                return await self._runtime_agent._run_command(tape=tape, line=prompt.strip())
            return await self._runtime_agent._agent_loop(
                tape=tape,
                prompt=prompt,
                allowed_tools=allowed_tools,
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
        payload = {
            "view_mode": view_mode,
            "anchor_name": anchor_name,
            "context_entry_count": context_entry_count,
            "estimated_tokens": estimated_tokens,
        }
        if self._framework.get_tape_store() is None:
            _run_async(self._runtime_agent.tapes.append_event(tape.name, "gradio.context_selection", payload))
            return
        self._append_entry_sync(tape.name, TapeEntry.event("gradio.context_selection", payload))

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
        if self._framework.get_tape_store() is None:
            _run_async(self._runtime_agent.tapes.handoff(tape.name, name=normalized, state=state or None))
            return normalized
        self._append_handoff_sync(tape.name, normalized, state or None)
        return normalized

    def reset(self, session_id: str) -> None:
        tape = self._session_tape(session_id)
        if self._framework.get_tape_store() is None:
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
            return
        self._append_entry_sync(
            tape.name,
            TapeEntry.event(
                "gradio.tape_archived",
                {
                    "old_tape": tape.name,
                    "reason": "user_reset",
                },
            ),
        )
        self._reset_tape_sync(tape.name)
        self._append_handoff_sync(tape.name, AUTO_BOOTSTRAP_ANCHOR, {"owner": "human"})

    def _session_tape(self, session_id: str) -> Tape:
        return self._runtime_agent.tapes.session_tape(session_id, self._workspace)

    def _read_entries(self, tape: Tape) -> list[TapeEntry]:
        entries = _run_async(tape.query_async.all())
        return list(entries)

    async def _read_entries_async(self, tape: Tape) -> list[TapeEntry]:
        entries = await tape.query_async.all()
        return list(entries)

    def _read_context_entries(self, tape: Tape, view_mode: ViewMode, anchor_name: str | None) -> list[TapeEntry]:
        context = build_runtime_tape_context(view_mode=view_mode, anchor_name=anchor_name, state={})
        query = context.build_query(tape.query_async)
        return list(_run_async(query.all()))

    async def _read_context_entries_async(
        self, tape: Tape, view_mode: ViewMode, anchor_name: str | None
    ) -> list[TapeEntry]:
        context = build_runtime_tape_context(view_mode=view_mode, anchor_name=anchor_name, state={})
        query = context.build_query(tape.query_async)
        return list(await query.all())

    def _create_bootstrap_anchor(self, tape: Tape) -> tuple[list[TapeEntry], list[AnchorState], AnchorState | None]:
        if self._framework.get_tape_store() is None:
            _run_async(self._runtime_agent.tapes.ensure_bootstrap_anchor(tape.name))
        else:
            self._append_handoff_sync(tape.name, AUTO_BOOTSTRAP_ANCHOR, dict(AUTO_BOOTSTRAP_STATE))
        entries = self._read_entries(tape)
        anchors = extract_anchors(entries)
        created = find_anchor_by_name(anchors, AUTO_BOOTSTRAP_ANCHOR)
        if created is None and anchors:
            created = anchors[-1]
        return entries, anchors, created

    async def _create_bootstrap_anchor_async(
        self, tape: Tape
    ) -> tuple[list[TapeEntry], list[AnchorState], AnchorState | None]:
        if self._framework.get_tape_store() is None:
            await self._runtime_agent.tapes.ensure_bootstrap_anchor(tape.name)
        else:
            await self._append_handoff_async(tape.name, AUTO_BOOTSTRAP_ANCHOR, dict(AUTO_BOOTSTRAP_STATE))
        entries = await self._read_entries_async(tape)
        anchors = extract_anchors(entries)
        created = find_anchor_by_name(anchors, AUTO_BOOTSTRAP_ANCHOR)
        if created is None and anchors:
            created = anchors[-1]
        return entries, anchors, created

    def _append_handoff_sync(self, tape_name: str, name: str, state: dict[str, Any] | None) -> None:
        self._append_entry_sync(tape_name, TapeEntry.anchor(name, state=state))
        self._append_entry_sync(tape_name, TapeEntry.event("handoff", {"name": name, "state": state or {}}))

    async def _append_handoff_async(self, tape_name: str, name: str, state: dict[str, Any] | None) -> None:
        await self._append_entry_async(tape_name, TapeEntry.anchor(name, state=state))
        await self._append_entry_async(tape_name, TapeEntry.event("handoff", {"name": name, "state": state or {}}))

    def _append_entry_sync(self, tape_name: str, entry: TapeEntry) -> None:
        store = self._framework.get_tape_store()
        if store is None:
            raise RuntimeError("tape store is not configured")
        result = store.append(tape_name, entry)
        if inspect.isawaitable(result):
            _run_async(result)

    async def _append_entry_async(self, tape_name: str, entry: TapeEntry) -> None:
        store = self._framework.get_tape_store()
        if store is None:
            raise RuntimeError("tape store is not configured")
        result = store.append(tape_name, entry)
        if inspect.isawaitable(result):
            await result

    def _reset_tape_sync(self, tape_name: str) -> None:
        store = self._framework.get_tape_store()
        if store is None:
            raise RuntimeError("tape store is not configured")
        result = store.reset(tape_name)
        if inspect.isawaitable(result):
            _run_async(result)

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
            if not anchors and ensure_anchor and not entries:
                entries, anchors, _ = self._create_bootstrap_anchor(tape)
            resolved_anchor_name = anchors[-1].name if anchors else None
            return "latest", resolved_anchor_name, entries, anchors

        target = find_anchor_by_name(anchors, anchor_name) if anchor_name else None
        if target is None and anchors:
            target = anchors[-1]
        if target is None and ensure_anchor and not entries:
            entries, anchors, target = self._create_bootstrap_anchor(tape)
        resolved_anchor_name = target.name if target else None
        return "from-anchor", resolved_anchor_name, entries, anchors

    async def _resolve_view_async(
        self,
        *,
        tape: Tape,
        view_mode: ViewMode,
        anchor_name: str | None,
        ensure_anchor: bool,
    ) -> tuple[ViewMode, str | None, list[TapeEntry], list[AnchorState]]:
        entries = await self._read_entries_async(tape)
        anchors = extract_anchors(entries)

        if view_mode == "full":
            return "full", None, entries, anchors

        if view_mode == "latest":
            if not anchors and ensure_anchor and not entries:
                entries, anchors, _ = await self._create_bootstrap_anchor_async(tape)
            resolved_anchor_name = anchors[-1].name if anchors else None
            return "latest", resolved_anchor_name, entries, anchors

        target = find_anchor_by_name(anchors, anchor_name) if anchor_name else None
        if target is None and anchors:
            target = anchors[-1]
        if target is None and ensure_anchor and not entries:
            entries, anchors, target = await self._create_bootstrap_anchor_async(tape)
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


def resolve_active_anchor(
    anchors: list[AnchorState],
    view_mode: ViewMode,
    anchor_name: str | None,
) -> AnchorState | None:
    if view_mode == "full" or not anchors:
        return None
    if view_mode == "latest":
        return anchors[-1]
    return find_anchor_by_name(anchors, anchor_name) or anchors[-1]


def build_runtime_tape_context(
    *,
    view_mode: ViewMode,
    anchor_name: str | None,
    state: dict[str, object],
    has_anchor: bool = True,
) -> TapeContext:
    anchor: str | None | object
    if view_mode == "full" or not has_anchor:
        anchor = None
    elif view_mode == "latest":
        anchor = LAST_ANCHOR
    else:
        anchor = anchor_name or LAST_ANCHOR
    return TapeContext(
        anchor=anchor,
        select=default_tape_context().select,
        state=state,
    )


def extract_conversation_messages(entries: list[TapeEntry]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for entry in entries:
        kind = getattr(entry, "kind", "")
        payload = getattr(entry, "payload", {})
        if not isinstance(payload, dict):
            continue

        if kind == "message":
            role = payload.get("role")
            content = payload.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                if role == "user":
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


def _coerce_view_mode(value: object) -> ViewMode:
    if value in {"full", "latest", "from-anchor"}:
        return value
    return "latest"


def _coerce_anchor_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


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
