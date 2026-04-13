from __future__ import annotations

import asyncio
import contextlib
import html
import json
import threading
import time
import uuid
from typing import Any

from bub.channels.base import Channel
from bub.channels.message import ChannelMessage
from bub.framework import BubFramework
from bub.types import MessageHandler
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from republic import ErrorPayload
from republic.core.errors import ErrorKind

from endless_context.agent import BubAgent, ConversationSnapshot, ViewMode


class PendingTurn:
    def __init__(self) -> None:
        self.outputs: list[str] = []
        self.completed = threading.Event()

    def append(self, kind: str, content: str) -> None:
        rendered = content.strip()
        if not rendered:
            return
        if kind == "error" and not rendered.startswith("Error:"):
            rendered = f"Error: {rendered}"
        self.outputs.append(rendered)
        self.completed.set()

    def render(self) -> str:
        return "\n\n".join(item for item in self.outputs if item)


def _preview_text(content: str, limit: int = 120) -> str:
    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class GradioChannelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BUB_GRADIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 7860
    response_timeout_seconds: float = 300.0
    enqueue_timeout_seconds: float = 10.0
    output_settle_seconds: float = 0.1


class GradioChannel(Channel):
    name = "gradio"

    def __init__(self, on_receive: MessageHandler, framework: BubFramework) -> None:
        self._on_receive = on_receive
        self._framework = framework
        self._config = GradioChannelSettings()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._demo: Any | None = None
        self._gr: Any | None = None
        self._pending_lock = threading.RLock()
        self._pending: dict[str, PendingTurn] = {}
        self._agent = BubAgent(framework)

    async def start(self, stop_event: asyncio.Event) -> None:
        del stop_event
        try:
            import gradio as gr
        except ImportError:
            logger.error("gradio is not installed; the gradio channel cannot start")
            return

        if self._loop is not None:
            return

        self._loop = asyncio.get_running_loop()
        self._gr = gr
        demo = self._build_demo(gr)
        await asyncio.to_thread(
            demo.launch,
            server_name=self._config.host,
            server_port=self._config.port,
            prevent_thread_lock=True,
            show_error=True,
            quiet=True,
            share=False,
            theme=gr.themes.Soft(),
            css=CSS,
            head=INPUT_HEAD_JS,
        )
        self._demo = demo
        logger.info("gradio.start complete host={} port={}", self._config.host, self._config.port)

    async def stop(self) -> None:
        if self._demo is None:
            return
        await asyncio.to_thread(self._demo.close)
        self._demo = None
        self._loop = None
        logger.info("gradio.stop complete")

    async def send(self, message: ChannelMessage) -> None:
        with self._pending_lock:
            pending = self._pending.get(message.session_id)
        if pending is None:
            logger.debug("gradio.send dropped outbound for inactive session_id={}", message.session_id)
            return
        logger.info(
            "gradio.outbound session_id={} kind={} content={}",
            message.session_id,
            message.kind,
            _preview_text(message.content or ""),
        )
        pending.append(message.kind or "normal", message.content or "")

    def _build_demo(self, gr: Any) -> Any:
        with gr.Blocks(title="Endless Context") as demo:
            session_state = gr.State("")
            pending_message = gr.State("")

            gr.Markdown("## Endless Context\nAppend-only tape &middot; Handoff anchors &middot; Context assembly")

            with gr.Row(elem_id="main-row"):
                with gr.Column(scale=3, elem_id="tape-col"):
                    gr.Markdown("#### Tape")
                    view_mode = gr.Radio(
                        choices=["latest", "full", "from-anchor"],
                        value="latest",
                        label="Context view",
                    )
                    show_system_events = gr.Checkbox(
                        value=False,
                        label="Show events",
                        info="Only affects tape rendering; does not affect context selection.",
                    )
                    anchor_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Anchor",
                        info="Active when view is from-anchor.",
                        interactive=False,
                    )
                    log_html = gr.HTML()
                    tape_footer = gr.Markdown()

                with gr.Column(scale=6, elem_id="conversation-col"):
                    gr.Markdown("#### Conversation")
                    context_indicator = gr.HTML()
                    chatbot = gr.Chatbot(height=480, label="Messages", elem_id="conv-chatbot")
                    user_input = gr.Textbox(
                        label="Message",
                        placeholder="Type a message, Shift+Enter to send",
                        lines=3,
                        elem_id="user-input",
                    )
                    with gr.Row():
                        send_button = gr.Button("Send", variant="primary", elem_id="send-btn")
                        refresh_button = gr.Button("Refresh", variant="secondary")

                with gr.Column(scale=3):
                    gr.Markdown("#### Anchors")
                    with gr.Row():
                        full_tape_button = gr.Button("Full tape", size="sm")
                        latest_anchor_button = gr.Button("Latest", size="sm")
                    anchors_table = gr.Dataframe(
                        headers=["", "Label", "Name", "Summary"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        row_count=6,
                        column_count=(4, "fixed"),
                        value=[],
                    )
                    with gr.Accordion("Create Handoff", open=False):
                        handoff_name = gr.Textbox(label="Name", placeholder="e.g. impl-details")
                        handoff_phase = gr.Textbox(label="Phase", placeholder="e.g. Implementation")
                        handoff_summary = gr.Textbox(label="Summary")
                        handoff_facts = gr.Textbox(label="Facts (one per line)", lines=3)
                        handoff_button = gr.Button("Create", variant="secondary")

            status_text = gr.Markdown()

            core = [chatbot, log_html, anchor_selector, anchors_table, tape_footer, context_indicator, status_text]

            demo.load(
                fn=self._init_session,
                inputs=[show_system_events],
                outputs=[session_state] + core,
                show_progress="hidden",
            )
            view_mode.input(
                fn=self._refresh,
                inputs=[session_state, view_mode, anchor_selector, show_system_events],
                outputs=core,
                show_progress="hidden",
            )
            anchor_selector.input(
                fn=self._refresh,
                inputs=[session_state, view_mode, anchor_selector, show_system_events],
                outputs=core,
                show_progress="hidden",
            )
            show_system_events.input(
                fn=self._refresh,
                inputs=[session_state, view_mode, anchor_selector, show_system_events],
                outputs=core,
                show_progress="hidden",
            )
            refresh_button.click(
                fn=self._refresh,
                inputs=[session_state, view_mode, anchor_selector, show_system_events],
                outputs=core,
                show_progress="hidden",
            )

            send_button.click(
                fn=self._send_stage1,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot, status_text, pending_message],
                queue=False,
                show_progress="hidden",
            ).then(
                fn=self._send_stage2,
                inputs=[session_state, pending_message, view_mode, anchor_selector, show_system_events],
                outputs=core + [pending_message],
                show_progress="hidden",
            )
            user_input.submit(
                fn=self._send_stage1,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot, status_text, pending_message],
                queue=False,
                show_progress="hidden",
            ).then(
                fn=self._send_stage2,
                inputs=[session_state, pending_message, view_mode, anchor_selector, show_system_events],
                outputs=core + [pending_message],
                show_progress="hidden",
            )

            handoff_button.click(
                fn=self._create_handoff,
                inputs=[session_state, handoff_name, handoff_phase, handoff_summary, handoff_facts, show_system_events],
                outputs=[handoff_name, handoff_phase, handoff_summary, handoff_facts, view_mode] + core,
                show_progress="hidden",
            )

            full_tape_button.click(
                fn=lambda session_id, show: self._switch_view(session_id, "full", show),
                inputs=[session_state, show_system_events],
                outputs=[view_mode] + core,
                show_progress="hidden",
            )
            latest_anchor_button.click(
                fn=lambda session_id, show: self._switch_view(session_id, "latest", show),
                inputs=[session_state, show_system_events],
                outputs=[view_mode] + core,
                show_progress="hidden",
            )
            anchors_table.select(
                fn=self._select_anchor_from_table,
                inputs=[session_state, anchors_table, show_system_events],
                outputs=[view_mode] + core,
                show_progress="hidden",
            )
        return demo

    def _new_session_id(self) -> str:
        return f"{self.name}:{uuid.uuid4().hex[:12]}"

    def _chat_id_from_session(self, session_id: str) -> str:
        if ":" not in session_id:
            return session_id
        return session_id.split(":", 1)[1]

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("gradio channel is not running yet")
        return self._loop

    def _build_inbound_message(self, session_id: str, content: str) -> ChannelMessage:
        return ChannelMessage(
            session_id=session_id,
            channel=self.name,
            chat_id=self._chat_id_from_session(session_id),
            content=content,
            kind="command" if content.startswith(",") else "normal",
            is_active=True,
            context={"surface": "gradio"},
        )

    def _dispatch_and_wait(self, session_id: str, content: str) -> str:
        loop = self._require_loop()
        pending = PendingTurn()
        with self._pending_lock:
            if session_id in self._pending:
                raise RuntimeError(f"session {session_id} already has a running turn")
            self._pending[session_id] = pending

        try:
            logger.info("gradio.inbound session_id={} content={}", session_id, _preview_text(content))
            future = asyncio.run_coroutine_threadsafe(
                self._on_receive(self._build_inbound_message(session_id=session_id, content=content)),
                loop,
            )
            future.result(timeout=self._config.enqueue_timeout_seconds)
            if not pending.completed.wait(self._config.response_timeout_seconds):
                raise TimeoutError(
                    f"timed out waiting for Bub output after {self._config.response_timeout_seconds:.1f}s"
                )
            if self._config.output_settle_seconds > 0:
                time.sleep(self._config.output_settle_seconds)
            rendered = pending.render()
            if not rendered:
                raise RuntimeError("Bub returned no outbound message")
            return rendered
        finally:
            with self._pending_lock:
                self._pending.pop(session_id, None)

    def _build_view(
        self,
        session_id: str,
        view_mode: ViewMode,
        anchor_name: str | None,
        show_system_events: bool = False,
    ) -> tuple[list[dict[str, str]], str, dict[str, Any], list[list[str]], str, str]:
        snapshot = self._agent.snapshot(session_id, view_mode=view_mode, anchor_name=anchor_name)
        anchor_choices = [anchor.name for anchor in snapshot.anchors]
        gr = self._gr
        if gr is None:
            raise RuntimeError("gradio is not initialized")

        if view_mode == "from-anchor":
            resolved_anchor = (
                anchor_name if anchor_name in anchor_choices else (anchor_choices[-1] if anchor_choices else None)
            )
            if resolved_anchor != anchor_name:
                snapshot = self._agent.snapshot(session_id, view_mode=view_mode, anchor_name=resolved_anchor)
            anchor_update = gr.update(choices=anchor_choices, value=resolved_anchor, interactive=True)
        else:
            anchor_update = gr.update(choices=anchor_choices, value=None, interactive=False)

        return (
            snapshot.messages,
            _render_log_html(snapshot, show_system_events),
            anchor_update,
            _anchor_rows(snapshot),
            _render_tape_footer(snapshot, view_mode),
            _render_context(snapshot, view_mode),
        )

    def _init_session(self, show_system_events: bool = False):
        session_id = self._new_session_id()
        chat, log, anchor_upd, anchors, footer, ctx = self._build_view(session_id, "latest", None, show_system_events)
        return session_id, chat, log, anchor_upd, anchors, footer, ctx, ""

    def _refresh(self, session_id: str, view_mode: ViewMode, anchor_name: str | None, show_system_events: bool = False):
        chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
            session_id, view_mode, anchor_name, show_system_events
        )
        return chat, log, anchor_upd, anchors, footer, ctx, ""

    def _send_stage1(self, message: str, chat_history: list[dict[str, str]] | None):
        text = message.strip()
        if not text:
            return "", chat_history or [], "", ""
        history = list(chat_history or [])
        history.append({"role": "user", "content": text})
        return "", history, "", text

    def _send_stage2(
        self,
        session_id: str,
        pending_message: str,
        view_mode: ViewMode,
        anchor_name: str | None,
        show_system_events: bool,
    ):
        text = pending_message.strip()
        if not text:
            chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
                session_id, view_mode, anchor_name, show_system_events
            )
            yield chat, log, anchor_upd, anchors, footer, ctx, "", ""
            return

        snapshot = self._agent.snapshot(session_id, view_mode=view_mode, anchor_name=anchor_name)
        self._agent.append_context_selection_event(
            session_id,
            view_mode=view_mode,
            anchor_name=snapshot.active_anchor.name if snapshot.active_anchor else anchor_name,
            context_entry_count=len(snapshot.context_entries),
            estimated_tokens=snapshot.estimated_tokens,
        )

        state: dict[str, Any] = {"done": False, "reply": "", "error": None}

        def _worker() -> None:
            try:
                state["reply"] = self._dispatch_and_wait(session_id, text)
            except Exception as exc:
                state["error"] = exc
            finally:
                state["done"] = True

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while not bool(state["done"]):
            chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
                session_id, view_mode, anchor_name, show_system_events
            )
            if not chat or chat[-1].get("role") != "user" or chat[-1].get("content") != text:
                chat = [*chat, {"role": "user", "content": text}]
            yield chat, log, anchor_upd, anchors, footer, ctx, "", text
            time.sleep(0.2)

        reply = str(state.get("reply") or "")
        if state.get("error") is not None:
            reply = f"Error: {state['error']}"
        status = reply if reply.startswith("Error:") else ""
        chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
            session_id, view_mode, anchor_name, show_system_events
        )
        yield chat, log, anchor_upd, anchors, footer, ctx, status, ""

    def _create_handoff(
        self,
        session_id: str,
        name: str,
        phase: str,
        summary: str,
        facts_text: str,
        show_system_events: bool = False,
    ):
        if not name.strip():
            chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
                session_id, "latest", None, show_system_events
            )
            return (
                name,
                phase,
                summary,
                facts_text,
                "latest",
                chat,
                log,
                anchor_upd,
                anchors,
                footer,
                ctx,
                "Name is required.",
            )

        facts = [line.strip() for line in facts_text.splitlines() if line.strip()]
        normalized = self._agent.handoff(session_id, name=name, phase=phase, summary=summary, facts=facts)
        chat, log, anchor_upd, anchors, footer, ctx = self._build_view(session_id, "latest", None, show_system_events)
        return "", "", "", "", "latest", chat, log, anchor_upd, anchors, footer, ctx, f"Handoff created: {normalized}"

    def _switch_view(self, session_id: str, target_mode: ViewMode, show_system_events: bool = False):
        chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
            session_id, target_mode, None, show_system_events
        )
        return target_mode, chat, log, anchor_upd, anchors, footer, ctx, ""

    def _select_anchor_from_table(
        self,
        session_id: str,
        rows: list[list[str]],
        show_or_evt: bool | Any,
        evt: Any | None = None,
    ):
        if evt is None:
            show_system_events = False
            if not hasattr(show_or_evt, "index"):
                return self._switch_view(session_id, "latest", show_system_events)
            evt = show_or_evt
        else:
            show_system_events = bool(show_or_evt)
        row_index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(rows):
            return self._switch_view(session_id, "latest", show_system_events)
        anchor_name = rows[row_index][2]
        if not isinstance(anchor_name, str) or not anchor_name.strip():
            return self._switch_view(session_id, "latest", show_system_events)
        chat, log, anchor_upd, anchors, footer, ctx = self._build_view(
            session_id, "from-anchor", anchor_name, show_system_events
        )
        return "from-anchor", chat, log, anchor_upd, anchors, footer, ctx, ""


def _kind_label(kind: str) -> str:
    return kind.upper()[:10]


_ORDERED_KEYS: dict[str, list[str]] = {
    "message": ["role", "content"],
    "event": ["name", "data"],
    "anchor": ["name", "state"],
    "system": ["content"],
    "error": ["kind", "message", "details"],
    "tool_call": ["calls"],
    "tool_result": ["results"],
}


def _format_error_payload_data(data: dict[str, Any]) -> str:
    """Format dicts matching republic 0.5.x ErrorPayload.as_dict() (tape error entries)."""
    kind_raw = data.get("kind")
    message = data.get("message")
    details = data.get("details")
    details_dict = details if isinstance(details, dict) else None

    kind: ErrorKind | None = None
    if isinstance(kind_raw, str) and kind_raw.strip():
        with contextlib.suppress(ValueError):
            kind = ErrorKind(kind_raw.strip())

    if kind is not None and isinstance(message, str):
        text = str(ErrorPayload(kind, message, details_dict))
        if details_dict:
            extra = json.dumps(details_dict, ensure_ascii=False, separators=(",", ":"))
            if len(extra) > 200:
                extra = extra[:197] + "…"
            return f"{text}\n{extra}"
        return text

    parts: list[str] = []
    if isinstance(message, str) and message.strip():
        line = message.strip()
        if isinstance(kind_raw, str) and kind_raw.strip():
            line = f"[{kind_raw.strip()}] {line}"
        parts.append(line)
    elif isinstance(kind_raw, str) and kind_raw.strip():
        parts.append(f"[{kind_raw.strip()}]")
    if not parts:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))[:200]
    head = parts[0]
    if isinstance(details, dict) and details:
        det = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
        if len(det) > 120:
            det = det[:117] + "…"
        return f"{head}\n{det}"
    return head


def _args_summary(arguments: Any, max_values: int = 4, max_len: int = 24) -> str:
    obj: dict[str, Any] | None = None
    if isinstance(arguments, dict):
        obj = arguments
    elif isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            obj = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    if not obj:
        return ""
    parts: list[str] = []
    for value in list(obj.values())[:max_values]:
        text = str(value).strip().replace("\n", " ")
        if len(text) > max_len:
            text = text[: max_len - 1] + "…"
        parts.append(text)
    return ", ".join(parts)


def _human_text(kind: str, payload: dict[str, Any]) -> str:
    if kind == "error":
        return _format_error_payload_data(payload)

    calls = payload.get("calls")
    if isinstance(calls, list) and calls:
        parts: list[str] = []
        for call in calls[:3]:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            name = fn.get("name") if isinstance(fn, dict) else "?"
            args_raw = fn.get("arguments") if isinstance(fn, dict) else None
            summary = _args_summary(args_raw)
            parts.append(f"{name}({summary})" if summary else f"{name}()")
        if len(calls) > 3:
            parts.append("…")
        return ", ".join(parts) if parts else "tool_call"

    results = payload.get("results")
    if isinstance(results, list):
        if not results:
            return "tool_result (0 results)"
        first = results[0]
        if isinstance(first, dict):
            for key in ("message", "error", "content"):
                value = first.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:90]
            return f"results: {len(results)} item(s)"
        if isinstance(first, str) and first.strip():
            return first.strip()[:90]
        return f"results: {len(results)} item(s)"

    role = payload.get("role")
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        prefix = f"{role}: " if isinstance(role, str) and role.strip() else ""
        return f"{prefix}{content.strip().replace(chr(10), ' ')}"

    for key in ("message", "name", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            line = value.strip()[:120]
            if key == "name" and kind == "event":
                data = payload.get("data")
                if isinstance(data, dict) and data:
                    line += " (" + ", ".join(str(item) for item in list(data.keys())[:3]) + ")"
                line = "event: " + line
            elif key == "name" and kind == "anchor":
                state = payload.get("state")
                if isinstance(state, dict):
                    phase = state.get("phase")
                    if isinstance(phase, str) and phase.strip():
                        line += f" ({phase.strip()})"
            return line

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("message", "error", "name", "status"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:120]

    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return compact[:120]


def _kv_row(key: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        shown = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        shown = str(value)
    return (
        "<div class='entry-kv'>"
        f"<span class='entry-k'>{html.escape(str(key))}</span>"
        f"<span class='entry-v'>{html.escape(shown)}</span>"
        "</div>"
    )


def _parse_arguments_for_display(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _structured_value(key: str, value: Any) -> str:
    if not isinstance(value, list):
        return _kv_row(key, value)
    blocks: list[str] = []
    for index, item in enumerate(value):
        if key == "calls" and isinstance(item, dict):
            fn = item.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            args_raw = fn.get("arguments") if isinstance(fn, dict) else None
            rows = (
                _kv_row("id", item.get("id"))
                + _kv_row("name", name)
                + _kv_row("arguments", _parse_arguments_for_display(args_raw))
            )
            blocks.append(
                f"<div class='entry-call-block'><div class='entry-call-title'>{key} {index + 1}</div>{rows}</div>"
            )
        elif isinstance(item, dict):
            rows = "".join(_kv_row(item_key, item_value) for item_key, item_value in item.items())
            blocks.append(
                f"<div class='entry-result-block'><div class='entry-result-title'>{key} {index + 1}</div>{rows}</div>"
            )
        else:
            blocks.append(
                "<div class='entry-result-block'>"
                f"<div class='entry-result-title'>{key} {index + 1}</div>"
                f"{_kv_row('value', item)}</div>"
            )
    return "".join(blocks) if blocks else _kv_row(key, value)


def _render_structured(kind: str, payload: dict[str, Any]) -> str:
    ordered = _ORDERED_KEYS.get(kind, list(payload.keys()))
    seen: set[str] = set()
    out: list[str] = []
    for key in ordered:
        if key not in payload:
            continue
        seen.add(key)
        out.append(_structured_value(key, payload[key]))
    for key, value in payload.items():
        if key not in seen:
            out.append(_structured_value(key, value))
    if not out:
        out.append(_kv_row("payload", "(empty)"))
    return "<div class='entry-structured'>{}</div>".format("".join(out))


def _render_entry_meta_block(entry: Any) -> str:
    date_val = getattr(entry, "date", None)
    meta = getattr(entry, "meta", None)
    if not date_val and not (isinstance(meta, dict) and meta):
        return ""
    rows: list[str] = []
    if date_val:
        rows.append(_kv_row("date", date_val))
    if isinstance(meta, dict):
        for meta_key, meta_value in meta.items():
            rows.append(_kv_row(str(meta_key), meta_value))
    return f"<div class='entry-meta-block'><div class='entry-meta-title'>Entry metadata</div>{''.join(rows)}</div>"


def _render_log_html(snapshot: ConversationSnapshot, show_system_events: bool = False) -> str:
    context_ids = {entry.id for entry in snapshot.context_entries}
    active_anchor_id = snapshot.active_anchor.entry_id if snapshot.active_anchor else None
    rows: list[str] = []
    for entry in snapshot.entries:
        if not show_system_events and getattr(entry, "kind", "") == "event":
            continue
        is_context = entry.id in context_ids
        is_active_anchor = entry.kind == "anchor" and entry.id == active_anchor_id
        classes = ["tape-entry"]
        if is_context:
            classes.append("in-context")
        if is_active_anchor:
            classes.append("active-anchor")
        payload = getattr(entry, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        human = html.escape(_human_text(entry.kind, payload))
        meta_block = _render_entry_meta_block(entry)
        structured = _render_structured(entry.kind, payload)
        raw_payload = html.escape(json.dumps(payload, ensure_ascii=False, indent=2))
        meta_raw = html.escape(
            json.dumps(
                getattr(entry, "meta", {}) if isinstance(getattr(entry, "meta", None), dict) else {},
                ensure_ascii=False,
                indent=2,
            )
        )
        date_raw = html.escape(str(getattr(entry, "date", "")))
        rows.append(
            f"<details class='{' '.join(classes)}' title='Entry #{entry.id}'>"
            "<summary class='entry-summary'>"
            f"<span class='entry-badge'>{_kind_label(entry.kind)}</span>"
            f"<span class='entry-text'>{human}</span>"
            "</summary>"
            f"{meta_block}"
            f"{structured}"
            "<details class='entry-raw-block'>"
            "<summary class='entry-raw-summary'>Raw fields</summary>"
            f"<pre class='entry-raw'>{meta_raw}</pre>"
            f"<div class='entry-raw-label'>date</div><pre class='entry-raw'>{date_raw}</pre>"
            f"<div class='entry-raw-label'>payload</div><pre class='entry-raw'>{raw_payload}</pre>"
            "</details>"
            "</details>"
        )
    if not rows:
        rows.append("<div class='tape-empty'>Tape is empty. Send a message to begin.</div>")
    return "<div class='tape-list'>{}</div>".format("".join(rows))


def _context_source_label(snapshot: ConversationSnapshot, view_mode: ViewMode) -> str:
    if view_mode == "full":
        return "Full Context"
    if view_mode == "latest":
        if snapshot.active_anchor:
            return f"Latest: {snapshot.active_anchor.label}"
        return "Latest (no anchor)"
    if snapshot.active_anchor:
        return f"Anchor: {snapshot.active_anchor.label}"
    return "Anchor: not found"


def _token_health(estimated_tokens: int) -> tuple[str, str]:
    if estimated_tokens > 3000:
        return "HIGH", "ctx-high"
    if estimated_tokens > 2000:
        return "MODERATE", "ctx-moderate"
    return "OK", "ctx-ok"


def _render_context(snapshot: ConversationSnapshot, view_mode: ViewMode) -> str:
    source = html.escape(_context_source_label(snapshot, view_mode))
    status_label, status_class = _token_health(snapshot.estimated_tokens)
    progress = min(int((snapshot.estimated_tokens / 4000) * 100), 100)
    return (
        "<div class='ctx-bar'>"
        "<div class='ctx-info'>"
        f"<span class='ctx-source'>{source}</span>"
        f"<span class='ctx-stats'>{snapshot.context_entry_count} / {snapshot.total_entries} entries"
        f" &middot; ~{snapshot.estimated_tokens} tok"
        f" &middot; <b class='{status_class}'>{status_label}</b></span>"
        "</div>"
        "<div class='ctx-track'>"
        f"<div class='ctx-fill {status_class}' style='width:{progress}%'></div>"
        "</div>"
        "</div>"
    )


def _render_tape_footer(snapshot: ConversationSnapshot, view_mode: ViewMode) -> str:
    if view_mode == "full":
        left = "All entries in context"
    elif view_mode == "latest":
        left = "From latest anchor"
    else:
        left = f"From: {snapshot.active_anchor.label}" if snapshot.active_anchor else "From: anchor (missing)"
    return (
        f"**{left}** &nbsp; "
        f"{snapshot.context_entry_count} in context &middot; "
        f"{snapshot.total_entries} total &middot; "
        f"~{snapshot.estimated_tokens} tokens"
    )


def _anchor_rows(snapshot: ConversationSnapshot) -> list[list[str]]:
    active_name = snapshot.active_anchor.name if snapshot.active_anchor else ""
    rows: list[list[str]] = []
    for anchor in snapshot.anchors:
        rows.append(
            [
                "\u2713" if anchor.name == active_name else "",
                anchor.label,
                anchor.name,
                anchor.summary or "-",
            ]
        )
    return rows


CSS = """
#main-row { align-items: stretch !important; min-height: 640px; }
#tape-col, #conversation-col { display: flex !important; flex-direction: column !important; min-height: 0; }
#conversation-col > div:nth-child(3) {
  flex: 1 1 0 !important; min-height: 0 !important;
  display: flex !important; flex-direction: column !important;
}
#conv-chatbot { flex: 1 1 0 !important; min-height: 400px !important; }
.tape-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 560px;
  overflow-y: auto;
  padding: 2px 0;
}
.tape-entry {
  display: block;
  padding: 5px 10px; border-radius: 6px;
  border-left: 3px solid transparent;
  transition: background 0.15s;
}
.entry-summary { display: flex; align-items: center; gap: 8px; cursor: pointer; list-style: none; }
.entry-summary::-webkit-details-marker { display: none; }
.tape-entry:hover { background: color-mix(in srgb, var(--body-text-color) 6%, transparent); }
.tape-entry.in-context { border-left-color: #2ea043; background: color-mix(in srgb, #2ea043 8%, transparent); }
.tape-entry.active-anchor { border-left-color: #d29922; background: color-mix(in srgb, #d29922 8%, transparent); }
.tape-empty { padding: 20px; text-align: center; opacity: 0.5; }
.entry-badge {
  font-size: 10px; font-weight: 600; padding: 1px 6px; border-radius: 4px;
  text-transform: uppercase; white-space: nowrap; flex-shrink: 0;
  background: color-mix(in srgb, #8b949e 18%, transparent); color: #8b949e;
}
.entry-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; opacity: 0.85; }
.entry-structured {
  margin: 8px 0 6px 0;
  padding: 8px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--body-text-color) 4%, transparent);
  border: 1px solid var(--border-color-primary);
}
.entry-kv { display: grid; grid-template-columns: 120px 1fr; gap: 8px; padding: 2px 0; font-size: 12px; }
.entry-k { opacity: 0.65; font-family: monospace; }
.entry-v { white-space: pre-wrap; word-break: break-word; }
.entry-call-block, .entry-result-block {
  margin: 8px 0; padding: 8px; border-radius: 6px;
  border: 1px solid var(--border-color-primary);
  background: color-mix(in srgb, var(--body-text-color) 3%, transparent);
}
.entry-call-title, .entry-result-title {
  font-size: 11px; font-weight: 600; opacity: 0.8; margin-bottom: 6px;
}
.entry-meta-block {
  margin: 8px 0 6px 0;
  padding: 8px;
  border-radius: 6px;
  border: 1px dashed var(--border-color-primary);
  background: color-mix(in srgb, var(--body-text-color) 2%, transparent);
}
.entry-meta-title {
  font-size: 11px;
  font-weight: 600;
  opacity: 0.75;
  margin-bottom: 6px;
}
.entry-raw-block { margin: 0 0 2px 0; }
.entry-raw-summary { cursor: pointer; font-size: 12px; opacity: 0.8; }
.entry-raw-label { font-size: 11px; opacity: 0.65; margin: 8px 0 2px 0; }
.entry-raw {
  margin: 6px 0 0 0;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid var(--border-color-primary);
  background: color-mix(in srgb, var(--body-text-color) 2%, transparent);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.ctx-bar { padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border-color-primary); margin-bottom: 6px; }
.ctx-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  flex-wrap: wrap;
}
.ctx-source { font-weight: 600; }
.ctx-stats { opacity: 0.65; }
.ctx-track {
  height: 4px;
  border-radius: 4px;
  background: var(--border-color-primary);
  margin-top: 6px;
  overflow: hidden;
}
.ctx-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.ctx-ok { color: #2ea043; } .ctx-fill.ctx-ok { background: #2ea043; }
.ctx-moderate { color: #d29922; } .ctx-fill.ctx-moderate { background: #d29922; }
.ctx-high { color: #f85149; } .ctx-fill.ctx-high { background: #f85149; }
"""


INPUT_HEAD_JS = """
<script>
(function() {
  function attach() {
    var el = document.getElementById('user-input');
    if (!el) return false;
    var textarea = el.querySelector('textarea');
    if (!textarea || textarea.dataset.ecAttached) return false;
    textarea.dataset.ecAttached = '1';
    textarea.addEventListener('keydown', function(e) {
      if (e.key !== 'Enter') return;
      if (e.shiftKey) {
        e.preventDefault();
        var btn = document.getElementById('send-btn');
        if (btn) btn.click();
      } else {
        e.preventDefault();
        var start = textarea.selectionStart, end = textarea.selectionEnd;
        var val = textarea.value;
        textarea.value = val.slice(0, start) + '\\n' + val.slice(end);
        textarea.selectionStart = textarea.selectionEnd = start + 1;
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    return true;
  }
  var attempts = 0;
  function tryAttach() {
    if (attach() || attempts > 6) return;
    attempts++;
    setTimeout(tryAttach, 300);
  }
  if (document.readyState === 'complete') tryAttach();
  else window.addEventListener('load', tryAttach);
})();
</script>
"""
