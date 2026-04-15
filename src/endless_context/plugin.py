from __future__ import annotations

from typing import Any

from bub import hookimpl
from bub.channels import Channel
from bub.envelope import field_of
from bub.framework import BubFramework
from bub.tools import resolve_tool_names
from bub.types import MessageHandler

from endless_context.agent import DEFAULT_SYSTEM_PROMPT, BubAgent
from endless_context.channel import GradioChannel

GRADIO_SYSTEM_PROMPT = (
    "You are responding through Bub's native gradio channel. "
    "A normal plain-text assistant reply will be delivered to the user automatically via Bub outbound routing. "
    "Do not look for a separate channel skill, do not call quit, and do not stop the session "
    "unless the user explicitly asks. "
    "Use tools only when they are actually needed, then finish with a direct answer."
)


class EndlessContextPlugin:
    def __init__(self, framework: BubFramework) -> None:
        from bub.builtin import tools as _builtin_tools  # noqa: F401

        self._framework = framework
        self._runtime_agent = BubAgent(framework)

    @hookimpl
    def provide_channels(self, message_handler: MessageHandler) -> list[Channel]:
        return [GradioChannel(message_handler, self._framework)]

    @hookimpl
    def load_state(self, message: Any, session_id: str) -> dict[str, object]:
        del session_id
        state: dict[str, object] = {}
        channel_name = field_of(message, "channel")
        if channel_name is not None:
            state["_channel_name"] = str(channel_name)

        view_mode = field_of(message, "view_mode")
        if isinstance(view_mode, str) and view_mode in {"full", "latest", "from-anchor"}:
            state["_gradio_view_mode"] = view_mode

        anchor_name = field_of(message, "anchor_name")
        if isinstance(anchor_name, str) and anchor_name.strip():
            state["_gradio_anchor_name"] = anchor_name.strip()

        return state

    @hookimpl(tryfirst=True)
    async def run_model(self, prompt: str | list[dict], session_id: str, state: dict[str, object]):
        if state.get("_channel_name") != "gradio":
            return None
        allowed_tools = resolve_tool_names(None, exclude={"quit"})
        return await self._runtime_agent.run(
            session_id=session_id,
            prompt=prompt,
            state=state,
            allowed_tools=allowed_tools,
        )

    @hookimpl
    def system_prompt(self, prompt: str | list[dict], state: dict[str, object]) -> str:
        del prompt
        if state.get("_channel_name") == "gradio":
            return f"{DEFAULT_SYSTEM_PROMPT}\n\n{GRADIO_SYSTEM_PROMPT}"
        return DEFAULT_SYSTEM_PROMPT


def register(framework: BubFramework) -> EndlessContextPlugin:
    return EndlessContextPlugin(framework)
