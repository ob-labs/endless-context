from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from endless_context.plugin import EndlessContextPlugin


class FakeRuntimeAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, *, session_id: str, prompt: str | list[dict], state: dict[str, object], allowed_tools):  # noqa: ANN001
        self.calls.append(
            {
                "session_id": session_id,
                "prompt": prompt,
                "state": dict(state),
                "allowed_tools": set(allowed_tools),
            }
        )
        return "stream"


def test_load_state_exposes_channel_name() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace())

    state = plugin.load_state({"channel": "gradio"}, "gradio:test")

    assert state == {"_channel_name": "gradio"}


def test_run_model_stream_excludes_quit_for_gradio() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace())
    plugin._runtime_agent = FakeRuntimeAgent()  # type: ignore[assignment]

    result = asyncio.run(plugin.run_model_stream("hello", "gradio:test", {"_channel_name": "gradio"}))

    assert result == "stream"
    assert "quit" not in plugin._runtime_agent.calls[0]["allowed_tools"]  # type: ignore[index]


def test_run_model_stream_skips_non_gradio() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace())
    plugin._runtime_agent = FakeRuntimeAgent()  # type: ignore[assignment]

    result = asyncio.run(plugin.run_model_stream("hello", "telegram:test", {"_channel_name": "telegram"}))

    assert result is None
    assert plugin._runtime_agent.calls == []  # type: ignore[attr-defined]


def test_system_prompt_clarifies_gradio_direct_reply() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace())

    prompt = plugin.system_prompt("hello", {"_channel_name": "gradio"})

    assert "plain-text assistant reply will be delivered" in prompt
    assert "do not call quit" in prompt.lower()
