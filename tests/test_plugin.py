from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from endless_context.plugin import EndlessContextPlugin


class FakeRuntimeAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        session_id: str,
        prompt: str | list[dict],
        state: dict[str, object],
        allowed_tools,
    ):  # noqa: ANN001
        self.calls.append(
            {
                "mode": "run",
                "session_id": session_id,
                "prompt": prompt,
                "state": dict(state),
                "allowed_tools": set(allowed_tools),
            }
        )
        return "text"


def test_load_state_exposes_channel_name() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace(workspace=Path(".")))

    state = plugin.load_state({"channel": "gradio"}, "gradio:test")

    assert state == {"_channel_name": "gradio"}


def test_load_state_preserves_gradio_view_selection() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace(workspace=Path(".")))

    state = plugin.load_state({"channel": "gradio", "view_mode": "from-anchor", "anchor_name": "handoff:x"}, "s")

    assert state == {
        "_channel_name": "gradio",
        "_gradio_view_mode": "from-anchor",
        "_gradio_anchor_name": "handoff:x",
    }


def test_run_model_excludes_quit_for_gradio() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace(workspace=Path(".")))
    plugin._runtime_agent = FakeRuntimeAgent()  # type: ignore[assignment]

    result = asyncio.run(plugin.run_model("hello", "gradio:test", {"_channel_name": "gradio"}))

    assert result == "text"
    assert plugin._runtime_agent.calls[0]["mode"] == "run"  # type: ignore[index]
    assert "quit" not in plugin._runtime_agent.calls[0]["allowed_tools"]  # type: ignore[index]


def test_run_model_skips_non_gradio() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace(workspace=Path(".")))
    plugin._runtime_agent = FakeRuntimeAgent()  # type: ignore[assignment]

    result = asyncio.run(plugin.run_model("hello", "telegram:test", {"_channel_name": "telegram"}))

    assert result is None
    assert plugin._runtime_agent.calls == []  # type: ignore[attr-defined]


def test_system_prompt_clarifies_gradio_direct_reply() -> None:
    plugin = EndlessContextPlugin(SimpleNamespace(workspace=Path(".")))

    prompt = plugin.system_prompt("hello", {"_channel_name": "gradio"})

    assert "plain-text assistant reply will be delivered" in prompt
    assert "do not call quit" in prompt.lower()
