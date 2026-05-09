from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import resolve_enabled_channels


class _FakeFramework:
    def get_channels(self, _message_handler):  # noqa: ANN001
        return {
            "gradio": object(),
            "mcp.lifecycle": object(),
            "cli": object(),
            "other.lifecycle": object(),
        }


def test_resolve_enabled_channels_adds_lifecycle_channels() -> None:
    enabled = resolve_enabled_channels(_FakeFramework(), ["gradio"])

    assert enabled == ["gradio", "mcp.lifecycle", "other.lifecycle"]


def test_resolve_enabled_channels_preserves_explicit_entries() -> None:
    enabled = resolve_enabled_channels(_FakeFramework(), ["gradio", "mcp.lifecycle"])

    assert enabled == ["gradio", "mcp.lifecycle", "other.lifecycle"]
