from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import build_framework, resolve_enabled_channels, runtime_workspace_path


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


def test_runtime_workspace_path_uses_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BUB_WORKSPACE_PATH", str(tmp_path))

    assert runtime_workspace_path() == tmp_path.resolve()


def test_build_framework_applies_runtime_workspace(monkeypatch, tmp_path: Path) -> None:
    class FakeFramework:
        def __init__(self) -> None:
            self.workspace = Path("initial")
            self.loaded = False

        def load_hooks(self) -> None:
            self.loaded = True

    monkeypatch.setattr("app.BubFramework", FakeFramework)
    monkeypatch.setenv("BUB_WORKSPACE_PATH", str(tmp_path))

    framework = build_framework()

    assert framework.workspace == tmp_path.resolve()
    assert framework.loaded is True
