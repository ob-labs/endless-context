from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from pathlib import Path

from bub.channels.manager import ChannelManager
from bub.framework import BubFramework


def runtime_workspace_path() -> Path:
    raw_path = os.environ.get("BUB_WORKSPACE_PATH")
    if raw_path and raw_path.strip():
        return Path(raw_path).expanduser().resolve()
    return Path.cwd().resolve()


def build_framework() -> BubFramework:
    framework = BubFramework()
    framework.workspace = runtime_workspace_path()
    framework.load_hooks()
    return framework


def resolve_enabled_channels(framework: BubFramework, primary_channels: Iterable[str]) -> list[str]:
    enabled = list(dict.fromkeys(primary_channels))
    for channel_name in framework.get_channels(lambda _message: None):
        if channel_name.endswith(".lifecycle") and channel_name not in enabled:
            enabled.append(channel_name)
    return enabled


def main() -> None:
    framework = build_framework()
    manager = ChannelManager(
        framework,
        enabled_channels=resolve_enabled_channels(framework, ["gradio"]),
    )
    asyncio.run(manager.listen_and_run())


if __name__ == "__main__":
    main()
