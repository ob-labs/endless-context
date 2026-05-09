from __future__ import annotations

import asyncio
from collections.abc import Iterable

from bub.channels.manager import ChannelManager
from bub.framework import BubFramework


def build_framework() -> BubFramework:
    framework = BubFramework()
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
