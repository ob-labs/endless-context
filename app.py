from __future__ import annotations

import asyncio

from bub.channels.manager import ChannelManager
from bub.framework import BubFramework


def build_framework() -> BubFramework:
    framework = BubFramework()
    framework.load_hooks()
    return framework


def main() -> None:
    framework = build_framework()
    manager = ChannelManager(framework, enabled_channels=["gradio"])
    asyncio.run(manager.listen_and_run())


if __name__ == "__main__":
    main()
