"""Headless smoke test for the converted data + engine pipeline.

Run from the repository root:  python tests/smoke.py
Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402

from mario_py.game.assets import load_assets  # noqa: E402
from mario_py.game.buffers import NH, NV  # noqa: E402
from mario_py.main import build_figures  # noqa: E402

EXPECTED_PACKS = {"8g34", "sunyu", "testrooms", "webangelo", "mario4"}
LEVELS = (
    "1a",                    # main 23-level campaign
    "mario4/1-1a",           # 2004 four-level build
    "mario4/1-4b",
    "testrooms/8-5a",        # ghost-house vision layer
    "testrooms/11-1a",       # water level, style 1
    "testrooms/11-2a",       # water level, style 2
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    pygame.init()
    try:
        assets = load_assets()
        check(len(assets.palette) == 256, "palette must have 256 colours")
        check(len(assets.levels) >= 50, "main campaign data is missing")
        check(len(assets.idx) > 100, "sprite index data is missing")

        missing = EXPECTED_PACKS - set(assets.mod_packs)
        check(not missing, f"missing level packs: {sorted(missing)}")

        page = pygame.Surface((NH * 20, NV * 14))
        for name in LEVELS:
            level = assets.level(name)
            world, options, figures, backgr = build_figures(assets, level)
            check(world.x_size > 0, f"{name}: empty world")
            for col in range(min(NH, world.x_size + 1)):
                for row in range(NV):
                    figures.redraw(page, world, col, row)
        print(f"SMOKE OK: {len(assets.levels)} main levels, "
              f"{len(assets.mod_levels)} mod/test levels, "
              f"{len(EXPECTED_PACKS)} packs, {len(LEVELS)} worlds rendered")
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
