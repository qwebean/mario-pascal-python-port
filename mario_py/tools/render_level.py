"""Headless render of a level with the Figures port — the figures.py smoke test.

Renders a level to a PNG using exactly the code path the game will use, but
with no window:

  levels.json -> WorldMap -> Figures.init_sky / init_walls / init_pipes /
    build_world / set_sky_palette -> Figures.redraw per tile -> PNG

Usage:
  py -3.12 mario_py/tools/render_level.py [level] [--cols N] [--out PATH]
  py -3.12 mario_py/tools/render_level.py 1a --water-top 0 --water-style 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame                              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mario_py.game.assets import load_assets   # noqa: E402
from mario_py.game.backgr import BackGr          # noqa: E402
from mario_py.game.buffers import TILE_H, TILE_W, WorldMap, \
    WorldOptions                              # noqa: E402
from mario_py.game.figures import Figures       # noqa: E402


def render(level: str, cols: int | None, out: Path,
         deco_frame: int = 0, water_top: int | None = None,
         water_style: int | None = None) -> Path:
    assets = load_assets()
    lvl = assets.level(level)
    world = WorldMap()
    opts = dict(lvl.get("options") or {})
    if water_top is not None:
        opts["WaterTop"] = water_top
    if water_style is not None:
        opts["WaterStyle"] = water_style
    options = WorldOptions.from_dict(opts)
    rows = int(lvl.get("rows") or len(lvl["map"][0]))
    world.load_columns(lvl["map"], options, rows=rows,
                       vertical=lvl.get("vertical"))

    fig = Figures(assets)
    fig.water_seed = 0          # deterministic preview layout
    backgr = BackGr(fig, assets)
    fig.backgr = backgr
    fig.init_sky(options.SkyType)
    fig.init_walls(options.WallType1, options.WallType2, options.WallType3,
                   options)
    fig.init_pipes(options.PipeColor)
    backgr.init_backgr(options.BackGrType, options.Clouds)
    fig.build_world(world, options)
    fig.set_sky_palette()
    backgr.draw_pal_backgr(0)
    fig.init_grass()
    fig.deco_frame = deco_frame

    n = world.x_size if cols is None else max(1, min(cols, world.x_size))
    surf = pygame.Surface((n * TILE_W, rows * TILE_H))
    surf.fill((0, 0, 0))
    for x in range(n):
        for y in range(rows):
            fig.redraw(surf, world, x, y)
    # a static full-world render: the mountain is view-dependent (parallax),
    # so cover the whole image with the view-0 mapping
    backgr.draw_backgr(surf, 0, 0, span=(0, n * TILE_W))
    # clouds are a layer on top of the tiles (they used to be painted before
    # them, where Redraw simply erased them again)
    backgr.draw_clouds(surf, 0)
    backgr.read_color_map(surf, 0)
    fig.draw_decorations(surf, world, 0, n - 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(out))
    print(f"rendered {out}  ({surf.get_width()}x{surf.get_height()})")
    print(f"level {level}: {world.x_size} columns x {rows} rows, "
          f"SkyType={options.SkyType} WallType1={options.WallType1} "
          f"Design={options.Design} BackGrType={options.BackGrType} "
          f"Horizon={options.Horizon}")
    # quick tile-code census after BuildWorld
    from collections import Counter
    codes: Counter[int] = Counter()
    for x in range(n):
        for y in range(rows):
            codes[world.get(x, y)] += 1
    shown = {k: v for k, v in codes.items() if k not in (0x20,)}
    print("tile codes after build_world:",
          ", ".join(f"{k}({chr(k) if 32 <= k < 127 else '?'})={v}"
                    for k, v in sorted(shown.items())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("level", nargs="?", default="1a")
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--deco-frame", type=int, default=0,
                    help="Charlie scenery frame (the game alternates 0/1)")
    ap.add_argument("--water-top", type=int, default=None,
                    help="override Options.WaterTop (-1 off, 0 whole dive, N surface row)")
    ap.add_argument("--water-style", type=int, default=None,
                    help="override Options.WaterStyle (1 SMB3, 2 DOS)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / f"render_{args.level}.png"
    render(args.level, args.cols, out, args.deco_frame,
           args.water_top, args.water_style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
