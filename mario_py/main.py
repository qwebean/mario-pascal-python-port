"""Minimal runnable entry point — renders Level 1a and lets you scroll it.

The smoke test for the engine + data + figures pipeline: FIGURES.PAS is now
ported, so the level is drawn exactly the way the game will draw it
(BuildWorld auto-tiling, SmoothFill sky, per-level palette included).

Controls: LEFT/RIGHT to scroll, ESC to quit.
Run with:  py -3.12 -m mario_py.main
"""
from __future__ import annotations

import sys

import pygame

from .engine.input import Keyboard
from .engine.vga import (VIR_SCREEN_HEIGHT, VIR_SCREEN_WIDTH, VGAScreen,
                         VgaConfig)
from .game.assets import load_assets
from .game.backgr import BackGr
from .game.buffers import NH, NV, TILE_W, WorldMap, WorldOptions
from .game.figures import Figures


def build_figures(assets, level: dict) -> tuple[WorldMap, WorldOptions,
                                                Figures, BackGr]:
    """Run the PLAY.PAS BuildLevel setup order for one level."""
    world = WorldMap()
    options = WorldOptions.from_dict(level.get("options"))
    world.load_columns(level["map"], options)

    figures = Figures(assets)
    backgr = BackGr(figures, assets)
    figures.backgr = backgr
    figures.init_sky(options.SkyType)
    figures.init_walls(options.WallType1, options.WallType2,
                       options.WallType3, options)
    figures.init_pipes(options.PipeColor)
    backgr.init_backgr(options.BackGrType, options.Clouds)
    figures.build_world(world, options)
    figures.set_sky_palette()
    backgr.draw_pal_backgr(0)
    figures.init_grass()
    return world, options, figures, backgr


def main() -> int:
    pygame.init()
    assets = load_assets()
    level = assets.levels["1a"]
    world, options, figures, backgr = build_figures(assets, level)

    # the original addresses video memory directly with world coordinates,
    # so the page needs to be world-wide; the 320x200 view scrolls over it
    page_w = max(VIR_SCREEN_WIDTH, (world.x_size + 1) * TILE_W)
    screen = VGAScreen(VgaConfig(width=page_w, height=VIR_SCREEN_HEIGHT,
                                 scale=2))
    screen.set_palette(figures.palette)

    keyboard = Keyboard()
    view_x = 0
    last_view_x = 0
    max_x = max(0, (world.x_size - NH) * TILE_W)
    clock = pygame.time.Clock()

    running = True
    while running and not keyboard.quit:
        keyboard.pump()
        if keyboard.left:
            view_x = max(0, view_x - 4)
        if keyboard.right:
            view_x = min(max_x, view_x + 4)

        screen.set_view(view_x, 0)
        screen.page.fill((0, 0, 0, 255))
        backgr.start_clouds(screen.page, view_x)
        col0 = view_x // TILE_W
        for col in range(col0 - 1, col0 + NH + 2):
            for row in range(NV):
                figures.redraw(screen.page, world, col, row)
        # PLAY.PAS page-init order: tiles, then DrawBackGr, then ReadColorMap
        backgr.draw_backgr(screen.page, view_x, last_view_x)
        backgr.read_color_map(screen.page, view_x)
        if backgr.draw_pal_backgr(view_x):
            screen.set_palette(figures.palette)
        last_view_x = view_x
        screen.show_page()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
