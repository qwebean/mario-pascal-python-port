"""Port of STARS.PAS — the parallax star field.

NOTE (provenance): no shipped level enables ``Options.Stars`` (all 45 levels
store 0), and the original's draw asm addresses video memory linearly
(Mode 13h style) while the game actually runs the planar 360-px page — so this
unit never executes in the retail game.  The port therefore implements the
apparent intent with the port's normal page model:

* one star per screen column, at a random row above Options.Horizon;
* ``Options.Stars`` 1/2 also drops ~70% of the columns and picks the two
  palette entries used for the twinkle (C1/C2);
* the field parallaxes with the camera at 1/STAR_SPEED (20) of its speed, so
  the stars stay far behind the mountains, and carries into the next row at
  the 320-px boundary, like the original linear layout;
* stars only paint over bright sky pixels (index >= 0xA0, except 0xF0) and the
  covered index is saved per page / per column so HideStars restores it.

The random layout uses Python's RNG seeded with 0 (the original does
``RandSeed := 0``); it is deterministic but not bit-identical to Turbo Pascal's
LCG.  This is purely cosmetic and unreachable in the shipped levels.
"""
from __future__ import annotations

import random

from ..engine.vga import MAX_PAGE

STAR_SPEED = 20        # the field scrolls at 1/20 of the camera (user
                       # 2026-10-25: the original 0.8x reads as foreground)
SCREEN_W = 320


class Stars:
    def __init__(self, screen, figures, options, rng=None) -> None:
        self.screen = screen
        self.figures = figures
        self.options = options
        self.rng = rng if rng is not None else random.Random(0)
        self.star_map = [0] * SCREEN_W
        self.last_x = [0] * (MAX_PAGE + 1)
        self.backgr = [[0] * SCREEN_W for _ in range(MAX_PAGE + 1)]
        self.blink_counter = 0
        self.c1 = 0
        self.c2 = 0
        self._offset = [0] * (MAX_PAGE + 1)

    # ------------------------------------------------------------------ #
    # palette-index pixel helpers
    # ------------------------------------------------------------------ #
    def _reverse(self) -> dict[tuple[int, int, int], int]:
        rev: dict[tuple[int, int, int], int] = {}
        for i, rgb in enumerate(self.figures.palette):
            rev.setdefault((rgb[0], rgb[1], rgb[2]), i)
        return rev

    def _get_index(self, x: int, y: int, rev) -> int:
        page = self.screen.page
        if not (0 <= x < page.get_width() and 0 <= y < page.get_height()):
            return 0
        c = page.get_at((x, y))
        return rev.get((c.r, c.g, c.b), 0)

    def _put_index(self, x: int, y: int, index: int) -> None:
        page = self.screen.page
        if 0 <= x < page.get_width() and 0 <= y < page.get_height():
            page.set_at((x, y), self.figures.palette[index])

    # ------------------------------------------------------------------ #
    # init / clear
    # ------------------------------------------------------------------ #
    def clear_stars(self) -> None:
        for page in self.backgr:
            for i in range(SCREEN_W):
                page[i] = 0
        self.last_x = [0] * (MAX_PAGE + 1)
        self._offset = [0] * (MAX_PAGE + 1)

    def init_stars(self) -> None:
        self.clear_stars()
        self.rng.seed(0)
        horizon = self.options.Horizon
        for i in range(SCREEN_W):
            self.star_map[i] = (self.rng.randrange(horizon) * SCREEN_W + i
                                if horizon > 0 else 0)
        if self.options.Stars in (1, 2):
            for i in range(SCREEN_W):
                if self.rng.randrange(10) > 2:
                    self.star_map[i] = 0
        if self.options.Stars == 1:
            self.c1, self.c2 = 29, 31
        elif self.options.Stars == 2:
            self.c1, self.c2 = 90, 92

    # ------------------------------------------------------------------ #
    # per-frame draw / hide
    # ------------------------------------------------------------------ #
    def show_stars(self, x_view: int) -> None:
        page = self.screen.current
        self.last_x[page] = x_view
        # Distant layer: the field parallaxes with the camera only (never on
        # its own clock) and at 1/STAR_SPEED of it, so the stars sit far behind
        # the mountains.  Twinkling still runs while Mario stands still.
        offset = x_view // STAR_SPEED
        self._offset[page] = offset
        self.blink_counter = self.rng.randrange(SCREEN_W)
        rev = self._reverse()
        for i in range(SCREEN_W):
            pos = self.star_map[i]
            if pos == 0:
                continue
            linear = pos + offset
            sy = linear // SCREEN_W
            sx = linear % SCREEN_W
            index = self._get_index(sx + x_view, sy, rev)
            if index == 0 or index == 0xF0 or index < 0xA0:
                self.backgr[page][i] = 0
                continue
            self.blink_counter = (self.blink_counter - 1) & 0xFFFF
            color = self.c2 if self.blink_counter == 0 else self.c1
            self.backgr[page][i] = index
            self._put_index(sx + x_view, sy, color)

    def hide_stars(self) -> None:
        page = self.screen.current
        offset = self._offset[page]
        for i in range(SCREEN_W):
            pos = self.star_map[i]
            if pos == 0:
                continue
            saved = self.backgr[page][i]
            if saved == 0:
                continue
            linear = pos + offset
            sy = linear // SCREEN_W
            sx = linear % SCREEN_W
            self._put_index(sx + self.last_x[page], sy, saved)
