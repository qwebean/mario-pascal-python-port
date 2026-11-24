"""Port of GLITTER.PAS — the tiny sparkle pixels.

Three layers, exactly like the original:

* ``NewGlitter``  — one pixel with a lifetime.
* ``NewStar``     — a five-pixel cross (centre lives 4 ticks longer).
* ``CoinGlitter`` / ``StartGlitter`` — the fixed coin burst and the random
  sparkle used by enemies / the star power-up.

ShowGlitter runs at the very end of the draw phase: it saves the pixel that is
underneath and writes the sparkle colour.  HideGlitter runs at the start of
the next frame on that page and restores the saved palette index.  A saved 0
means "nothing to restore" (the original uses 0 as its sentinel, so a sparkle
over colour index 0 is simply erased).

Coordinates are world pixels; the original compares X against XView and stores
``Y * VIR_SCREEN_WIDTH + X`` in video memory.  The port draws on the world-wide
page at (X, Y) directly, which is the same mapping figures/enemies use.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..engine.vga import MAX_PAGE
from .buffers import NH, NV, TILE_H, TILE_W

W = TILE_W
H = TILE_H

MAX_GLITTER = 75
STAR_ATTR = 0x1F


@dataclass
class GlitterRec:
    attr: int = 0
    x: int = 0
    y: int = 0
    backgr: list[int] = field(
        default_factory=lambda: [0] * (MAX_PAGE + 1))


class Glitter:
    """GLITTER.PAS: pool of at most 74 sparkle pixels (slot 75 is unused by
    the original's scan loop)."""

    def __init__(self, screen, figures, view_x=None, rows=None, rng=None) -> None:
        self.screen = screen
        self.figures = figures
        self._view_x = view_x if view_x is not None else (lambda: 0)
        self._rows = rows if rows is not None else (lambda: NV)
        self.rng = rng if rng is not None else random.Random()
        self.counts = [0] * (MAX_GLITTER + 1)       # 1..75, like Pascal
        self.items = [GlitterRec() for _ in range(MAX_GLITTER + 1)]
        self.num_glitter = 0

    # ------------------------------------------------------------------ #
    # pixel helpers (palette indices, not RGB)
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
    # creation
    # ------------------------------------------------------------------ #
    def clear_glitter(self) -> None:
        for i in range(1, MAX_GLITTER + 1):
            self.counts[i] = 0
        self.num_glitter = 0

    def new_glitter(self, x: int, y: int, new_attr: int,
                    duration: int) -> None:
        view = self._view_x()
        if x < view or x >= view + NH * W:
            return
        i = 1
        while self.counts[i] > 0 and i < MAX_GLITTER:
            i += 1
        if i < MAX_GLITTER:
            if y < 0 or y > self._rows() * H:
                return
            self.counts[i] = duration
            self.num_glitter += 1
            rec = self.items[i]
            rec.x = x
            rec.y = y
            rec.backgr = [0] * (MAX_PAGE + 1)
            rec.attr = new_attr

    def new_star(self, x: int, y: int, new_attr: int,
                 duration: int) -> None:
        self.new_glitter(x, y, new_attr, duration + 4)
        self.new_glitter(x + 1, y, new_attr, duration)
        self.new_glitter(x, y + 1, new_attr, duration)
        self.new_glitter(x - 1, y, new_attr, duration)
        self.new_glitter(x, y - 1, new_attr, duration)

    def coin_glitter(self, x: int, y: int) -> None:
        self.new_star(x + 5, y + 2, STAR_ATTR, 20)
        self.new_star(x + W - 6, y + 6, STAR_ATTR, 18)
        self.new_star(x + 10, y + H - 3, STAR_ATTR, 16)
        self.new_glitter(x + W - 9, y + 2, STAR_ATTR, 15)
        self.new_glitter(x + 6, y + 7, STAR_ATTR, 17)
        self.new_glitter(x + 3, y + 9, STAR_ATTR, 15)

    def start_glitter(self, x: int, y: int, w: int, h: int) -> None:
        self.new_star(x + self.rng.randrange(w), y + self.rng.randrange(h),
                      STAR_ATTR, 10 + self.rng.randrange(10))
        for _ in range(4):
            self.new_glitter(x + self.rng.randrange(w),
                             y + self.rng.randrange(h),
                             STAR_ATTR, 5 + self.rng.randrange(10))

    # ------------------------------------------------------------------ #
    # per-frame draw / hide
    # ------------------------------------------------------------------ #
    def show_glitter(self) -> None:
        if self.num_glitter <= 0:
            return
        page = self.screen.current
        rev = self._reverse()
        for i in range(1, MAX_GLITTER + 1):
            if self.counts[i] > MAX_PAGE + 1:
                rec = self.items[i]
                rec.backgr[page] = self._get_index(rec.x, rec.y, rev)
                self._put_index(rec.x, rec.y, rec.attr)
            elif self.counts[i] > 0:
                self.items[i].backgr[page] = 0

    def hide_glitter(self) -> None:
        if self.num_glitter <= 0:
            return
        page = self.screen.current
        for i in range(MAX_GLITTER, 0, -1):
            if self.counts[i] > 0:
                rec = self.items[i]
                saved = rec.backgr[page]
                if saved != 0:
                    self._put_index(rec.x, rec.y, saved)
                self.counts[i] -= 1
                if self.counts[i] == 0:
                    self.num_glitter -= 1
