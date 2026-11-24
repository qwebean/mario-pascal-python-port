"""Port of PALETTES.PAS — palette state, blinking animations and fades.

PALETTES.PAS works on a shared 6-bit palette array and writes DAC entries
directly.  The port keeps the same 6-bit array in Figures (``palette6`` /
``palette``) and separates the *display* palette:

* the page surfaces are RGB, so a palette change only becomes visible after
  the affected surfaces are re-baked and redrawn — ``apply()`` tells play.py
  whether that is necessary;
* ``BlinkPalette`` touches a small set of entries (star 1, waterfall 7..11,
  coins 12..14, grass 153..159); Figures.invalidate_indices() drops only the
  baked surfaces that use one of them instead of the whole cache;
* fades subtract a constant from every DAC channel, which is implemented in
  VGAScreen.set_fade() as a display-time lookup table.

InitGrass / SetSkyPalette already live in Figures (they are also used by the
rendering tools); this unit delegates to them.
"""
from __future__ import annotations

import os
import random
import time

from .figures import Figures

STEPS = 32
BLINK_SPEED = 25
GRASS_SPEED = 40
COIN_SPEED = 25
WATERFALL_SPEED = 10
WATER_FRAME_SPEED = 80      # frames per 91898 water frame (10x slower)

# --------------------------------------------------------------------------- #
# Frame-timing jitter (2026-10-25, user request: "can a pose be held for a
# random time inside a range instead of a fixed period?").  Off by default so
# every regression keeps the original fixed cadence; ``--anim-jitter`` or
# MARIO_ANIM_JITTER=1 turns it on.  JITTER_SPAN is the +/- fraction of the base
# hold length that gets randomised.
# --------------------------------------------------------------------------- #
ANIM_JITTER = os.environ.get("MARIO_ANIM_JITTER", "0") not in ("0", "", "false")
JITTER_SPAN = 0.35

PE_NO_EFFECT = 0
PE_BLACK_WHITE = 1
PE_EGA_MODE = 2


class Palettes:
    def __init__(self, figures: Figures, screen=None, rng=None) -> None:
        self.figures = figures
        self.screen = screen
        self.rng = rng if rng is not None else random.Random()

        self.lock_palette = False
        self.modify_palette = True
        self.fading_done = True
        self.fading_up = False
        self.fading_down = False
        self.fading_pos = 0
        self.effect = PE_NO_EFFECT

        self.blink_counter = 0
        self.grass_counter = 0
        self.coin_counter = 0
        self.waterfall_counter = 0
        self.water_anim_counter = 0
        # Hold length of each pose.  Equal to the original *_SPEED constants
        # unless ANIM_JITTER is on, in which case a new one is drawn per cycle.
        self.blink_limit = BLINK_SPEED
        self.grass_limit = GRASS_SPEED
        self.coin_limit = COIN_SPEED

        self.dirty = True

    # ------------------------------------------------------------------ #
    # frame timing
    # ------------------------------------------------------------------ #
    def _hold(self, base: int) -> int:
        """Frames a pose is held.

        ``ANIM_JITTER`` off (the default, and what every regression pins)
        returns ``base`` unchanged, so the counters below keep the original
        PALETTES.PAS arithmetic exactly.  On, it draws a new length inside
        ``base * (1 +- JITTER_SPAN)`` each cycle -- the user's "does a random
        swap time look more natural?" experiment.
        """
        if not ANIM_JITTER:
            return base
        lo = max(1, round(base * (1.0 - JITTER_SPAN)))
        hi = max(lo + 1, round(base * (1.0 + JITTER_SPAN)))
        return self.rng.randint(lo, hi)

    # ------------------------------------------------------------------ #
    # palette storage / display
    # ------------------------------------------------------------------ #
    def display_palette(self) -> list[list[int]]:
        if self.effect == PE_NO_EFFECT:
            return self.figures.palette
        out: list[list[int]] = []
        for rgb6 in self.figures.palette6:
            if self.effect == PE_BLACK_WHITE:
                i = sum(rgb6) // 3
                rgb = (i, i, i)
            else:                       # PE_EGA_MODE
                rgb = (rgb6[0] & 0xF0, rgb6[1] & 0xF0, rgb6[2] & 0xF0)
            v = [c * 255 // 63 for c in rgb]
            out.append(v)
        return out

    def apply(self) -> bool:
        """Push the current display palette to the screen.  Returns whether
        anything changed since the last call (caller should redraw then)."""
        changed = self.dirty
        if self.screen is not None:
            self.screen.set_palette(self.display_palette())
        self.dirty = False
        return changed

    def read_palette(self) -> None:
        self.apply()

    def out_palette(self, color: int, r: int, g: int, b: int) -> None:
        if self.modify_palette:
            self.figures.change_palette(color, r, g, b)
        if not self.lock_palette:
            self.dirty = True

    def change_palette(self, color: int, r: int, g: int, b: int) -> None:
        self.figures.change_palette(color, r, g, b)
        self.dirty = True

    def copy_palette(self, c1: int, c2: int) -> None:
        self.out_palette(c2, *self.figures.palette6[c1])

    def new_palette(self) -> None:
        self.figures.reset_palette()
        self.fading_up = False
        self.fading_down = False
        self.dirty = True

    def clear_palette(self) -> None:
        # PALETTES.ClearPalette calls ReadPalette, which reads the DAC when no
        # effect is active -- i.e. it does not clear anything.  Keep the quirk.
        if self.effect == PE_NO_EFFECT:
            return
        if self.screen is not None:
            self.screen.set_palette([(0, 0, 0)] * 256)
        self.dirty = False

    def init_grass(self) -> None:
        self.figures.init_grass()
        self.dirty = True

    def lock_pal(self) -> None:
        self.lock_palette = True

    def unlock_pal(self) -> None:
        self.lock_palette = False

    def set_effect(self, effect: int) -> None:
        self.effect = effect
        self.dirty = True

    # ------------------------------------------------------------------ #
    # BlinkPalette (star / waterfall / coin / grass)
    # ------------------------------------------------------------------ #
    def blink_palette(self) -> bool:
        if self.fading_up or self.fading_down:
            return False
        f = self.figures
        rng = self.rng

        self.out_palette(1, 60 + rng.randrange(4), 55 + rng.randrange(8),
                         30 + rng.randrange(25))

        self.waterfall_counter += 1
        if self.waterfall_counter >= 5 * WATERFALL_SPEED:
            self.waterfall_counter = 0
        i = self.waterfall_counter % WATERFALL_SPEED
        if i == 0:
            j = self.waterfall_counter // WATERFALL_SPEED
            for i in range(5):
                j -= 1
                if j < 0:
                    j = 4
                k = 5 - j
                sky = f.options.SkyType if f.options else 0
                if sky == 0:
                    c = (40 + 3 * k, 50 + 2 * k, 53 + 2 * k)
                elif sky == 1:
                    c = (45 + 3 * k, 52 + 2 * k, 51 + 2 * k)
                elif sky == 2:
                    c = (44 + 3 * k, 53 + 2 * k, 53 + 2 * k)
                elif sky == 3:
                    c = (34 + 3 * k, 40 + 2 * k, 40 + 2 * k)
                elif sky == 4:
                    c = (38 + 3 * k, 47 + 2 * k, 47 + 2 * k)
                elif sky == 5:
                    c = (53 + 2 * k, 53 + 2 * k, 44 + 3 * k)
                elif 6 <= sky <= 8:
                    c = (42 + 4 * k, 5 + k * k, 2 * k)
                elif sky == 10:
                    c = (40 + 4 * k, 45 + 3 * k, 63)
                else:
                    c = (50 + 2 * k, 50 + 2 * k, 50 + 2 * k)
                self.out_palette(7 + i, *c)
        opt = f.options
        if opt is not None and opt.WaterTop >= 0:
            # the 91898 water set has three real frames; each tile cell adds
            # its own stable random offset (Figures._water_cell_frame)
            self.water_anim_counter += 1
            if self.water_anim_counter >= WATER_FRAME_SPEED:
                self.water_anim_counter = 0
                f.water_frame = (f.water_frame + 1) % 3
                self.dirty = True

        self.blink_counter += 1
        if self.blink_counter > self.blink_limit:
            self.blink_limit = self._hold(BLINK_SPEED)
            self.blink_counter = -self.blink_limit
            self.out_palette(159, 52, 43, 21)
        elif self.blink_counter == 0:
            self.out_palette(159, 55, 46, 24)

        self.grass_counter += 1
        if self.grass_counter > self.grass_limit:
            self.grass_limit = self._hold(GRASS_SPEED)
            self.grass_counter = -self.grass_limit
            self.copy_palette(2, 153)
            self.copy_palette(3, 154)
            self.copy_palette(2, 155)
            self.copy_palette(3, 156)
            self.copy_palette(self._horizon_index(), 157)
            self.copy_palette(self._horizon_index(), 158)
        elif self.grass_counter == 0:
            self.copy_palette(self._horizon_index(), 153)
            self.copy_palette(self._horizon_index(), 154)
            self.copy_palette(3, 155)
            self.copy_palette(2, 156)
            self.copy_palette(2, 157)
            self.copy_palette(3, 158)
        # Charlie scenery has real two-frame art instead of a palette swap, so
        # it rides this exact clock: the frame flips on the same tick the
        # original's grass / palm shadows change state (and honours
        # --anim-jitter the same way).
        f.deco_frame = 1 if self.grass_counter >= 0 else 0

        self.coin_counter += 1
        step = self.coin_limit
        if self.coin_counter > 3 * step:
            self.coin_limit = step = self._hold(COIN_SPEED)
            self.coin_counter = 0
            self.out_palette(12, 62, 56, 20)
            self.out_palette(13, 60, 56, 22)
            self.out_palette(14, 63, 63, 36)
        elif self.coin_counter == step:
            self.out_palette(14, 62, 56, 20)
            self.out_palette(12, 60, 56, 22)
            self.out_palette(13, 63, 63, 36)
        elif self.coin_counter == 2 * step:
            self.out_palette(13, 62, 56, 20)
            self.out_palette(14, 60, 56, 22)
            self.out_palette(12, 63, 63, 36)

        self.figures.invalidate_indices(
            {1, 7, 8, 9, 10, 11, 12, 13, 14,
             153, 154, 155, 156, 157, 158, 159})
        if not self.lock_palette:
            self.dirty = True
        return True

    def _horizon_index(self) -> int:
        opt = self.figures.options
        return 0xF0 - (1 if (opt and opt.SkyType == 10) else 0)

    # ------------------------------------------------------------------ #
    # fades
    # ------------------------------------------------------------------ #
    def fade_up(self, n: int = 64, delay_ms: int = 0) -> None:
        if self.effect == PE_EGA_MODE:
            return
        for k in range(n - 1, -1, -1):
            self._fade_step(k, delay_ms)
        self._fade_step(0, 0)

    def fade_down(self, n: int = 64, delay_ms: int = 0) -> None:
        if self.effect == PE_EGA_MODE:
            return
        for k in range(n):
            self._fade_step(k, delay_ms)

    def _fade_step(self, k: int, delay_ms: int) -> None:
        if self.screen is None:
            return
        self.screen.set_fade(k)
        self.screen.show_page()
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    def start_fade_up(self) -> None:
        self.fading_up = True
        self.fading_down = False
        self.fading_pos = 63
        self.fading_done = False

    def start_fade_down(self) -> None:
        self.fading_down = True
        self.fading_up = False
        self.fading_pos = 0
        self.fading_done = False

    def fade(self, delay_ms: int = 0) -> bool:
        """Per-frame fade (unused in the shipped game).  Returns True while a
        fade is in progress."""
        if not (self.fading_up or self.fading_down):
            return False
        self._fade_step(self.fading_pos, delay_ms)
        if self.fading_up:
            if self.fading_pos == 0:
                self.fading_up = False
                self.fading_done = True
            else:
                self.fading_pos -= 1
        if self.fading_down:
            if self.fading_pos == 63:
                self.fading_down = False
                self.fading_done = True
            else:
                self.fading_pos += 1
        return self.fading_up or self.fading_down
