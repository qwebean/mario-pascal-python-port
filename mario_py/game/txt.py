"""Port of TXT.PAS — the two bitmap fonts and WriteText.

Font format (see tools/convert_fonts.py): each glyph starts with a width byte,
a height byte, then a continuous LSB-first bit stream.  SetFont(0) selects
F8X8 (chars 0..127, used for the status bar and its little coin icons),
SetFont(1) selects SWISS (chars 32..127, 24 px high, used for menus).

WriteText mirrors the original's three passes: shadow at (+1,+1) in colour 16,
bold at (-1,0) in the text colour, then the normal glyph at (0,0).  The
horizontal advance is glyph width + 1 per enabled style, exactly like
TextWidth()."""

from __future__ import annotations

import numpy as np
import pygame

SCREEN_WIDTH = 320


NORMAL = 0x00
BOLD = 0x01
SHADOW = 0x02

SHADOW_ATTR = 16


class Txt:
    def __init__(self, screen, figures, assets) -> None:
        self.screen = screen
        self.figures = figures
        self.fonts = assets.fonts
        self.font_name = "f8x8"
        self.bold = False
        self.shadow = False
        # (font, code, attr) -> (rgb, surface); rebuilt when the palette moves
        self._glyphs: dict[tuple[str, int, int],
                           tuple[tuple[int, int, int], pygame.Surface]] = {}
        self.set_font(0, NORMAL)

    # ------------------------------------------------------------------ #
    # font selection / metrics
    # ------------------------------------------------------------------ #
    def set_font(self, i: int, style: int) -> None:
        if i == 0:
            self.font_name = "f8x8"
        elif i == 1:
            self.font_name = "swiss"
        self.bold = (style | BOLD) == style
        self.shadow = (style | SHADOW) == style

    def letter(self, code: int) -> dict | None:
        # glyphs.json is keyed by the actual character code (the converter
        # already applied the font's first_char offset)
        return self.fonts[self.font_name]["glyphs"].get(str(code))

    def text_width(self, s: str) -> int:
        extra = int(self.bold) + int(self.shadow)
        total = 0
        for ch in s:
            g = self.letter(ord(ch))
            if g is not None:
                total += g["w"]
            total += extra
        return total

    # ------------------------------------------------------------------ #
    # glyph surfaces
    # ------------------------------------------------------------------ #
    def _glyph_surface(self, code: int, attr: int) -> pygame.Surface | None:
        g = self.letter(code)
        if g is None:
            return None
        rgb = tuple(self.figures.palette[attr])
        key = (self.font_name, code, attr)
        cached = self._glyphs.get(key)
        if cached is not None and cached[0] == rgb:
            return cached[1]
        h, w = g["h"], g["w"]
        mask = np.array([[c == "1" for c in row] for row in g["rows"]],
                        dtype=bool)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(mask, 255, 0)
        surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA")
        self._glyphs[key] = (rgb, surf)
        return surf

    def _draw(self, code: int, x: int, y: int, attr: int) -> None:
        surf = self._glyph_surface(code, attr)
        if surf is not None:
            self.screen.page.blit(surf, (x, y))

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #
    def write_text(self, x: int, y: int, s: str, attr: int) -> None:
        extra = int(self.bold) + int(self.shadow)
        for ch in s:
            code = ord(ch)
            g = self.letter(code)
            if g is not None:
                if self.shadow:
                    self._draw(code, x + 1, y + 1, SHADOW_ATTR)
                if self.bold:
                    if self.shadow:
                        self._draw(code, x, y + 1, SHADOW_ATTR)
                    self._draw(code, x - 1, y, attr)
                self._draw(code, x, y, attr)
                x += g["w"] + extra
            else:
                x += extra

    def center_x(self, s: str, view_x: int) -> int:
        return view_x + (SCREEN_WIDTH - self.text_width(s)) // 2

    def center_text(self, y: int, s: str, attr: int, view_x: int) -> None:
        self.write_text(self.center_x(s, view_x), y, s, attr)
