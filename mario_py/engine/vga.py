"""VGA/planar emulation layer for the Python port.

Replaces VGA256.PAS. The original drives Mode 13h (320x200, 256 colours) with
page flipping and a planar DrawImage blitter. Here we keep two logical pages
as pygame Surfaces and implement the same primitives the game logic uses:

  - set_view / show_page / swap_pages
  - draw_image / put_image / get_image / fill / draw_part / up_side_down
  - push_backgr / pop_backgr  (background save/restore stack)
  - palette set/fade

Coordinate convention is kept identical to the original: (0,0) top-left of the
*virtual* screen, which is VIR_SCREEN_HEIGHT tall and scrolled by set_view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 200
VIR_SCREEN_WIDTH = SCREEN_WIDTH + 2 * 20
VIR_SCREEN_HEIGHT = 182
YBASE = 9

# The original programs the CRTC (YStart $12 / YEnd $7D) so the 182-line
# virtual screen is stretched over all 200 scanlines: 200/182, no horizontal
# scaling.  Nearest-neighbour source row for each destination row, with the
# display phase bias fitted against the DOSBox-X capture (b ~= 45..60).
STRETCH_BIAS = 50

MAX_PAGE = 1

# Ghost-house vision (Options.Vision).  The sight is the diamond that joins the
# midpoints of the outer edges of the four tiles around Mario's head: vertices
# at (+-1.5 tiles horizontally, 0) and (0, +-1.5 tiles vertically), i.e.
# |dx|/TILE_W + |dy|/TILE_H <= r.  Tiles are 20x14 so that is 30 px sideways and
# 21 px up/down -- the budget is exactly r, no half-tile slack, or the diamond
# swallows a whole extra ring.
# The top rows carry the HUD, which must stay readable in a pitch-black room:
# Status.ShowStatus draws at view_y + HEIGHT (6) over a BAND_HEIGHT (20) band,
# so the always-lit strip is rows 0..25 -- the level's hint text below it does
# *not* escape the dark.
TILE_W = 20
TILE_H = 14
HUD_ROWS = 26
# SkyType 3 / palette index 0xE0, the near-black the black room is filled with.
VISION_DARK = (0, 20, 12)


@dataclass
class VgaConfig:
    width: int = VIR_SCREEN_WIDTH
    height: int = VIR_SCREEN_HEIGHT
    scale: int = 3


class VGAScreen:
    """Two-page 8-bit indexed display emulating the original hardware."""

    def __init__(self, cfg: VgaConfig | None = None) -> None:
        self.cfg = cfg or VgaConfig()
        pygame.init()
        self.window = pygame.display.set_mode(
            (SCREEN_WIDTH * self.cfg.scale, SCREEN_HEIGHT * self.cfg.scale),
            pygame.SCALED)
        pygame.display.set_caption("Mario (Python port)")
        # pages are 32-bit RGBA; the 256-colour palette is applied when sprite
        # surfaces are loaded, so fill()/blit() stay straightforward.
        self.pages: list[pygame.Surface] = [
            pygame.Surface((self.cfg.width, self.cfg.height),
                           pygame.SRCALPHA).convert_alpha()
            for _ in range(MAX_PAGE + 1)
        ]
        self.current = 0
        self.view_x = 0
        self.view_y = 0
        self.palette: list[list[int]] = []
        self._stack: dict[int, tuple[int, int, int, int, bytes]] = {}
        self._next_addr = 1
        self._tick = 0
        # PALETTES.FadeUp/FadeDown subtract a constant from every 6-bit DAC
        # channel.  The port's pages already contain RGB, so the fade is a
        # display-time lookup table instead of a palette write.
        self._fade = 0
        self._fade_lut: np.ndarray | None = None
        self._fade_luts: dict[int, np.ndarray] = {}
        self._vision: list[tuple[int, int, int]] = []
        self.vision_dark = VISION_DARK
        rows = ((VIR_SCREEN_HEIGHT * np.arange(SCREEN_HEIGHT) + STRETCH_BIAS)
                // SCREEN_HEIGHT)
        self._stretch_rows = np.clip(rows, 0, VIR_SCREEN_HEIGHT - 1)
        self._stretch_dst = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    # --- palette ----------------------------------------------------------
    def set_palette(self, rgb: list[list[int]]) -> None:
        self.palette = rgb

    def palette_color(self, index: int) -> tuple[int, int, int]:
        r, g, b = self.palette[index]
        return (r, g, b)

    # --- page handling ----------------------------------------------------
    @property
    def page(self) -> pygame.Surface:
        return self.pages[self.current]

    def set_view(self, x: int, y: int) -> None:
        self.view_x = x
        self.view_y = y

    def set_fade(self, k: int) -> None:
        """Dim the displayed picture by subtracting k from every 6-bit DAC
        channel (PALETTES.FadeUp/FadeDown/Fade).  0 restores normal output."""
        k = max(0, min(63, int(k)))
        self._fade = k
        if k == 0:
            self._fade_lut = None
            return
        lut = self._fade_luts.get(k)
        if lut is None:
            to6 = np.array([round(c * 63 / 255) for c in range(256)])
            lut = (np.clip(to6 - k, 0, 63) * 255 // 63).astype(np.uint8)
            self._fade_luts[k] = lut
        self._fade_lut = lut

    @property
    def fade(self) -> int:
        return self._fade

    # --- ghost-house vision ----------------------------------------------
    def clear_vision(self) -> None:
        """Back to a fully lit picture (every level without Options.Vision)."""
        self._vision = []

    def set_vision(self, cx: int, cy: int, r_tiles: float,
                   lights: list[tuple[int, int, float]] | None = None) -> None:
        """Cut a hard-edged black field around the lit centres for this frame.

        ``cx``/``cy`` are **page** pixels -- the same space every entity is
        drawn in (``Enemies.show_enemies`` blits at ``e.x_pos`` and only uses
        ``view_x`` to cull), so ``_vision_mask`` is the one that subtracts the
        camera.  ``r_tiles`` is Mario's sight radius in tiles, applied per axis
        against the 20x14 tile, and may be fractional (the base is 1.5).

        ``lights`` are ``(x, y, r)`` triples: each light carries **its own**
        radius.  They must not inherit Mario's -- he grows when he eats a coin,
        and deriving the glow from his live radius made every torch, coin and
        mechanism in the room brighten with him, which is absurd.  The mask only
        touches what is *shown*: the page keeps its pixels, so a hidden tile
        still blocks and still hurts.
        """
        self._vision = [(cx, cy, r_tiles)]
        self._vision.extend(lights or ())

    def _vision_mask(self) -> np.ndarray:
        """True = the pixel may be seen; the HUD band is always lit.

        ``(width, height)`` -- the order pygame's ``surfarray.array3d``
        actually returns, verified against a filled test surface: axis 0 is x,
        axis 1 is y.  So the HUD is ``[:, :HUD_ROWS]`` and the sight is the
        diamond |x - cx|/20 + |y - cy|/14 <= r, whose vertices sit on the
        midpoints of the outer edges of the four tiles around the anchor.
        Page centres become screen ones by subtracting the view, which is what
        the ``+ view_x`` below does -- getting that sign wrong is what made the
        circle crawl backwards.
        """
        mask = np.zeros((SCREEN_WIDTH, VIR_SCREEN_HEIGHT), dtype=bool)
        mask[:, :HUD_ROWS] = True
        # whole pixels, no +0.5: the diamond's vertices are meant to land
        # exactly on the outer-edge midpoints, and a half-pixel offset pushes
        # them one step outside their own budget
        dx = np.abs(np.arange(SCREEN_WIDTH) + self.view_x)[:, None]
        dy = np.abs(np.arange(VIR_SCREEN_HEIGHT) + self.view_y)[None, :]
        for cx, cy, r in self._vision:
            mask |= (np.abs(dx - cx) / TILE_W
                     + np.abs(dy - cy) / TILE_H) <= r
        return mask

    def show_page(self) -> None:
        src = self.page
        view_rect = pygame.Rect(self.view_x, self.view_y,
                                SCREEN_WIDTH, VIR_SCREEN_HEIGHT)
        clipped = view_rect.clip(src.get_rect())
        if clipped == view_rect:
            sub = src.subsurface(clipped)
        else:
            # pad with black when the view runs past the page edge
            sub = pygame.Surface((SCREEN_WIDTH, VIR_SCREEN_HEIGHT))
            sub.fill((0, 0, 0))
            if clipped.width > 0 and clipped.height > 0:
                sub.blit(src, (max(0, -self.view_x), max(0, -self.view_y)),
                         clipped)
        # 182 virtual rows -> 200 scanlines (CRTC stretching), then integer
        # upscale to the window.
        pixels = pygame.surfarray.array3d(sub)
        if self._vision:
            # the mask goes on *before* the fade LUT: the black field is a
            # colour of the picture, so during a fade-up it must rise out of
            # pitch black with everything else.  Applied afterwards it stayed
            # at full-strength VISION_DARK, i.e. the whole room glowed dim
            # green while Mario's circle was still invisible.
            mask = self._vision_mask()          # (height, width), like pixels
            pixels = np.where(mask[..., None], pixels,
                              np.array(self.vision_dark, dtype=np.uint8))
        if self._fade_lut is not None:
            pixels = self._fade_lut[pixels]
        pygame.surfarray.blit_array(self._stretch_dst,
                                    pixels[:, self._stretch_rows])
        k = self.cfg.scale
        if k > 1:
            # scale_by() is exact integer nearest-neighbour; transform.scale()
            # filters, which smears every 1-px dark outline (the ground's top
            # edge) into a wide grey band -- the "thick black shadow" the DOS
            # original never shows.
            scaled = pygame.transform.scale_by(self._stretch_dst, k)
        else:
            scaled = pygame.transform.scale(
                self._stretch_dst,
                (SCREEN_WIDTH * k, SCREEN_HEIGHT * k))
        self.window.blit(scaled, (0, 0))
        pygame.display.flip()
        self._tick += 1

    def swap_pages(self) -> None:
        self.current ^= 1

    def clear(self) -> None:
        for page in self.pages:
            page.fill(0)

    # --- drawing primitives ----------------------------------------------
    def fill(self, x: int, y: int, w: int, h: int, attr: int) -> None:
        if self.palette:
            self.page.fill(self.palette_color(attr), pygame.Rect(x, y, w, h))
        else:
            self.page.fill(attr, pygame.Rect(x, y, w, h))

    def put_pixel(self, x: int, y: int, attr: int) -> None:
        if 0 <= x < self.cfg.width and 0 <= y < self.cfg.height:
            color = self.palette_color(attr) if self.palette else (attr, attr, attr)
            self.page.set_at((x, y), color)

    def get_pixel(self, x: int, y: int) -> int:
        if 0 <= x < self.cfg.width and 0 <= y < self.cfg.height:
            return int(self.page.get_at((x, y))[0])
        return 0

    def draw_image(self, x: int, y: int, surf: pygame.Surface) -> None:
        """Blit with colour-key transparency (index 0 == transparent)."""
        self.page.blit(surf, (x, y))

    def put_image(self, x: int, y: int, surf: pygame.Surface) -> None:
        self.page.blit(surf, (x, y))

    def get_image(self, x: int, y: int, w: int, h: int) -> pygame.Surface:
        out = pygame.Surface((w, h))
        out.blit(self.page, (0, 0), pygame.Rect(x, y, w, h))
        return out

    def up_side_down(self, x: int, y: int, surf: pygame.Surface) -> None:
        self.page.blit(pygame.transform.flip(surf, False, True), (x, y))

    # --- background stack (PushBackGr/PopBackGr) --------------------------
    def push_backgr(self, x: int, y: int, w: int, h: int) -> int:
        addr = self._next_addr
        self._next_addr += 1
        rect = pygame.Rect(x, y, w, h)
        clipped = rect.clip(self.page.get_rect())
        if clipped.width <= 0 or clipped.height <= 0:
            self._stack[addr] = (x, y, 0, 0, b"")
            return addr
        pixels = pygame.image.tostring(
            self.page.subsurface(clipped).copy(), "RGBA", False)
        self._stack[addr] = (clipped.x, clipped.y, clipped.width,
                             clipped.height, pixels)
        return addr

    def pop_backgr(self, addr: int) -> None:
        if addr not in self._stack:
            return
        x, y, w, h, pixels = self._stack.pop(addr)
        if w == 0 or h == 0:
            return
        tmp = pygame.image.fromstring(pixels, (w, h), "RGBA")
        self.page.blit(tmp, (x, y))

    def reset_stack(self) -> None:
        self._stack.clear()

    # --- timing -----------------------------------------------------------
    @property
    def tick(self) -> int:
        return self._tick


def load_sprite(path: Path, palette: list[list[int]]) -> pygame.Surface:
    """Load a converted RGBA PNG as an 8-bit colour-keyed surface."""
    surf = pygame.image.load(str(path)).convert_alpha()
    return surf
