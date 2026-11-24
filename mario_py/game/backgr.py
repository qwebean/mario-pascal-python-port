"""Port of BACKGR.PAS — the parallax background behind every level.

Owns, in original order of importance:

  * ``init_backgr``       — the height map (BOGEN / MOUNT / BOGEN7 / BOGEN26)
                            plus the cloud table;
  * ``draw_backgr``       — PutBackGr, the mountain silhouette that sits on
                            the horizon and scrolls at 1/3 speed (parallax),
                            followed by the clouds;
  * ``draw_sky``          — the textured skies: DrawBricks (PALBRICK000),
                            LargeBricks, Pillar (PALPILL000-002), Windows;
  * ``draw_pal_backgr``   — the palette animators that make those textured
                            skies *move*: the image is static, the palette
                            rotates (BrickPalette / LargeBrickPalette /
                            PillarPalette / WindowPalette);
  * ``draw_backgr_block`` — background restore for BumpBlock / TempObj;
  * ``read_color_map``    — the column DrawBackGrBlock samples per row;
  * ``smooth_fill``       — delegated to Figures (already pixel-verified).

Coordinate convention: world coordinates on a world-wide page, identical to
Figures.  The original addressed a 360-px cylindrical VGA page with XView;
the port keeps a world-wide page so the wrap disappears, and PutBackGr's
``XStart = XView div Speed`` parallax mapping is preserved exactly.

Clouds: the original ``StartClouds``/``PutClouds``/``TraceCloud`` pair is
ported but never exercised by the shipped levels (``Options.Clouds`` is 0
everywhere) and its incremental paint/erase pass is internally inconsistent
(the full-body trace and the two edge traces disagree, and StartClouds runs
with ``N = -CloudSpeed`` so it only erases).  The port therefore keeps the
same seven ``CLOUD_MAP`` cloud positions and the two 19-row edge lists, but
draws them as a normal layer after the tiles: each cloud is rebuilt from its
left/right edge offsets and only paints over sky pixels ($E0).  ``Clouds`` is
a palette index, not a boolean (15 = white).  Screen parallax is the
original's 1/4 camera speed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from .buffers import MAX_WORLD_SIZE, NH, NV, TILE_H, TILE_W

MOON_SKY_FILE = (Path(__file__).resolve().parents[1] / "data" / "backgr"
                 / "xp_night_moon.npy")

# --- BACKGR.PAS constants ---------------------------------------------------
LEFT, RIGHT = 0, 1
SPEED = 3                    # mountain scroll divisor (parallax)
BRICK_SPEED = 2              # palette animation divisor
HEIGHT = 26                  # height-map band depth, in rows
CLOUD_SPEED = 4
MAX_CLOUDS = 7
MIN_CLOUD_SIZE = 30
MAX_CLOUD_SIZE = 70
CLOUD_HEIGHT = 19
SHIFT = 16                   # ReadColorMap's sample column offset

SCREEN_WIDTH = NH * TILE_W   # 320
MAP_SIZE = (MAX_WORLD_SIZE // SPEED) * TILE_W     # BackGrMap[0..Max] = 1560

# InitBackGr: BackGrType -> (height map, inverted).  Types 1, 9, 10 store
# ``Height - BackGrMap[i] + 1``; type 11 leaves the (never-loaded, zero) map
# alone, which renders as a solid silhouette.
HEIGHT_SOURCES = {1: ("bogen", True), 2: ("bogen", False),
                  3: ("mount", False), 9: ("bogen7", True),
                  10: ("bogen26", True)}
MOUNTAIN_BACKGROUNDS = frozenset({1, 2, 3, 9, 10, 11})
TEXTURED_SKIES = frozenset({4, 5, 6, 7})
SMOOTH_SKIES = frozenset({2, 5, 9, 10, 11})

# TraceCloud shape lists, read byte-for-byte from the inline data.  Each of
# the 19 rows starts at the previous row's start plus its entry, so the
# entries accumulate.  The left list is the left cap inset, the right list the
# right cap outset (they sum to 9 on every row).
CLOUD_LEFT = (9, -3, -2, -1, -1, -1, 0, -1, 0, 0, 0, 0, 1,
              0, 1, 1, 1, 2, 3)
CLOUD_RIGHT = (0, 3, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, -1, 0, -1,
               -1, -1, -2, -3)


def _cumulative(values: tuple[int, ...]) -> tuple[int, ...]:
    out, run = [], 0
    for v in values:
        run += v
        out.append(run)
    return tuple(out)


CLOUD_LEFT_OFF = _cumulative(CLOUD_LEFT)
CLOUD_RIGHT_OFF = _cumulative(CLOUD_RIGHT)

# InitClouds: the random generator in the source is commented out and the
# table is hard-coded.  CloudMap[i, 0] / [i, 1] = left x / y of cloud i,
# CloudMap[i + MaxClouds, 0] = right x (its y is never read).
CLOUD_MAP = {
    1: (50, 58), 2: (180, 20), 3: (430, 40), 4: (570, 15),
    5: (840, 30), 6: (980, 60), 7: (1200, 20),
    8: (92, 0), 9: (228, 0), 10: (484, 0), 11: (600, 0),
    12: (900, 0), 13: (1040, 0), 14: (1240, 0),
}


class BackGr:
    """Everything BACKGR.PAS owns: height maps, clouds, textured skies and
    the palette animators behind them."""

    def __init__(self, figures, assets) -> None:
        self.figures = figures
        self.assets = assets
        self.background = 0
        self.clouds = 0
        self.moon_sky: pygame.Surface | None = None
        self.height_map: np.ndarray = np.zeros(0, dtype=np.int16)
        # ColorMap[i] = pixel[XView+16, i] * 256 + pixel[XView+17, i]
        self.color_map: list[int] = [0] * (NV * TILE_H)

    # ------------------------------------------------------------------ #
    # level setup
    # ------------------------------------------------------------------ #
    def init_backgr(self, new_background: int, b_clouds: int) -> None:
        self.background = new_background
        if new_background in HEIGHT_SOURCES:
            name, inverted = HEIGHT_SOURCES[new_background]
            raw = self.assets.backgr[name]
            if inverted:
                # BackGrMap[i] := Height - BackGrMap[i] + 1
                self.height_map = HEIGHT + 1 - raw
            else:
                self.height_map = raw.copy()
        else:
            # type 11 (and anything unlisted): the map is never loaded
            self.height_map = np.zeros(MAP_SIZE, dtype=np.int16)
        self.clouds = b_clouds
        if self.clouds != 0:
            self._init_clouds()

    def init_moon_sky(self, enabled: int) -> None:
        """Load the port-only XP moon backdrop when a level opts in."""
        if not enabled:
            self.moon_sky = None
            return
        if self.moon_sky is None:
            arr = np.load(MOON_SKY_FILE)              # (h, w, 3)
            self.moon_sky = pygame.surfarray.make_surface(
                np.ascontiguousarray(arr.swapaxes(0, 1)))

    def draw_moon_sky(self, target: pygame.Surface, x: int,
                      y: int = 0) -> None:
        """Screen-fixed backdrop; the caller skips the normal sky fill."""
        if self.moon_sky is not None:
            target.blit(self.moon_sky, (x, y))

    def _init_clouds(self) -> None:
        # the original seeds a random table here, but ships it hard-coded in
        # CLOUD_MAP, so there is nothing to compute
        # the unloaded height map (type 11) is zero-filled to the map size so
        # the parallax indexer can clamp against it
        pass

    # ------------------------------------------------------------------ #
    # the mountain (PutBackGr / DrawBackGr)
    # ------------------------------------------------------------------ #
    def draw_backgr(self, target: pygame.Surface, view_x: int,
                    last_view_x: int = 0, first_time: bool = False,
                    horizon: int | None = None,
                    span: tuple[int, int] | None = None) -> None:
        """DrawBackGr (FirstTime): the mountain band, then the clouds.

        The original redraws incrementally (only columns whose height changed
        plus the scrolled-in strip); the port re-draws the visible window
        every frame, which produces the same pixels.  ``horizon`` is the
        already-YOffset-adjusted horizon PLAY.PAS uses when calling this from
        MoveScreen (``Horizon + GetYOffset - YBASE``).  ``span`` overrides the
        world columns covered (render_level uses it to paint the whole static
        reference with the view-0 parallax).
        """
        opt = self.figures.options
        if opt is not None and opt.WaterTop >= 0:
            # an underwater room has no distant mountain; the water layer is
            # the backdrop (a waterline level keeps flat sky above it)
            return
        if self.background in MOUNTAIN_BACKGROUNDS:
            self._put_backgr(target, view_x,
                             self.figures.options.Horizon
                             if horizon is None else horizon, span)

    def _put_backgr(self, target: pygame.Surface, view_x: int,
                    horizon: int,
                    span: tuple[int, int] | None = None) -> None:
        top = horizon - HEIGHT
        if top < 0:
            top = 0
        # the fill loop runs rows 0..Height inclusive (27 rows; the last one
        # lands on the first ground row, where the conditional write is a
        # no-op over the tiles)
        band = min(HEIGHT + 1, target.get_height() - top)
        width = target.get_width()
        if span is None:
            # the visible window, in world coordinates
            x0, x1 = max(view_x, 0), min(view_x + SCREEN_WIDTH, width)
        else:
            x0, x1 = max(0, min(span[0], width)), max(0, min(span[1], width))
        if x1 <= x0 or band <= 0:
            return
        # XStart := XView div Speed, +1 per column right of the view's left
        # edge (the plane/bank bookkeeping in the asm works out to exactly
        # this for every XView mod 4)
        delta = np.arange(x0, x1, dtype=np.int32) - view_x
        idx = np.clip(view_x // SPEED + delta, 0, len(self.height_map) - 1)
        sky = self.height_map[idx]
        rows = np.arange(band, dtype=np.int32)[:, None]
        band_idx = np.where(rows < sky[None, :], 0xE0, 0xF0).astype(np.uint8)
        # the conditional stores: only pixels currently $E0 or $F0 flip
        self.figures.blit_masked(target, x0, top, band_idx, (0xE0, 0xF0))

    # ------------------------------------------------------------------ #
    # clouds (Options.Clouds = palette index; 0 disables)
    # ------------------------------------------------------------------ #
    def draw_clouds(self, target: pygame.Surface, view_x: int) -> None:
        """Paint the seven CLOUD_MAP clouds for the visible window.

        Clouds move at 1/4 camera speed (the original ``XView - XView div 4``
        parallax), so they only move while the camera moves.  Each cloud is a
        19-row lens: row r spans ``x1 + left[r] .. x2 + right[r]`` from the
        accumulated edge lists.  Pixels are replaced only where the page
        currently holds the sky colour $E0, which keeps clouds behind palms
        and the mountain silhouette.
        """
        if self.clouds == 0:
            return
        if self.figures.options is not None and self.figures.options.WaterTop >= 0:
            return
        f = self.figures
        attr = np.asarray(f.palette[self.clouds], dtype=np.uint8)
        sky = np.asarray(f.palette[0xE0], dtype=np.uint8)
        par = view_x - view_x // CLOUD_SPEED
        right_edge = view_x + SCREEN_WIDTH
        height = target.get_height()
        for i in range(1, MAX_CLOUDS + 1):
            x1 = par + CLOUD_MAP[i][0]
            x2 = par + CLOUD_MAP[i + MAX_CLOUDS][0]
            y = CLOUD_MAP[i][1]
            bx0 = max(view_x, x1)
            bx1 = min(right_edge, x2 + max(CLOUD_RIGHT_OFF) + 1)
            by0 = max(0, y)
            by1 = min(height, y + len(CLOUD_LEFT_OFF))
            if bx0 >= bx1 or by0 >= by1:
                continue
            w, h = bx1 - bx0, by1 - by0
            sub = target.subsurface(pygame.Rect(bx0, by0, w, h))
            px = np.frombuffer(pygame.image.tostring(sub, "RGB", False),
                               dtype=np.uint8).reshape(h, w, 3).copy()
            mask = np.zeros((h, w), dtype=bool)
            for r, (lo, ro) in enumerate(zip(CLOUD_LEFT_OFF,
                                             CLOUD_RIGHT_OFF)):
                yy = y + r - by0
                if not (0 <= yy < h):
                    continue
                lx = max(x1 + lo, bx0)
                rx = min(x2 + ro, bx1 - 1)
                if lx <= rx:
                    mask[yy, lx - bx0:rx - bx0 + 1] = True
            paint = mask & np.all(px == sky, axis=-1)
            if not paint.any():
                continue
            out = px.copy()
            out[paint] = attr
            surf = pygame.image.frombuffer(
                np.ascontiguousarray(out).tobytes(), (w, h), "RGB")
            target.blit(surf, (bx0, by0))

    # ------------------------------------------------------------------ #
    # textured skies (FIGURES.DrawSky delegates here for Sky 6, 7, 8)
    # ------------------------------------------------------------------ #
    def draw_sky(self, target: pygame.Surface, x: int, y: int,
                 w: int, h: int) -> None:
        """The textured sky for one rectangle (FIGURES.DrawSky dispatches)."""
        b = self.background
        if b == 4:
            self.draw_bricks(target, x, y, w, h)
        elif b == 5:
            self.large_bricks(target, x, y, w, h)
        elif b == 6:
            self.pillar(target, x, y, w, h)
        elif b == 7:
            self.windows(target, x, y, w, h)
        else:
            # The original has no else branch: types 0/8 (and anything above
            # 7) draw nothing and leave whatever was in VRAM.  The port's page
            # persists across frames, so that stale content reads as motion
            # blur behind Mario -- fill the flat sky colour instead.
            self._fill(target, x, y, w, h, 0xE0)

    def draw_bricks(self, target: pygame.Surface, x: int, y: int,
                    w: int, h: int) -> None:
        # PutImage (X, Y, W, H, PALBRICK000) -- opaque, top-left W*H of the
        # 20x14 tile (the plane-major blitter reads the same pixels)
        target.blit(self.figures.sprite_surf("PALBRICK000", True), (x, y),
                    pygame.Rect(0, 0, w, h))

    def pillar(self, target: pygame.Surface, x: int, y: int,
               w: int, h: int) -> None:
        # case (X div W) mod 3 of -- one of the three 20x14 pillar frames
        frame = (x // w) % 3
        target.blit(self.figures.sprite_surf(f"PALPILL00{frame}", True),
                    (x, y), pygame.Rect(0, 0, w, h))

    def large_bricks(self, target: pygame.Surface, x: int, y: int,
                     w: int, h: int) -> None:
        """LargeBricks: a 32-px-period gradient with two solid mortar rows
        and a stagger that flips every 16 rows."""
        out = np.empty((h, w), dtype=np.uint8)
        bl = (x & 0x1F) + 0xE0
        if (y + 14) & 0x10:
            bl ^= 0x10
        cols = np.arange(w, dtype=np.int32)
        for row in range(h):
            d = (y + row) & 0x0F
            if d == 2:
                out[row, :] = 0xD4
            elif d == 1:
                out[row, :] = 0xD1
            elif d == 0:
                out[row, :] = 0xD6
                bl ^= 0x10
            else:
                out[row, :] = ((bl + cols) & 0x1F) + 0xE0
        target.blit(self.figures.bake(out, True), (x, y))

    def windows(self, target: pygame.Surface, x: int, y: int,
                w: int, h: int) -> None:
        """Windows: the same 32-px gradient, with three white rows every 32
        (offset by 22) -- the lit window bands."""
        out = np.empty((h, w), dtype=np.uint8)
        cols = np.arange(w, dtype=np.int32)
        base = x & 0x1F
        for row in range(h):
            if ((y + 22 + row) & 0x1F) < 3:
                out[row, :] = 1
            else:
                out[row, :] = ((base + cols) & 0x1F) | 0xE0
        target.blit(self.figures.bake(out, True), (x, y))

    # ------------------------------------------------------------------ #
    # background restore (BumpBlock / TempObj)
    # ------------------------------------------------------------------ #
    def draw_backgr_block(self, target: pygame.Surface, x: int, y: int,
                          w: int, h: int) -> None:
        opt = self.figures.options
        wy = opt.water_y
        if opt.WaterFillTop and wy >= 0 and y < wy and y + h <= wy:
            self.figures.draw_water_fill(target, x, y, w, h)
            return
        if wy >= 0 and y + h > wy:
            if y < wy:
                if opt.WaterFillTop:
                    self.figures.draw_water_fill(target, x, y, w, wy - y)
                else:
                    self._draw_backgr_block_plain(target, x, y, w, wy - y)
            self.figures.draw_water(target, x, max(y, wy), w, y + h - max(y, wy))
            return
        self._draw_backgr_block_plain(target, x, y, w, h)

    def _draw_backgr_block_plain(self, target: pygame.Surface, x: int, y: int,
                                 w: int, h: int) -> None:
        sky = self.figures.options.SkyType
        if sky in SMOOTH_SKIES:
            self.figures.smooth_fill(target, x, y, w, h)
            return
        b = self.background
        if b == 4:
            self.draw_bricks(target, x, y, w, h)
        elif b == 5:
            self.large_bricks(target, x, y, w, h)
        elif b == 6:
            self.pillar(target, x, y, w, h)
        elif b == 7:
            self.windows(target, x, y, w, h)
        else:
            # Fill (X, Y+i, W, 1, ColorMap [Y+i]) -- a Word truncated to its
            # low byte, i.e. the pixel at XView+17 on that row
            for i in range(h):
                row = y + i
                c = (self.color_map[row] & 0xFF
                     if 0 <= row < len(self.color_map) else 0)
                self._fill(target, x, row, w, 1, c)

    def draw_backgr_map(self, target: pygame.Surface, y1: int, y2: int,
                        shift: int, c: int) -> None:
        """MARIO.PAS only: tint the mountain shape on the title screen."""
        n = len(self.height_map)
        for i in range(SCREEN_WIDTH):
            top = y1 - int(self.height_map[min(i + shift, n - 1)])
            if top < 0:
                top = 0
            for j in range(top, y2 + 1):
                if self._get_index(target, i, j) >= 0xC0:
                    self._put_pixel(target, i, j, c)

    # ------------------------------------------------------------------ #
    # palette animators (DrawPalBackGr)
    # ------------------------------------------------------------------ #
    def draw_pal_backgr(self, view_x: int) -> bool:
        """i := Round (XView / BrickSpeed); rotates the sky palette.  Returns
        whether the palette changed (so the caller can re-apply it)."""
        b = self.background
        if b not in TEXTURED_SKIES:
            return False
        # The DOS skies are distant background: their palette only rotates with
        # XView, exactly like the parallax mountains.  Standing still freezes
        # them (no port-only clock term).
        i = round(view_x / BRICK_SPEED)
        if b == 4:
            self.brick_palette(i)
        elif b == 5:
            self.large_brick_palette(i)
        elif b == 6:
            self.pillar_palette(i)
        else:
            self.window_palette(i)
        self.figures.bump_palette(0xC0)
        return True

    def brick_palette(self, i: int) -> None:
        i %= 20
        for j in range(20):
            if i == j:
                self.figures.copy_palette(0xFE, 0xE0 + j)
            elif (i + 2) % 20 == j:
                self.figures.copy_palette(0xFF, 0xE0 + j)
            else:
                self.figures.copy_palette(0xFD, 0xE0 + j)

    def large_brick_palette(self, i: int) -> None:
        i %= 32
        for j in range(32):
            if j == i or j == (i + 1) % 32:
                self.figures.copy_palette(0xD6, 0xE0 + j)
            elif j == (i + 3) % 32 or j == (i + 4) % 32:
                self.figures.copy_palette(0xD4, 0xE0 + j)
            else:
                self.figures.copy_palette(0xD1, 0xE0 + j)

    def window_palette(self, i: int) -> None:
        i %= 32
        for j in range(6):
            self.figures.copy_palette(1, 0xE0 + ((i + j) % 32))
        for j in range(6, 32):
            self.figures.copy_palette(16, 0xE0 + ((i + j) % 32))

    def pillar_palette(self, i: int) -> None:
        shadow_pos, shadow_end = 28, 36
        pal6 = self.figures.palette6
        opt = self.figures.options
        c1, c2, c3 = pal6[opt.BackGrColor1]
        i %= 60
        j, k = 0, 1
        while k < 15:
            for l in range(j, k + 1):
                c = (c1 + k, c2 + k, c3 + k)
                self.figures.change_palette(0xC0 + ((l + i) % 60), *c)
                self.figures.change_palette(
                    0xC0 + ((shadow_pos + i - l) % 60), *c)
            j, k = k, k + 1
        for jj in range(shadow_pos, shadow_end + 1):
            c1 = c1 - 1 if c1 > 0 else 0
            c2 = c2 - 1 if c2 > 0 else 0
            c3 = c3 - 1 if c3 > 0 else 0
            self.figures.change_palette(0xC0 + ((jj + i) % 60), c1, c2, c3)
        c1, c2, c3 = pal6[opt.BackGrColor2]
        for jj in range(shadow_end + 1, 60):
            self.figures.change_palette(0xC0 + ((i + jj) % 60), c1, c2, c3)

    # ------------------------------------------------------------------ #
    # ColorMap (ReadColorMap)
    # ------------------------------------------------------------------ #
    def read_color_map(self, target: pygame.Surface, view_x: int) -> None:
        rev = self._reverse_palette()
        # Sample the level's full height (the page may be taller than the
        # level because it is allocated for the tallest level in the game).
        rows = self.figures.options.YSize or NV
        size = min(target.get_height(), rows * TILE_H)
        self.color_map = []
        for i in range(size):
            hi = rev.get(tuple(target.get_at((view_x + SHIFT, i))[:3]), 0)
            lo = rev.get(tuple(target.get_at((view_x + SHIFT + 1, i))[:3]), 0)
            self.color_map.append(hi * 256 + lo)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _fill(self, target: pygame.Surface, x: int, y: int,
              w: int, h: int, attr: int) -> None:
        target.fill(self.figures.palette[attr], pygame.Rect(x, y, w, h))

    def _put_pixel(self, target: pygame.Surface, x: int, y: int,
                   attr: int) -> None:
        if 0 <= x < target.get_width() and 0 <= y < target.get_height():
            target.set_at((x, y), self.figures.palette[attr])

    def _get_index(self, target: pygame.Surface, x: int, y: int) -> int:
        if not (0 <= x < target.get_width() and 0 <= y < target.get_height()):
            return 0
        return self._reverse_palette().get(
            tuple(target.get_at((x, y))[:3]), 0)

    def _reverse_palette(self) -> dict[tuple[int, int, int], int]:
        # exact-colour -> index; first index wins on duplicates
        rev: dict[tuple[int, int, int], int] = {}
        for i, rgb in enumerate(self.figures.palette):
            key = (rgb[0], rgb[1], rgb[2])
            rev.setdefault(key, i)
        return rev
