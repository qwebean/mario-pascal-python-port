"""Port of FIGURES.PAS — tile figure tables, sky palettes and Redraw().

The original builds whole tile families at level load by manipulating palette
*indices* (ReColor / ReColor2 / Mirror / Rotate / ConvertGrass) and stores them
in FigList[1..3, 1..13] and Bricks[0..3].  This keeps the same structure: every
table holds a raw (h, w) uint8 index array, and RGB surfaces are produced on
demand from a *private* palette copy (SetSkyPalette rewrites $D1/$D4/$D6 and
$E0..$FF) and cached.

Coordinate convention (identical to the original): Redraw(X, Y) draws world
column X / row Y at pixel (X*W, Y*H).  The original addresses VGA memory
directly with world coordinates, so the caller's target surface must be at
least (XSize*W) wide; the visible 320x200 window is then cut out of it by
VGAScreen.set_view / show_page.
"""
from __future__ import annotations

import random

import numpy as np
import pygame

from .buffers import (CAN_HOLD_YOU, CAN_STAND_ON, NH, TILE_H, TILE_W,
                      WorldMap, WorldOptions)

N1 = 3          # number of wall sets (FigList[1..N1])
N2 = 13         # slots per wall set (FigList[..., 1..N2])

# cp865 tile codes, as byte values.
CH_SHARP2 = 225      # 'ß'  (placed by '²' in BuildWorld)
CH_INVISIBLE = 173   # '¡'  (extend the tile below one row upwards)
CH_LGUILLEMET = 174  # '«'  (copy tile above; mark row world.rows with #254)
CH_CURRENCY = 175    # '¤'  (copy tile above; mark row world.rows with #255)
CH_SUPER_2 = 253     # '²'  (question block + 'ß' marker)
CH_SUPER_N = 252     # 'ⁿ'  (coin)
CH_EXIT_TOP = 254    # '■'
CH_GRASS_TOP = 247   # '≈'
CH_FENCE = 240       # '≡'
CH_PALM_TOP = 246    # '÷'
CH_PALM_TRUNK_T = 244  # '⌠'
CH_PALM_TRUNK_B = 245  # '⌡'
CH_PALM_SIDE = 249    # '∙'
CH_PALM_MID = 250     # '·'

# FigList slots that carry a connecting piece: they are drawn transparently on
# top of the neighbour's underlay instead of being blitted opaquely.
CONNECTOR_SLOTS = (1, 3, 4, 6, 7, 9)

# horizontal pipe tiles: '4' mouth top, '5' mouth bottom, '6' body top,
# '7' body bottom (port-only codes, free in every level pack)
HPIPE_CODES = {ord("4"), ord("5"), ord("6"), ord("7")}

# Charlie the Duck scenery (port-only).  Deliberately outside CAN_HOLD_YOU and
# CAN_STAND_ON: these are pure decoration, walked through, never solid, never
# drawn on the entity layer.
CH_CHAR_GRASS = ord("g")     # 20x14 tuft, 2 frames
CH_CHAR_FLOWER = ord("o")    # 14x12 daisy, 2 frames, bottom-centred in its cell
CH_CHAR_CATTAIL = ord("t")   # 20x14 segment, stacks up automatically
CH_CHAR_PALM = ord("p")      # one marker, auto-lays crown + leaf layers
CH_CHAR_BUSH = ord("s")      # 20x14 round bush, 2 frames
DECO_CODES = {CH_CHAR_GRASS, CH_CHAR_FLOWER, CH_CHAR_CATTAIL, CH_CHAR_PALM,
              CH_CHAR_BUSH}
# Palm alternation: both layers are 59x41 and draw at the same origin (top
# aligned), the leaf fan in front of the crown on this deco frame only, so it
# covers the crown's upper part and the pair swaps like the original's grass.
PALM_LEAF_FRAME = 0
# how far a scenery sprite can reach past its own marker column (palm crown
# 59 px on a 20 px cell)
DECO_OVERHANG = 2

# WallType -> base sprite name for InitWall (FIGURES.PAS InitWall case list).
WALL_SPRITES = {0: "GREEN", 1: "SAND", 2: "GREEN", 3: "BROWN",
                4: "GRASS", 5: "DES"}
# base slots for the five wall frames
WALL_SLOTS = (1, 2, 4, 5, 10)

# Bricks[] source frames for WallType1 in {100, 101, 102} (FIGURES.PAS
# BuildWorld: 100 -> Brick0000..2, 101 -> Brick1000..2, 102 -> Brick2000..2)
BRICK_WALL_TYPES = {100: "BRICK000", 101: "BRICK100", 102: "BRICK200"}

# Port-only water palette (6-bit, like every ChangePalette call).  0xF8..0xFD
# are the six colours of the user's 91898 water tiles (0xFB is normally a
# power-up white; in a water level it becomes the brightest foam, which is
# visually identical).  Style 2 reuses the same six entries for the original
# DOS water/waterfall ramp.
WATER_DEEP = 0xF8
WATER_MID = 0xF9
WATER_LIGHT = 0xFA
WATER_CREST = 0xFB
WATER_CREST_B = 0xFC
WATER_WHITE = 0xFD
# 0xF8 is unused by the 91898 tiles (they only reference 0xF9..0xFD), so the
# ramp is exactly the sheet's own six colours.
WATER_RAMP_91898 = ((36, 44, 46), (46, 54, 54), (49, 56, 56),
                    (52, 58, 58), (55, 60, 60), (58, 62, 62))
WATER_RAMP_DOS = ((4, 8, 12), (16, 34, 44), (28, 52, 58),
                  (42, 64, 66), (52, 74, 70), (58, 80, 76))
WATER_RAMPS = {1: WATER_RAMP_91898, 2: WATER_RAMP_DOS}


def _dos_water_palette() -> np.ndarray:
    """8-bit colours of the DOS water background, independent of the level
    palette: style 1 draws that background too, but its 0xF9..0xFD belong to
    the pale 91898 tile art."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for index, rgb in zip((WATER_DEEP, WATER_MID, WATER_LIGHT, WATER_CREST,
                           WATER_CREST_B, WATER_WHITE), WATER_RAMP_DOS):
        pal[index] = [max(0, min(63, int(c))) * 255 // 63 for c in rgb]
    return pal


WATER_DOS_PAL = _dos_water_palette()


def _wrap_add(arr: np.ndarray, c: int) -> np.ndarray:
    """byte arithmetic: (arr + c) mod 256"""
    return ((arr.astype(np.int16) + int(c)) & 0xFF).astype(np.uint8)


class Figures:
    """Everything FIGURES.PAS owns: figure tables, sky palette, tile drawing."""

    def __init__(self, assets) -> None:
        self.assets = assets
        # private palette copy (list of [r, g, b] in 8-bit), plus the derived
        # lookup table used when baking index arrays into surfaces.
        self.palette: list[list[int]] = [list(c) for c in assets.palette]
        self.pal: np.ndarray = np.asarray(self.palette, dtype=np.uint8)
        self._palette_dirty = False
        # exact 6-bit mirror of the palette.  The original stores Palette as
        # 6-bit values and computes the background gradients from them; the
        # 8-bit values are v*255//63, which round()s back to v exactly.
        self.palette6: list[list[int]] = [
            [round(c * 63 / 255) for c in rgb] for rgb in self.palette]
        # master MPAL256, restored by PALETTES.NewPalette at level start
        self.base_palette = [list(c) for c in self.palette]
        self.base_palette6 = [list(c) for c in self.palette6]
        # Fixed-palette baking: a few port-only sprites (the blue Dry Bones)
        # must keep their authored colours even though BlinkPalette rewrites
        # index 1 every frame (star/coin flicker) and SetSkyPalette rewrites
        # $E0..$FF.  They are baked with this master palette instead.
        self.base_pal: np.ndarray = np.asarray(self.base_palette, dtype=np.uint8)
        self.sky = 0
        self.lava_counter = 0
        # 0..2: which of the three 91898 water frames is being drawn; every
        # cell adds a stable random offset so the tiles never pulse in lockstep
        self.water_frame = 0
        self.water_seed = random.randrange(3)
        # Charlie scenery frame (0/1), driven by PALETTES.BlinkPalette's grass
        # two-state clock -- the same alternation the original's grass and palm
        # shadows run on.
        self.deco_frame = 0
        self.options: WorldOptions | None = None
        # BACKGR.PAS owns the textured skies / mountain background; set by the
        # caller so DrawSky can delegate to it (mirrors FIGURES.PAS `uses
        # BackGr`).
        self.backgr: "BackGr | None" = None

        # FigList[set][slot] -> index array; [0] unused on both axes
        self.fig: list[list[np.ndarray | None]] = [[None] * (N2 + 1)
                                                   for _ in range(N1 + 1)]
        self.bricks: list[np.ndarray | None] = [None] * 4    # Bricks[0..3]
        self.pipes: list[np.ndarray | None] = [None] * 4     # recoloured copies
        # horizontal pipes: 0=mouth top, 1=mouth bottom, 2=body top,
        # 3=body bottom, 4/5 = horizontally mirrored mouths (opening left)
        self.hpipes: list[np.ndarray | None] = [None] * 6
        # sprites recoloured in place by BuildWorld
        self.block001: np.ndarray | None = None
        self.wood000: np.ndarray | None = None
        self.xblock000: np.ndarray | None = None
        # ConvertGrass output (Grass1000/2000/3000 and Palm0000..3000 replaced)
        self.grass: dict[int, np.ndarray] = {}
        self.palm: dict[int, np.ndarray] = {}

        self._surf: dict[tuple, pygame.Surface] = {}
        self._surf_max: dict[tuple, int] = {}
        self._surf_idx: dict[tuple, frozenset[int]] = {}
        # surfaces baked with a fixed palette (pal is not None) do not depend
        # on the live level/blink palette, so bump/invalidate must keep them.
        self._surf_fixed: set[tuple] = set()
        self._sky: dict[tuple[int, int, int, int], pygame.Surface] = {}

    # ------------------------------------------------------------------ #
    # palette mutations (PALETTES.PAS ChangePalette / CopyPalette /
    # OutPalette all work on the shared 6-bit Palette array)
    # ------------------------------------------------------------------ #
    def change_palette(self, index: int, r: int, g: int, b: int) -> None:
        """OutPalette/ChangePalette: store 6-bit values, mirror as 8-bit."""
        self._change(index, r, g, b)

    def copy_palette(self, c1: int, c2: int) -> None:
        """CopyPalette (C1, C2): Palette[C2] := Palette[C1] (both mirrors)."""
        self.palette[c2] = list(self.palette[c1])
        self.palette6[c2] = list(self.palette6[c1])
        self._palette_dirty = True

    def bump_palette(self, low: int = 0xC0) -> None:
        """Call after directly mutating the palette: rebuild the lookup table
        and drop every baked surface that used an index >= ``low``.  BackGr's
        palette animators only ever touch the sky range $C0..$FF, so the tile
        surfaces (indices below that) stay cached."""
        self._rebuild_lookup()
        keep = {k for k, m in self._surf_max.items()
                if m < low or k in self._surf_fixed}
        self._surf = {k: v for k, v in self._surf.items() if k in keep}
        self._surf_max = {k: m for k, m in self._surf_max.items()
                          if k in keep}
        self._surf_idx = {k: v for k, v in self._surf_idx.items()
                          if k in keep}
        self._surf_fixed &= keep
        if low <= 0xF0:
            self._sky.clear()

    def invalidate_indices(self, indices) -> None:
        """Drop only the baked surfaces that actually use one of ``indices``.

        PALETTES.BlinkPalette changes a handful of palette entries each frame
        (star / waterfall / coins / grass); re-baking every sprite would be
        wasteful, so the cache remembers which indices each surface touched."""
        drop = set(indices)
        keep = {k for k, used in self._surf_idx.items()
                if k in self._surf_fixed or not (used & drop)}
        self._surf = {k: v for k, v in self._surf.items() if k in keep}
        self._surf_max = {k: m for k, m in self._surf_max.items()
                          if k in keep}
        self._surf_idx = {k: v for k, v in self._surf_idx.items()
                          if k in keep}
        self._surf_fixed &= keep

    def reset_palette(self) -> None:
        """PALETTES.NewPalette(P256): restore the master palette and drop
        every baked surface (the previous level may have animated indices)."""
        self.palette = [list(c) for c in self.base_palette]
        self.palette6 = [list(c) for c in self.base_palette6]
        self._rebuild_lookup()
        self._surf.clear()
        self._surf_max.clear()
        self._surf_idx.clear()
        self._surf_fixed.clear()
        self._sky.clear()

    def bake(self, arr: np.ndarray, opaque: bool = False) -> pygame.Surface:
        """Index array -> RGBA Surface using the current palette."""
        return self._to_surface(arr, opaque)

    def smooth_fill(self, target: pygame.Surface, x: int, y: int,
                    w: int, h: int) -> None:
        target.blit(self._smooth_surf(x, y, w, h), (x, y))

    def blit_masked(self, target: pygame.Surface, x: int, y: int,
                    idx: np.ndarray, allowed: tuple[int, ...]) -> None:
        """Composite ``idx`` onto ``target`` at (x, y), but only over pixels
        whose current colour equals one of the ``allowed`` palette entries.

        BACKGR.PAS writes its mountain with conditional stores (``cmp [di],
        $F0; jnz`` -- it only flips pixels between $E0 and $F0), which is how
        it leaves tiles untouched despite drawing after them.  The port keeps
        pages as RGB, so the same test is done on colours."""
        rect = pygame.Rect(x, y, idx.shape[1], idx.shape[0])
        rect = rect.clip(target.get_rect())
        if rect.width <= 0 or rect.height <= 0:
            return
        sub = target.subsurface(rect)
        px = np.frombuffer(pygame.image.tostring(sub, "RGBA", False),
                           dtype=np.uint8).reshape(rect.height, rect.width, 4)
        mask = np.zeros(px.shape[:2], dtype=bool)
        for a in allowed:
            c = np.asarray(self.palette[a], dtype=np.uint8)
            mask |= np.all(px[..., :3] == c, axis=-1)
        if not mask.any():
            return
        part = np.ascontiguousarray(idx[:rect.height, :rect.width])
        rgb = self.pal[part]
        out = np.where(mask[..., None], rgb, px[..., :3])
        rgba = np.ascontiguousarray(np.dstack([out, px[..., 3]]))
        surf = pygame.image.frombuffer(rgba.tobytes(),
                                       (rect.width, rect.height), "RGBA")
        target.blit(surf, (rect.x, rect.y))

    # ------------------------------------------------------------------ #
    # index-level operations (direct translations of the inline asm)
    # ------------------------------------------------------------------ #
    @staticmethod
    def recolor(src: np.ndarray, c: int) -> np.ndarray:
        """ReColor(P1, P2, C): dst = copy; where src > $10, dst = (src & 7) + C."""
        dst = src.copy()
        m = src > 0x10
        dst[m] = _wrap_add(src[m] & 7, c)
        return dst

    @staticmethod
    def recolor2(src: np.ndarray, c1: int, c2: int) -> np.ndarray:
        """ReColor2(P1, P2, C1, C2): v = src & $0F; v >= 8 -> (v & 7) + C2,
        else v + C1; only where src > $10."""
        dst = src.copy()
        m = src > 0x10
        v = (src & 0x0F).astype(np.int16)
        hi = v >= 8
        sel = m & hi
        dst[sel] = _wrap_add((v[sel] & 7), c2)
        sel = m & ~hi
        dst[sel] = _wrap_add(v[sel], c1)
        return dst

    @staticmethod
    def replace(src: np.ndarray, n1: int, n2: int) -> np.ndarray:
        """Replace(P1, P2, N1, N2): dst[src == N1] = N2."""
        dst = src.copy()
        dst[src == n1] = n2
        return dst

    @staticmethod
    def mirror(src: np.ndarray) -> np.ndarray:
        """Mirror(P1, P2): horizontal flip, dest[y, x] = src[y, W-1-x]."""
        return np.ascontiguousarray(src[:, ::-1])

    @staticmethod
    def rotate(src: np.ndarray) -> np.ndarray:
        """Rotate(P1, P2): 180 degrees, dest[y, x] = src[H-1-y, W-1-x]."""
        return np.ascontiguousarray(src[::-1, ::-1])

    @staticmethod
    def convert_grass(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """ConvertGrass(P0, P1, P2) from FIGURES.PAS: returns the shadow frame
        P0 computed from the two input frames P1 and P2.

        C0 = C1; if C1 = C2 keep C1; C1 = 2 -> 153 (C2 = 0) else 155;
        C1 = 3 -> 154 (C2 = 0) else 156; else C2 = 2 -> 157 else 155.
        """
        c1 = p1
        c2 = p2
        c0 = c1.copy()
        diff = c1 != c2
        is2 = diff & (c1 == 2)
        c0[is2] = 155
        c0[is2 & (c2 == 0)] = 153
        is3 = diff & (c1 == 3)
        c0[is3] = 156
        c0[is3 & (c2 == 0)] = 154
        other = diff & (c1 != 2) & (c1 != 3)
        c0[other] = 155
        c0[other & (c2 == 2)] = 157
        return c0

    # ------------------------------------------------------------------ #
    # level setup
    # ------------------------------------------------------------------ #
    def init_sky(self, new_sky: int) -> None:
        self.sky = new_sky

    def init_pipes(self, new_color: int) -> None:
        # ReColor is applied in place to Pipe000..003 in the original; we keep
        # recoloured copies so the asset tables stay pristine.  The horizontal
        # tiles use the same 112..119 ramp + 16 outline, so one shift covers
        # every PipeColor; the mirrored mouths are baked from the shifted
        # arrays (Mirror on indices, like the original's HighMirror).
        self.pipes = [self.recolor(self.assets.sprite_idx(f"PIPE00{i}"),
                                   new_color) for i in range(4)]
        self.hpipes = [self.recolor(self.assets.sprite_idx(f"HPIPE00{i}"),
                                    new_color) for i in range(4)]
        self.hpipes.append(self.mirror(self.hpipes[0]))
        self.hpipes.append(self.mirror(self.hpipes[1]))

    def init_walls(self, w1: int, w2: int, w3: int,
                   options: WorldOptions | None = None) -> None:
        # PLAY.PAS parses the level options before InitWalls, so the ground
        # colours for wall type 2 are already known here -- accept them
        # explicitly instead of silently falling back to 0.
        if options is not None:
            self.options = options
        for n, wall_type in ((1, w1), (2, w2), (3, w3)):
            self.init_wall(n, wall_type)

    def init_wall(self, n: int, wall_type: int) -> None:
        slots: dict[int, np.ndarray] = {}
        if wall_type in WALL_SPRITES:
            base = WALL_SPRITES[wall_type]
            frames = (f"{base}000", f"{base}001", f"{base}002",
                      f"{base}003", f"{base}004")
            if wall_type == 2:
                # case 2: Green frames run through ReColor2 with the two
                # ground colours from the level options
                c1 = self.options.GroundColor1 if self.options else 0
                c2 = self.options.GroundColor2 if self.options else 0
                for slot, name in zip(WALL_SLOTS, frames):
                    slots[slot] = self.recolor2(self.assets.sprite_idx(name),
                                                c1, c2)
            else:
                for slot, name in zip(WALL_SLOTS, frames):
                    slots[slot] = self.assets.sprite_idx(name).copy()
        for slot, arr in slots.items():
            self.fig[n][slot] = arr
        # derived pieces.  The original runs these unconditionally even for
        # WallTypes outside 0..5, mirroring whatever FigList held from the
        # previous level; those levels (WallType1 >= 100) never read
        # FigList -- BuildWall is skipped and their 'A' walls use Bricks[] --
        # so leaving the slots empty is both safe and equivalent.
        if slots:
            f = self.fig[n]
            f[3] = self.mirror(f[1])
            f[6] = self.rotate(f[4])
            f[9] = self.rotate(f[1])
            f[8] = self.rotate(f[2])
            f[7] = self.rotate(f[3])
            f[11] = self.mirror(f[10])
            f[12] = self.rotate(f[11])
            f[13] = self.mirror(f[12])

    # ------------------------------------------------------------------ #
    # palette (SetSkyPalette / ChangePalette store 6-bit values)
    # ------------------------------------------------------------------ #
    def _change(self, index: int, r: int, g: int, b: int) -> None:
        v = lambda c: max(0, min(63, int(c))) * 255 // 63
        self.palette[index] = [v(r), v(g), v(b)]
        w = lambda c: max(0, min(63, int(c)))
        self.palette6[index] = [w(r), w(g), w(b)]
        self._palette_dirty = True

    def _rebuild_lookup(self) -> None:
        self.pal = np.asarray(self.palette, dtype=np.uint8)
        self._palette_dirty = False

    def set_sky_palette(self) -> None:
        """Rewrite the sky range of the palette. Call after build_world; the
        caller should apply figures.palette to the screen afterwards."""
        assert self.options is not None, "set_sky_palette needs options"
        cp = self._change
        sky = self.sky
        bgt = self.options.BackGrType
        if sky == 0:
            cp(0xE0, 35, 45, 63); cp(0xF0, 20, 38, 48); cp(0xFF, 54, 57, 60)
        elif sky == 1:
            cp(0xE0, 52, 55, 55); cp(0xF0, 42, 48, 45); cp(0xFF, 61, 61, 61)
        elif sky == 2:
            for j in range(16):
                cp(0xE0 + j, 48 - 2 * j, 58 - j, 58)
            cp(0xF0, 35, 48, 46)
        elif sky == 3:
            cp(0xE0, 0, 5, 3); cp(0xF0, 8, 12, 10); cp(0xFF, 8, 13, 13)
        elif sky == 4:
            cp(0xE0, 35, 45, 53); cp(0xF0, 23, 39, 43); cp(0xFF, 58, 60, 60)
        elif sky == 5:
            for j in range(16):
                cp(0xE0 + j, 58 - j // 2, 56 - j, 38 - j)
            cp(0xF0, 52, 49, 32)
        elif sky == 6:        # brown bricks
            if bgt == 4:
                for i in range(0xE0, 0xF0):
                    cp(i, 22, 15, 11)
                cp(0xFD, 22, 15, 11); cp(0xFE, 19, 12, 8); cp(0xFF, 25, 18, 14)
            else:
                for i in range(0xE0, 0x100):
                    cp(i, 19, 9, 8)
                cp(0xD1, 19, 9, 8); cp(0xD6, 21, 11, 10); cp(0xD4, 17, 7, 6)
        elif sky == 7:        # gray bricks
            if bgt == 4:
                for i in range(0xE0, 0xF0):
                    cp(i, 18, 18, 22)
                cp(0xFD, 18, 18, 22); cp(0xFF, 23, 23, 27); cp(0xFE, 13, 13, 17)
            else:
                for i in range(0xE0, 0x100):
                    cp(i, 15, 15, 18)
                cp(0xD1, 15, 15, 18); cp(0xD4, 18, 18, 21); cp(0xD6, 12, 12, 15)
        elif sky == 8:        # dark brown bricks
            if bgt == 4:
                for i in range(0xE0, 0xF0):
                    cp(i, 17, 10, 10)
                cp(0xFD, 17, 10, 10); cp(0xFE, 11, 5, 5); cp(0xFF, 20, 14, 14)
            else:
                for i in range(0xE0, 0x100):
                    cp(i, 15, 5, 5)
                cp(0xD1, 15, 5, 5); cp(0xD4, 20, 10, 10); cp(0xD6, 10, 0, 0)
        elif sky == 9:
            for j in range(16):
                cp(0xE0 + j, 63 - j // 3, 50 - j, 25 - j)
            cp(0xF0, 48, 35, 18)
        elif sky == 10:
            for j in range(16):
                cp(0xE0 + j, 27 - j, 43 - j, 63 - j)
            cp(0xF0, 58, 58, 63)
        elif sky == 11:
            for j in range(16):
                cp(0xE0 + j, 60 - j, 63 - j, 63 - j)
            cp(0xF0, 42, 48, 45)
        elif sky == 12:
            for j in range(16):
                cp(0xE0 + j, 55 - j, 63 - j, 63)
            cp(0xF0, 30, 50, 58)
            cp(0xF0, 36, 45, 41)   # second assignment wins in the original
        if self.options.WaterTop >= 0:
            ramp = WATER_RAMPS.get(self.options.WaterStyle,
                                   WATER_RAMPS[1])
            for index, rgb in zip((WATER_DEEP, WATER_MID, WATER_LIGHT,
                                   WATER_CREST, WATER_CREST_B,
                                   WATER_WHITE), ramp):
                cp(index, *rgb)
        self._rebuild_lookup()
        # every baked surface depends on the palette
        self._surf.clear()
        self._surf_max.clear()
        self._surf_idx.clear()
        self._surf_fixed.clear()
        self._sky.clear()

    def init_grass(self) -> None:
        """PALETTES.InitGrass: per-level grass / palm colours.

        Indices 2 and 3 take the level's C2/C3 options, 153..158 copy those
        plus the horizon colour (so ConvertGrass shadows animate), and 6 is
        the champ.  Must run after set_sky_palette (157/158 read $E0/$EF);
        the blinking animation itself (BlinkPalette) is not ported yet, so
        this is the pre-animation base state."""
        assert self.options is not None, "init_grass needs options"
        opt = self.options
        cp = self._change
        cp(2, opt.C2r, opt.C2g, opt.C2b)
        cp(3, opt.C3r, opt.C3g, opt.C3b)
        p = self.palette
        p[153] = list(p[2]); p[154] = list(p[3])
        p[155] = list(p[2]); p[156] = list(p[3])
        horizon = 0xF0 - (1 if opt.SkyType == 10 else 0)
        p[157] = list(p[horizon]); p[158] = list(p[horizon])
        cp(6, 60, 40, 35)                       # champ
        self._rebuild_lookup()
        self._surf.clear()
        self._surf_max.clear()
        self._surf_idx.clear()

    # ------------------------------------------------------------------ #
    # surfaces (index array -> RGBA, cached)
    # ------------------------------------------------------------------ #
    def _to_surface(self, arr: np.ndarray, opaque: bool,
                    pal: np.ndarray | None = None) -> pygame.Surface:
        if self._palette_dirty:
            self._rebuild_lookup()
        p = self.pal if pal is None else pal
        rgb = p[np.ascontiguousarray(arr)]                           # (h,w,3)
        if opaque:
            alpha = np.full(arr.shape, 255, dtype=np.uint8)
        else:
            alpha = np.where(arr != 0, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        rgba = np.ascontiguousarray(rgba)
        # NOTE: no convert_alpha() — this module must also work headless.
        return pygame.image.frombuffer(rgba.tobytes(),
                                       (arr.shape[1], arr.shape[0]), "RGBA")

    def _cached(self, key: str, arr: np.ndarray | None,
                opaque: bool, pal: np.ndarray | None = None
                ) -> pygame.Surface | None:
        if arr is None:
            return None
        ck = ((key, opaque) if pal is None
              else (key, opaque, id(pal)))
        s = self._surf.get(ck)
        if s is None:
            s = self._to_surface(arr, opaque, pal)
            self._surf[ck] = s
            self._surf_max[ck] = int(arr.max())
            self._surf_idx[ck] = frozenset(
                int(v) for v in np.unique(arr))
            if pal is not None:
                self._surf_fixed.add(ck)
        return s

    def _fig_surf(self, n: int, slot: int, opaque: bool):
        return self._cached(f"F{n}:{slot}", self.fig[n][slot], opaque)

    def _sprite_surf(self, name: str, opaque: bool = False):
        return self._cached(f"S:{name}", self.assets.sprite_idx(name), opaque)

    def sprite_surf(self, name: str, opaque: bool = False):
        """Asset sprite -> cached Surface (BackGr's DrawBricks / Pillar)."""
        return self._sprite_surf(name, opaque)

    def cache_surface(self, key: str, arr: np.ndarray,
                      opaque: bool = False,
                      pal: np.ndarray | None = None) -> pygame.Surface:
        """Bake an arbitrary index array into a palette-aware cached Surface.

        ENEMIES.PAS keeps its picture tables as raw palette indices and lets
        the VGA hardware apply the palette at blit time; the port has to bake
        RGB surfaces instead, and they must be invalidated whenever the sky
        range of the palette is rewritten (SetSkyPalette / InitGrass / the
        BackGr palette animators).  Routing them through the same cache as the
        tile figures gives exactly that.

        ``pal`` overrides the live level palette with a fixed one (the master
        ``base_pal``); the blue Dry Bones uses it so BlinkPalette's per-frame
        rewrite of index 1 cannot turn its outline yellow.
        """
        return self._cached(f"X:{key}", arr, opaque, pal)

    # ------------------------------------------------------------------ #
    # BuildWorld (post-processing of the level map)
    # ------------------------------------------------------------------ #
    def build_world(self, world: WorldMap,
                    options: WorldOptions | None = None) -> None:
        if options is not None:
            self.options = options
        assert self.options is not None, "build_world needs options"
        x_size = world.x_size

        # 1. marker expansion, column-major like the original
        rows = world.rows
        for x in range(x_size):
            for y in range(rows):
                ch = world.get(x, y)
                if ch == CH_SUPER_2:          # '²'
                    world.set(x, y - 5, ord("?"))
                    world.set(x, y - 6, CH_SHARP2)
                    world.set(x, y, ord(" "))
                elif ch == CH_SUPER_N:        # 'ⁿ'
                    world.set(x, y - 2, ord("*"))
                    world.set(x, y, ord(" "))
                elif ch == CH_INVISIBLE:      # '¡'
                    k = y + 1
                    for l in range(y, -2, -1):
                        world.set(x, l, world.get(x, k))
                elif ch == CH_LGUILLEMET:     # '«'
                    world.set(x, y, world.get(x, y - 1))
                    world.set(x, rows, 254)
                elif ch == CH_CURRENCY:       # '¤'
                    world.set(x, y, world.get(x, y - 1))
                    world.set(x, rows, 255)

        # 2. automatic wall tiling (or the brick-table fallback)
        if self.options.WallType1 < 100:
            for x in range(x_size):
                for y in range(rows):
                    self._build_wall(world, x, y)
        else:
            base = BRICK_WALL_TYPES.get(self.options.WallType1)
            if base is not None:
                for i in range(3):
                    self.bricks[i] = self.recolor(
                        self.assets.sprite_idx(f"{base}{i}"),
                        self.options.GroundColor1)

        # 3. ConvertGrass on the grass/palm families.
        #    ConvertGrass(P0, P1, P2) replaces P0 with a shadow frame derived
        #    from P1 and P2; note the 3xxx family passes its last two frames
        #    swapped.
        a = self.assets
        self.grass = {
            1: self.convert_grass(a.sprite_idx("GRASS1001"),
                                  a.sprite_idx("GRASS1002")),
            2: self.convert_grass(a.sprite_idx("GRASS2001"),
                                  a.sprite_idx("GRASS2002")),
            3: self.convert_grass(a.sprite_idx("GRASS3002"),
                                  a.sprite_idx("GRASS3001")),
        }
        self.palm = {
            k: self.convert_grass(a.sprite_idx(f"PALM{k}001"),
                                  a.sprite_idx(f"PALM{k}002"))
            for k in range(4)
        }

        # 4. in-place ReColors of the recolourable sprites
        self.block001 = self.recolor(a.sprite_idx("BLOCK001"),
                                     self.options.BrickColor)
        self.wood000 = self.recolor(a.sprite_idx("WOOD000"),
                                    self.options.WoodColor)
        self.xblock000 = self.recolor(a.sprite_idx("XBLOCK000"),
                                      self.options.XBlockColor)

    def _build_wall(self, world: WorldMap, x: int, y: int) -> None:
        """BuildWall (FIGURES.PAS): replace A/B/C/D with the matching piece of
        the wall family.  Column-major in-place: the left/upper neighbours have
        already been converted when they are read."""
        c = world.get(x, y)
        ignore_above = {CH_GRASS_TOP}
        if c == ord("A") or c == ord("B"):
            ch = {c} | set(range(1, 14))
            # the original evaluates `if LastAB <> C then ChLeft := Ch - [3,6,9]
            # else ChLeft := Ch`, but the trailing `LastAB := AB; LastAB := CD;
            # LastAB := EF;` triple assigns only LastAB and never reads AB/CD,
            # so ChLeft is always Ch - [3,6,9].  The bug is not reproduced.
            ch_left = ch - {3, 6, 9}
            n = 0
        elif c == ord("C") or c == ord("D"):
            ch = ({c} | set(range(1, 27)) | {ord("A"), ord("B")}
                  | ignore_above)
            ch_left = ch
            n = 13
        else:
            return

        x_size = self.options.XSize if self.options else world.x_size
        rows = world.rows
        a = 1 if (y > 0 and world.get(x, y - 1) not in (ch - ignore_above)) else 0
        b = 2 if not (y == rows - 1 or world.get(x, y + 1) in ch) else 0
        left = 4 if not (x == 0 or world.get(x - 1, y) in ch_left) else 0
        r = 8 if not (x == x_size - 1 or world.get(x + 1, y) in ch) else 0

        code = a + b + left + r
        if code == 0:
            if x > 0 and y > 0 and world.get(x - 1, y - 1) not in ch:
                world.set(x, y, 10 + n); return
            if x < x_size - 1 and y > 0 and world.get(x + 1, y - 1) not in ch:
                world.set(x, y, 11 + n); return
            if x > 0 and y < rows - 1 and world.get(x - 1, y + 1) not in ch:
                world.set(x, y, 12 + n); return
            if x < x_size - 1 and y < rows - 1 and world.get(x + 1, y + 1) not in ch:
                world.set(x, y, 13 + n); return
            world.set(x, y, 5 + n); return
        mapping = {1: 2 + n, 2: 8 + n, 4: 4 + n, 8: 6 + n,
                   5: 1 + n, 6: 7 + n, 9: 3 + n, 10: 9 + n}
        world.set(x, y, mapping.get(code, 5 + n))

    # ------------------------------------------------------------------ #
    # sky
    # ------------------------------------------------------------------ #
    def _rgb(self, index: int) -> tuple[int, int, int]:
        return tuple(self.palette[index])

    def draw_sky(self, target: pygame.Surface, x: int, y: int,
                 w: int, h: int) -> None:
        opt = self.options
        assert opt is not None, "draw_sky needs options"
        if opt.MoonSky:
            # the XP backdrop was blitted as the page base; repainting sky
            # here (space tiles call Redraw for every cell) would erase it
            return
        wy = opt.water_y
        if wy >= 0:
            # split: above-water part, then water below it.  WaterFillTop
            # replaces the dry rows with static body tiles (test rooms only).
            if y < wy:
                fill_h = min(y + h, wy) - y
                if opt.WaterFillTop:
                    self.draw_water_fill(target, x, y, w, fill_h)
                else:
                    self._draw_sky_plain(target, x, y, w, fill_h)
            if y + h > wy:
                top = max(y, wy)
                self.draw_water(target, x, top, w, y + h - top)
            return
        self._draw_sky_plain(target, x, y, w, h)

    def _draw_sky_plain(self, target: pygame.Surface, x: int, y: int,
                        w: int, h: int) -> None:
        opt = self.options
        assert opt is not None, "draw_sky needs options"
        if opt.BackGrType == 0:
            target.fill(self._rgb(0xE0), (x, y, w, h))
            return
        sky = self.sky
        if sky in (0, 1, 3, 4):
            horizon = opt.Horizon
            if horizon < y:
                target.fill(self._rgb(0xF0), (x, y, w, h))
            elif horizon > y + h - 1:
                target.fill(self._rgb(0xE0), (x, y, w, h))
            else:
                j = horizon - y
                target.fill(self._rgb(0xE0), (x, y, w, j))
                target.fill(self._rgb(0xF0), (x, horizon, w, h - j))
        elif sky in (2, 5, 9, 10, 11, 12):
            target.blit(self._smooth_surf(x, y, w, h), (x, y))
        elif sky in (6, 7, 8):
            # textured skies: DrawBricks / LargeBricks / Pillar / Windows live
            # in BACKGR.PAS (FIGURES.PAS DrawSky just dispatches on
            # BackGrType 4/5/6/7 and draws nothing for other values).
            if self.backgr is not None:
                self.backgr.draw_sky(target, x, y, w, h)

    def draw_water(self, target: pygame.Surface, x: int, y: int,
                   w: int, h: int) -> None:
        """Paint one rectangle below the waterline (port-only).

        Style 1 is the DOS water background plus one 20x13 row of 91898
        waterfall tiles (WATER_BODY_0..2) cycled through three frames, plus
        the grass/cattail overlay; every tile cell gets a stable per-cell
        frame offset (water_seed + cell hash), so the animation is a random
        patchwork instead of a synchronized pulse.  Style 2 is the original
        DOS water/waterfall fallback (no overlay).
        """
        opt = self.options
        assert opt is not None, "draw_water needs options"
        wy = opt.water_y
        if wy < 0 or h <= 0 or w <= 0 or y + h <= wy:
            return
        if opt.WaterStyle == 1:
            self._draw_water_91898(target, x, y, w, h, wy)
        else:
            self._draw_water_dos(target, x, y, w, h, wy)

    @staticmethod
    def _water_clip(target: pygame.Surface, surf: pygame.Surface,
                    px: int, py: int, rect: pygame.Rect) -> None:
        x0 = max(rect.left, px)
        y0 = max(rect.top, py)
        x1 = min(rect.right, px + surf.get_width())
        y1 = min(rect.bottom, py + surf.get_height())
        if x1 <= x0 or y1 <= y0:
            return
        target.blit(surf, (x0, y0),
                    area=pygame.Rect(x0 - px, y0 - py, x1 - x0, y1 - y0))

    def draw_water_fill(self, target: pygame.Surface, x: int, y: int,
                        w: int, h: int) -> None:
        """Static water above the waterline (port-only WaterFillTop)."""
        if h <= 0 or w <= 0:
            return
        if self.options is None or self.options.WaterStyle != 1:
            target.fill(self._rgb(WATER_DEEP), (x, y, w, h))
            return
        rect = pygame.Rect(x, y, w, h)
        start_x = x - (x % TILE_W)
        first_y = y - (y % 13)
        for cy in range(first_y, y + h, 13):
            for cx in range(start_x, x + w, TILE_W):
                if cx + TILE_W <= x or cx >= x + w:
                    continue
                frame = self._water_static_cell_frame(cx // TILE_W, cy // 13)
                self._water_clip(target,
                                 self._sprite_surf(f"WATER_BODY_{frame}"),
                                 cx, cy, rect)

    def _water_hash_offset(self, cx: int, cy: int) -> int:
        # mixed integer hash, not cx*7+cy*13: the latter repeats visibly
        # every three cells
        h = (cx * 374761393 + cy * 668265263
             + self.water_seed * 2246822519) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        return (h ^ (h >> 16)) % 3

    def _water_cell_frame(self, cx: int, cy: int) -> int:
        return (self.water_frame + self._water_hash_offset(cx, cy)) % 3

    def _water_static_cell_frame(self, cx: int, cy: int) -> int:
        return self._water_hash_offset(cx, cy)

    def _draw_water_91898(self, target: pygame.Surface, x: int, y: int,
                          w: int, h: int, wy: int) -> None:
        """The DOS water background plus one animated 91898 waterfall row.

        The background (shallow band + diagonal dashes over deep navy) is the
        style 2 look, baked from WATER_DOS_PAL because the tile ramp owns
        0xF9..0xFD.  The top row is WATER_BODY_0..2 -- the last three of the
        six imported tiles -- cycled off water_frame.  The seaweed overlay is
        added later by draw_decorations.
        """
        out, top = self._water_dos_array(x, y, w, h, wy)
        # the background never moves, and redraw() revisits the same cells
        # every frame, so cache the baked surface per world rect
        key = f"WBG:{x}:{y}:{w}:{h}:{wy}"
        target.blit(self.cache_surface(key, out, True, WATER_DOS_PAL),
                    (x, top))
        rect = pygame.Rect(x, y, w, h)
        body_h = 13
        if wy < rect.bottom and wy + body_h > rect.top:
            start_x = x - (x % TILE_W)
            for cx in range(start_x, x + w, TILE_W):
                if cx + TILE_W <= x or cx >= x + w:
                    continue
                frame = self._water_cell_frame(cx // TILE_W, 0)
                self._water_clip(target,
                                 self._sprite_surf(f"WATER_BODY_{frame}"),
                                 cx, wy, rect)

    def _water_dos_array(self, x: int, y: int, w: int, h: int,
                         wy: int) -> tuple[np.ndarray, int]:
        """Index plane of the DOS water background, plus its first row."""
        top = max(y, wy)
        hh = y + h - top
        yy = np.arange(top, top + hh, dtype=np.int32)[:, None]
        xx = np.arange(x, x + w, dtype=np.int32)[None, :]
        depth = (yy - wy) // TILE_H
        out = np.full((hh, w), WATER_DEEP, dtype=np.uint8)
        shallow = (depth[:, 0] <= 2)[:, None]
        out[np.broadcast_to(shallow, out.shape)] = WATER_MID
        dash = ((xx + 2 * yy) % 41) < 2
        out[dash & (depth >= 1)] = WATER_LIGHT
        local = yy - wy
        if top < wy + 7:
            crest_h = 2 + ((xx // 4) % 3)
            m = (local >= 0) & (local < crest_h)
            out[m] = WATER_CREST
            m = (local >= crest_h) & (local < crest_h + 2)
            out[m] = WATER_LIGHT
        return out, top

    def _draw_water_dos(self, target: pygame.Surface, x: int, y: int,
                        w: int, h: int, wy: int) -> None:
        """Original DOS water palette + waterfall surface (fallback style)."""
        out, top = self._water_dos_array(x, y, w, h, wy)
        target.blit(self.bake(out, True), (x, top))
        if wy >= top and wy < top + out.shape[0]:
            for cx in range(x - (x % TILE_W), x + w, TILE_W):
                if cx + TILE_W <= x or cx >= x + w:
                    continue
                surf = self._sprite_surf("FALL000" if (cx // TILE_W) % 2
                                         else "FALL001")
                target.blit(surf, (cx, wy))

    def _smooth_indices(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """SmoothFill (BACKGR.PAS), fully vectorised.

        c(y) = $EF - (y div 6), floored at $E0; $F0 from Options.Horizon - 4.
        Rows with (y mod 6) >= 3 are dithered: the second pass writes c-1
        through map mask 0101 (planes 0/2) when (y mod 6) is even and 1010
        (planes 1/3) when odd, i.e. every other column: even x for even
        (y mod 6), odd x for odd.  Skipped when c is $E0/$F0.  All
        coordinates are absolute."""
        horizon = self.options.Horizon - 4 if self.options else 0
        ys = np.arange(y, y + h, dtype=np.int32)
        xs = np.arange(x, x + w, dtype=np.int32)
        col = np.maximum(0xEF - (ys // 6), 0xE0)
        col = np.where(ys >= horizon, 0xF0, col)
        dh = ys % 6
        dither = ((dh >= 3) & (col != 0xE0) & (col != 0xF0))[:, None]
        sel = (xs[None, :] & 1) == (dh[:, None] & 1)
        out = np.where(dither & sel, col[:, None] - 1, col[:, None])
        return np.ascontiguousarray(out.astype(np.uint8))

    def _smooth_surf(self, x: int, y: int, w: int, h: int) -> pygame.Surface:
        key = (x, y, w, h)
        s = self._sky.get(key)
        if s is None:
            s = self._to_surface(self._smooth_indices(x, y, w, h), True)
            self._sky[key] = s
        return s

    # ------------------------------------------------------------------ #
    # Redraw
    # ------------------------------------------------------------------ #
    def redraw(self, target: pygame.Surface, world: WorldMap,
               x: int, y: int) -> None:
        if x < 0 or y < 0 or y >= world.rows:
            return
        ch = world.get(x, y)
        xpos, ypos = x * TILE_W, y * TILE_H

        if ch != 0:
            if ch == ord("%") and self.options.Design == 4:
                self.draw_sky(target, xpos, ypos, TILE_W, TILE_H // 2)
            else:
                self.draw_sky(target, xpos, ypos, TILE_W, TILE_H)
        if ch == ord(" "):
            return

        # underlay for tiles sitting below an interior C/D wall piece (#18)
        if world.get(x, y - 1) == 18:
            self._put(target, xpos, ypos, self._fig_surf(1, 5, True))

        fig: pygame.Surface | None = None
        opt = self.options

        if 1 <= ch <= 26:
            c = ch - 13 if ch > 13 else ch
            if ch <= 13:
                left = world.get(x - 1, y)
                if 14 <= left <= 26 and c in (1, 4, 7):
                    self._put(target, xpos, ypos,
                              self._fig_surf(1, left - 13, True))
                else:
                    right = world.get(x + 1, y)
                    if 14 <= right <= 26 and c in (3, 6, 9):
                        self._put(target, xpos, ypos,
                                  self._fig_surf(1, right - 13, True))
            if c in CONNECTOR_SLOTS:
                fig = self._fig_surf(1, c, False)
            else:
                self._put(target, xpos, ypos, self._fig_surf(1, c, True))

        elif ch == ord("?"):
            fig = self._sprite_surf("QUEST000")
        elif ch == ord("@"):
            fig = self._sprite_surf("QUEST001")
        elif ch == ord("x"):
            fig = self._sprite_surf("X000")

        elif ch == ord("A"):
            l = world.get(x - 1, y) == ord("A")
            r = world.get(x + 1, y) == ord("A")
            if (x + y) % 2:
                ls, rs = False, True
            else:
                ls, rs = True, False
            if ls and r:
                fig = self._cached("B1", self.bricks[1], False)
            elif rs and l:
                fig = self._cached("B2", self.bricks[2], False)
            else:
                fig = self._cached("B0", self.bricks[0], False)

        elif ch == ord("I"):
            fig = self._sprite_surf("BLOCK000")
        elif ch == ord("J"):
            fig = self._cached("J", self.block001, False)
        elif ch == ord("K"):
            fig = self._sprite_surf("NOTE000")
        elif ch == ord("X"):
            fig = self._cached("X", self.xblock000, False)
        elif ch == ord("W"):
            fig = self._cached("W", self.wood000, False)

        elif ch == ord("="):
            pin = self._sprite_surf("PIN000")
            if world.get(x, y + 1) in CAN_HOLD_YOU:
                self._draw(target, xpos, ypos, pin)
            else:
                self._upside_down(target, xpos, ypos, pin)

        elif ch == ord("0"):
            fig = self._cached("P0", self.pipes[0], False)
        elif ch == ord("1"):
            fig = self._cached("P1", self.pipes[1], False)
        elif ch == ord("2"):
            fig = self._cached("P2", self.pipes[2], False)
        elif ch == ord("3"):
            fig = self._cached("P3", self.pipes[3], False)

        elif ch in HPIPE_CODES:
            # '4'/'5' mouth top/bottom, '6'/'7' body top/bottom.  The stored
            # mouth art opens right; mirror it when the pipe runs right.
            if ch in (ord("4"), ord("5")):
                slot = (0 if ch == ord("4") else 1)
                if world.get(x + 1, y) in HPIPE_CODES:
                    slot += 4
            else:
                slot = 2 if ch == ord("6") else 3
            fig = self._cached(f"HP{slot}", self.hpipes[slot], False)

        elif ch == ord("*"):
            fig = self._sprite_surf("COIN000")

        elif ch == CH_EXIT_TOP:
            if world.get(x, y - 1) == CH_EXIT_TOP:
                fig = self._sprite_surf("EXIT001")
            else:
                fig = self._sprite_surf("EXIT000")

        elif ch == CH_GRASS_TOP:                     # '≈'
            above = world.get(x, y - 1)
            if above == CH_FENCE and opt.Design == 2:
                self._draw(target, xpos, ypos, self._sprite_surf("SMTREE001"))
            if above == CH_PALM_TOP and opt.Design == 1:
                self._draw(target, xpos, ypos, self._sprite_surf("WPALM000"))
            right = world.get(x + 1, y)
            if x == 0 or world.get(x - 1, y) == ch:
                key = 2 if right == ch else 3
            else:
                key = 1 if right == ch else 3
            fig = self._cached(f"G{key}", self.grass[key], False)

        elif ch == CH_FENCE:                         # '≡'
            above = world.get(x, y - 1)
            if opt.Design == 1:
                fig = self._sprite_surf("FENCE001" if above != ch
                                        else "FENCE000")
            elif opt.Design == 2:
                fig = self._sprite_surf("SMTREE000" if above != ch
                                        else "SMTREE001")

        elif ch == CH_PALM_TOP:                      # '÷'
            if opt.Design == 1:
                fig = self._sprite_surf("WPALM000")
            elif opt.Design == 2:
                above = world.get(x, y - 1)
                fig = self._sprite_surf("BUSH001" if above != ch
                                        else "BUSH000")

        elif ch == CH_PALM_MID:                      # '·'
            if opt.Design == 1:
                if world.get(x - 1, y) == CH_PALM_SIDE:
                    self._draw(target, xpos, ypos,
                               self._cached("A3", self.palm[3], False))
                elif world.get(x + 1, y) == CH_PALM_SIDE:
                    self._draw(target, xpos, ypos,
                               self._cached("A1", self.palm[1], False))
                fig = self._cached("A0", self.palm[0], False)

        elif ch == CH_PALM_TRUNK_T:                  # '⌠'
            if opt.Design == 1:
                if world.get(x, y + 1) == CH_PALM_TOP:
                    self._draw(target, xpos, ypos,
                               self._sprite_surf("WPALM000"))
                fig = self._cached("A1", self.palm[1], False)

        elif ch == CH_PALM_SIDE:                     # '∙'
            if opt.Design == 1:
                fig = self._cached("A2", self.palm[2], False)

        elif ch == CH_PALM_TRUNK_B:                  # '⌡'
            if opt.Design == 1:
                if world.get(x, y + 1) == CH_PALM_TOP:
                    self._draw(target, xpos, ypos,
                               self._sprite_surf("WPALM000"))
                fig = self._cached("A3", self.palm[3], False)

        elif ch == ord("#"):
            above = world.get(x, y - 1)
            if opt.Design == 1:
                fig = self._sprite_surf("FALL000")
            elif opt.Design == 2:
                if above == ord("#"):
                    self._put(target, xpos, ypos,
                              self._sprite_surf("TREE001", True))
                elif above == ord("%"):
                    self._put(target, xpos, ypos,
                              self._sprite_surf("TREE000", True))
                    fig = self._sprite_surf("TREE003")
                else:
                    fig = self._sprite_surf("TREE003")
            elif opt.Design == 3:
                fig = self._sprite_surf("WINDOW001")
            elif opt.Design == 4:
                fig = self._sprite_surf("LAVA000")
            elif opt.Design == 5:
                target.fill(self._rgb(5), (xpos, ypos, TILE_W, TILE_H))

        elif ch == ord("%"):
            above = world.get(x, y - 1)
            if opt.Design == 1:
                fig = self._sprite_surf("FALL001")
            elif opt.Design == 2:
                if above == ord("%"):
                    self._put(target, xpos, ypos,
                              self._sprite_surf("TREE000", True))
                elif above == ord("#"):
                    self._put(target, xpos, ypos,
                              self._sprite_surf("TREE001", True))
                    fig = self._sprite_surf("TREE002")
                else:
                    fig = self._sprite_surf("TREE002")
            elif opt.Design == 3:
                fig = self._sprite_surf("WINDOW000")
            elif opt.Design == 4:
                fig = self._sprite_surf("LAVA001")
            elif opt.Design == 5:
                frame = (x + self.lava_counter // 8) % 5
                fig = self._sprite_surf(f"LAVA200{frame + 1}")

        if fig is not None:
            self._draw(target, xpos, ypos, fig)

    # ------------------------------------------------------------------ #
    # Charlie scenery (port-only decoration layer)
    # ------------------------------------------------------------------ #
    def decoration_columns(self, x_view: int, view_cols: int) -> tuple[int, int]:
        """Column range to paint so no sprite is cut off at the screen edge.

        The palm crown is 59 px on a 20 px cell, so it reaches two columns past
        its marker on either side; a range that only covers the visible columns
        would leave that overhang unpainted.
        """
        return (x_view // TILE_W - DECO_OVERHANG,
                x_view // TILE_W + view_cols + DECO_OVERHANG)

    def draw_decorations(self, target: pygame.Surface, world: WorldMap,
                         x0: int, x1: int) -> None:
        """Paint the Charlie plants over the tile layer.

        They cannot go through Redraw: the crown and the stacked cattail are
        taller/wider than one cell, and Redraw walks columns left to right, so
        the next column's sky fill would wipe the overhang.  Redraw leaves the
        decoration codes alone (it only fills their cell with sky).  Every path
        that repaints tiles (the repaint, the scroll's entering column, the
        initial page) has to follow up with this call, or the frame shows the
        tile layer where a plant should be.
        """
        rows = world.rows
        frame = self.deco_frame
        for x in range(x0, x1 + 1):
            for y in range(rows):
                ch = world.get(x, y)
                if ch not in DECO_CODES:
                    continue
                xpos, ypos = x * TILE_W, y * TILE_H
                if ch == CH_CHAR_GRASS:
                    self._put(target, xpos, ypos,
                              self._sprite_surf(f"CHRGRS00{frame}"))
                elif ch == CH_CHAR_FLOWER:
                    s = self._sprite_surf(f"CHRFLW00{frame}")
                    self._put(target, xpos + (TILE_W - s.get_width()) // 2,
                              ypos + TILE_H - s.get_height(), s)
                elif ch == CH_CHAR_CATTAIL:
                    self._draw_cattail(target, world, x, y, xpos, ypos, frame)
                elif ch == CH_CHAR_PALM:
                    self._draw_palm(target, world, x, y, xpos, ypos, frame)
                elif ch == CH_CHAR_BUSH:
                    self._put(target, xpos, ypos,
                              self._sprite_surf(f"CHRBSH00{frame}"))
        if (self.options is not None and self.options.WaterTop >= 0
                and self.options.WaterStyle == 1):
            # the 91898 water gets the grass/cattail overlay; it is drawn
            # after the tiles (so it covers the water) and before entities
            self._draw_water_plants(target, world, x0, x1, frame)

    def _draw_water_plants(self, target, world, x0: int, x1: int,
                           frame: int) -> None:
        """Water overlay: sparse grass tufts and cattails on empty water
        cells, drawn after the tiles so they cover the water, before Mario.
        The frame is the same grass/leaf two-state clock as the land decor."""
        opt = self.options
        assert opt is not None
        rows = world.rows
        for x in range(x0, x1 + 1):
            for y in range(rows - 1, max(opt.WaterTop, 0) - 1, -1):
                if world.get(x, y) != ord(" "):
                    continue
                # only draw above a solid floor, never on top of a coin or a
                # decoration tile
                if y + 1 < rows:
                    below = world.get(x, y + 1)
                    if below not in CAN_HOLD_YOU and below not in CAN_STAND_ON:
                        continue
                key = (x * 7 + y * 3) % 13
                if key in (0, 9):
                    self._put(target, x * TILE_W, y * TILE_H,
                              self._sprite_surf(f"CHRGRS00{frame}"))
                elif key in (3, 6):
                    seg = 2 if key == 3 else 1
                    self._put(target, x * TILE_W, y * TILE_H,
                              self._sprite_surf(f"CHRCAT{seg}00{frame}"))
                    if (y - 1 >= opt.WaterTop
                            and world.get(x, y - 1) == ord(" ")):
                        self._put(target, x * TILE_W, (y - 1) * TILE_H,
                                  self._sprite_surf(
                                      f"CHRCAT{3 - seg}00{frame}"))
                break

    def _draw_palm(self, target, world, x, y, xpos, ypos, frame) -> None:
        """One 'p' cell is a crown on the ground; a vertical run grows a trunk.

        Only the topmost marker draws: it puts the crown (and the leaf overlay)
        bottom-aligned on itself and one 16x14 trunk segment in every cell
        below.  The crown's last bulge is rows 34..40 with the same
        10/14/16/16/16/14/10 width profile as a segment's lower disc, and both
        are centred in the cell, so the stack is seamless.
        """
        up = 0
        while world.get(x, y - up - 1) == CH_CHAR_PALM:
            up += 1
        if up:
            return
        down = 0
        while world.get(x, y + down + 1) == CH_CHAR_PALM:
            down += 1
        trunk = self._sprite_surf("CHRTRNK000")
        for k in range(1, down + 1):
            self._put(target, xpos + (TILE_W - trunk.get_width()) // 2,
                      ypos + k * TILE_H, trunk)
        crown = self._sprite_surf("CHRPALM000")
        cx = xpos + (TILE_W - crown.get_width()) // 2
        cy = ypos + TILE_H - crown.get_height()
        self._put(target, cx, cy, crown)
        if frame == PALM_LEAF_FRAME:
            self._put(target, cx, cy, self._sprite_surf("CHRPALM001"))

    def _draw_cattail(self, target, world, x, y, xpos, ypos, frame) -> None:
        """Stack rule the user set: 1-2 cells repeat one segment, 3+ alternate
        segments 1/2 bottom-up, and the topmost cell is always segment 2."""
        down = 0
        while world.get(x, y + down + 1) == CH_CHAR_CATTAIL:
            down += 1
        up = 0
        while world.get(x, y - up - 1) == CH_CHAR_CATTAIL:
            up += 1
        if down + up + 1 <= 2:
            seg = 1
        else:
            seg = 2 if (up == 0 or down % 2) else 1
        self._put(target, xpos, ypos,
                  self._sprite_surf(f"CHRCAT{seg}00{frame}"))

    # -- blit helpers (PutImage = opaque, DrawImage = colour-keyed) ----------
    @staticmethod
    def _put(target: pygame.Surface, x: int, y: int,
             surf: pygame.Surface | None) -> None:
        if surf is not None:
            target.blit(surf, (x, y))

    @staticmethod
    def _draw(target: pygame.Surface, x: int, y: int,
              surf: pygame.Surface | None) -> None:
        if surf is not None:
            target.blit(surf, (x, y))

    @staticmethod
    def _upside_down(target: pygame.Surface, x: int, y: int,
                     surf: pygame.Surface | None) -> None:
        if surf is not None:
            target.blit(pygame.transform.flip(surf, False, True), (x, y))
