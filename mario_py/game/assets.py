"""Asset loader — converted data (JSON / NPY / PNG) -> runtime objects.

Sprites are held as raw palette-index arrays (``data/sprites/<NAME>.npy``).
FIGURES.PAS builds whole tile families at load time by manipulating palette
*indices* (ReColor / ReColor2 / Mirror / Rotate / ConvertGrass), so the baked
RGB PNGs are not sufficient.  RGB Surfaces are therefore produced from the
indices + palette on demand and cached.

Palette index 0 is always transparent (the original's DrawImage blitter skips
zero bytes with ``or al,al; jz``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Assets:
    palette: list[list[int]] = field(default_factory=list)
    palette_lookup: np.ndarray | None = None
    meta: dict[str, dict] = field(default_factory=dict)
    idx: dict[str, np.ndarray] = field(default_factory=dict)
    levels: dict[str, dict] = field(default_factory=dict)
    # Mod level packs (data/levels/levels_mod.json): names are "<pack>/<world><a|b>"
    # so they cannot collide with the original level names.
    mod_levels: dict[str, dict] = field(default_factory=dict)
    mod_packs: dict[str, dict] = field(default_factory=dict)
    # BACKGR.PAS height maps (BOGEN / MOUNT / BOGEN7 / BOGEN26), see
    # tools/convert_backgr.py
    backgr: dict[str, np.ndarray] = field(default_factory=dict)
    # TXT.PAS fonts (F8X8 / SWISS), decoded by tools/convert_fonts.py
    fonts: dict[str, dict] = field(default_factory=dict)
    _surf: dict[tuple[str, bool], pygame.Surface] = field(default_factory=dict)

    # --- loading ----------------------------------------------------------
    def load(self) -> "Assets":
        pal = json.loads(
            (DATA / "palettes" / "palette.json").read_text(encoding="utf-8"))
        self.palette = pal["pal256"]
        self.palette_lookup = np.asarray(self.palette, dtype=np.uint8)

        # sprite names are case-insensitive for callers: index.json has a few
        # mixed-case keys (e.g. Start001), so normalise once.
        raw_meta = json.loads(
            (DATA / "sprites" / "index.json").read_text(encoding="utf-8"))
        self.meta = {name.upper(): m for name, m in raw_meta.items()}
        for name, m in self.meta.items():
            self.idx[name] = np.load(DATA / "sprites" / m["idx_file"])

        self.levels = json.loads(
            (DATA / "levels" / "levels.json").read_text(encoding="utf-8"))
        mod_path = DATA / "levels" / "levels_mod.json"
        if mod_path.exists():
            mod = json.loads(mod_path.read_text(encoding="utf-8"))
            self.mod_levels = mod.get("levels", {})
            self.mod_packs = mod.get("packs", {})
        # Port-only test packs; kept out of the generated mod file so
        # convert_mod_levels.py can rewrite that freely.
        for extra in ("levels_webangelo.json", "levels_test.json",
                      "levels_mario4.json"):
            extra_path = DATA / "levels" / extra
            if not extra_path.exists():
                continue
            pack = json.loads(extra_path.read_text(encoding="utf-8"))
            self.mod_levels.update(pack.get("levels", {}))
            self.mod_packs.update(pack.get("packs", {}))

        bg = json.loads((DATA / "backgr.json").read_text(encoding="utf-8"))
        self.backgr = {k: np.asarray(v, dtype=np.int16)
                       for k, v in bg.items()}

        font_path = DATA / "fonts" / "glyphs.json"
        if font_path.exists():
            self.fonts = json.loads(font_path.read_text(encoding="utf-8"))
        return self

    # --- sprites ----------------------------------------------------------
    def sprite_idx(self, name: str) -> np.ndarray:
        """Raw (h, w) uint8 palette indices; 0 == transparent."""
        return self.idx[name.upper()]

    def sprite_size(self, name: str) -> tuple[int, int]:
        m = self.meta[name.upper()]
        return m["w"], m["h"]

    def to_surface(self, img: np.ndarray, opaque: bool = False) -> pygame.Surface:
        """indices -> RGBA Surface (index 0 transparent unless opaque)."""
        rgb = self.palette_lookup[img]                     # (h, w, 3)
        if opaque:
            alpha = np.full(img.shape, 255, dtype=np.uint8)
        else:
            alpha = np.where(img != 0, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        surf = pygame.image.frombuffer(rgba.tobytes(),
                                       (img.shape[1], img.shape[0]), "RGBA")
        return surf.convert_alpha()

    def sprite(self, name: str) -> pygame.Surface:
        key = (name.upper(), False)
        s = self._surf.get(key)
        if s is None:
            s = self.to_surface(self.sprite_idx(name))
            self._surf[key] = s
        return s

    def sprite_opaque(self, name: str) -> pygame.Surface:
        """For PutImage call sites: index 0 is NOT transparent in the original."""
        key = (name.upper(), True)
        s = self._surf.get(key)
        if s is None:
            s = self.to_surface(self.sprite_idx(name), opaque=True)
            self._surf[key] = s
        return s

    def has(self, name: str) -> bool:
        return name.upper() in self.idx

    # --- levels -----------------------------------------------------------
    def level(self, name: str) -> dict:
        """Look a level up in the original set, then in the mod packs."""
        if name in self.mod_levels:
            return self.mod_levels[name]
        return self.levels[name]


def load_assets() -> Assets:
    return Assets().load()
