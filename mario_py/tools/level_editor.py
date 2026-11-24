"""Level editor: raw JSON map -> tile grid -> click to paint -> save.

Features
  * PALETTE / LEVELS / PLATFORM tabs (the original MVP);
  * OPTIONS: every level option (SkyType, BackGrType, WallType1, Design,
    Horizon, Stars, Clouds, MoonSky, colours, InitX/Y, no_time, hint) plus the
    level size -- width in columns and rows in tiles;
  * OBJECT: right-click a cell to expose per-object parameters --
    enemy spawn direction (enemy_dirs), pipe type/link key/destination, and
    the selected long platform's type/speed/length/range/axis/variant/loop
    plus its path points;
  * warp pipes are drawn as links (link key on the line);
  * markers are drawn with the art the game spawns, not abbreviations: the
    sheet is rendered by the game's own spawn/draw code and cached, so an art
    or draw-code change rebuilds it (see tools/marker_sheet.py).
Ctrl+S writes the original JSON, F5 saves and launches the game.
Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) undo/redo; one drag stroke is one step and
every option, size, pipe, platform and path edit is its own step.

Run:  py -3.12 -m mario_py.tools.level_editor [--level testrooms/2-4a]
      py -3.12 -m mario_py.tools.level_editor --selftest
Manual: 关卡编辑器手册.md (repo root) -- update it whenever behaviour changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mario_py.game import enemies as E                       # noqa: E402
from mario_py.game.assets import load_assets                 # noqa: E402
from mario_py.game.backgr import BackGr                      # noqa: E402
from mario_py.game.buffers import (MAX_WORLD_SIZE, NV, TILE_H, TILE_W,  # noqa: E402
                                  WorldMap, WorldOptions)
from mario_py.game.figures import (CH_EXIT_TOP, DECO_CODES, CH_FENCE,  # noqa: E402
                                   CH_GRASS_TOP,
                                   CH_PALM_MID, CH_PALM_SIDE, CH_PALM_TOP,
                                   CH_PALM_TRUNK_B, CH_PALM_TRUNK_T, Figures)
from mario_py.tools.marker_sheet import get_marker_sheet  # noqa: E402

DATA = ROOT / "mario_py" / "data" / "levels"
LEVEL_FILES = [DATA / "levels.json", DATA / "levels_mod.json",
               DATA / "levels_webangelo.json", DATA / "levels_test.json",
               DATA / "levels_mario4.json"]

SPACE = ord(" ")
SCALE = 2
TW = TILE_W * SCALE          # tiles are 20x14, not square
TH = TILE_H * SCALE
TOP_BAR = 30
PANEL_W = 320
WINDOW = (1280, 800)
# BackGrMap is MAX_WORLD_SIZE columns wide; taller than a screen is fine, the
# vertical camera takes over (see WorldMap.rows).
MAX_WIDTH = MAX_WORLD_SIZE
MAX_ROWS = 40

# The palette: (label, code) with ("group", title) separators.  Labels are
# ASCII on purpose (pygame's default font has no CJK glyphs).
PALETTE: list[tuple[str, int]] = [
    ("group", "TILES"),
    ("Empty", SPACE),
    ("Ground A", ord("A")),
    ("Grey I", ord("I")),
    ("Solid X", ord("X")),
    ("Question ?", ord("?")),
    ("Hidden Q @", ord("@")),
    ("Coin *", ord("*")),
    ("Pin =", ord("=")),
    ("Wood W", ord("W")),
    ("Note K", ord("K")),
    ("Brick J", ord("J")),
    ("Random x", ord("x")),
    ("Pipe 0", ord("0")),
    ("Pipe 1", ord("1")),
    ("Pipe 2", ord("2")),
    ("Pipe 3", ord("3")),
    ("HPipe mT", ord("4")),
    ("HPipe mB", ord("5")),
    ("HPipe bT", ord("6")),
    ("HPipe bB", ord("7")),
    ("Charlie grass", ord("g")),
    ("Charlie flower", ord("o")),
    ("Charlie cattail", ord("t")),
    ("Charlie palm", ord("p")),
    ("Charlie bush", ord("s")),
    ("Grass top", CH_GRASS_TOP),
    ("Fence", CH_FENCE),
    ("Palm top", CH_PALM_TOP),
    ("Palm mid", CH_PALM_MID),
    ("Palm side", CH_PALM_SIDE),
    ("Palm trunk T", CH_PALM_TRUNK_T),
    ("Palm trunk B", CH_PALM_TRUNK_B),
    ("Exit top", CH_EXIT_TOP),
    ("group", "ENEMIES / MARKERS"),
    ("Chibibo", E.CH_CHIBIBO),
    ("Vert fish", E.CH_VERT_FISH),
    ("Vert fireball", E.CH_VERT_FIREBALL),
    ("Chibibo B", E.CH_CHIBIBO_B),
    ("Plant 0", E.CH_PLANT0),
    ("Plant 1", E.CH_PLANT1),
    ("Plant 2", E.CH_PLANT2),
    ("Spiny", E.CH_RED),
    ("Koopa green", E.CH_KOOPA0),
    ("Koopa red", E.CH_KOOPA1),
    ("Koopa black", E.CH_KOOPA2),
    ("Fly green", E.CH_FLY_GREEN),
    ("Fly red H", E.CH_FLY_RED_H),
    ("Fly red V", E.CH_FLY_RED_V),
    ("Fly black", E.CH_FLY_BLACK),
    ("Spiny black", E.CH_SPINY_BLACK),
    ("Fly Spiny", E.CH_FLY_SPINY),
    ("Fly Spiny blk", E.CH_FLY_SPINY_BLACK),
    ("Fish spout (sw)", E.CH_FLY_FISH),
    ("Fly Chibibo", E.CH_FLY_CHIBIBO),
    ("Buzzy", E.CH_BUZZY),
    ("Fly Buzzy", E.CH_FLY_BUZZY),
    ("Dry Bones", E.CH_DRY_BONES),
    ("Hammer Bro", E.CH_HAMMER_BRO),
    ("Boo", E.CH_BOO),
    ("Bomb stand", E.CH_BOMB_STAND),
    ("Thwomp", E.CH_THWOMP),
    ("Lakitu (sw)", E.CH_LAKITU),
    ("Fire plant", E.CH_FIRE_PLANT),
    ("Fire plant aim", E.CH_FIRE_PLANT_AIM),
    ("Fast plant", E.CH_FAST_PLANT),
    ("Side plant", E.CH_SIDE_PLANT),
    ("Black plant", E.CH_BLACK_PLANT),
    ("Ghost chibibo", E.CH_GHOST_CHIBIBO),
    ("group", "SPECIAL"),
    ("Gate left", 174),
    ("Gate right", 175),
    ("Gate up", 173),
    ("Exit pipe", 0xE7),
    ("Block lift", E.CH_BLOCK_LIFT),
    ("Donut lift", E.CH_DONUT),
]

SHORT: dict[int, str] = {
    E.CH_CHIBIBO: "CH", E.CH_VERT_FISH: "VF", E.CH_VERT_FIREBALL: "VFB",
    E.CH_CHIBIBO_B: "CHb", E.CH_PLANT0: "P0", E.CH_PLANT1: "P1",
    E.CH_PLANT2: "P2", E.CH_RED: "SP", E.CH_KOOPA0: "KG", E.CH_KOOPA1: "KR",
    E.CH_KOOPA2: "KB", E.CH_FLY_GREEN: "FG", E.CH_FLY_RED_H: "FRh",
    E.CH_FLY_RED_V: "FRv", E.CH_FLY_BLACK: "FK", E.CH_SPINY_BLACK: "SPk",
    E.CH_FLY_SPINY: "FS", E.CH_FLY_SPINY_BLACK: "FSk", E.CH_FLY_FISH: "SW",
    E.CH_FLY_CHIBIBO: "FC", E.CH_BUZZY: "BZ", E.CH_FLY_BUZZY: "FB",
    E.CH_DRY_BONES: "DB", E.CH_HAMMER_BRO: "HB", E.CH_BOO: "BO",
    E.CH_BOMB_STAND: "BS", E.CH_THWOMP: "TW", E.CH_LAKITU: "LK", E.CH_FIRE_PLANT: "FP", E.CH_FIRE_PLANT_AIM: "FA",
    E.CH_FAST_PLANT: "FS", E.CH_SIDE_PLANT: "SS",
    E.CH_BLACK_PLANT: "Pb", E.CH_GHOST_CHIBIBO: "Gh",
    174: "G<", 175: "G>", 173: "G^", 0xE7: "EX",
    E.CH_BLOCK_LIFT: "LF", E.CH_DONUT: "DN",
}

# --- per-object parameters (port-only where the original had none) -------- #
TABS = ("palette", "levels", "platforms", "options", "object")
TAB_LABELS = ("PAL", "LEVELS", "PLAT", "OPT", "OBJ")

# Markers whose initial walking / patrol direction is spawn-time data.
DIR_CODES = {
    E.CH_CHIBIBO, E.CH_CHIBIBO_B, E.CH_GHOST_CHIBIBO, E.CH_RED,
    E.CH_KOOPA0, E.CH_KOOPA1, E.CH_KOOPA2, E.CH_BUZZY,
    E.CH_FLY_GREEN, E.CH_FLY_RED_H, E.CH_FLY_RED_V, E.CH_FLY_BLACK,
    E.CH_SPINY_BLACK, E.CH_FLY_SPINY, E.CH_FLY_SPINY_BLACK,
    E.CH_FLY_CHIBIBO, E.CH_FLY_BUZZY, E.CH_DRY_BONES, E.CH_HAMMER_BRO,
}
DIR_NAMES = {0: "AUTO", 1: "RIGHT", -1: "LEFT"}
VERT_DIR_NAMES = {0: "AUTO", 1: "UP FIRST", -1: "DOWN FIRST"}

PIPE_TYPES = (("WARP", 0xE0), ("SUBLEVEL", 0xE1), ("EXIT", 0xE7))

# Options exposed in the OPTIONS tab: (key, lo, hi, step, label).  Values that
# differ from 1a are highlighted; MoonSky is port-only.
OPTION_FIELDS = (
    ("group", "LEVEL SIZE"),
    ("Width", 1, MAX_WIDTH, 5, "Width (cols)"),
    ("Rows", NV, MAX_ROWS, 1, "Rows (tiles)"),
    ("group", "SKY / BACKGROUND"),
    ("SkyType", 0, 12, 1, "SkyType"),
    ("BackGrType", 0, 12, 1, "BackGrType"),
    ("Design", 0, 5, 1, "Design"),
    ("Horizon", 0, 400, 14, "Horizon"),
    ("Stars", 0, 2, 1, "Stars"),
    ("Clouds", 0, 255, 1, "Clouds (pal)"),
    ("MoonSky", 0, 1, 1, "MoonSky (XP)"),
    ("Vision", 0, 1, 1, "Ghost-house sight"),
    ("group", "WATER"),
    ("WaterTop", -1, 400, 14, "WaterTop (-1 off)"),
    ("WaterStyle", 1, 2, 1, "WaterStyle 1 91898 / 2 DOS"),
    ("WaterFillTop", 0, 1, 1, "Fill rows above water"),
    ("BackGrColor1", 0, 63, 1, "BackGrColor1"),
    ("BackGrColor2", 0, 63, 1, "BackGrColor2"),
    ("group", "WALLS / TILES"),
    ("WallType1", 0, 110, 1, "WallType1"),
    ("GroundColor1", 0, 255, 1, "GroundColor1"),
    ("GroundColor2", 0, 255, 1, "GroundColor2"),
    ("BrickColor", 0, 255, 1, "BrickColor"),
    ("WoodColor", 0, 255, 1, "WoodColor"),
    ("XBlockColor", 0, 255, 1, "XBlockColor"),
    ("PipeColor", 0, 255, 1, "PipeColor"),
    ("group", "PLANT COLOURS (C2/C3)"),
    ("C2r", 0, 63, 1, "C2 red"),
    ("C2g", 0, 63, 1, "C2 green"),
    ("C2b", 0, 63, 1, "C2 blue"),
    ("C3r", 0, 63, 1, "C3 red"),
    ("C3g", 0, 63, 1, "C3 green"),
    ("C3b", 0, 63, 1, "C3 blue"),
    ("group", "START / FLAGS"),
    ("InitX", 0, 4000, 20, "InitX"),
    ("InitY", 0, 4000, 14, "InitY"),
    ("no_time", 0, 1, 1, "No time limit"),
    ("hint", 0, 0, 0, "Hint text"),
)

# Options that change the baked tile/sky/palette art and need a full reload.
STYLE_OPTIONS = {
    "SkyType", "BackGrType", "Design", "Horizon", "WallType1", "Clouds",
    "GroundColor1", "GroundColor2", "BackGrColor1", "BackGrColor2",
    "BrickColor", "WoodColor", "XBlockColor", "PipeColor",
    "C2r", "C2g", "C2b", "C3r", "C3g", "C3b", "MoonSky",
    "WaterTop", "WaterStyle", "WaterFillTop",
}

_DEFAULT_OPTIONS: dict | None = None


def default_options() -> dict:
    global _DEFAULT_OPTIONS
    if _DEFAULT_OPTIONS is None:
        raw = json.loads((DATA / "levels.json").read_text(encoding="utf-8"))
        _DEFAULT_OPTIONS = dict(raw["1a"]["options"])
    return _DEFAULT_OPTIONS


def fmt_value(key: str, value) -> str:
    if key == "hint":
        return repr(value)[:28]
    if key == "no_time":
        return "YES" if value else "NO"
    return str(value)


def _ui_font(size: int) -> pygame.font.Font:
    """A CJK-capable UI font when the system has one (pack titles are Chinese)."""
    path = pygame.font.match_font(
        "microsoftyahei,simhei,notosanscjksc,pingfangsc,arial")
    return pygame.font.Font(path, size) if path else pygame.font.Font(None, size)


@dataclass
class LevelDoc:
    name: str
    path: Path
    container: dict
    level: dict
    wrapped: bool
    compact: bool


def _load_docs(files: list[Path]) -> tuple[dict[str, LevelDoc],
                                          list[tuple[str, list[str]]]]:
    docs: dict[str, LevelDoc] = {}
    groups: list[tuple[str, list[str]]] = []
    for path in files:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        root = json.loads(raw)
        wrapped = "levels" in root
        levels = root["levels"] if wrapped else root
        packs = root.get("packs", {})
        by_group: dict[str, list[str]] = {}
        for name, level in levels.items():
            docs[name] = LevelDoc(name, path, root, level, wrapped,
                                  compact="\n" not in raw)
            prefix = name.split("/")[0] if "/" in name else ""
            by_group.setdefault(prefix, []).append(name)
        for prefix, names in by_group.items():
            if not prefix:
                title = f"ORIGINAL ({path.name})"
            else:
                title = packs.get(prefix, {}).get("title", prefix)
                title = f"{prefix} - {title}"
            groups.append((title, names))
    return docs, groups


class Editor:
    def __init__(self, assets, files: list[Path], initial: str | None = None,
                 headless: bool = False) -> None:
        self.assets = assets
        self.docs, self.groups = _load_docs(files)
        if not self.docs:
            raise SystemExit("no levels found in the given JSON files")
        self.headless = headless
        self.screen = (None if headless
                       else pygame.display.set_mode(WINDOW))
        if not headless:
            pygame.display.set_caption("MARIO LEVEL EDITOR")
        self.font = _ui_font(16)
        self.small = _ui_font(13)

        self.name = ""
        self.doc: LevelDoc | None = None
        self.raw: list[list[int]] = []
        self.rows = 13
        self.width = 0
        self.options: WorldOptions | None = None
        self.fig: Figures | None = None
        self.backgr: BackGr | None = None
        self.surf: pygame.Surface | None = None
        self.scaled: pygame.Surface | None = None
        self.need_rebuild = False
        self.dirty = False
        self.selected = ord("X")
        self.platforms: list[dict] = []
        self.sel_platform = -1
        self.sel_cell: tuple[int, int] | None = None
        self.enemy_dirs: dict[tuple[int, int], int] = {}
        self.prop_index = 0
        self.prop_rows: list[tuple] = []
        self.text_edit: dict | None = None
        self.cam_x = self.cam_y = 0
        self.grid_on = True
        self.tab = "palette"          # see TABS
        self.panel_scroll = 0
        self.scroll_to_current = False
        self.panel_rows: list[tuple] = []
        self.thumb: dict[int, pygame.Surface | None] = {}
        self.marker_imgs: dict[int, pygame.Surface] = {}
        self.marker_map: dict[int, pygame.Surface] = {}
        self.status = ""
        self.esc_armed = False
        self.painting = False
        self.panning = False
        self.launched: subprocess.Popen | None = None
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._stroke_pending = False
        self._saved_snap: dict | None = None

        if initial is None:
            initial = ("testrooms/1-1a" if "testrooms/1-1a" in self.docs
                       else next(iter(self.docs)))
        if initial not in self.docs:
            raise SystemExit(f"unknown level {initial!r}; "
                             f"known: {', '.join(list(self.docs)[:8])} ...")
        self.load_level(initial)

    # -- level / rendering ------------------------------------------------ #
    def load_level(self, name: str) -> None:
        self.name = name
        self.doc = self.docs[name]
        lvl = self.doc.level
        self.raw = lvl["map"]
        self.platforms = lvl.get("platforms") or []
        self.sel_platform = -1
        self.sel_cell = None
        self.prop_index = 0
        self.text_edit = None
        self.enemy_dirs = {}
        for e in lvl.get("enemy_dirs") or []:
            try:
                self.enemy_dirs[(int(e["x"]), int(e["y"]))] = int(e["dir"])
            except (KeyError, TypeError, ValueError):
                pass
        self.width = int(lvl.get("width") or len(self.raw))
        self.rows = int(lvl.get("rows") or
                        (len(self.raw[0]) if self.raw else 13))
        self.options = WorldOptions.from_dict(lvl.get("options"))
        self._reset_style()     # BuildWorld first: it bakes grass/palm art
        self.dirty = False
        self._undo = []
        self._redo = []
        self._stroke_pending = False
        self._saved_snap = self._snapshot()
        self.cam_x = self.cam_y = 0
        if self.tab == "levels":
            self.scroll_to_current = True
        if not self.headless:
            pygame.display.set_caption(f"LEVEL EDITOR - {name}")

    def display_world(self) -> WorldMap:
        world = WorldMap()
        cols = [list(c) for c in self.raw]
        world.load_columns(cols, self.options, rows=self.rows,
                           vertical=self.doc.level.get("vertical"))
        return world

    def rebuild(self) -> None:
        """Re-run BuildWorld and repaint the level surface (raw data stays)."""
        assert self.fig is not None and self.options is not None
        world = self.display_world()
        self.fig.build_world(world, self.options)
        self.fig.set_sky_palette()
        self.disp = world
        w, h = self.width * TILE_W, self.rows * TILE_H
        surf = pygame.Surface((w, h))
        if self.options.MoonSky:
            self.backgr.draw_moon_sky(surf, 0)
        for y in range(self.rows):
            for x in range(self.width):
                self.fig.draw_sky(surf, x * TILE_W, y * TILE_H,
                                  TILE_W, TILE_H)
                self.fig.redraw(surf, world, x, y)
        self.backgr.draw_clouds(surf, 0)
        self.fig.draw_decorations(surf, world, 0, self.width - 1)
        self.surf = surf
        self.scaled = pygame.transform.scale(surf, (w * SCALE, h * SCALE))
        self.clamp_cam()

    @staticmethod
    def marker_codes() -> list[int]:
        """Palette codes that are creatures/mechanisms with real art."""
        return sorted({code for kind, code in PALETTE
                       if kind != "group" and 0x80 <= code <= 0xB1})

    def ensure_markers(self) -> dict[int, pygame.Surface]:
        """The preview sheet, rendered from the game's own spawn/draw code."""
        if self.marker_imgs or self.fig is None or self.options is None:
            return self.marker_imgs
        self.marker_imgs = get_marker_sheet(
            self.assets, self.fig, self.options, self.marker_codes())
        self.marker_map = {
            code: pygame.transform.scale(
                surf, (surf.get_width() * SCALE, surf.get_height() * SCALE))
            for code, surf in self.marker_imgs.items()}
        return self.marker_imgs

    @staticmethod
    def _fit(surf: pygame.Surface, w: int, h: int) -> pygame.Surface:
        sw, sh = surf.get_size()
        k = min(w / sw, h / sh)
        return pygame.transform.scale(surf, (max(1, round(sw * k)),
                                             max(1, round(sh * k))))

    def build_thumbs(self) -> None:
        assert self.fig is not None and self.options is not None
        pw = WorldMap()
        # load_columns() writes XSize/YSize, so keep the level's options intact.
        pw.load_columns([[SPACE] * 13], replace(self.options), rows=13)
        thumbs: dict[int, pygame.Surface | None] = {}
        markers = self.ensure_markers()
        for entry in PALETTE:
            if entry[0] == "group":
                continue
            code = entry[1]
            img = markers.get(code)
            if img is not None:
                thumbs[code] = self._fit(img, TW - 4, TH - 4)
                continue
            if 0x80 <= code <= 0xB1 or code in (173, 174, 175, 0xE7):
                thumbs[code] = None
                continue
            pw.set(0, 0, code)
            if code in DECO_CODES:
                # the scenery sprites are bigger than the cell they hang on
                cell = pygame.Surface((TILE_W * 4, TILE_H * 4))
                self.fig.draw_sky(cell, 0, 0, TILE_W * 4, TILE_H * 4)
                pw.set(2, 2, code)
                self.fig.draw_decorations(cell, pw, 2, 2)
                pw.set(2, 2, SPACE)
            else:
                cell = pygame.Surface((TILE_W, TILE_H))
                self.fig.draw_sky(cell, 0, 0, TILE_W, TILE_H)
                self.fig.redraw(cell, pw, 0, 0)
            thumbs[code] = pygame.transform.scale(
                cell, (TILE_W * SCALE - 4, TILE_H * SCALE - 4))
            pw.set(0, 0, SPACE)
        self.thumb = thumbs

    # -- option / object properties --------------------------------------- #
    def _reset_style(self) -> None:
        """(Re)build figures/backdrop for the current options and repaint."""
        assert self.options is not None
        self.fig = Figures(self.assets)
        self.backgr = BackGr(self.fig, self.assets)
        self.fig.backgr = self.backgr
        self.fig.init_sky(self.options.SkyType)
        self.fig.init_walls(self.options.WallType1, self.options.WallType2,
                            self.options.WallType3, self.options)
        self.fig.init_pipes(self.options.PipeColor)
        self.backgr.init_backgr(self.options.BackGrType, self.options.Clouds)
        self.backgr.init_moon_sky(self.options.MoonSky)
        self.marker_imgs = {}
        self.marker_map = {}
        self.rebuild()
        self.build_thumbs()

    def reload_style(self) -> None:
        """Re-init sky/walls/pipes/backdrop after an art-affecting option."""
        self._reset_style()

    def _field_value(self, key: str):
        if key == "Width":
            return self.width
        if key == "Rows":
            return self.rows
        if key == "no_time":
            return bool(self.doc.level.get("no_time"))
        if key == "hint":
            return str(self.doc.level.get("hint") or "")
        return getattr(self.options, key)

    def set_option(self, key: str, value, record: bool = True) -> None:
        if key == "Width":
            self.set_width(int(value))
            return
        if key == "Rows":
            self.set_rows(int(value))
            return
        cur = self._field_value(key)
        if key == "no_time":
            if bool(value) == bool(cur):
                return
        elif key == "hint":
            if str(value) == str(cur):
                return
        elif int(value) == int(cur):
            return
        if record:
            self._checkpoint()
        if key == "no_time":
            if value:
                self.doc.level["no_time"] = True
            else:
                self.doc.level.pop("no_time", None)
        elif key == "hint":
            self.doc.level["hint"] = str(value)
        else:
            value = int(value)
            setattr(self.options, key, value)
            self.doc.level.setdefault("options", {})[key] = value
        self.dirty = True
        if key in STYLE_OPTIONS:
            self.reload_style()
            self.status = f"{key} = {self._field_value(key)} (rebuilt)"
        else:
            self.status = f"{key} = {self._field_value(key)}"

    def adjust_option(self, key: str, delta: int) -> None:
        spec = next((f for f in OPTION_FIELDS
                     if len(f) == 5 and f[0] == key), None)
        if spec is None:
            return
        _k, lo, hi, step, _label = spec
        if key == "hint":
            self.text_edit = {"key": "hint",
                              "buf": str(self._field_value("hint"))}
            self.status = "type the hint, Enter saves, Esc cancels"
            return
        if key == "no_time":
            self.set_option(key, not self._field_value(key))
            return
        value = int(self._field_value(key)) + delta * step
        self.set_option(key, max(lo, min(hi, value)))

    def _write_option(self, key: str, value: int) -> None:
        setattr(self.options, key, value)
        self.doc.level.setdefault("options", {})[key] = value

    def set_width(self, new: int) -> None:
        """Add/remove columns at the right edge (never shifts existing cells)."""
        new = max(1, min(MAX_WIDTH, int(new)))
        if new == self.width:
            return
        cur = len(self.raw)
        if new < cur:
            for x in range(new, cur):
                if any(c != SPACE for c in self.raw[x][:self.rows]):
                    self.status = (f"width: column {x} is not empty "
                                   f"(erase it first)")
                    return
            for i, spec in enumerate(self.platforms):
                end = int(spec.get("x", 0)) + max(1, int(spec.get("len", 1)))
                if end > new:
                    self.status = f"width: platform #{i} ends past column {new}"
                    return
                for point in spec.get("path") or []:
                    if len(point) >= 2 and int(point[0]) >= new:
                        self.status = (f"width: platform #{i} has a path point "
                                       f"past column {new}")
                        return
            for item in self.doc.level.get("webangelo") or []:
                if int(item.get("x", 0)) >= new * TILE_W:
                    self.status = "width: a firebar would leave the level"
                    return
            battle = self.doc.level.get("battle_room") or {}
            barrier = battle.get("barrier") or {}
            used = [int(c) for c in barrier.get("cols", [])]
            used += [int(t[0]) for t in battle.get("targets", [])]
            if used and max(used) >= new:
                self.status = f"width: battle_room uses column {max(used)}"
                return
            self._checkpoint()
            self.enemy_dirs = {k: v for k, v in self.enemy_dirs.items()
                               if k[0] < new}
            if self.options.InitX > (new - 1) * TILE_W:
                self._write_option("InitX", (new - 1) * TILE_W)
            while len(self.raw) > new:
                self.raw.pop()
        else:
            self._checkpoint()
            while len(self.raw) < new:
                self.raw.append([SPACE] * self.rows)
        self.width = new
        self.doc.level["width"] = new
        self.dirty = True
        self.rebuild()
        self.status = f"width = {new} cols"

    def set_rows(self, new: int) -> None:
        """Add/remove rows at the top; bottom-anchored data keeps its height.

        Source rows count from the floor up, so the map itself needs no shift.
        Absolute top-down fields do: WaterTop and InitY, battle-room rows and
        targets, and the webangelo firebars' pixel y.
        """
        new = max(NV, min(MAX_ROWS, int(new)))
        if new == self.rows:
            return
        delta = new - self.rows
        battle = self.doc.level.get("battle_room") or {}
        barrier = battle.get("barrier") or {}
        if delta < 0:
            for x, col in enumerate(self.raw):
                if any(c != SPACE for c in col[new:self.rows]):
                    self.status = (f"rows: column {x} has content above the "
                                   f"new ceiling")
                    return
            for i, spec in enumerate(self.platforms):
                if int(spec.get("y", 0)) >= new:
                    self.status = f"rows: platform #{i} is above the ceiling"
                    return
                for point in spec.get("path") or []:
                    if len(point) >= 2 and int(point[1]) >= new:
                        self.status = (f"rows: platform #{i} has a path point "
                                       f"above the ceiling")
                        return
            for t in battle.get("targets", []):
                if int(t[1]) + delta < 0:
                    self.status = (f"rows: battle_room target {t[0]},{t[1]} "
                                   f"leaves the room")
                    return
            for item in self.doc.level.get("webangelo") or []:
                if int(item.get("y", 0)) + delta * TILE_H < 0:
                    self.status = "rows: a firebar would leave the level"
                    return
        self._checkpoint()
        if delta < 0:
            for col in self.raw:
                del col[new:]
            self.enemy_dirs = {k: v for k, v in self.enemy_dirs.items()
                               if k[1] < new}
        if self.options.WaterTop > 0:
            water = max(0, min(new - 1, self.options.WaterTop + delta))
            self._write_option("WaterTop", water)
        # WaterTop == 0 means "the whole level is water"; it stays 0.
        start = max(0, min(new * TILE_H - TILE_H,
                           self.options.InitY + delta * TILE_H))
        self._write_option("InitY", start)
        if self.options.Horizon > 0:
            self._write_option(
                "Horizon",
                max(0, self.options.Horizon + delta * TILE_H))
        if barrier.get("rows"):
            rows = sorted({max(0, int(r) + delta) for r in barrier["rows"]})
            barrier["rows"] = rows
        if battle.get("targets"):
            battle["targets"] = [[int(t[0]), int(t[1]) + delta]
                                 for t in battle["targets"]]
        for item in self.doc.level.get("webangelo") or []:
            if "y" in item:
                item["y"] = max(0, int(item["y"]) + delta * TILE_H)
        self.rows = new
        self.doc.level["rows"] = new
        self.dirty = True
        self.rebuild()
        vert = self.doc.level.get("vertical", new > NV)
        self.status = f"rows = {new} (vertical={'on' if vert else 'off'})"

    # -- undo / redo ------------------------------------------------------ #
    UNDO_LIMIT = 50

    def _snapshot(self) -> dict:
        """Everything an edit can touch, deep-copied for one undo step."""
        level = copy.deepcopy(self.doc.level)
        level["map"] = copy.deepcopy(self.raw)
        level["width"] = self.width
        if self.rows != NV:
            level["rows"] = self.rows
        else:
            level.pop("rows", None)
        if self.enemy_dirs:
            level["enemy_dirs"] = [
                {"x": x, "y": y, "dir": d}
                for (x, y), d in sorted(self.enemy_dirs.items())]
        else:
            level.pop("enemy_dirs", None)
        return {"level": level, "options": replace(self.options),
                "width": self.width, "rows": self.rows}

    def _push(self, stack: list[dict], snap: dict) -> None:
        stack.append(snap)
        if len(stack) > self.UNDO_LIMIT:
            stack.pop(0)

    def _checkpoint(self) -> None:
        self._push(self._undo, self._snapshot())
        self._redo.clear()

    def _restore(self, snap: dict) -> None:
        # update in place: LevelDoc.container still points at this dict
        self.doc.level.clear()
        self.doc.level.update(copy.deepcopy(snap["level"]))
        self.raw = self.doc.level["map"]
        self.platforms = self.doc.level.get("platforms") or []
        self.width = int(snap["width"])
        self.rows = int(snap["rows"])
        self.options = replace(snap["options"])
        self.enemy_dirs = {}
        for e in self.doc.level.get("enemy_dirs") or []:
            try:
                self.enemy_dirs[(int(e["x"]), int(e["y"]))] = int(e["dir"])
            except (KeyError, TypeError, ValueError):
                pass
        self.sel_cell = None
        self.sel_platform = -1
        self.prop_index = 0
        self.text_edit = None
        self._reset_style()
        self.dirty = True
        self.clamp_cam()

    def undo(self) -> None:
        if not self._undo:
            self.status = "nothing to undo"
            return
        self._push(self._redo, self._snapshot())
        self._restore(self._undo.pop())
        self._sync_dirty()
        self.status = f"undo ({len(self._undo)} left) - Ctrl+Y redo"

    def redo(self) -> None:
        if not self._redo:
            self.status = "nothing to redo"
            return
        self._push(self._undo, self._snapshot())
        self._restore(self._redo.pop())
        self._sync_dirty()
        self.status = f"redo ({len(self._redo)} left)"

    def _sync_dirty(self) -> None:
        """Undoing back to the last saved state clears the UNSAVED flag."""
        self.dirty = (self._saved_snap is None
                      or self._snapshot() != self._saved_snap)

    def option_rows(self) -> list[tuple]:
        rows: list[tuple] = []
        defaults = default_options()
        for entry in OPTION_FIELDS:
            if entry[0] == "group":
                rows.append(("group", entry[1]))
                continue
            key, _lo, _hi, _step, label = entry
            value = self._field_value(key)
            if key == "hint":
                changed = bool(value)
            elif key == "no_time":
                changed = bool(value)
            elif key in ("Width", "Rows"):
                changed = False
            else:
                changed = int(value) != int(defaults.get(key, 0))
            rows.append(("field", key, label, fmt_value(key, value), changed))
        return rows

    # -- raw-cell helpers -------------------------------------------------- #
    def _raw_src(self, col: int, src: int) -> int:
        if not (0 <= col < self.width and 0 <= src < self.rows):
            return SPACE
        column = self.raw[col]
        return column[src] if src < len(column) else SPACE

    def _write_src(self, col: int, src: int, code: int) -> None:
        if not (0 <= col < self.width and 0 <= src < self.rows):
            return
        column = self.raw[col]
        while len(column) < self.rows:
            column.append(SPACE)
        if column[src] != code:
            column[src] = code
            self.dirty = True
            self.need_rebuild = True

    def pipe_pair_at(self, col: int, disp_row: int):
        """The vertical-pipe mouth pair for the cell under the cursor.

        A mouth is two code bytes ($E0..$EF) in adjacent cells with a pipe rim
        ('0'/'1') directly below.  Selecting the rim or the body looks upward.
        """
        if not (0 <= col < self.width and 0 <= disp_row < self.rows):
            return None
        src = self.rows - 1 - disp_row
        code = self.cell_code(col, disp_row)
        if 0xE0 <= code <= 0xEF:
            for lc in (col - 1, col):
                if 0 <= lc < self.width - 1:
                    c1, c2 = self._raw_src(lc, src), self._raw_src(lc + 1, src)
                    if 0xE0 <= c1 <= 0xEF and 0xE0 <= c2 <= 0xEF:
                        return lc, src, c1, c2
        for dy in (1, 2):
            sr = src + dy
            if sr >= self.rows or col >= self.width - 1:
                continue
            if (self._raw_src(col, sr - 1) not in (ord("0"), ord("1"))
                    or self._raw_src(col + 1, sr - 1) not in (ord("0"), ord("1"))):
                continue
            c1, c2 = self._raw_src(col, sr), self._raw_src(col + 1, sr)
            if 0xE0 <= c1 <= 0xEF and 0xE0 <= c2 <= 0xEF:
                return col, sr, c1, c2
        return None

    def pipe_target(self, c1: int, c2: int, exclude: tuple[int, int]):
        """Mirror PLAY.FindPipeExit: type + matching mouth position."""
        if c1 == 0xE7:
            return "EXIT", None
        if c1 == 0xE1:
            return "SUBMAP", None
        if c1 != 0xE0:
            # 0xE8 and the other spare codes are only ever *destinations*:
            # PLAY's pipe switch does not act on them.
            return "DESTINATION", None
        for x in range(self.width - 1):
            for y in range(self.rows):
                if (x, y) == exclude:
                    continue
                if (0xE0 <= self._raw_src(x, y) <= 0xEF
                        and self._raw_src(x + 1, y) == c2):
                    return "WARP", (x, y)
        return "NONE", None

    def pipe_mouths(self) -> list[tuple[int, int, int, int]]:
        """All vertical-pipe mouth pairs (rim '0'/'1' directly below)."""
        out = []
        for x in range(self.width - 1):
            for y in range(1, self.rows):
                if (self._raw_src(x, y - 1) in (ord("0"), ord("1"))
                        and self._raw_src(x + 1, y - 1) in (ord("0"), ord("1"))
                        and 0xE0 <= self._raw_src(x, y) <= 0xEF
                        and 0xE0 <= self._raw_src(x + 1, y) <= 0xEF):
                    out.append((x, y, self._raw_src(x, y),
                                self._raw_src(x + 1, y)))
        return out

    def set_pipe_pair(self, lc: int, src: int, c1: int, c2: int) -> None:
        self._write_src(lc, src, c1)
        self._write_src(lc + 1, src, c2)
        self.status = (f"pipe {lc},{src}: type 0x{c1:02X} link 0x{c2:02X}")

    def link_pipe(self, src_cell: tuple[int, int],
                  dst_cell: tuple[int, int]) -> bool:
        """Point the source warp pipe at the mouth under the mouse.

        The link key is the second code byte; PLAY.FindPipeExit matches a
        destination mouth whose right byte equals it.  A free key is chosen so
        existing links are left alone.
        """
        sp = self.pipe_pair_at(*src_cell)
        dp = self.pipe_pair_at(*dst_cell)
        if sp is None or dp is None or sp[:2] == dp[:2]:
            self.status = "LINK: select a pipe, then hover another pipe"
            return False
        used = {c2 for lc, sr, c1, c2 in self.pipe_mouths()
                if c1 == 0xE0 and (lc, sr) != sp[:2]}
        key = next((0xE0 + k for k in range(16) if 0xE0 + k not in used),
                   None)
        if key is None:
            self.status = "LINK: all 16 keys are in use"
            return False
        self._checkpoint()
        self.set_pipe_pair(sp[0], sp[1], 0xE0, key)
        self.set_pipe_pair(dp[0], dp[1], 0xE8, key)   # destination-only
        self.status = (f"pipe {sp[0]},{sp[1]} -> {dp[0]},{dp[1]} "
                       f"(key {key - 0xE0})")
        return True

    # -- per-marker direction --------------------------------------------- #
    def cycle_enemy_dir(self, col: int, src: int, delta: int) -> None:
        order = (0, 1, -1)
        cur = self.enemy_dirs.get((col, src), 0)
        nxt = order[(order.index(cur) + delta) % 3]
        if nxt == cur:
            return
        self._checkpoint()
        if nxt == 0:
            self.enemy_dirs.pop((col, src), None)
        else:
            self.enemy_dirs[(col, src)] = nxt
        self.dirty = True
        self.status = (f"enemy {col},{src} direction "
                       f"{DIR_NAMES[nxt]}")

    # -- platform selection / path ---------------------------------------- #
    def platform_at(self, col: int, disp_row: int) -> int | None:
        src = self.rows - 1 - disp_row
        for i, spec in enumerate(self.platforms):
            try:
                x = int(spec.get("x", 0))
                y = int(spec.get("y", 0))
                n = max(1, int(spec.get("len", 3)))
            except (TypeError, ValueError):
                continue
            if x <= col < x + n and y <= src <= y + 1:
                return i
        return None

    def add_path_point(self, col: int, src: int) -> None:
        if not (0 <= self.sel_platform < len(self.platforms)):
            self.status = "select a platform first"
            return
        spec = self.platforms[self.sel_platform]
        self._checkpoint()
        spec.setdefault("path", []).append([col, src])
        spec["type"] = "path"
        self.dirty = True
        self.status = f"platform #{self.sel_platform} path point {col},{src}"

    def pop_path_point(self) -> None:
        if not (0 <= self.sel_platform < len(self.platforms)):
            return
        spec = self.platforms[self.sel_platform]
        path = spec.get("path") or []
        if path:
            self._checkpoint()
            path.pop()
            self.dirty = True
            self.status = f"platform #{self.sel_platform} path point removed"

    def object_rows(self) -> list[tuple]:
        rows: list[tuple] = []
        cell = self.sel_cell
        if cell is not None:
            col, row = cell
            src = self.rows - 1 - row
            code = self.cell_code(col, row)
            name = SHORT.get(code) or f"0x{code:02X}"
            rows.append(("group", f"CELL {col},{src}  {name}"))
            rows.append(("action", "pick", "PICK AS BRUSH"))
            rows.append(("action", "erase", "ERASE CELL"))
            rows.append(("action", "start", "SET START HERE"))
            if code in DIR_CODES:
                names = (VERT_DIR_NAMES if code == E.CH_FLY_RED_V
                         else DIR_NAMES)
                d = self.enemy_dirs.get((col, src), 0)
                rows.append(("field", "enemy_dir", "Direction", names[d], True))
            pair = self.pipe_pair_at(col, row)
            if pair:
                lc, sr, c1, c2 = pair
                pname = next((n for n, c in PIPE_TYPES if c == c1),
                             "DESTINATION")
                kind, dest = self.pipe_target(c1, c2, (lc, sr))
                dtext = (f"COL {dest[0]} ROW {dest[1]}" if dest else kind)
                rows.append(("group", f"PIPE {lc},{sr}  -> {dtext}"))
                rows.append(("field", "pipe_type", "Type", pname, True))
                rows.append(("field", "pipe_link", "Link key",
                             f"{c2 - 0xE0} (0x{c2:02X})", True))
                rows.append(("field", "pipe_dest", "Destination", dtext, False))
                rows.append(("action", "pipe_link_cursor",
                             "LINK TO PIPE UNDER MOUSE (L)"))

        if 0 <= self.sel_platform < len(self.platforms):
            spec = self.platforms[self.sel_platform]
            rows.append(("group", f"PLATFORM #{self.sel_platform}"))
            rows.append(("field", "plat_type", "Type",
                         str(spec.get("type", "patrol")).upper(), True))
            rows.append(("field", "plat_speed", "Speed",
                         str(spec.get("speed", 0.8)), True))
            rows.append(("field", "plat_len", "Length (seg)",
                         str(spec.get("len", 3)), True))
            rows.append(("field", "plat_range", "Range (patrol)",
                         str(spec.get("range", 80)), True))
            rows.append(("field", "plat_axis", "Axis",
                         str(spec.get("axis", "h")).upper(), True))
            rows.append(("field", "plat_variant", "Variant",
                         str(spec.get("variant", 0)), True))
            rows.append(("field", "plat_loop", "Loop",
                         "YES" if spec.get("loop") else "NO", True))
            path = spec.get("path") or []
            rows.append(("group", f"PATH {len(path)} POINTS"))
            for k, point in enumerate(path):
                if len(point) >= 2:
                    rows.append(("info", f"{k}: col {point[0]} row {point[1]}"))
            rows.append(("action", "path_add", "ADD POINT AT CURSOR"))
            rows.append(("action", "path_pop", "REMOVE LAST POINT"))
            rows.append(("action", "path_clear", "CLEAR PATH"))
        if not rows:
            rows.append(("group", "RIGHT-CLICK A CELL OR PLATFORM"))
        return rows

    def adjust_prop(self, delta: int) -> None:
        if not (0 <= self.prop_index < len(self.prop_rows)):
            return
        entry = self.prop_rows[self.prop_index]
        if entry[0] == "action":
            self.run_action(entry[1])
            return
        if entry[0] != "field":
            return
        key = entry[1]
        cell = self.sel_cell
        if key == "enemy_dir" and cell is not None:
            col, row = cell
            self.cycle_enemy_dir(col, self.rows - 1 - row, delta)
            return
        if key in ("pipe_type", "pipe_link") and cell is not None:
            pair = self.pipe_pair_at(*cell)
            if pair is None:
                return
            lc, src, c1, c2 = pair
            if key == "pipe_type":
                names = [c for _n, c in PIPE_TYPES]
                i = names.index(c1) if c1 in names else 0
                c1 = names[(i + delta) % len(names)]
            else:
                c2 = 0xE0 + ((c2 - 0xE0 + delta) % 16)
            self._checkpoint()
            self.set_pipe_pair(lc, src, c1, c2)
            return
        if key == "pipe_dest":
            return
        if key.startswith("plat_") and 0 <= self.sel_platform < len(self.platforms):
            spec = self.platforms[self.sel_platform]
            self._checkpoint()
            if key == "plat_type":
                order = ("patrol", "osc", "fall", "path")
                cur = str(spec.get("type", "patrol")).lower()
                i = order.index(cur) if cur in order else 0
                spec["type"] = order[(i + delta) % len(order)]
            elif key == "plat_speed":
                v = float(spec.get("speed", 0.8)) + 0.1 * delta
                spec["speed"] = round(max(0.1, min(4.0, v)), 2)
            elif key == "plat_len":
                n = int(spec.get("len", 3)) + delta
                spec["len"] = max(1, min(n, self.width - int(spec.get("x", 0))))
            elif key == "plat_range":
                r = int(spec.get("range", 80)) + 20 * delta
                spec["range"] = max(0, min(400, r))
            elif key == "plat_axis":
                spec["axis"] = "v" if str(spec.get("axis", "h")) == "h" else "h"
            elif key == "plat_variant":
                spec["variant"] = 0 if int(spec.get("variant", 0)) else 1
            elif key == "plat_loop":
                spec["loop"] = not spec.get("loop")
            self.dirty = True
            self.status = (f"platform #{self.sel_platform}: "
                           f"{spec.get('type')} L{spec.get('len')} "
                           f"spd {spec.get('speed')}")

    def run_action(self, action: str) -> None:
        cell = self.sel_cell
        if action == "pick" and cell is not None:
            self.selected = self.cell_code(*cell)
            self.status = f"picked {self.selected} (0x{self.selected:02X})"
        elif action == "erase" and cell is not None:
            if self.cell_code(*cell) != SPACE:
                self._checkpoint()
                self.paint_cell(cell[0], cell[1], SPACE)
        elif action == "start" and cell is not None:
            col, row = cell
            # one undo step for the pair (InitX + InitY)
            self._checkpoint()
            self.set_option("InitX", col * TILE_W, record=False)
            # InitY is a top-down pixel coordinate (a 13-row level's ground
            # start is 126, not 0), so the clicked cell's top is row * TILE_H.
            self.set_option("InitY", row * TILE_H, record=False)
        elif action == "path_add" and cell is not None:
            self.add_path_point(cell[0], self.rows - 1 - cell[1])
        elif action == "path_pop":
            self.pop_path_point()
        elif action == "path_clear" and 0 <= self.sel_platform < len(self.platforms):
            if self.platforms[self.sel_platform].get("path"):
                self._checkpoint()
                self.platforms[self.sel_platform].pop("path", None)
                self.dirty = True
                self.status = "path cleared"
        elif action == "pipe_link_cursor" and cell is not None:
            hover = self.cell_at(pygame.mouse.get_pos())
            if hover is not None:
                self.link_pipe(cell, hover)

    # -- editing ---------------------------------------------------------- #
    def paint_cell(self, col: int, disp_row: int, code: int) -> None:
        if not (0 <= col < self.width and 0 <= disp_row < self.rows):
            return
        src_row = self.rows - 1 - disp_row
        column = self.raw[col]
        while len(column) < self.rows:
            column.append(SPACE)
        if column[src_row] == code:
            return
        if self._stroke_pending:
            # one undo step per drag stroke, taken at its first real change
            self._checkpoint()
            self._stroke_pending = False
        column[src_row] = code
        self.dirty = True
        self.need_rebuild = True

    def cell_code(self, col: int, disp_row: int) -> int:
        if not (0 <= col < self.width and 0 <= disp_row < self.rows):
            return SPACE
        src_row = self.rows - 1 - disp_row
        column = self.raw[col]
        return column[src_row] if src_row < len(column) else SPACE

    def add_platform(self, col: int, src_row: int) -> None:
        self._checkpoint()
        if "platforms" not in self.doc.level:
            self.doc.level["platforms"] = self.platforms
        spec = {"x": col, "y": src_row, "len": 3, "type": "patrol",
                "speed": 0.8, "range": 80}
        self.platforms.append(spec)
        self.sel_platform = len(self.platforms) - 1
        self.dirty = True
        self.status = f"added platform #{self.sel_platform} at {col},{src_row}"

    def edit_platform(self, key: int) -> None:
        if not (0 <= self.sel_platform < len(self.platforms)):
            self.status = "no platform selected (PLATFORM tab / P to add)"
            return
        spec = self.platforms[self.sel_platform]
        self._checkpoint()
        order = ("patrol", "osc", "fall", "path")
        cur = str(spec.get("type", "patrol")).lower()
        if cur not in order:
            cur = "patrol"
        if key == pygame.K_t:
            spec["type"] = order[(order.index(cur) + 1) % len(order)]
        elif key == pygame.K_DELETE or key == pygame.K_BACKSPACE:
            self.platforms.pop(self.sel_platform)
            self.sel_platform = min(self.sel_platform,
                                    len(self.platforms) - 1)
            self.dirty = True
            self.status = "platform deleted"
            return
        elif key in (pygame.K_MINUS, pygame.K_EQUALS):
            speed = float(spec.get("speed", 0.8))
            speed = max(0.1, min(4.0, speed + (0.1 if key == pygame.K_EQUALS
                                               else -0.1)))
            spec["speed"] = round(speed, 2)
        elif key in (pygame.K_COMMA, pygame.K_PERIOD):
            n = int(spec.get("len", 3))
            n = n + (1 if key == pygame.K_PERIOD else -1)
            spec["len"] = max(1, min(n, self.width - int(spec.get("x", 0))))
        elif key in (pygame.K_SEMICOLON, pygame.K_QUOTE):
            r = int(spec.get("range", 80))
            r = r + (20 if key == pygame.K_QUOTE else -20)
            spec["range"] = max(0, min(400, r))
        elif key == pygame.K_a:
            spec["axis"] = "v" if str(spec.get("axis", "h")) == "h" else "h"
        elif key == pygame.K_v:
            spec["variant"] = 0 if int(spec.get("variant", 0)) else 1
        self.dirty = True
        self.status = (f"platform #{self.sel_platform}: {spec.get('type')} "
                       f"L{spec.get('len')} spd {spec.get('speed')} "
                       f"rng {spec.get('range', '-')} ax {spec.get('axis', 'h')}")

    def save(self) -> None:
        assert self.doc is not None
        self.doc.level["map"] = self.raw
        self.doc.level["width"] = self.width
        if self.rows != NV:
            for column in self.raw:
                while len(column) < self.rows:
                    column.append(SPACE)
            self.doc.level["rows"] = self.rows
        else:
            self.doc.level.pop("rows", None)
        if self.enemy_dirs:
            self.doc.level["enemy_dirs"] = [
                {"x": x, "y": y, "dir": d}
                for (x, y), d in sorted(self.enemy_dirs.items())]
        else:
            self.doc.level.pop("enemy_dirs", None)
        kwargs = {"ensure_ascii": False}
        if self.doc.compact:
            kwargs["separators"] = (",", ":")
        else:
            kwargs["indent"] = 1
        text = json.dumps(self.doc.container, **kwargs)
        if not self.doc.compact:
            text += "\n"
        tmp = self.doc.path.with_suffix(self.doc.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.doc.path)
        self.dirty = False
        self._saved_snap = self._snapshot()
        self.status = f"saved {self.doc.path.name}"

    def launch(self) -> None:
        self.save()
        if self.launched is not None and self.launched.poll() is None:
            self.status = f"still running: {self.name}"
            return
        cmd = [sys.executable, "-m", "mario_py.game.mario",
               "--level", self.name, "--big"]
        self.launched = subprocess.Popen(cmd, cwd=str(ROOT))
        self.status = f"launched {self.name}"

    def switch(self, delta: int) -> None:
        if self.dirty:
            self.status = "UNSAVED - press Ctrl+S first"
            return
        names = list(self.docs)
        i = (names.index(self.name) + delta) % len(names)
        self.load_level(names[i])

    # -- camera ----------------------------------------------------------- #
    def view_size(self) -> tuple[int, int]:
        h = WINDOW[1] - TOP_BAR
        return WINDOW[0] - PANEL_W, h

    def clamp_cam(self) -> None:
        vw, vh = self.view_size()
        max_x = max(0, self.width * TW - vw)
        max_y = max(0, self.rows * TH - vh)
        self.cam_x = max(0, min(self.cam_x, max_x))
        self.cam_y = max(0, min(self.cam_y, max_y))

    def cell_at(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        vw, vh = self.view_size()
        x, y = pos
        if x >= vw or y < TOP_BAR or y >= TOP_BAR + vh:
            return None
        col = (x + self.cam_x) // TW
        row = (y - TOP_BAR + self.cam_y) // TH
        if 0 <= col < self.width and 0 <= row < self.rows:
            return col, row
        return None

    # -- drawing ---------------------------------------------------------- #
    def draw_grid(self, view: pygame.Rect) -> None:
        vw, vh = self.view_size()
        for col in range(self.cam_x // TW, (self.cam_x + vw) // TW + 2):
            x = col * TW - self.cam_x
            pygame.draw.line(self.screen, (40, 44, 52),
                             (x, TOP_BAR), (x, TOP_BAR + vh))
        for row in range(self.cam_y // TH, (self.cam_y + vh) // TH + 2):
            y = row * TH - self.cam_y + TOP_BAR
            pygame.draw.line(self.screen, (40, 44, 52),
                             (0, y), (vw, y))

    def draw_markers(self) -> None:
        """Markers drawn with the art the game will spawn; badges as fallback."""
        vw, vh = self.view_size()
        self.ensure_markers()
        maps = self.marker_map
        for row in range(self.cam_y // TH, (self.cam_y + vh) // TH + 2):
            for col in range(self.cam_x // TW, (self.cam_x + vw) // TW + 2):
                if not (0 <= col < self.width and 0 <= row < self.rows):
                    continue
                code = self.cell_code(col, row)
                label = SHORT.get(code)
                if label is None:
                    continue
                x = col * TW - self.cam_x
                y = row * TH - self.cam_y + TOP_BAR
                img = maps.get(code)
                if img is not None:
                    # the game draws the enemy at (map_x*W, map_y*H), so the
                    # sprite's top-left is the cell's top-left
                    self.screen.blit(img, (x, y))
                    pygame.draw.rect(self.screen, (150, 70, 170),
                                     (x, y, TW, TH), 1)
                    continue
                colour = ((200, 60, 60) if 0x80 <= code <= 0xB1
                          else (200, 60, 200))
                pygame.draw.rect(self.screen, colour,
                                 (x + 2, y + 2, TW - 4, TH - 4), 1)
                txt = self.small.render(label, True, (255, 255, 255))
                self.screen.blit(txt, (x + TW // 2 - txt.get_width() // 2,
                                       y + TH // 2 - txt.get_height() // 2))

    def draw_platforms(self) -> None:
        """Boards plus their JSON paths; platform data lives in ``platforms``."""
        vw, vh = self.view_size()
        h = E.PLATFORM_SEG_H * SCALE
        for i, spec in enumerate(self.platforms):
            try:
                col = int(spec.get("x", 0))
                src = int(spec.get("y", 0))
                n = max(1, int(spec.get("len", 3)))
            except (TypeError, ValueError):
                continue
            row = self.rows - 1 - src
            x = col * TW - self.cam_x
            y = row * TH - self.cam_y + TOP_BAR
            w = n * TW
            if x + w < 0 or x > vw or y + h < TOP_BAR or y > TOP_BAR + vh:
                continue
            pts = spec.get("path") or []
            if pts:
                pxy = []
                for point in pts:
                    if len(point) >= 2:
                        px = int(point[0]) * TW - self.cam_x
                        py = (self.rows - 1 - int(point[1])) * TH \
                            - self.cam_y + TOP_BAR
                        pxy.append((px, py))
                if len(pxy) >= 2:
                    pygame.draw.lines(self.screen, (255, 180, 60), False,
                                      pxy, 1)
                for px, py in pxy:
                    pygame.draw.circle(self.screen, (255, 180, 60),
                                       (px, py), 3)
            colour = (90, 230, 255) if i == self.sel_platform \
                else (50, 150, 190)
            pygame.draw.rect(self.screen, colour, (x, y, w, h), 2)
            label = (f"{str(spec.get('type', 'patrol')).upper()} "
                     f"{spec.get('speed', '')} L{n}")
            txt = self.small.render(label, True, (255, 255, 255))
            self.screen.blit(txt, (x + 3, y + 2))

    def _row_h(self, entry: tuple) -> int:
        return {"group": 26, "item": 30, "field": 28,
                "action": 28, "info": 22}.get(entry[0], 28)

    def draw_panel(self) -> None:
        x0 = WINDOW[0] - PANEL_W
        pygame.draw.rect(self.screen, (28, 30, 36),
                         (x0, 0, PANEL_W, WINDOW[1]))
        for i, name in enumerate(TAB_LABELS):
            r = pygame.Rect(x0 + 4 + i * 63, 6, 61, 26)
            active = self.tab == TABS[i]
            pygame.draw.rect(self.screen, (70, 76, 90) if active
                             else (44, 48, 58), r, border_radius=4)
            txt = self.font.render(name, True,
                                   (255, 255, 255) if active else (170, 175, 185))
            self.screen.blit(txt, (r.centerx - txt.get_width() // 2,
                                   r.centery - txt.get_height() // 2))

        if self.tab == "palette":
            self.panel_rows = [("group", item[1]) if item[0] == "group"
                               else ("item", item[0], item[1])
                               for item in PALETTE]
        elif self.tab == "levels":
            self.panel_rows = []
            for title, names in self.groups:
                self.panel_rows.append(("group", title))
                for name in names:
                    self.panel_rows.append(("item", name, None))
        elif self.tab == "platforms":
            self.panel_rows = [("group", "P: ADD AT CURSOR")]
            for i, spec in enumerate(self.platforms):
                label = (f"#{i} {str(spec.get('type', 'patrol')).upper()} "
                         f"L{spec.get('len', 3)} @{spec.get('speed', '')} "
                         f"x{spec.get('x', 0)},y{spec.get('y', 0)}")
                self.panel_rows.append(("item", label, i))
        elif self.tab == "options":
            self.panel_rows = self.option_rows()
            self.prop_rows = self.panel_rows
        else:
            self.panel_rows = self.object_rows()
            self.prop_rows = self.panel_rows

        list_y = 40
        list_h = WINDOW[1] - list_y
        if self.scroll_to_current:
            self.scroll_to_current = False
            y_scan = target = 0
            for entry in self.panel_rows:
                if entry[0] == "item" and entry[1] == self.name:
                    target = y_scan
                    break
                y_scan += self._row_h(entry)
            self.panel_scroll = max(0, target - list_h // 2)
        clip = pygame.Rect(x0, list_y, PANEL_W, list_h)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(clip)
        y = list_y - self.panel_scroll
        for idx, entry in enumerate(self.panel_rows):
            h = self._row_h(entry)
            visible = y + h >= list_y and y < list_y + list_h
            if visible:
                kind = entry[0]
                if kind == "group":
                    txt = self.small.render(entry[1], True, (140, 170, 210))
                    self.screen.blit(txt, (x0 + 10, y + 6))
                    pygame.draw.line(self.screen, (60, 66, 78),
                                     (x0 + 8, y + h - 1),
                                     (x0 + PANEL_W - 8, y + h - 1))
                elif kind == "item":
                    label, code = entry[1], entry[2]
                    selected = (self.tab == "palette" and code == self.selected) \
                        or (self.tab == "levels" and label == self.name) \
                        or (self.tab == "platforms" and code == self.sel_platform)
                    if selected:
                        pygame.draw.rect(self.screen, (52, 76, 110),
                                         (x0 + 4, y, PANEL_W - 8, h))
                    if self.tab == "palette":
                        thumb = self.thumb.get(code)
                        if thumb is not None:
                            self.screen.blit(thumb, (x0 + 10, y + 1))
                        else:
                            pygame.draw.rect(self.screen, (90, 45, 45),
                                             (x0 + 10, y + 3, TW - 4, TH - 4))
                            t = self.small.render(SHORT.get(code, "?"), True,
                                                  (255, 255, 255))
                            self.screen.blit(t, (x0 + 10 + (TW - 4) // 2
                                                 - t.get_width() // 2,
                                                 y + 3 + (TH - 4) // 2
                                                 - t.get_height() // 2))
                        txt = self.font.render(label, True, (225, 228, 235))
                        self.screen.blit(txt, (x0 + 14 + TW, y + 7))
                    else:
                        txt = self.font.render(label, True, (225, 228, 235))
                        self.screen.blit(txt, (x0 + 12, y + 7))
                elif kind == "field":
                    _k, key, label, value, changed = entry
                    selected = (self.tab in ("options", "object")
                                and idx == self.prop_index)
                    if selected:
                        pygame.draw.rect(self.screen, (52, 76, 110),
                                         (x0 + 4, y, PANEL_W - 8, h))
                    txt = self.small.render(label, True, (210, 214, 222))
                    self.screen.blit(txt, (x0 + 12, y + 7))
                    col = (255, 220, 90) if changed else (170, 220, 190)
                    val = self.small.render(str(value), True, col)
                    self.screen.blit(val, (x0 + PANEL_W - 14 - val.get_width(),
                                           y + 7))
                elif kind == "action":
                    selected = (self.tab in ("options", "object")
                                and idx == self.prop_index)
                    pygame.draw.rect(self.screen,
                                     (70, 76, 90) if selected else (44, 48, 58),
                                     (x0 + 10, y + 2, PANEL_W - 20, h - 4),
                                     border_radius=4)
                    txt = self.small.render(entry[2], True, (235, 238, 244))
                    self.screen.blit(txt, (x0 + 18, y + 7))
                elif kind == "info":
                    txt = self.small.render(entry[1], True, (160, 166, 178))
                    self.screen.blit(txt, (x0 + 20, y + 4))
            y += h
        self.screen.set_clip(old_clip)
        content = y + self.panel_scroll - list_y
        max_scroll = max(0, content - list_h)
        self.panel_scroll = max(0, min(self.panel_scroll, max_scroll))

    def draw_pipe_links(self) -> None:
        """Draw warp-pipe links (alpha mouths) as lines with their link key."""
        selected = self.pipe_pair_at(*self.sel_cell) if self.sel_cell else None
        for lc, sr, c1, c2 in self.pipe_mouths():
            if c1 != 0xE0:
                continue
            _kind, dest = self.pipe_target(c1, c2, (lc, sr))
            if dest is None:
                continue
            x1 = lc * TW + TW - self.cam_x
            y1 = (self.rows - 1 - sr) * TH + TH // 2 - self.cam_y + TOP_BAR
            x2 = dest[0] * TW + TW - self.cam_x
            y2 = (self.rows - 1 - dest[1]) * TH + TH // 2 \
                - self.cam_y + TOP_BAR
            hot = selected is not None and selected[:2] == (lc, sr)
            pygame.draw.line(self.screen,
                             (255, 190, 70) if hot else (80, 130, 100),
                             (x1, y1), (x2, y2), 2 if hot else 1)
            txt = self.small.render(f"k{c2 - 0xE0}", True,
                                    (255, 210, 120) if hot
                                    else (140, 190, 160))
            self.screen.blit(txt, ((x1 + x2) // 2, (y1 + y2) // 2))

    def draw(self) -> None:
        self.screen.fill((18, 20, 24))
        vw, vh = self.view_size()
        if self.scaled is not None:
            self.screen.blit(self.scaled, (0, TOP_BAR),
                             pygame.Rect(self.cam_x, self.cam_y, vw, vh))
        if self.grid_on:
            self.draw_grid(pygame.Rect(0, TOP_BAR, vw, vh))
        m = pygame.mouse.get_pos()
        cell = self.cell_at(m)
        self.draw_markers()
        self.draw_platforms()
        self.draw_pipe_links()
        if self.sel_cell is not None:
            sc, sr = self.sel_cell
            pygame.draw.rect(self.screen, (90, 230, 140),
                             (sc * TW - self.cam_x,
                              sr * TH - self.cam_y + TOP_BAR, TW, TH), 2)
        if cell is not None:
            col, row = cell
            pygame.draw.rect(self.screen, (255, 220, 80),
                             (col * TW - self.cam_x,
                              row * TH - self.cam_y + TOP_BAR,
                              TW, TH), 2)
            code = self.cell_code(col, row)
            info = (f"{self.name}  [{self.doc.path.name}]  "
                    f"col {col}  src row {self.rows - 1 - row}  "
                    f"code {code} (0x{code:02X})  "
                    f"pick {self.selected} (0x{self.selected:02X})")
        else:
            info = f"{self.name}  [{self.doc.path.name}]  " \
                   f"pick {self.selected} (0x{self.selected:02X})"
        if self.tab in ("options", "object"):
            info += ("  | Up/Down select, Left/Right change (Shift x10), "
                     "Enter edit")
        if self.dirty:
            info += "  *UNSAVED*"
            if self._undo or self._redo:
                info += "  | Ctrl+Z undo, Ctrl+Y redo"
        if self.text_edit is not None:
            info = (f"HINT: {self.text_edit['buf']}_   "
                    f"(Enter saves, Esc cancels)")
        if 0 <= self.sel_platform < len(self.platforms):
            spec = self.platforms[self.sel_platform]
            info += (f"  | PLAT#{self.sel_platform} "
                     f"{str(spec.get('type', 'patrol')).upper()} "
                     f"L{spec.get('len', 3)} spd {spec.get('speed', '')} "
                     f"rng {spec.get('range', '-')}")
        if self.status:
            info += f"   | {self.status}"
        txt = self.font.render(info, True, (230, 232, 238))
        self.screen.blit(txt, (8, 8))
        self.draw_panel()

    # -- events ----------------------------------------------------------- #
    def move_prop(self, delta: int) -> None:
        if not self.prop_rows:
            return
        n = len(self.prop_rows)
        i = self.prop_index
        for _ in range(n):
            i = (i + delta) % n
            if self.prop_rows[i][0] in ("field", "action", "info"):
                break
        self.prop_index = i
        list_h = WINDOW[1] - 40
        y = sum(self._row_h(e) for e in self.prop_rows[:i])
        h = self._row_h(self.prop_rows[i])
        if y < self.panel_scroll:
            self.panel_scroll = y
        elif y + h > self.panel_scroll + list_h:
            self.panel_scroll = y + h - list_h

    def activate_prop(self) -> None:
        if not (0 <= self.prop_index < len(self.prop_rows)):
            return
        entry = self.prop_rows[self.prop_index]
        if entry[0] == "action":
            self.run_action(entry[1])
        elif entry[0] == "field":
            if entry[1] == "hint":
                self.text_edit = {"key": "hint",
                                  "buf": str(self._field_value("hint"))}
                self.status = "type the hint, Enter saves, Esc cancels"
            elif self.tab == "options":
                self.adjust_option(entry[1], 1)
            else:
                self.adjust_prop(1)

    def handle(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and self.text_edit is not None:
            if event.key == pygame.K_RETURN:
                self.set_option(self.text_edit["key"], self.text_edit["buf"])
                self.text_edit = None
            elif event.key == pygame.K_ESCAPE:
                self.text_edit = None
                self.status = "hint edit cancelled"
            elif event.key == pygame.K_BACKSPACE:
                self.text_edit["buf"] = self.text_edit["buf"][:-1]
            elif event.unicode and event.unicode.isprintable():
                self.text_edit["buf"] += event.unicode
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.dirty and not self.esc_armed:
                    self.esc_armed = True
                    self.status = "UNSAVED - Esc again to quit, Ctrl+S to save"
                else:
                    return False
                return True
            self.esc_armed = False
            if event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
                if event.mod & pygame.KMOD_SHIFT:
                    self.redo()
                else:
                    self.undo()
            elif event.key == pygame.K_y and event.mod & pygame.KMOD_CTRL:
                self.redo()
            elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                self.save()
            elif event.key == pygame.K_F5:
                self.launch()
            elif event.key == pygame.K_TAB:
                self.tab = TABS[(TABS.index(self.tab) + 1) % len(TABS)]
                self.prop_index = 0
                self.panel_scroll = 0
                self.scroll_to_current = self.tab == "levels"
            elif self.tab in ("options", "object") and event.key in (
                    pygame.K_UP, pygame.K_DOWN):
                self.move_prop(-1 if event.key == pygame.K_UP else 1)
            elif self.tab in ("options", "object") and event.key in (
                    pygame.K_LEFT, pygame.K_RIGHT, pygame.K_MINUS,
                    pygame.K_EQUALS):
                delta = -1 if event.key in (pygame.K_LEFT, pygame.K_MINUS) \
                    else 1
                if 0 <= self.prop_index < len(self.prop_rows):
                    entry = self.prop_rows[self.prop_index]
                    if entry[0] == "field":
                        if self.tab == "options":
                            mult = 10 if event.mod & pygame.KMOD_SHIFT else 1
                            self.adjust_option(entry[1], delta * mult)
                        else:
                            self.adjust_prop(delta)
            elif self.tab in ("options", "object") and event.key in (
                    pygame.K_RETURN, pygame.K_SPACE):
                self.activate_prop()
            elif event.key == pygame.K_l and self.tab == "object":
                if self.sel_cell is not None:
                    hover = self.cell_at(pygame.mouse.get_pos())
                    if hover is not None:
                        self.link_pipe(self.sel_cell, hover)
            elif event.key == pygame.K_p:
                cell = self.cell_at(pygame.mouse.get_pos())
                if cell is not None:
                    self.add_platform(cell[0], self.rows - 1 - cell[1])
                else:
                    self.status = "P: put the cursor on the grid first"
            elif self.tab == "platforms" and event.key in (
                    pygame.K_t, pygame.K_MINUS, pygame.K_EQUALS,
                    pygame.K_COMMA, pygame.K_PERIOD,
                    pygame.K_SEMICOLON, pygame.K_QUOTE,
                    pygame.K_a, pygame.K_v,
                    pygame.K_DELETE, pygame.K_BACKSPACE):
                self.edit_platform(event.key)
            elif event.key == pygame.K_g:
                self.grid_on = not self.grid_on
            elif event.key == pygame.K_HOME:
                self.cam_x = self.cam_y = 0
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_PAGEUP):
                self.switch(-1)
            elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PAGEDOWN):
                self.switch(1)
        if event.type == pygame.MOUSEWHEEL:
            m = pygame.mouse.get_pos()
            if m[0] >= WINDOW[0] - PANEL_W:
                self.panel_scroll -= event.y * 40
                return True
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.cam_x -= event.y * TW * 3
            else:
                self.cam_y -= event.y * TH * 3
            self.clamp_cam()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:
                self.panning = True
            elif event.button == 1:
                m = event.pos
                if m[0] >= WINDOW[0] - PANEL_W:
                    self.panel_click(m)
                else:
                    self.painting = True
                    self._stroke_pending = True
                    cell = self.cell_at(m)
                    if cell is not None:
                        self.sel_cell = cell
                        self.paint_cell(cell[0], cell[1], self.selected)
            elif event.button == 3:
                cell = self.cell_at(event.pos)
                if cell is not None:
                    self.sel_cell = cell
                    self.tab = "object"
                    self.prop_index = 0
                    plat = self.platform_at(*cell)
                    if plat is not None:
                        self.sel_platform = plat
                    self.status = "object properties (OPT tab = level options)"
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.painting = False
                self._stroke_pending = False
            elif event.button == 2:
                self.panning = False
        if event.type == pygame.MOUSEMOTION:
            if self.panning:
                self.cam_x -= event.rel[0]
                self.cam_y -= event.rel[1]
                self.clamp_cam()
            elif self.painting:
                cell = self.cell_at(event.pos)
                if cell is not None:
                    self.paint_cell(cell[0], cell[1], self.selected)
        return True

    def panel_click(self, pos: tuple[int, int]) -> None:
        x0 = WINDOW[0] - PANEL_W
        y = 40 - self.panel_scroll
        for idx, entry in enumerate(self.panel_rows):
            h = self._row_h(entry)
            if y <= pos[1] < y + h:
                if entry[0] == "item":
                    if self.tab == "palette":
                        self.selected = entry[2]
                    elif self.tab == "platforms":
                        self.sel_platform = int(entry[2])
                        self.status = f"platform #{self.sel_platform}"
                    elif self.tab == "levels":
                        if self.dirty:
                            self.status = "UNSAVED - press Ctrl+S first"
                        else:
                            self.load_level(entry[1])
                elif self.tab in ("options", "object"):
                    self.prop_index = idx
                    if entry[0] == "action":
                        self.run_action(entry[1])
                    elif entry[0] == "field" and entry[1] == "hint":
                        self.text_edit = {"key": "hint",
                                          "buf": str(self._field_value("hint"))}
                        self.status = "type the hint, Enter saves, Esc cancels"
                return
            y += h

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if not self.handle(event):
                    running = False
            if self.need_rebuild:
                self.rebuild()
                self.need_rebuild = False
            self.draw()
            pygame.display.flip()
            clock.tick(60)

    # -- selftest --------------------------------------------------------- #
    def selftest(self) -> int:
        """Load -> paint -> save -> reload, on a throwaway copy of testrooms."""
        tmpdir = Path(tempfile.mkdtemp(prefix="mario_editor_"))
        try:
            copy = tmpdir / "levels_test.json"
            shutil.copyfile(DATA / "levels_test.json", copy)
            docs, groups = _load_docs([copy])
            name = "testrooms/1-1a"
            if name not in docs:
                name = next(iter(docs))
            ed = Editor(self.assets, [copy], initial=name, headless=True)
            ed.load_level(name)
            col, row = 4, ed.rows - 2       # source row 1: empty in 1-1a
            old = ed.cell_code(col, row)
            if old != SPACE:
                col += 3
                old = ed.cell_code(col, row)
            ed.selected = ord("X")
            ed.paint_cell(col, row, ord("X"))
            ed.add_platform(4, 2)
            ed.sel_platform = len(ed.platforms) - 1
            ed.edit_platform(pygame.K_PERIOD)     # len 3 -> 4
            ed.edit_platform(pygame.K_EQUALS)     # speed 0.8 -> 0.9
            ed.edit_platform(pygame.K_t)          # patrol -> osc
            ed.save()
            data = json.loads(copy.read_text(encoding="utf-8"))
            src = ed.rows - 1 - row
            got = data["levels"][name]["map"][col][src]
            ok = (got == ord("X") and not ed.dirty)
            print(f"selftest paint/save: {ok} (old {old} -> {got})")
            if not ok:
                return 1
            plats = data["levels"][name].get("platforms") or []
            pok = (len(plats) == 1 and plats[0].get("type") == "osc"
                   and plats[0].get("len") == 4
                   and abs(float(plats[0].get("speed", 0)) - 0.9) < 1e-6)
            print(f"selftest platform add/edit/save: {pok} {plats[:1]}")
            if not pok:
                return 1
            ed2 = Editor(self.assets, [copy], initial=name, headless=True)
            if ed2.cell_code(col, row) != ord("X"):
                print("selftest reload: FAILED")
                return 1
            if len(ed2.platforms) != 1 or ed2.platforms[0].get("type") != "osc":
                print("selftest platform reload: FAILED")
                return 1
            if ed2.scaled is None or ed2.scaled.get_width() != ed2.width * TW:
                print("selftest render surface: FAILED")
                return 1
            ed2.launch = lambda: None          # never start the game here
            print("selftest reload/render: OK")

            # --- options / pipes / enemy dirs / platform paths ---------- #
            ed3 = Editor(self.assets, [copy], initial="testrooms/2-1a",
                         headless=True)
            marker = None
            for c in range(ed3.width):
                for r in range(ed3.rows):
                    if ed3.cell_code(c, r) in DIR_CODES:
                        marker = (c, r)
                        break
                if marker is not None:
                    break
            if marker is None:
                print("selftest enemy dir: FAILED (no marker in 2-1a)")
                return 1
            ed3.cycle_enemy_dir(marker[0], ed3.rows - 1 - marker[1], 1)
            ed3.save()

            ed3.load_level("testrooms/4-1a")
            ed3.set_option("SkyType", 9)
            ed3.set_option("hint", "SELFTEST HINT")
            mouths = ed3.pipe_mouths()
            if not mouths:
                print("selftest pipe edit: FAILED (no pipe mouth)")
                return 1
            lc, sr, _c1, _c2 = mouths[0]
            ed3.set_pipe_pair(lc, sr, 0xE0, 0xE0 + 7)
            ok_pipe = (ed3.raw[lc][sr] == 0xE0 and ed3.raw[lc + 1][sr] == 0xE7)
            # a synthetic destination pipe, then point the exit pipe at it
            ed3._write_src(20, 1, ord("0"))
            ed3._write_src(21, 1, ord("1"))
            ed3._write_src(20, 2, 0xE8)
            ed3._write_src(21, 2, 0xEA)
            src_cell = (lc, ed3.rows - 1 - sr)
            dst_cell = (20, ed3.rows - 1 - 2)
            ok_link = ed3.link_pipe(src_cell, dst_cell)
            sp = ed3.pipe_pair_at(*src_cell)
            dp = ed3.pipe_pair_at(*dst_cell)
            ok_link = bool(ok_link and sp and dp and sp[2] == 0xE0
                           and sp[3] == dp[3]
                           and ed3.pipe_target(sp[2], sp[3],
                                               (sp[0], sp[1]))[1]
                           == (dp[0], dp[1]))
            ed3.add_platform(4, 2)
            ed3.sel_platform = len(ed3.platforms) - 1
            ed3.add_path_point(6, 3)
            ed3.add_path_point(8, 3)
            ed3.save()

            data = json.loads(copy.read_text(encoding="utf-8"))
            lv = data["levels"]["testrooms/4-1a"]
            ok_opt = (lv["options"].get("SkyType") == 9
                      and lv.get("hint") == "SELFTEST HINT")
            ok_pipe_saved = (lv["map"][lc][sr] == 0xE0
                             and lv["map"][lc + 1][sr] == sp[3])
            plats = lv.get("platforms") or []
            ok_path = (bool(plats)
                       and plats[-1].get("path") == [[6, 3], [8, 3]])
            ed4 = Editor(self.assets, [copy], initial="testrooms/2-1a",
                         headless=True)
            ok_dir = ed4.enemy_dirs.get(
                (marker[0], ed4.rows - 1 - marker[1])) == 1
            print(f"selftest options/pipe/link/path: "
                  f"{ok_opt}/{ok_pipe and ok_pipe_saved}/{ok_link}/{ok_path}")
            print(f"selftest enemy dirs: {ok_dir}")
            if not (ok_opt and ok_pipe and ok_pipe_saved and ok_link
                    and ok_path and ok_dir):
                return 1

            # --- level size + marker preview art ----------------------- #
            ed5 = Editor(self.assets, [copy], initial=name, headless=True)
            w0, r0 = ed5.width, ed5.rows
            y0 = ed5.options.InitY
            ed5.set_width(w0 + 3)
            ok_w = (ed5.width == w0 + 3 and len(ed5.raw) == w0 + 3)
            ed5.set_rows(r0 + 2)
            ok_h = (ed5.rows == r0 + 2
                    and ed5.options.InitY == y0 + 2 * TILE_H)
            ed5.sel_cell = (3, 9)
            ed5.run_action("start")
            ok_start = ed5.options.InitY == 9 * TILE_H
            imgs = ed5.ensure_markers()
            ok_art = (len(imgs) >= 30
                      and imgs.get(E.CH_PLANT0) is not None
                      and imgs[E.CH_PLANT0].get_height() >= 16
                      and imgs.get(E.CH_THWOMP) is not None)
            ed5.save()
            data = json.loads(copy.read_text(encoding="utf-8"))
            lv = data["levels"][name]
            ok_saved = (lv.get("width") == w0 + 3
                        and lv.get("rows") == r0 + 2)
            ed6 = Editor(self.assets, [copy], initial=name, headless=True)
            ok_reload = (ed6.width == w0 + 3 and ed6.rows == r0 + 2)
            ed6.paint_cell(2, 0, ord("X"))
            ed6.set_rows(ed6.rows - 1)
            ok_refuse_rows = ed6.rows == r0 + 2
            ed6.paint_cell(ed6.width - 1, 6, ord("X"))
            ed6.set_width(ed6.width - 1)
            ok_refuse_w = ed6.width == w0 + 3
            print(f"selftest size: w={ok_w} rows={ok_h} start={ok_start} "
                  f"saved={ok_saved} reload={ok_reload} "
                  f"refuse={ok_refuse_rows}/{ok_refuse_w} "
                  f"art={len(imgs)}")
            if not (ok_w and ok_h and ok_start and ok_saved and ok_reload
                    and ok_refuse_rows and ok_refuse_w and ok_art):
                return 1

            # --- undo / redo ------------------------------------------- #
            ed7 = Editor(self.assets, [copy], initial=name, headless=True)
            col, row = 5, ed7.rows - 2
            old = ed7.cell_code(col, row)
            brush = ord("X") if old != ord("X") else ord("I")
            ed7._stroke_pending = True          # what MOUSEBUTTONDOWN does
            ed7.paint_cell(col, row, brush)
            ed7._stroke_pending = False
            ed7.undo()
            ok_u_paint = ed7.cell_code(col, row) == old
            ed7.redo()
            ok_r_paint = ed7.cell_code(col, row) == brush
            sky = ed7.options.SkyType
            ed7.set_option("SkyType", 9 if sky != 9 else 8)
            ed7.undo()
            ok_u_opt = ed7.options.SkyType == sky
            w1 = ed7.width
            ed7.set_width(w1 + 2)
            ed7.undo()
            ok_u_width = ed7.width == w1
            ed7.redo()
            ok_r_width = ed7.width == w1 + 2
            ed8 = Editor(self.assets, [copy], initial=name, headless=True)
            ok_reset = not ed8._undo and not ed8._redo
            print(f"selftest undo/redo: paint={ok_u_paint}/{ok_r_paint} "
                  f"opt={ok_u_opt} width={ok_u_width}/{ok_r_width} "
                  f"reset={ok_reset}")
            if not (ok_u_paint and ok_r_paint and ok_u_opt and ok_u_width
                    and ok_r_width and ok_reset):
                return 1
            print(f"docs: {len(docs)} levels in {len(groups)} groups")
            return 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", help="level key to open, e.g. testrooms/2-4a")
    ap.add_argument("--file", action="append", default=[],
                    help="only read these level JSON files (repeatable)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    pygame.init()
    try:
        assets = load_assets()
        files = [Path(p) for p in args.file] if args.file else LEVEL_FILES
        if args.selftest:
            return Editor(assets, files, initial=args.level,
                          headless=True).selftest()
        Editor(assets, files, initial=args.level).run()
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
