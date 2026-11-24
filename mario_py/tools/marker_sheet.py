"""Render the editor's enemy-marker preview sheet from the game code itself.

Every map marker in the editor is spawned by ``Enemies.start_enemies`` into a
throwaway world and painted by ``Enemies._draw_enemy``.  The frame with the
largest visible rectangle is cropped, so a plant is shown fully out of its
pipe and a flyer with its wings open -- no per-enemy art table that could go
stale when a sprite or a draw branch changes.

The result is cached in ``data/cache/marker_sheet.png`` + ``.json``; the key
covers the source files and the palette entries the art reads, so any art
change rebuilds it once on the next editor start.

    py -3.12 -m mario_py.tools.marker_sheet [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mario_py" / "data"
CACHE_DIR = DATA / "cache"
SHEET_PNG = CACHE_DIR / "marker_sheet.png"
SHEET_JSON = CACHE_DIR / "marker_sheet.json"

# Anything below can change what a marker looks like.
SOURCE_GLOBS = (
    "mario_py/tools/marker_sheet.py",
    "mario_py/game/enemies.py",
    "mario_py/game/figures.py",
    "mario_py/game/assets.py",
    "mario_py/game/buffers.py",
    "mario_py/game/backgr.py",
    "mario_py/data/palettes/palette.json",
    "mario_py/data/sprites/index.json",
    "mario_py/data/newenemies/idx/*.npy",
    "mario_py/data/refs/wing/*.png",
)

PREVIEW_ROWS = 15
PREVIEW_COLS = 8
PREVIEW_COL = 3
PREVIEW_ROW = 12          # world row of the marker (floor is rows 13..14)
CANVAS = (220, 220)

PLANT_MARKERS = {0x84, 0x85, 0x86, 0x9C, 0x9D, 0x9E, 0x9F, 0xA0}


def _source_key(codes: list[int], options) -> str:
    h = hashlib.sha1()
    files: list[Path] = []
    for pattern in SOURCE_GLOBS:
        p = ROOT / pattern
        files.extend(sorted(p.parent.glob(p.name)) if "*" in pattern
                     else [p])
    for p in files:
        if not p.exists():
            continue
        st = p.stat()
        h.update(p.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    for name in ("C2r", "C2g", "C2b", "C3r", "C3g", "C3b"):
        h.update(name.encode())
        h.update(str(int(getattr(options, name, 0))).encode())
    h.update(",".join(str(c) for c in sorted(codes)).encode())
    return h.hexdigest()


def _preview_world(options):
    from mario_py.game.buffers import WorldMap

    cols = []
    for _ in range(PREVIEW_COLS):
        col = [ord(" ")] * PREVIEW_ROWS
        col[0] = col[1] = ord("A")       # a floor to land on
        cols.append(col)
    world = WorldMap()
    world.load_columns(cols, options, rows=PREVIEW_ROWS)
    return world


def _spawn_preview(assets, figures, options, code: int):
    """Spawn the object ``code`` stands for; None if this is not a creature."""
    from dataclasses import replace

    from mario_py.game import enemies as E

    # Work on a copy: load_columns() writes XSize/YSize, and the caller's
    # WorldOptions belongs to the level being edited.
    opts = replace(options)
    world = _preview_world(opts)
    en = E.Enemies(figures, assets, world, opts)
    en.view_x = en.view_y = 0
    en.player_x1, en.player_y1 = -1000, 0
    en.player_x2, en.player_y2 = -986, 14
    en.player_xvel = 0
    en.dir_overrides = {}
    if code == E.CH_LAKITU:
        en.new_enemy(E.TP_LAKITU, 0, PREVIEW_COL, PREVIEW_ROW, 0, 0, 0)
        return en, 1                    # the cloud must not fly/tick away
    if code == E.CH_FLY_FISH:
        en.new_enemy(E.TP_FLY_FISH, 0, PREVIEW_COL, PREVIEW_ROW, 1, 0, 0)
        return en, 1
    world.set(PREVIEW_COL, PREVIEW_ROW, code)
    en.start_enemies(PREVIEW_COL, 1)
    if code in (E.CH_VERT_FISH, E.CH_VERT_FIREBALL):
        frames = 240          # they wait up to 150 frames before jumping
    elif code in PLANT_MARKERS:
        frames = 150
    else:
        frames = 70
    return en, frames


def _render(assets, figures, options, code: int):
    spawned = _spawn_preview(assets, figures, options, code)
    if spawned is None:
        return None
    en, frames = spawned
    if not en.active:
        return None
    # Only what the marker itself stands for: a Hammer Bro's thrown hammer or
    # a fire plant's fireball would otherwise blow up the icon's bounds.
    base = list(en.active)
    best = None
    best_area = 0
    for _ in range(frames):
        canvas = pygame.Surface(CANVAS, pygame.SRCALPHA)
        for idx in base:
            if idx in en.active:
                en._draw_enemy(canvas, en.pool[idx])
        rect = canvas.get_bounding_rect(min_alpha=1)
        if rect.w and rect.h and rect.w * rect.h > best_area:
            best_area = rect.w * rect.h
            best = canvas.subsurface(rect).copy()
        en.move_enemies()
    return best


def build(assets, figures, options, codes: list[int]) -> dict[int, pygame.Surface]:
    items: dict[int, pygame.Surface] = {}
    for code in codes:
        try:
            surf = _render(assets, figures, options, code)
        except (KeyError, FileNotFoundError):
            surf = None
        if surf is not None:
            items[code] = surf
    return items


def _write(items: dict[int, pygame.Surface], key: str) -> None:
    pad = 4
    width = 512
    placed: dict[int, list[int]] = {}
    x = y = pad
    row_h = 0
    for code in sorted(items):
        w, h = items[code].get_size()
        if x + w + pad > width:
            x = pad
            y += row_h + pad
            row_h = 0
        placed[code] = [x, y, w, h]
        x += w + pad
        row_h = max(row_h, h)
    sheet = pygame.Surface((width, y + row_h + pad), pygame.SRCALPHA)
    for code, (sx, sy, _w, _h) in placed.items():
        sheet.blit(items[code], (sx, sy))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pygame.image.save(sheet, str(SHEET_PNG))
    SHEET_JSON.write_text(json.dumps({"key": key, "items": placed},
                                     indent=1), encoding="utf-8")


def _read(key: str) -> dict[int, pygame.Surface] | None:
    if not (SHEET_PNG.exists() and SHEET_JSON.exists()):
        return None
    try:
        meta = json.loads(SHEET_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("key") != key:
        return None
    try:
        sheet = pygame.image.load(str(SHEET_PNG))
    except (pygame.error, OSError):
        return None
    out: dict[int, pygame.Surface] = {}
    for code, (x, y, w, h) in meta.get("items", {}).items():
        out[int(code)] = sheet.subsurface(
            pygame.Rect(x, y, w, h)).copy()
    return out


def get_marker_sheet(assets, figures, options, codes: list[int],
                     force: bool = False) -> dict[int, pygame.Surface]:
    """Marker code -> cropped 1x sprite; rebuilds the cache when stale."""
    key = _source_key(codes, options)
    if not force:
        cached = _read(key)
        if cached is not None:
            return cached
    items = build(assets, figures, options, codes)
    try:
        _write(items, key)
    except OSError:
        pass            # read-only cache dir: rebuild on every start
    return items


def main(argv: list[str] | None = None) -> int:
    from mario_py.game.assets import load_assets
    from mario_py.game.backgr import BackGr
    from mario_py.game.buffers import WorldOptions
    from mario_py.game.figures import Figures
    from mario_py.tools.level_editor import PALETTE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    pygame.init()
    try:
        assets = load_assets()
        figures = Figures(assets)
        figures.backgr = BackGr(figures, assets)
        options = WorldOptions()
        codes = [code for kind, *rest in PALETTE
                 if kind != "group" for code in rest]
        start = time.perf_counter()
        items = get_marker_sheet(assets, figures, options, codes,
                                 force=args.force)
        print(f"{len(items)}/{len(codes)} markers in "
              f"{time.perf_counter() - start:.2f}s -> {SHEET_PNG}")
        missing = [c for c in codes if c not in items]
        if missing:
            print("no art (fallback badge):",
                  " ".join(f"0x{c:02X}" for c in missing))
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
