"""Shared game data — port of BUFFERS.PAS.

Holds the world map, level options, player/game state, and the constants every
other module refers to. Kept as plain dataclasses + module-level constants so
the ported logic can reference them the same way the Pascal did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TILE_W = 20
TILE_H = 14
NV = 13
NH = 16
MAX_WORLD_SIZE = 236
EX = 1
EY1 = 8
EY2 = 3

DIR_LEFT, DIR_RIGHT = 0, 1
MD_SMALL, MD_LARGE, MD_FIRE = 0, 1, 2
PL_MARIO, PL_LUIGI = 0, 1

PLAYER_NAME = ("MARIO", "LUIGI")

# collision sets (BUFFERS.PAS) as byte tile codes
CAN_HOLD_YOU = bytes(range(0, 14)) + b"0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CAN_HOLD_YOU = set(CAN_HOLD_YOU)
CAN_STAND_ON = set(bytes(range(14, 17)) + b"abcdef")
HIDDEN = {ord("$")}


@dataclass
class WorldOptions:
    InitX: int = 0
    InitY: int = 0
    SkyType: int = 0
    WallType1: int = 0
    WallType2: int = 0
    WallType3: int = 0
    PipeColor: int = 0
    GroundColor1: int = 0
    GroundColor2: int = 0
    Horizon: int = 0
    BackGrType: int = 0
    BackGrColor1: int = 0
    BackGrColor2: int = 0
    Stars: int = 0
    Clouds: int = 0
    # Port-only: blit the XP moon-night backdrop instead of drawing the sky.
    # The image is screen-fixed, so such levels must be one screen wide.
    MoonSky: int = 0
    # Port-only: ghost-house sight.  The room is painted black except for a
    # Manhattan-radius circle around Mario; coins widen it, it decays back.
    Vision: int = 0
    # Port-only underwater levels.  WaterTop is the world row of the water
    # surface: -1 = no water, 0 = the surface is at the top of the screen (a
    # whole-level dive), N = rows N..YSize-1 are water.  WaterStyle 1 is the
    # user's 91898 water set (three surface + three body frames, random per
    # cell, with the grass/cattail overlay); 2 is the original DOS water palette.
    WaterTop: int = -1
    WaterStyle: int = 1
    # Port-only switch: fill the dry rows above WaterTop with static water
    # tiles (kept for future waterline levels; current test rooms are
    # whole-level water and leave this at 0).
    WaterFillTop: int = 0
    Design: int = 1
    C2r: int = 0
    C2g: int = 0
    C2b: int = 0
    C3r: int = 0
    C3g: int = 0
    C3b: int = 0
    BrickColor: int = 0
    WoodColor: int = 0
    XBlockColor: int = 0
    BuildWall: bool = False
    XSize: int = 0
    YSize: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "WorldOptions":
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    @property
    def water_y(self) -> int:
        """Pixel Y of the water surface, or -1 when the level has no water."""
        return self.WaterTop * TILE_H if self.WaterTop >= 0 else -1

    def in_water_row(self, row: int) -> bool:
        return self.WaterTop >= 0 and row >= self.WaterTop

    def in_water_box(self, y: int, height: int) -> bool:
        """Does a body [y, y+height) reach into the water?"""
        return self.WaterTop >= 0 and y + height > self.water_y


@dataclass
class GameData:
    """What the original persists to MARIO.CFG between sessions."""
    NumPlayers: int = 1
    Progress: list[int] = field(default_factory=lambda: [0, 0])
    Lives: list[int] = field(default_factory=lambda: [5, 5])
    Coins: list[int] = field(default_factory=lambda: [0, 0])
    Score: list[int] = field(default_factory=lambda: [0, 0])
    Mode: list[int] = field(default_factory=lambda: [MD_SMALL, MD_SMALL])
    # Mod level packs, kept apart from the original Progress so a slot can
    # still be loaded by the original game (port-only fields, not in MARIO.CFG).
    ModPack: str = ""
    ModProgress: dict[str, list[int]] = field(default_factory=dict)

    def reset(self) -> None:
        self.Lives = [5, 5]
        self.Coins = [0, 0]
        self.Score = [0, 0]
        self.Progress = [0, 0]
        self.Mode = [MD_SMALL, MD_SMALL]


class WorldMap:
    """WorldMap[-EX .. size+EX-1][-EY1 .. rows+EY2-1] of byte tile codes.

    ``rows`` is the level height in tiles (13 for every original/mod level).
    ``vertical`` is the opt-in switch for the vertical camera; tall custom
    levels enable it unless the level explicitly turns it off.

    Stored as a flat dict keyed by (col, row); out-of-range reads return the
    solid-border tile '@' (as ReadWorld fills the borders).
    """

    def __init__(self, size: int = MAX_WORLD_SIZE, rows: int = NV) -> None:
        self.size = size
        self.rows = rows
        self.vertical = False
        self._cells: dict[tuple[int, int], int] = {}
        self.x_size = 0

    def _default(self, col: int, row: int) -> int:
        # ReadWorld (BUFFERS.PAS): the level occupies columns 0..XSize-1,
        # columns -EX..-1 and XSize..XSize+EX-1 are filled with the solid
        # '@' border.  (Pascal leaves columns past XSize+EX as ' ', but the
        # game never reads out there, so returning the safe solid tile is
        # equivalent -- and it fixes an off-by-one that used to leave the
        # right border column XSize reading as ' ' instead of '@'.)
        if row < -EY1 or row >= self.rows + EY2:
            return ord("@")
        if col < 0 or col >= self.x_size:
            return ord("@")
        return ord(" ")

    def get(self, col: int, row: int) -> int:
        return self._cells.get((col, row), self._default(col, row))

    def set(self, col: int, row: int, code: int) -> None:
        self._cells[(col, row)] = code

    def load_columns(self, cols: list[list[int]],
                     options: "WorldOptions | None" = None,
                     rows: int | None = None,
                     vertical: bool | None = None) -> None:
        """Fill the map from the converted level data (list of columns).

        ``rows`` is the level height; when omitted it is inferred from the
        column length (the original 13).  Every column must fit in ``rows``;
        shorter columns are padded with empty sky.  ``options`` gets
        ``XSize``/``YSize``, exactly as ReadWorld does (``Options.XSize := X``);
        StartEnemies uses XSize as its spawn boundary.
        """
        if rows is None:
            rows = len(cols[0]) if cols else NV
        rows = int(rows)
        if rows < NV:
            raise ValueError(f"level rows {rows} < viewport rows {NV}")
        self._cells.clear()
        self.rows = rows
        self.vertical = (rows > NV) if vertical is None else bool(vertical)
        self.x_size = len(cols)
        if options is not None:
            options.XSize = self.x_size
            options.YSize = rows
        for x, col in enumerate(cols):
            if len(col) > rows:
                raise ValueError(
                    f"column {x} has {len(col)} rows, level rows={rows}")
            padded = list(col) + [ord(" ")] * (rows - len(col))
            for src_row, code in enumerate(padded):
                row = rows - 1 - src_row      # source row 0 == bottom
                self.set(x, row, code)
            # ReadWorld: `W^[X, -EY1] := #0` -- one solid (#0 is in
            # CanHoldYou) row above the level, above the plain ' ' sky rows.
            # Enemies collide against it (Check() indexes rows down to -EY1).
            self.set(x, -EY1, 0)
            # extra rows below repeat the bottom row (ReadWorld behaviour)
            bottom = padded[0]
            for i in range(EY2):
                self.set(x, rows + i, bottom)
