"""Port of TMPOBJ.PAS — temporary objects and the "remove/redraw" queue.

Two things live here:

* ``TempObj[1..20]``: short-lived sprites — the four brick fragments of a
  broken block, the coin that pops out of a block, the white "hit" flash and
  the note that flies out of a music block.  Each one saves the background it
  covers (per video page) and restores it when hidden.
* ``RemList[1..10]``: a small queue of map cells that have to be repainted on
  *both* video pages after the map changed (used block, note block, broken
  block).  Every entry is drawn ``MAX_PAGE + 1`` times, once per frame.

The unit also owns the three helpers every other unit calls through the hooks:
``BreakBlock``, ``HitCoin`` and ``AddLife``.

All coordinates are world pixels (same convention as figures/enemies/players).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..engine.vga import MAX_PAGE
from .buffers import NH, TILE_H, TILE_W, WorldMap, WorldOptions
from .enemies import EnemyHooks

W = TILE_W
H = TILE_H

TP_BROKEN = 1
TP_COIN = 2
TP_HIT = 3
TP_FIRE = 4
TP_NOTE = 5

BROKEN_DELAY = 3
COIN_SPEED = -4
COIN_DELAY = 12
MAX_COIN_YVEL = 6
HIT_TIME = 4

MAX_TEMP_OBJ = 20
MAX_REMOVE = 10


@dataclass
class TempRec:
    alive: bool = False
    visible: list[bool] = field(
        default_factory=lambda: [False] * (MAX_PAGE + 1))
    tp: int = 0
    backgr_addr: list[int] = field(
        default_factory=lambda: [-1] * (MAX_PAGE + 1))
    x_pos: int = 0
    y_pos: int = 0
    h_size: int = 0
    v_size: int = 0
    x_vel: int = 0
    y_vel: int = 0
    delay_counter: int = 0


@dataclass
class RemoveRec:
    active: bool = False
    rem_count: int = 0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    new_image: int = 0


class TempObjects:
    """TMPOBJ.PAS: the temp-object pool plus the remove queue."""

    def __init__(self, screen, figures, backgr, world: WorldMap,
                 options: WorldOptions, data, player: int = 0,
                 hooks: EnemyHooks | None = None,
                 view_x=None) -> None:
        self.screen = screen
        self.figures = figures
        self.backgr = backgr
        self.world = world
        self.options = options
        self.data = data
        self.player = player
        self.hooks = hooks if hooks is not None else EnemyHooks()
        self._view_x = view_x if view_x is not None else (lambda: 0)

        self.pool: list[TempRec] = [TempRec()
                                    for _ in range(MAX_TEMP_OBJ + 1)]
        self.rem_list: list[RemoveRec] = [RemoveRec()
                                          for _ in range(MAX_REMOVE + 1)]
        # Part000, recoloured with Options.BrickColor at InitTempObj
        self.part = self.figures.recolor(
            self.figures.assets.sprite_idx("part000"),
            self.options.BrickColor)
        self.init_temp_obj()

    # ------------------------------------------------------------------ #
    # images
    # ------------------------------------------------------------------ #
    def _surf(self, name: str, arr: np.ndarray | None = None):
        if arr is None:
            arr = self.figures.assets.sprite_idx(name)
        return self.figures.cache_surface(f"TO:{name}", arr)

    def _image(self, tp: int):
        if tp == TP_BROKEN:
            return self._surf("PART", self.part)
        if tp == TP_COIN:
            return self._surf("COIN000")
        if tp == TP_HIT:
            return self._surf("WHHIT000")
        if tp == TP_FIRE:
            return self._surf("WHFIRE000")
        if tp == TP_NOTE:
            return self._surf("NOTE000")
        return None

    # ------------------------------------------------------------------ #
    # pool management
    # ------------------------------------------------------------------ #
    def init_temp_obj(self) -> None:
        for rec in self.pool:
            rec.alive = False
            rec.visible = [False] * (MAX_PAGE + 1)
        for rem in self.rem_list:
            rem.active = False
        # InitTempObj recolours Part000 with the *current* Options.BrickColor,
        # so a sub-level with a different brick colour gets new debris.
        self.part = self.figures.recolor(
            self.figures.assets.sprite_idx("part000"),
            self.options.BrickColor)

    def _read_backgr(self, rec: TempRec) -> None:
        page = self.screen.current
        rec.backgr_addr[page] = self.screen.push_backgr(
            rec.x_pos, rec.y_pos, rec.h_size + 4, rec.v_size)

    def _available(self, rec: TempRec) -> bool:
        return not (rec.alive or any(rec.visible))

    def new_temp_obj(self, new_type: int, x: int, y: int, xv: int, yv: int,
                     wid: int, ht: int) -> None:
        if new_type == TP_BROKEN:
            # do not spawn debris that will be off-screen for its whole life
            if xv > 0:
                if x + 32 * xv > self._view_x() + NH * W + 2 * W:
                    return
            elif x + 32 * xv + 2 * W < self._view_x():
                return
        for rec in self.pool[1:MAX_TEMP_OBJ + 1]:
            if self._available(rec):
                rec.alive = True
                rec.visible = [False] * (MAX_PAGE + 1)
                rec.tp = new_type
                rec.x_pos = x
                rec.y_pos = y
                rec.x_vel = xv
                rec.y_vel = yv
                rec.h_size = wid
                rec.v_size = ht
                self._read_backgr(rec)
                rec.delay_counter = 0
                return

    def show_temp_obj(self) -> None:
        page = self.screen.page
        for rec in self.pool[1:MAX_TEMP_OBJ + 1]:
            if not rec.alive:
                continue
            self._read_backgr(rec)
            surf = self._image(rec.tp)
            if surf is not None:
                page.blit(surf, (rec.x_pos, rec.y_pos))
            rec.visible[self.screen.current] = True

    def hide_temp_obj(self) -> None:
        page = self.screen.current
        for rec in reversed(self.pool[1:MAX_TEMP_OBJ + 1]):
            if rec.visible[page]:
                self.screen.pop_backgr(rec.backgr_addr[page])
                rec.visible[page] = False

    def move_temp_obj(self) -> None:
        for rec in self.pool[1:MAX_TEMP_OBJ + 1]:
            if not rec.alive:
                continue
            if rec.tp == TP_BROKEN:
                rec.delay_counter += 1
                if rec.delay_counter > BROKEN_DELAY:
                    rec.delay_counter = 0
                    rec.y_vel += 1
                    if rec.y_pos > self.world.rows * H:
                        rec.alive = False
            elif rec.tp == TP_COIN:
                rec.delay_counter += 1
                if rec.delay_counter > COIN_DELAY:
                    rec.y_vel += 1
                    if rec.y_vel > MAX_COIN_YVEL:
                        rec.alive = False
                        self.hooks.coin_glitter(rec.x_pos + rec.x_vel,
                                                rec.y_pos + rec.y_vel)
            elif rec.tp in (TP_HIT, TP_FIRE):
                rec.delay_counter += 1
                if rec.delay_counter > HIT_TIME:
                    rec.alive = False
            rec.x_pos += rec.x_vel
            rec.y_pos += rec.y_vel

    # ------------------------------------------------------------------ #
    # remove queue (RunRemove)
    # ------------------------------------------------------------------ #
    def remove(self, x: int, y: int, w: int, h: int, new_img: int) -> None:
        if y < 0:
            return
        for rem in self.rem_list[1:MAX_REMOVE + 1]:
            if not rem.active:
                rem.x = x
                rem.y = y
                rem.w = w
                rem.h = h
                rem.new_image = new_img
                rem.rem_count = MAX_PAGE + 1
                rem.active = True
                return

    def run_remove(self) -> None:
        page = self.screen.page
        for rem in self.rem_list[1:MAX_REMOVE + 1]:
            if not rem.active:
                continue
            if rem.new_image == 0:
                self.backgr.draw_backgr_block(page, rem.x, rem.y,
                                              rem.w, rem.h)
            elif rem.new_image == 1:
                page.blit(self._surf("QUEST001"), (rem.x, rem.y))
            elif rem.new_image == 2:
                page.blit(self._surf("QUEST000"), (rem.x, rem.y))
            elif rem.new_image == 5:
                page.blit(self._surf("NOTE000"), (rem.x, rem.y))
            rem.rem_count -= 1
            if rem.rem_count < 1:
                rem.active = False

    # ------------------------------------------------------------------ #
    # helpers used by every other unit
    # ------------------------------------------------------------------ #
    def break_block(self, map_x: int, map_y: int) -> None:
        self.world.set(map_x, map_y, ord(" "))
        x = map_x * W
        y = map_y * H
        self.remove(x, y, W, H, 0)
        x1 = x
        x2 = x + W // 2
        y1 = y
        y2 = y + H // 2
        self.new_temp_obj(TP_BROKEN, x1, y1, -2, -6, 12, H // 2)
        self.new_temp_obj(TP_BROKEN, x2, y1, 2, -6, 12, H // 2)
        self.new_temp_obj(TP_BROKEN, x1, y2, -2, -4, 12, H // 2)
        self.new_temp_obj(TP_BROKEN, x2, y2, 2, -4, 12, H // 2)
        self.hooks.beep(110)

    def hit_coin(self, x: int, y: int, throw_up: bool) -> None:
        map_x = x // W
        map_y = y // H
        if self.world.get(map_x, map_y) == ord(" "):
            return
        if throw_up:
            self.new_temp_obj(TP_COIN, x, y - H, 0, COIN_SPEED, W, H)
        else:
            self.world.set(map_x, map_y, ord(" "))
            self.remove(x, y, W, H, 0)
            self.hooks.coin_glitter(x, y)
        self.hooks.beep(2420)
        self.data.Coins[self.player] += 1
        self.hooks.add_score(50)
        if self.data.Coins[self.player] % 100 == 0:
            self.add_life()
            self.data.Coins[self.player] = 0

    def add_life(self) -> None:
        self.data.Lives[self.player] += 1
        self.hooks.start_music("life")
