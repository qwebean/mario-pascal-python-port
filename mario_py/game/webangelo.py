"""Angelo objects ported into the Mario engine: Boss, BossFire, FireRotator.

Source: WEBANGELO\\AngeloHTML5\\Angelo (ActionScript decompiled from
Mario.swf/Angelo.swf).  Coordinates, speeds and timers are taken from
Game.as (InitObject), Enemy.as and FireRotator.as:

* Boss     40x28, 10 player-fireball hits, slow walker, faces the player,
           contact hurts.  A hidden Spitter fires a 12x7 bullet every
           150..300 frames while the player is left of the boss.
* FireRotator  6 fireballs rotating around a pivot: a hub on i=0 plus five
           segments at radii 11*i / 10*i for i = 1..5.  ``angle += PI/100``
           per frame; each segment has a 12x8 hitbox and hurts on contact.
           (FireRotator.as loops 2..6 because its index is pre-incremented;
           the original game shows six balls starting at the pivot block.)

Every sprite saves the page underneath through VGAScreen.push_backgr() before
it is drawn, exactly like the ENEMIES.PAS objects do.  Sprite records are kept
for the whole level so the second video page is also restored after an object
disappears (PlayWorld alternates pages and hides one frame behind).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pygame

from ..engine.vga import MAX_PAGE
from .buffers import CAN_HOLD_YOU, TILE_H, TILE_W
from .enemies import TP_DYING_FIREBALL, TP_FIREBALL

W, H = TILE_W, TILE_H
FRAME_DIR = Path(__file__).resolve().parent.parent / "data" / "webangelo"

BOSS_W, BOSS_H = 40, 28
BOSS_HP = 10
FIRE_W, FIRE_H = 12, 7
SEG_W, SEG_H = 12, 8
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 200

_surfaces: dict[str, pygame.Surface] = {}


def _frame(kind: str, index: int) -> pygame.Surface:
    key = f"{kind}_{index}"
    surf = _surfaces.get(key)
    if surf is None:
        surf = pygame.image.load(str(FRAME_DIR / f"{key}.png"))
        try:
            surf = surf.convert_alpha()
        except pygame.error:
            pass
        _surfaces[key] = surf
    return surf


def _overlap(ax: float, ay: float, aw: float, ah: float,
             bx: float, by: float, bw: float, bh: float) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class _Sprite:
    """One drawn frame with its own PushBackGr address per video page."""

    __slots__ = ("surf", "addr")

    def __init__(self, surf: pygame.Surface) -> None:
        self.surf = surf
        self.addr = [0] * (MAX_PAGE + 1)

    def show(self, screen, x: float, y: float) -> None:
        ix, iy = int(x), int(y)
        page = screen.current
        self.addr[page] = screen.push_backgr(
            ix, iy, self.surf.get_width(), self.surf.get_height())
        screen.page.blit(self.surf, (ix, iy))

    def hide(self, screen) -> None:
        page = screen.current
        addr = self.addr[page]
        if addr:
            screen.pop_backgr(addr)
            self.addr[page] = 0


def _solid(world, x: int, y: int) -> bool:
    return world.get(x, y) in CAN_HOLD_YOU


def _on_screen(x: float, y: float, w: int, h: int,
               view_x: int, view_y: int) -> bool:
    return (x + w >= view_x and x <= view_x + SCREEN_WIDTH
            and y + h >= view_y and y <= view_y + SCREEN_HEIGHT)


class BossFire:
    w, h = FIRE_W, FIRE_H

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = -1.0
        self.dead = False
        self.sprite = _Sprite(_frame("bullet", 4))

    def update(self, world, player_box, enemies) -> None:
        self.x += self.vx
        px1, py1, px2, py2 = player_box
        if _overlap(self.x, self.y, self.w, self.h,
                    px1, py1, px2 - px1, py2 - py1):
            if not enemies.star:
                enemies.cd_hit = 1
            self.dead = True
            return
        cx = int((self.x + self.w / 2) // W)
        cy = int((self.y + self.h / 2) // H)
        if (self.x < -self.w or self.x > world.x_size * W
                or world.get(cx, cy) in CAN_HOLD_YOU):
            self.dead = True

    def show(self, screen, view_x: int, view_y: int) -> None:
        if _on_screen(self.x, self.y, self.w, self.h, view_x, view_y):
            self.sprite.show(screen, self.x, self.y)


class Boss:
    w, h = BOSS_W, BOSS_H

    def __init__(self, x: float, y: float, variant: int,
                 hooks, rng: random.Random) -> None:
        self.x = float(x)
        self.y = float(y)
        self.vx = -0.125
        self.vy = 0.0
        self.variant = variant if variant in (1, 2, 3) else 3
        first = (self.variant - 1) * 2
        self.frames = [_frame("boss", first), _frame("boss", first + 1)]
        self.flipped = [pygame.transform.flip(f, True, False)
                        for f in self.frames]
        self.hp = BOSS_HP
        self.anim = 0
        self.timer = 150 + rng.randrange(151)
        self.dying = False
        self.death_timer = 0
        self.gone = False
        self.face_right = False
        self.hooks = hooks
        self.rng = rng
        self.sprite = _Sprite(self.frames[0])

    # ------------------------------------------------------------------ #
    def _touching_wall(self, world, nx: float) -> bool:
        top = int((self.y + 2) // H)
        bottom = int((self.y + self.h - 3) // H)
        cx = int((nx + self.w - 1) // W) if self.vx > 0 else int(nx // W)
        return any(_solid(world, cx, cy) for cy in range(top, bottom + 1))

    def _on_ground(self, world, ny: float) -> bool:
        cy = int((ny + self.h) // H)
        left = int((self.x + 2) // W)
        right = int((self.x + self.w - 3) // W)
        return _solid(world, left, cy) or _solid(world, right, cy)

    def update(self, world, enemies, player_box, fires: list[BossFire],
               on_defeat) -> None:
        px1, py1, px2, py2 = player_box
        px = (px1 + px2) / 2
        self.face_right = px > self.x + self.w / 2
        self.anim += 1

        if self.dying:
            if self.gone:
                return
            self.death_timer += 1
            self.vy += 0.05
            self.y += self.vy
            if self.death_timer > 90:
                self.gone = True
                on_defeat()
            return

        self.vy = min(self.vy + 0.0125, 4.0)
        ny = self.y + self.vy
        if self._on_ground(world, ny):
            self.y = int((ny + self.h) // H) * H - self.h
            self.vy = 0.0
        else:
            self.y = ny

        nx = self.x + self.vx
        if self._touching_wall(world, nx):
            self.vx = -self.vx
        else:
            self.x = nx

        # Game.as sets canJumpOn = false: every contact hurts.
        if _overlap(self.x, self.y, self.w, self.h,
                    px1, py1, px2 - px1, py2 - py1):
            if not enemies.star:
                enemies.cd_hit = 1

        # Spitter.as: fire only when the player is to the left of the boss.
        self.timer -= 1
        if self.timer <= 0:
            self.timer = 150 + self.rng.randrange(151)
            if px < self.x and abs(px - self.x) < SCREEN_WIDTH + 20:
                fires.append(BossFire(self.x + 4, self.y + 12))
                self.hooks.beep(150)

    def hit_by_fireball(self, enemies) -> bool:
        """Consume one player fireball if it overlaps the boss."""
        if self.dying or self.gone:
            return False
        for idx in list(enemies.active):
            e = enemies.pool[idx]
            if e.tp != TP_FIREBALL:
                continue
            if _overlap(e.x_pos, e.y_pos, FIRE_W, FIRE_H,
                        self.x, self.y, self.w, self.h):
                e.tp = TP_DYING_FIREBALL
                e.delay_counter = -(MAX_PAGE + 1)
                self.hp -= 1
                self.hooks.beep(900)
                self.hooks.new_star(
                    int(self.x) + self.rng.randrange(self.w),
                    int(self.y) + self.rng.randrange(self.h),
                    57 + self.rng.randrange(7), 12 + self.rng.randrange(8))
                if self.hp <= 0:
                    self.dying = True
                    self.death_timer = 0
                    self.vx = 0.0
                    self.vy = -1.5
                    self.hooks.beep(400)
                return True
        return False

    def show(self, screen, view_x: int, view_y: int) -> None:
        if not _on_screen(self.x, self.y, self.w, self.h, view_x, view_y):
            return
        if self.dying:
            self.sprite.surf = pygame.transform.flip(
                self.frames[(self.anim // 18) & 1], False, True)
        else:
            self.sprite.surf = (self.flipped if self.face_right
                                else self.frames)[(self.anim // 18) & 1]
        self.sprite.show(screen, self.x, self.y)


class FireRotator:
    """Five fireballs rotating around a hub (FireRotator.as)."""

    def __init__(self, x: float, y: float, rng: random.Random) -> None:
        self.cx = float(x)
        self.cy = float(y)
        self.angle = rng.random() * math.tau
        self.time = 0
        self.rng = rng
        self.hub = _Sprite(_frame("enemy", 22))
        self.segments = [_Sprite(_frame("enemy", 22)) for _ in range(5)]
        self.pos: list[tuple[float, float]] = []
        self._place()

    def _place(self) -> None:
        # Original: hub on the pivot block plus five segments i = 1..5.
        # The decompiled FireRotator.as loop pre-increments and reads 2..6;
        # the game screenshots show six balls starting at the pivot, so the
        # loop indices are a decompiler artifact.  ``pos`` stores centres.
        self.pos = [(self.cx + i * 11 * math.sin(self.angle),
                     self.cy + i * 10 * math.cos(self.angle))
                    for i in range(1, 6)]

    def _hub_xy(self) -> tuple[float, float]:
        """Top-left of the hub sprite, centred on the pivot."""
        return self.cx - 10, self.cy - 7

    @staticmethod
    def _seg_hitbox(sx: float, sy: float) -> tuple[float, float, int, int]:
        """12x8 hitbox centred inside the 20x14 segment sprite."""
        return sx - SEG_W / 2, sy - SEG_H / 2, SEG_W, SEG_H

    def update(self, enemies, player_box, hooks) -> None:
        self.angle += math.pi / 100
        self.time += 1
        self._place()
        px1, py1, px2, py2 = player_box
        for sx, sy in self.pos:
            hx, hy, hw, hh = self._seg_hitbox(sx, sy)
            if _overlap(hx, hy, hw, hh,
                        px1, py1, px2 - px1, py2 - py1):
                if not enemies.star:
                    enemies.cd_hit = 1
                break
        if self.time % 35 == 0:
            sx, sy = self.pos[self.rng.randrange(5)]
            hooks.new_star(int(sx) + self.rng.randrange(5) - 2,
                           int(sy) + self.rng.randrange(5) - 2,
                           57 + self.rng.randrange(7),
                           10 + self.rng.randrange(8))

    def show(self, screen, view_x: int, view_y: int) -> None:
        frame = _frame("enemy", 22 + (self.time // 3) % 4)
        hx, hy = self._hub_xy()
        if _on_screen(hx, hy, 20, 14, view_x, view_y):
            self.hub.surf = frame
            self.hub.show(screen, hx, hy)
        for seg, (sx, sy) in zip(self.segments, self.pos):
            tx, ty = sx - 10, sy - 7
            if _on_screen(tx, ty, 20, 14, view_x, view_y):
                seg.surf = frame
                seg.show(screen, tx, ty)


class AngeloObjects:
    """Owns the WEBANGELO objects of one level."""

    def __init__(self, level, hooks, rng: random.Random | None = None) -> None:
        self.level = level
        self.hooks = hooks
        self.rng = rng or random.Random()
        self.bosses: list[Boss] = []
        self.bars: list[FireRotator] = []
        self.fires: list[BossFire] = []
        # Stable draw order for the LIFO background stack.  Sprites stay here
        # after their object dies so both video pages get restored.
        self._sprites: list[_Sprite] = []
        self.build()

    def build(self) -> None:
        self.bosses = []
        self.bars = []
        self.fires = []
        self._sprites = []
        for spec in getattr(self.level, "webangelo", []):
            kind = spec.get("type")
            if kind == "boss":
                boss = Boss(spec.get("x", 0), spec.get("y", 0),
                            int(spec.get("variant", 3)), self.hooks, self.rng)
                self.bosses.append(boss)
                self._sprites.append(boss.sprite)
            elif kind == "firebar":
                bar = FireRotator(spec.get("x", 0), spec.get("y", 0),
                                  self.rng)
                self.bars.append(bar)
                self._sprites.append(bar.hub)
                self._sprites.extend(bar.segments)

    def reset(self) -> None:
        self.build()

    def bind(self, level) -> None:
        self.level = level
        self.build()

    def move(self, world, enemies, on_defeat) -> None:
        box = (enemies.player_x1, enemies.player_y1,
               enemies.player_x2, enemies.player_y2)
        for boss in self.bosses:
            boss.update(world, enemies, box, self.fires, on_defeat)
            boss.hit_by_fireball(enemies)
        for bar in self.bars:
            bar.update(enemies, box, self.hooks)

        known = {id(s) for s in self._sprites}
        for fire in self.fires:
            if id(fire.sprite) not in known:
                self._sprites.append(fire.sprite)
                known.add(id(fire.sprite))

        for fire in list(self.fires):
            fire.update(world, box, enemies)
            if not fire.dead:
                self._fire_vs_fireball(fire, enemies)
        self.fires = [f for f in self.fires if not f.dead]
        self.bosses = [b for b in self.bosses if not b.gone]

    def _fire_vs_fireball(self, fire: BossFire, enemies) -> None:
        for idx in list(enemies.active):
            e = enemies.pool[idx]
            if e.tp != TP_FIREBALL:
                continue
            if _overlap(e.x_pos, e.y_pos, FIRE_W, FIRE_H,
                        fire.x, fire.y, fire.w, fire.h):
                e.tp = TP_DYING_FIREBALL
                e.delay_counter = -(MAX_PAGE + 1)
                fire.dead = True
                return

    def show(self, screen) -> None:
        vx, vy = screen.view_x, screen.view_y
        for boss in self.bosses:
            boss.show(screen, vx, vy)
        for bar in self.bars:
            bar.show(screen, vx, vy)
        for fire in self.fires:
            fire.show(screen, vx, vy)

    def hide(self, screen) -> None:
        for sprite in reversed(self._sprites):
            sprite.hide(screen)
