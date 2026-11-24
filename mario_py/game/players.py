"""Port of PLAYERS.PAS — the player character.

Holds everything the original unit owns: the 24 walk/jump frames per mode
(small / large / fire, Mario / Luigi, both directions), the player's movement
and collision state machine (walking, jumping, falling), the pipe demos, the
death animation, and the screen-scrolling rules.

The unit talks to the rest of the game through the same places the Pascal did:

* the shared ``cd*`` flags / ``Turbo`` / ``NewEnemy`` / ``HitAbove`` live on the
  ``Enemies`` instance (ENEMIES.PAS globals),
* ``Data`` (GameData) carries ``Mode[Player]``,
* the units not ported yet (TmpObj / Blocks / Music / Glitter) are reached via
  ``PlayerHooks`` — the default implementation does nothing, which makes the
  module testable headlessly.

Coordinate convention is identical to figures.py / enemies.py: ``x``/``y`` are
world pixels, a player frame is W x 2H (small players only occupy the lower H
rows), and drawing goes to the current VGA page.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from ..engine.vga import MAX_PAGE, VIR_SCREEN_HEIGHT
from .buffers import (CAN_HOLD_YOU, CAN_STAND_ON, DIR_LEFT, DIR_RIGHT, EY1,
                      MD_FIRE, MD_LARGE, MD_SMALL, NH, TILE_H, TILE_W,
                      WorldMap, WorldOptions)
from .enemies import (EnemyHooks, TP_FIREBALL, TP_RISING_CHAMP,
                      TP_RISING_FLOWER, TP_RISING_LIFE, TP_RISING_STAR, idiv)

W = TILE_W
H = TILE_H
SAFE = EY1
HSAFE = H * SAFE

# PLAYERS.PAS constants
ST_ON_THE_GROUND = 0
ST_JUMPING = 1
ST_FALLING = 2

SCROLL_AT = 112
# Vertical dead zone (screen pixels, 13-row viewport is 182 px).  The camera
# stays still while the player's 2-tile body fits between these rows.
V_DEAD_TOP = 48
V_DEAD_BOTTOM = 134
JUMP_VEL = 4
JUMP_DELAY = 6
MAX_Y_VEL = JUMP_VEL * 2
SLIP = 6
BLINK_TIME = 125
STAR_TIME = 750
GROW_TIME = 24
MAX_SPEED = 2

# Underwater movement (port-only).  Alt is the swim modifier: hold it to keep
# the current depth, Alt+Up/Down to swim at a constant speed, and left/right
# still steer.  Releasing Alt lets the reduced water gravity take over.  The
# two-frame swim delay is the user's "slightly sluggish but not extreme"
# control feel; horizontal acceleration uses every 8th frame (ground: 6).
SWIM_UP = 2
SWIM_INPUT_DELAY = 2
SWIM_ACCEL_DELAY = 8
SWIM_SINK_DELAY = 4
SWIM_SINK_MAX = 2
SWIM_ANIM_DIV = 8
SWIM_MAX_SPEED = 1
SWIM_BUBBLE_PERIOD = 28
SWIM_BUBBLES_MAX = 6

# 3x3 bubble: light-blue rim, white highlight.  Index 11 is the water
# animation entry, so bubbles shimmer with the rest of the water palette.
BUBBLE_ART = np.array([[0, 11, 0],
                       [11, 15, 11],
                       [0, 11, 0]], dtype=np.uint8)

# Ghost-house sight (Options.Vision), in tiles.  The diamond joins the midpoints
# of the outer edges of the four tiles around Mario's HEAD tile: one up, one
# left, one right, one down are lit (the user's rule, 2026-10-25).  At 1.5 that
# reaches 30 px sideways and 21 px up/down, and a big Mario's feet land on the
# lower edge of it.  A coin buys one more tile, stackable up to VISION_MAX, and
# one tile falls off every VISION_DECAY frames.
VISION_BASE = 1.5
VISION_MAX = 6
VISION_DECAY = 120           # frames -- "one tile every 2 s" at 60 Hz

# BUFFERS.PAS demo modes
DM_NO_DEMO = 0
DM_DOWN_IN_TO_PIPE = 1
DM_UP_OUT_OF_PIPE = 2
DM_UP_IN_TO_PIPE = 3
DM_DOWN_OUT_OF_PIPE = 4
DM_DEAD = 5
# port-only side demos for the horizontal pipe (same walking frames, the
# sprite is erased column by column like the side plant hides itself)
DM_LEFT_IN_TO_PIPE = 6
DM_RIGHT_IN_TO_PIPE = 7
DM_LEFT_OUT_OF_PIPE = 8
DM_RIGHT_OUT_OF_PIPE = 9

TP_NOTE = 5                     # TMPOBJ.PAS

# map byte codes (cp865)
CH_SPACE = 0x20
CH_COIN = ord("*")
CH_K = ord("K")
CH_J = ord("J")
CH_EQ = ord("=")
CH_QMARK = ord("?")
CH_DOLLAR = ord("$")
CH_Z = ord("z")
CH_0 = ord("0")
CH_1 = ord("1")
# horizontal pipe tiles (port-only codes, free in every level pack)
CH_HP_MOUTH_T = ord("4")
CH_HP_MOUTH_B = ord("5")
CH_HP_BODY_T = ord("6")
CH_HP_BODY_B = ord("7")
CH_AT = ord("@")
CH_ALPHA = 0xE0                 # 'α'  pipe entry / hidden item
CH_SS = 0xE1                    # 'ß'  hidden 1-up
CH_GAMMA = 0xE2                 # 'Γ'  hidden star
CH_PI = 0xE3                    # 'π'  multi-coin counter start
CH_TAU = 0xE7                   # 'τ'  end-of-level pipe
CH_INF = 0xEC                   # '∞'  multi-coin counter end
CH_PHI = 0xED                   # 'φ'  hidden champ (SubTp 1)
CH_INTER = 0xEF                 # '∩'  hidden note block
CH_LEFT_MARK = 254              # '■'  scroll stop at left edge
CH_RIGHT_MARK = 255             # ' '?  scroll stop at right edge

HOLD_OR_STAND = CAN_HOLD_YOU | CAN_STAND_ON
HOLD_OR_HIDDEN = CAN_HOLD_YOU | {ord("$")}


def pmod(a: int, b: int) -> int:
    """Pascal ``mod`` (truncates toward zero, like the x86 idiv)."""
    return a - idiv(a, b) * b


def pascal_round_div(a: int, b: int) -> int:
    """Pascal ``Round(a / b)`` for the non-negative values used here."""
    return (a + b // 2) // b


class PlayerHooks(EnemyHooks):
    """Call-outs to TmpObj / Blocks / Music / Glitter / Buffers.

    ENEMIES.PAS and PLAYERS.PAS need almost the same set of helpers, so this
    simply extends EnemyHooks with the two calls only the player makes."""

    def add_life(self) -> None:
        """TmpObj.AddLife — Inc(Data.Lives[Player]) + LifeMusic."""
        return None

    def bump_block(self, x: int, y: int) -> None:
        """Blocks.BumpBlock(X, Y) — pixel coordinates of the bumped tile."""
        return None


@dataclass
class ScreenRec:
    visible: bool = False
    x_pos: int = 0
    y_pos: int = 0
    backgr_addr: int = -1


class Player:
    """One player character (PLAYERS.PAS globals + procedures)."""

    def __init__(self, screen, world: WorldMap, options: WorldOptions,
                 figures, enemies, data, name: int = 0,
                 hooks: PlayerHooks | None = None) -> None:
        self.screen = screen
        self.world = world
        self.options = options
        self.figures = figures
        self.enemies = enemies
        self.data = data
        self.player = name
        self.hooks = hooks if hooks is not None else PlayerHooks()

        # Pictures[plMario..plLuigi, mdSmall..mdFire, 0..3, dirLeft..dirRight]
        self.pictures: list[list[list[list[np.ndarray | None]]]] = [
            [[[None, None] for _ in range(4)] for _ in range(3)]
            for _ in range(2)]
        # [player][mode][pose 0 stand / 1 walk / 2 jump / 3 jumpUp][direction]
        self.swim_pictures: list[list[list[list[np.ndarray | None]]]] = [
            [[[None, None] for _ in range(4)] for _ in range(3)]
            for _ in range(2)]
        self.init_player_figures()

        self.save_screen = [ScreenRec() for _ in range(MAX_PAGE + 1)]

        # movement state (PLAYERS.PAS vars)
        self.x = 0
        self.y = 0
        self.demo_y = 0
        self.demo_x = 0
        self.demo_counter1 = 0
        self.demo_counter2 = 0
        self.xvel = 0
        self.yvel = 0
        self.direction = DIR_RIGHT
        self.status = ST_ON_THE_GROUND
        self.walking_mode = 0
        self.counter = 0
        self.walk_count = 0
        # underwater state (port-only; the original has no water)
        self.in_water = False
        self.was_in_water = False
        self.water_y = -1
        self.swim_hold = 0
        self.swim_counter = 0
        self.swim_frame = 0
        self.swim_blocked = 0
        self.swim_hover = False
        self.water_tick = 0
        self.water_walk_phase = 0
        self.bubble_timer = 0
        self.bubbles: list[dict] = []
        self.high_jump = False
        self.hit_enemy = False
        self.jumped = False
        self.fired = False
        self.fire_counter = 0
        self.star_counter = 0
        self.grow_counter = 0
        self.blink_counter = 0
        self.at_ch1 = CH_SPACE
        self.at_ch2 = CH_SPACE
        self.below1 = CH_SPACE
        self.below2 = CH_SPACE

        # interface vars
        self.blinking = False
        self.growing = False
        self.invincible = False
        self.in_pipe = False
        self.pipe_code = [CH_SPACE, CH_SPACE]
        self.pipe_side = False      # last pipe entry was a horizontal pipe
        self.map_x = 0
        self.map_y = 0
        self.earth_quake = False
        self.earth_quake_counter = 0
        self.small = 0

        # BUFFERS.PAS globals this unit owns while play.py does not exist yet
        self.demo = DM_NO_DEMO
        self.passed = False
        self.text_counter = 0
        self.game_done = False
        self.x_view = 0
        self.view_y = 0

        # ghost-house sight (Options.Vision); 0 = the level is not dark
        self.vision_r = 0
        self.vision_decay = 0
        self._coins_seen = 0

        # shared box copied into Enemies
        self.player_x1 = 0
        self.player_x2 = W - 1
        self.player_y1 = 0
        self.player_y2 = 2 * H - 1
        self.player_xvel = 0
        self.player_yvel = 0

        # keyboard state (module vars keyLeft..keySpace)
        self.key_left = False
        self.key_right = False
        self.key_up = False
        self.key_down = False
        self.key_alt = False
        self.key_ctrl = False
        self.key_space = False

    # ------------------------------------------------------------------ #
    # InitPlayerFigures / InitPlayer
    # ------------------------------------------------------------------ #
    def init_player_figures(self) -> None:
        """Load the 24 frames and mirror every left-facing frame to the right.

        InitPlayerFigures fills Pictures, then calls HighMirror on each entry;
        palette indices are mirrored, so we keep raw arrays and let
        Figures.cache_surface() bake them."""
        a = self.figures.assets
        walk = ("SW", "LW", "FW")       # mdSmall, mdLarge, mdFire
        jump = ("SJ", "LJ", "FJ")
        who = ("MAR", "LUI")
        swim = (("SWIM_SMALL_MARIO", "SWIM_SMALL_LUIGI"),
                ("SWIM_BIG_MARIO", "SWIM_BIG_LUIGI"),
                ("SWIM_FIRE_MARIO", "SWIM_FIRE_LUIGI"))
        for pl in range(2):
            for md in range(3):
                for n in range(4):
                    pre = walk[md] if n < 2 else jump[md]
                    name = f"{pre}{who[pl]}{n % 2:03d}"
                    arr = a.sprite_idx(name)
                    self.pictures[pl][md][n][DIR_LEFT] = arr
                    self.pictures[pl][md][n][DIR_RIGHT] = \
                        np.ascontiguousarray(arr[:, ::-1])
                    # the 91898 sheet faces right, opposite to the original
                    # DOS frames; right is the raw art, left is its mirror
                    swim_arr = a.sprite_idx(f"{swim[md][pl]}_{n}")
                    self.swim_pictures[pl][md][n][DIR_RIGHT] = swim_arr
                    self.swim_pictures[pl][md][n][DIR_LEFT] = \
                        np.ascontiguousarray(swim_arr[:, ::-1])

    def init_player(self, init_x: int, init_y: int, name: int) -> None:
        self.player = name
        self.x = init_x
        self.y = init_y
        self.xvel = 0
        self.yvel = 0
        self.direction = DIR_RIGHT
        self.walking_mode = 0
        self.status = ST_ON_THE_GROUND
        self.jumped = False
        self.fired = False
        self.hit_enemy = False
        self.in_water = False
        self.was_in_water = False
        self.water_y = self.options.water_y
        self.swim_hold = 0
        self.swim_counter = 0
        self.swim_frame = 0
        self.swim_blocked = 0
        self.swim_hover = False
        self.water_tick = 0
        self.water_walk_phase = 0
        self.bubble_timer = 0
        self.bubbles.clear()
        for rec in self.save_screen:
            rec.visible = False
        self.player_x1 = self.x
        self.player_x2 = self.x + W - 1
        self.player_y1 = self.y + H
        self.player_y2 = self.y + 2 * H - 1
        self.player_xvel = self.xvel
        self.player_yvel = self.yvel
        # Start with the player inside the vertical dead zone (tall levels
        # usually begin at the bottom, so view_y snaps to the lower clamp).
        self.view_y = 0
        if self.world.vertical:
            max_y = max(0, self.world.rows * H - VIR_SCREEN_HEIGHT)
            self.view_y = max(
                0, min(max_y, self.y + 2 * H - V_DEAD_BOTTOM))
        self.blinking = False
        self.enemies.star = False
        self.growing = False
        self.earth_quake = False
        self.sync_enemies()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _mode(self) -> int:
        return int(self.data.Mode[self.player])

    def _is_small(self) -> bool:
        return self._mode() == MD_SMALL

    def _pic(self, mode: int, walk: int, direction: int,
             diff: int = 0) -> pygame.Surface:
        arr = self.pictures[self.player][mode][walk][direction]
        if diff:
            arr = np.where(arr != 0, (arr.astype(np.int16) + diff) & 0xFF,
                           0).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
        key = f"PL{self.player}{mode}{walk}{direction}:{diff & 0xFF}"
        return self.figures.cache_surface(key, arr)

    def _swim_pic(self, mode: int, pose: int, direction: int,
                  diff: int = 0) -> pygame.Surface:
        arr = self.swim_pictures[self.player][mode][pose][direction]
        if diff:
            arr = np.where(arr != 0, (arr.astype(np.int16) + diff) & 0xFF,
                           0).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
        key = f"SW{self.player}{mode}{pose}{direction}:{diff & 0xFF}"
        return self.figures.cache_surface(key, arr)

    def _update_view_y(self) -> None:
        """Vertical camera: dead-zone follow, clamped to the level height."""
        w = self.world
        if not w.vertical:
            self.view_y = 0
            return
        max_y = max(0, w.rows * H - VIR_SCREEN_HEIGHT)
        top = self.y - self.view_y
        if top < V_DEAD_TOP:
            self.view_y = self.y - V_DEAD_TOP
        elif top + 2 * H > V_DEAD_BOTTOM:
            self.view_y = self.y + 2 * H - V_DEAD_BOTTOM
        self.view_y = max(0, min(max_y, self.view_y))

    def sync_enemies(self) -> None:
        """Copy the shared PLAYERS.PAS globals into Enemies (the player box,
        the view and the Data/Player references).  move_player() and
        init_player() call this; play.py should call it after changing
        x_view/data on its own (PLAY does the same by writing the globals)."""
        en = self.enemies
        en.view_x = self.x_view
        en.view_y = self.view_y
        en.player_x1 = self.player_x1
        en.player_x2 = self.player_x2
        en.player_y1 = self.player_y1
        en.player_y2 = self.player_y2
        en.player_xvel = self.player_xvel
        en.player_yvel = self.player_yvel
        en.player = self.player
        en.data = self.data
        en.player_in_water = self.in_water

    @staticmethod
    def _draw(target: pygame.Surface, x: int, y: int,
              surf: pygame.Surface) -> None:
        target.blit(surf, (x, y))

    @staticmethod
    def _draw_part(target: pygame.Surface, x: int, y: int, y1: int, y2: int,
                   surf: pygame.Surface) -> None:
        """DrawPart: blit bitmap rows y1..y2 at (x, y+row).

        The asm compares the running row index (0..Height-1) against Y1/Y2;
        a Y1 above the bitmap (e.g. -DemoY with DemoY=-2H) draws nothing."""
        h = surf.get_height()
        y1 = max(0, y1)
        y2 = min(y2, h - 1)
        if y1 > y2:
            return
        target.blit(surf, (x, y + y1),
                    area=pygame.Rect(0, y1, surf.get_width(), y2 - y1 + 1))

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
    def _draw_part_x(self, target: pygame.Surface, x: int, y: int,
                     x1: int, x2: int, surf: pygame.Surface) -> None:
        """Column range of _draw_part: blit sprite columns x1..x2 at (x+x1, y)."""
        w = surf.get_width()
        x1 = max(0, x1)
        x2 = min(x2, w - 1)
        if x1 > x2:
            return
        target.blit(surf, (x + x1, y),
                    area=pygame.Rect(x1, 0, x2 - x1 + 1, surf.get_height()))

    def _side_draw_clip(self) -> tuple[int, int, int]:
        """(draw_x, first, last) visible sprite column for the side pipe demos.

        d = demo_x is the penetration depth: IN counts 0->24 (erase, the
        side-plant trick), OUT counts 24->0 (cannon reveal).  The clip plane
        is the mouth column's OUTER edge (the side Mario stands on), so the
        sprite is eaten exactly inside the mouth tile and never overlaps the
        pipe.  IN and OUT share one formula, only d runs the other way.
        """
        d = self.demo_x
        if self.demo in (DM_LEFT_IN_TO_PIPE, DM_RIGHT_OUT_OF_PIPE):
            # mouth on the left, he slides left: visible left edge stays at
            # self.x = the mouth column's right edge + 1
            return self.x - d, d, 2 * W + 3
        # mouth on the right: the plane is the collision-box edge x+W-1
        # (the sprite is 20 wide; the old 2*W+3 pinned the visible edge 4 px
        # INSIDE the mouth column, which read as Mario covering the mouth)
        return self.x + d, 0, W - 1 - d

    def _draw_demo(self) -> None:
        page = self.screen.page
        rec = self.save_screen[self.screen.current]
        surf = self._pic(self._mode(), self.walking_mode, self.direction)
        if self.demo in (DM_LEFT_IN_TO_PIPE, DM_RIGHT_IN_TO_PIPE,
                         DM_LEFT_OUT_OF_PIPE, DM_RIGHT_OUT_OF_PIPE):
            # walking frames; columns are erased/revealed at the mouth edge
            dx, x1, x2 = self._side_draw_clip()
            rec.backgr_addr = self.screen.push_backgr(dx, self.y, W + 4, 2 * H)
            rec.x_pos = dx
            rec.y_pos = self.y
            rec.visible = True
            self._draw_part_x(page, dx, self.y, x1, x2, surf)
            return
        rec.backgr_addr = self.screen.push_backgr(self.x, self.y, W + 4, 2 * H)
        rec.x_pos = self.x
        rec.y_pos = self.y
        rec.visible = True
        if self.demo in (DM_DOWN_IN_TO_PIPE, DM_UP_OUT_OF_PIPE):
            self._draw_part(page, self.x, self.y + self.demo_y, 0,
                            2 * H - self.demo_y - 1, surf)
        elif self.demo in (DM_UP_IN_TO_PIPE, DM_DOWN_OUT_OF_PIPE):
            self._draw_part(page, self.x, self.y + self.demo_y,
                            -self.demo_y, 2 * H, surf)
            self.figures.redraw(page, self.world, self.map_x, self.map_y - 1)
            self.figures.redraw(page, self.world, self.map_x + 1,
                                self.map_y - 1)
        elif self.demo == DM_DEAD:
            self._draw(page, self.x, self.y, surf)

    # ------------------------------------------------------------------ #
    # ghost-house sight (Options.Vision)
    # ------------------------------------------------------------------ #
    def begin_vision(self) -> None:
        """Reset the sight -- level start, restart, and every pipe travel.

        Levels without Options.Vision keep radius 0, which also means the
        renderer never cuts a mask for them.
        """
        self.vision_r = VISION_BASE if self.options.Vision else 0
        self.vision_decay = 0
        self._coins_seen = self.data.Coins[self.player]

    def tick_vision(self) -> None:
        """One frame of sight: a collected coin buys a tile, tiles decay back.

        The coin is read straight off Data.Coins rather than through a hook --
        TmpObj owns that counter and the test harnesses hand-roll their own
        hook stubs, so a new hook method would break them for no gain.
        """
        if not self.vision_r:
            return
        coins = self.data.Coins[self.player]
        if coins != self._coins_seen:
            # TmpObj wraps Coins back to 0 on every 100th coin (an extra life),
            # so a drop still means one was taken.
            self._coins_seen = coins
            self.vision_r = min(VISION_MAX, self.vision_r + 1)
            self.vision_decay = 0
        self.vision_decay += 1
        if self.vision_decay >= VISION_DECAY:
            self.vision_decay = 0
            self.vision_r = max(VISION_BASE, self.vision_r - 1)

    def draw_player(self) -> None:
        if self.demo != DM_NO_DEMO:
            self._draw_demo()
            return
        if self.blinking and (self.blink_counter % 2 != 0):
            return
        rec = self.save_screen[self.screen.current]
        page = self.screen.page
        rec.backgr_addr = self.screen.push_backgr(self.x, self.y, W + 4, 2 * H)
        rec.x_pos = self.x
        rec.y_pos = self.y
        rec.visible = True

        mode = self._mode()
        if self.in_water:
            # only the three swim poses; the walk/jump frames are never used
            # underwater (user's rule).  Star/grow still recolour the frame.
            pose = self.swim_frame
            if self.enemies.star or self.growing:
                n = self.grow_counter + self.star_counter
                diff = (((n & 1) << 4) - (1 if (n & 0xF) < 8 else 0)) & 0xFF
                self._draw(page, self.x, self.y,
                           self._swim_pic(mode, pose, self.direction, diff))
            else:
                self._draw(page, self.x, self.y,
                           self._swim_pic(mode, pose, self.direction))
            self._draw_bubbles(page)
            return
        if mode == MD_FIRE and self.key_space and self.fire_counter < 7:
            self.fire_counter += 1
            # the "kick" pose: lower 21 rows shifted down one, top 7 rows on top
            self._draw_part(page, self.x, self.y + 1, 0, 20,
                            self._pic(MD_FIRE, 1, self.direction))
            self._draw_part(page, self.x, self.y, 21, 2 * H,
                            self._pic(MD_FIRE, 0, self.direction))
        elif self.enemies.star or self.growing:
            n = self.grow_counter + self.star_counter
            diff = (((n & 1) << 4) - (1 if (n & 0xF) < 8 else 0)) & 0xFF
            self._draw(page, self.x, self.y,
                       self._pic(mode, self.walking_mode, self.direction, diff))
        else:
            self._draw(page, self.x, self.y,
                       self._pic(mode, self.walking_mode, self.direction))

    def erase_player(self) -> None:
        self._erase_bubbles()
        rec = self.save_screen[self.screen.current]
        if not rec.visible:
            return
        self.screen.pop_backgr(rec.backgr_addr)
        rec.visible = False

    # ------------------------------------------------------------------ #
    # underwater bubbles (port-only)
    # ------------------------------------------------------------------ #
    def _bubble_surf(self) -> pygame.Surface:
        return self.figures.cache_surface("BUBBLE000", BUBBLE_ART)

    def _erase_bubbles(self) -> None:
        page = self.screen.current
        keep = []
        for b in self.bubbles:
            addr = b["addr"][page]
            if addr != -1:
                self.screen.pop_backgr(addr)
                b["addr"][page] = -1
            if b["dead"]:
                # wait until every page has restored its saved pixels, or a
                # popped bubble leaves a ghost on the other back buffer
                b["dead_ticks"] += 1
                if b["dead_ticks"] <= MAX_PAGE + 2 or any(
                        a != -1 for a in b["addr"]):
                    keep.append(b)
            else:
                keep.append(b)
        self.bubbles = keep

    def _spawn_bubble(self) -> None:
        if len(self.bubbles) >= SWIM_BUBBLES_MAX:
            return
        mouth_x = self.x + (W - 3 if self.direction == DIR_RIGHT else 2)
        self.bubbles.append({
            "x": mouth_x,
            "y": self.y + (H + 7 if self._is_small() else 10),
            "dx": -1 if self.direction == DIR_LEFT else 1,
            "life": 70,
            "addr": [-1] * (MAX_PAGE + 1),
            "dead": False,
            "dead_ticks": 0,
        })

    def _move_bubbles(self) -> None:
        if not self.in_water:
            for b in self.bubbles:
                b["dead"] = True
            return
        for b in self.bubbles:
            if b["dead"]:
                continue
            b["y"] -= 1
            if b["life"] % 3 == 0:
                b["x"] += b["dx"]
            b["life"] -= 1
            if b["life"] <= 0 or b["y"] + 3 <= self.water_y:
                b["dead"] = True

    def _draw_bubbles(self, page: pygame.Surface) -> None:
        surf = self._bubble_surf()
        for b in self.bubbles:
            if b["dead"] or b["y"] + 3 <= self.water_y:
                continue
            addr = self.screen.push_backgr(b["x"], b["y"], 3, 3)
            b["addr"][self.screen.current] = addr
            page.blit(surf, (b["x"], b["y"]))

    # ------------------------------------------------------------------ #
    # demo state machine (DoDemo / StartDemo)
    # ------------------------------------------------------------------ #
    def start_demo(self, dm: int) -> None:
        self.demo = dm
        self.demo_counter1 = 0
        self.demo_counter2 = 0
        self.demo_y = 0
        self.demo_x = 0
        self.below1 = CH_SPACE
        self.below2 = CH_SPACE
        self.at_ch1 = CH_SPACE
        self.at_ch2 = CH_SPACE
        if dm in (DM_DOWN_IN_TO_PIPE, DM_UP_IN_TO_PIPE,
                  DM_DOWN_OUT_OF_PIPE, DM_UP_OUT_OF_PIPE,
                  DM_LEFT_IN_TO_PIPE, DM_RIGHT_IN_TO_PIPE,
                  DM_LEFT_OUT_OF_PIPE, DM_RIGHT_OUT_OF_PIPE):
            self.hooks.start_music("pipe")
        small = int(self._is_small())
        if dm == DM_UP_OUT_OF_PIPE:
            self.demo_y = 2 * H - 9 * small
        elif dm == DM_DOWN_OUT_OF_PIPE:
            self.demo_y = -2 * H
            self.y += H - 7 * small - 2
        elif dm == DM_DEAD:
            self.yvel = -3
            self.hooks.beep(220)
        elif dm in (DM_LEFT_IN_TO_PIPE, DM_RIGHT_IN_TO_PIPE):
            self.direction = DIR_LEFT if dm == DM_LEFT_IN_TO_PIPE else DIR_RIGHT
            self.walking_mode = 0
        elif dm in (DM_LEFT_OUT_OF_PIPE, DM_RIGHT_OUT_OF_PIPE):
            # emerge from the mouth beside the exit cell; start fully erased.
            # y stays (map_y-1)*H for both sizes -- the 28-tall picture frame
            # already carries the small sprite at its bottom rows.
            self.direction = DIR_LEFT if dm == DM_LEFT_OUT_OF_PIPE else DIR_RIGHT
            self.walking_mode = 0
            self.demo_x = 2 * W + 4
            self.x = self.map_x * W
        self.in_pipe = False

    def do_demo(self) -> None:
        self.small = 9 * int(self._is_small())
        small = self.small
        if self.demo in (DM_DOWN_IN_TO_PIPE, DM_UP_OUT_OF_PIPE):
            if self.pipe_code[0] == CH_TAU:
                if not self.passed:
                    self.passed = True
                    self.text_counter = 0
            self.demo_counter1 += 1
            if self.demo_counter1 % 3 == 0:
                if self.demo == DM_DOWN_IN_TO_PIPE:
                    self.demo_y += 1
                    if self.demo_y > 2 * H - small:
                        self.demo_counter2 += 1
                        self.demo_y -= 1
                        if self.demo_counter2 > 10:
                            self.in_pipe = True
                else:
                    self.demo_y -= 1
                    if self.demo_y < 0:
                        self.demo_y += 1
                        self.demo = DM_NO_DEMO
        elif self.demo in (DM_UP_IN_TO_PIPE, DM_DOWN_OUT_OF_PIPE):
            self.demo_counter1 += 1
            if self.demo_counter1 % 3 == 0:
                if self.demo == DM_DOWN_OUT_OF_PIPE:
                    self.demo_y += 1
                    if self.demo_y > -small:
                        self.demo = DM_NO_DEMO
                        self.demo_y -= 1
                else:
                    self.demo_y -= 1
                    if self.demo_y < -2 * H + small:
                        self.demo_counter2 += 1
                        self.demo_y += 1
                        if self.demo_counter2 > 10:
                            self.in_pipe = True
        elif self.demo in (DM_LEFT_IN_TO_PIPE, DM_RIGHT_IN_TO_PIPE):
            self.demo_counter1 += 1
            self.walking_mode = (self.demo_counter1 // 8) % 2
            if self.demo_counter1 % 3 == 0:
                self.demo_x += 1
                if self.demo_x >= 2 * W + 4:
                    self.demo_x = 2 * W + 4
                    self.demo_counter2 += 1
                    if self.demo_counter2 > 10:
                        self.in_pipe = True
        elif self.demo in (DM_LEFT_OUT_OF_PIPE, DM_RIGHT_OUT_OF_PIPE):
            self.demo_counter1 += 1
            self.walking_mode = (self.demo_counter1 // 8) % 2
            if self.demo_counter1 % 3 == 0:
                self.demo_x -= 1
                if self.demo_x < 0:
                    self.demo = DM_NO_DEMO
                    self.walking_mode = 0
        elif self.demo == DM_DEAD:
            self.demo_counter1 += 1
            if self.demo_counter1 % 7 == 0:
                self.yvel += 1
            self.y += self.yvel
            if self.y > self.world.rows * H:
                self.game_done = True

    # ------------------------------------------------------------------ #
    # pipe checks
    # ------------------------------------------------------------------ #
    def check_pipe_below(self) -> None:
        if self.xvel != 0 or self.yvel != 0 or self.y % H != 0:
            return
        mo = pmod(self.x, W)
        if not (4 <= mo <= W - 4):
            return
        if (self.below1 != CH_0 or self.below2 != CH_1
                or not (CH_ALPHA <= self.at_ch1 <= CH_TAU)
                or not (CH_ALPHA <= self.at_ch2 <= CH_INTER)):
            return
        self.pipe_code[0] = self.at_ch1
        self.pipe_code[1] = self.at_ch2
        self.pipe_side = False
        self.start_demo(DM_DOWN_IN_TO_PIPE)

    def check_pipe_above(self, c1: int, c2: int) -> None:
        mo = pmod(self.x, W)
        if not (4 <= mo <= W - 4):
            return
        if c1 != CH_0 or c2 != CH_1:
            return
        self.map_x = idiv(self.x, W)
        self.map_y = idiv(self.y, H) + 1
        if not (CH_ALPHA <= self.world.get(self.map_x, self.map_y) <= CH_TAU):
            return
        if not (CH_ALPHA <= self.world.get(self.map_x + 1, self.map_y)
                <= CH_INTER):
            return
        self.pipe_code[0] = self.world.get(self.map_x, self.map_y)
        self.pipe_code[1] = self.world.get(self.map_x + 1, self.map_y)
        self.pipe_side = False
        self.start_demo(DM_UP_IN_TO_PIPE)

    def check_pipe_side(self, direction: int) -> None:
        """Port-only entry for the horizontal pipe, the mirror of
        CheckPipeBelow: the collision box must touch the mouth pair, the
        entry axis is grid-aligned (X mod W = 0 <-> Y mod H = 0) and the
        player holds the key towards the pipe."""
        if self.xvel != 0 or self.x % W != 0:
            return
        if self.in_water:
            # rising or sinking past the mouth row; allow a couple of pixels
            # and snap onto the pipe grid when he pushes towards the mouth
            snap = round(self.y / H) * H
            if abs(self.y - snap) > 2:
                return
            self.y = snap
            self.yvel = 0
        elif self.yvel != 0 or self.y % H != 0:
            return
        map_x = idiv(self.x, W)
        map_y = idiv(self.y, H) + 1
        w = self.world
        front = map_x + (1 if direction == DIR_RIGHT else -1)
        if (w.get(front, map_y) != CH_HP_MOUTH_B
                or w.get(front, map_y - 1) != CH_HP_MOUTH_T):
            return
        if not (CH_ALPHA <= w.get(map_x, map_y) <= CH_TAU):
            return
        if not (CH_ALPHA <= w.get(map_x, map_y - 1) <= CH_INTER):
            return
        self.pipe_code[0] = w.get(map_x, map_y)
        self.pipe_code[1] = w.get(map_x, map_y - 1)
        self.map_x = map_x
        self.map_y = map_y
        self.pipe_side = True
        self.start_demo(DM_LEFT_IN_TO_PIPE if direction == DIR_LEFT
                        else DM_RIGHT_IN_TO_PIPE)

    # ------------------------------------------------------------------ #
    # collision (Check / CheckFall / CheckJump)
    # ------------------------------------------------------------------ #
    def _check_fall(self, hold1: bool, hold2: bool, new_x1: int,
                    new_x2: int, new_y: int, new_ch1: int,
                    new_ch2: int) -> None:
        hk = self.hooks
        w = self.world
        if not (hold1 or hold2):
            if new_ch1 == CH_COIN:
                hk.hit_coin(new_x1 * W, new_y * H, False)
            if new_ch2 == CH_COIN:
                hk.hit_coin(new_x2 * W, new_y * H, False)
            if self.in_water:
                if self.water_tick % SWIM_SINK_DELAY == 0:
                    self.yvel += 1
                if self.yvel > SWIM_SINK_MAX:
                    self.yvel = SWIM_SINK_MAX
            else:
                if self.counter % JUMP_DELAY == 0:
                    self.yvel += 1
                if self.yvel > MAX_Y_VEL:
                    self.yvel = MAX_Y_VEL
        else:
            if new_ch1 == CH_EQ or new_ch2 == CH_EQ:
                self.enemies.cd_hit = 1
            self.y = (idiv(self.y + self.yvel + 1 + HSAFE, H) - SAFE) * H
            self.yvel = 0
            self.status = ST_ON_THE_GROUND
            self.jumped = True
            if new_ch1 == CH_K or new_ch2 == CH_K:
                hk.start_music("note")
                if new_ch1 == CH_K:
                    hk.bump_block(new_x1 * W, new_y * H)
                    hk.remove(new_x1 * W, new_y * H, W, H, TP_NOTE)
                    w.set(new_x1, new_y, CH_K)
                if new_ch2 == CH_K:
                    hk.bump_block(new_x2 * W, new_y * H)
                    hk.remove(new_x2 * W, new_y * H, W, H, TP_NOTE)
                    w.set(new_x2, new_y, CH_K)
                self.counter = 0
                self.status = ST_JUMPING
                self.jumped = False
                self.high_jump = True
                self.yvel = -5
                self.hit_enemy = True
            # NOTE: the original then runs a `case Mo of` that rewrites
            # Ch/NewX2, but neither is read afterwards — dead code.

    def _check_jump(self) -> None:
        en = self.enemies
        if self.in_water:
            # no jump state underwater: Alt is the swim modifier instead
            en.cd_enemy = 0
            return
        if en.cd_enemy != 0:
            self.hit_enemy = True
            self.jumped = False
        if not self.jumped:
            if self.key_alt or self.hit_enemy:
                self.counter = 0
                self.status = ST_JUMPING
                self.high_jump = ((abs(self.xvel) == 2)
                                  or (self.hit_enemy and self.key_alt))
                self.yvel = (-JUMP_VEL
                             - 2 * int(self.hit_enemy and self.key_alt)
                             - int(en.turbo))
        en.cd_enemy = 0

    def _hit_block(self, ch: int, new_x2: int, new_y: int,
                   small: bool) -> int:
        """The `case Ch of '?','$','J','K'` branch of Check().  Returns the
        (possibly changed) Ch."""
        hk = self.hooks
        w = self.world
        en = self.enemies
        mo = 0
        above = w.get(new_x2, new_y - 1)
        if CH_ALPHA <= above <= CH_GAMMA:
            w.set(new_x2, new_y, CH_QMARK)
            ch = CH_QMARK
        elif above == CH_INTER:
            w.set(new_x2, new_y, CH_K)
            ch = CH_K
        elif (not small) and ch == CH_J:
            hk.break_block(new_x2, new_y)
            hk.add_score(10)
            mo = 1
        if mo == 0:
            hk.bump_block(new_x2 * W, new_y * H)
            hk.beep(110)

        above = w.get(new_x2, new_y - 1)
        if above == CH_SPACE or CH_PI <= above <= CH_INF:
            if ch not in (CH_J, CH_K):
                hk.hit_coin(new_x2 * W, new_y * H, True)
                if w.get(new_x2, new_y - 1) != CH_SPACE:
                    w.set(new_x2, new_y - 1,
                          (w.get(new_x2, new_y - 1) + 1) & 0xFF)
                    if w.get(new_x2, new_y) == CH_DOLLAR:
                        hk.remove(new_x2 * W, new_y * H, W, H, 2)
                        w.set(new_x2, new_y, CH_QMARK)
        elif above == CH_ALPHA:
            if self._mode() == MD_SMALL:
                en.new_enemy(TP_RISING_CHAMP, 0, new_x2, new_y, 0, -1, 2)
            else:
                en.new_enemy(TP_RISING_FLOWER, 0, new_x2, new_y, 0, -1, 2)
        elif above == CH_SS:
            en.new_enemy(TP_RISING_LIFE, 0, new_x2, new_y, 0, -1, 2)
        elif above == CH_GAMMA:
            en.new_enemy(TP_RISING_STAR, 0, new_x2, new_y, 0, -1, 1)
        elif above == CH_COIN:
            hk.hit_coin(new_x2 * W, (new_y - 1) * H, False)
        elif above == CH_PHI:
            en.new_enemy(TP_RISING_CHAMP, 1, new_x2, new_y, 0, -1, 2)
        en.hit_above(new_x2, new_y - 1)

        if ch == CH_K:
            hk.remove(new_x2 * W, new_y * H, W, H, TP_NOTE)
            w.set(new_x2, new_y, CH_K)
        elif ch != CH_J:
            if not (CH_PI <= w.get(new_x2, new_y - 1) <= CH_INF):
                hk.remove(new_x2 * W, new_y * H, W, H, 1)
                w.set(new_x2, new_y, CH_AT)
        return ch

    def check(self) -> None:
        w = self.world
        en = self.enemies
        hk = self.hooks
        new_ch1 = new_ch2 = new_ch3 = CH_SPACE
        small = self._is_small()

        side = (W - 1) if self.xvel > 0 else 0
        new_x1 = idiv(self.x + side, W)
        new_x2 = idiv(self.x + side + self.xvel, W)

        if new_x1 != new_x2:
            y1 = idiv(self.y + HSAFE + 4, H) - SAFE
            y2 = idiv(self.y + HSAFE + H, H) - SAFE
            y3 = idiv(self.y + HSAFE + 2 * H - 1, H) - SAFE
            new_ch1 = w.get(new_x2, y1)
            new_ch2 = w.get(new_x2, y2)
            new_ch3 = w.get(new_x2, y3)
            if new_ch3 == CH_COIN:
                hk.hit_coin(new_x2 * W, y3 * H, False)
            if new_ch2 == CH_COIN:
                hk.hit_coin(new_x2 * W, y2 * H, False)
            if new_ch2 == CH_Z:
                en.turbo = True
            if not small and new_ch1 == CH_COIN:
                hk.hit_coin(new_x2 * W, y1 * H, False)
            hold1 = (new_ch1 in CAN_HOLD_YOU) and (not small)
            hold2 = new_ch2 in CAN_HOLD_YOU
            hold3 = new_ch3 in CAN_HOLD_YOU
            if hold1 or hold2 or hold3:
                self.xvel = 0
                self.walking_mode = 0

        new_x1 = idiv(self.x + self.xvel, W)
        new_x2 = idiv(self.x + self.xvel + W - 1, W)

        if en.cd_enemy != 0:
            self._check_jump()

        if self.status == ST_JUMPING:
            new_y = (idiv(self.y + 1 + 4 + (H - 1 - 4) * int(small)
                          + self.yvel + HSAFE, H) - SAFE)
        else:
            new_y = idiv(self.y + 1 + 2 * H + self.yvel + HSAFE, H) - SAFE

        new_ch1 = w.get(new_x1, new_y)
        new_ch2 = w.get(new_x2, new_y)
        new_ch3 = w.get(idiv(self.x + self.xvel + W // 2, W), new_y)
        hold1 = new_ch1 in HOLD_OR_STAND
        hold2 = new_ch2 in HOLD_OR_STAND
        hold3 = new_ch3 in HOLD_OR_STAND

        if self.status == ST_FALLING:
            self._check_fall(hold1, hold2, new_x1, new_x2, new_y,
                             new_ch1, new_ch2)
            return

        if self.status == ST_ON_THE_GROUND:
            if en.cd_lift == 0:
                if not (hold1 or hold2):
                    self.status = ST_FALLING
                    if abs(self.xvel) < 2:
                        self.y += 1
                elif new_ch1 == CH_K or new_ch2 == CH_K:
                    self._check_fall(hold1, hold2, new_x1, new_x2, new_y,
                                     new_ch1, new_ch2)
                else:
                    if self.xvel == 0:
                        self.below1 = new_ch1
                        self.below2 = new_ch2
                        self.map_x = new_x1
                        self.map_y = new_y - 1
                        self.at_ch1 = w.get(self.map_x, self.map_y)
                        self.at_ch2 = w.get(self.map_x + 1, self.map_y)
                        mo = pmod(self.x, W)
                        if not hold1 and 1 <= mo <= 5:
                            self.xvel -= 1
                        if not hold2 and W - 5 <= mo <= W - 1:
                            self.xvel += 1
                    self._check_jump()
            else:
                self.yvel = self.player_yvel
                self._check_jump()
            return

        # ---- ST_JUMPING ------------------------------------------------ #
        hold1 = new_ch1 in HOLD_OR_HIDDEN
        hold2 = new_ch2 in HOLD_OR_HIDDEN
        hold3 = new_ch3 in HOLD_OR_HIDDEN
        hit = hold1 or hold2
        mo = pmod(self.x + self.xvel, W)
        if hit:
            if (1 <= mo <= 4 or W - 4 <= mo <= W - 1) and not hold3:
                if not (new_ch1 in HOLD_OR_HIDDEN
                        and new_ch2 in HOLD_OR_HIDDEN):
                    hit = False
                if mo < W // 2 and new_ch2 not in HOLD_OR_HIDDEN:
                    self.x -= mo
                elif mo >= W // 2 and new_ch1 not in HOLD_OR_HIDDEN:
                    self.x += W - mo

        if not hit:
            if new_ch1 == CH_COIN:
                hk.hit_coin(new_x1 * W, new_y * H, False)
            if new_ch2 == CH_COIN:
                hk.hit_coin(new_x2 * W, new_y * H, False)
            if (self.counter % (JUMP_DELAY + int(self.high_jump)) == 0
                    or ((not self.key_alt) and (not self.hit_enemy))):
                self.yvel += 1
            if self.yvel >= 0:
                self.yvel = 0
                self.status = ST_FALLING
            return

        ch = 0
        if 0 <= mo <= W // 2 - 1:
            if new_ch1 in HOLD_OR_HIDDEN:
                ch = new_ch1
                new_x2 = new_x1
            else:
                ch = new_ch2
        elif W // 2 <= mo <= W - 1:
            ch = new_ch2
            if ch not in HOLD_OR_HIDDEN:
                ch = new_ch1
                new_x2 = new_x1

        if ch == CH_EQ:
            en.cd_hit = 1
        elif ch in (CH_0, CH_1):
            if self.key_up:
                self.check_pipe_above(new_ch1, new_ch2)
        elif ch in (CH_QMARK, CH_DOLLAR, CH_J, CH_K):
            ch = self._hit_block(ch, new_x2, new_y, small)
        else:
            hk.beep(30)

        if (ch != CH_J) or (self._mode() == MD_SMALL):
            self.yvel = 0
            self.status = ST_FALLING
        if ch == CH_K:
            self.yvel = 3

    # ------------------------------------------------------------------ #
    # water state (port-only)
    # ------------------------------------------------------------------ #
    def _update_water_state(self) -> None:
        """Recompute in_water from the body centre and the level waterline."""
        was = self.in_water
        wy = self.options.water_y
        self.water_y = wy
        if wy < 0:
            self.in_water = False
            self.swim_hover = False
            self.swim_hold = 0
            for b in self.bubbles:
                b["dead"] = True
            return
        self.in_water = self.y + 3 * H // 2 >= wy
        if self.in_water and not was:
            # entering: kill most of the fall, release a little air, and
            # charge the two-frame swim delay from zero
            if self.yvel > 1:
                self.yvel = 1
            self.swim_hold = 0
            for _ in range(3):
                self._spawn_bubble()
            self.swim_counter = 0
            self.swim_frame = 0
        elif not self.in_water:
            self.swim_hover = False
            self.swim_hold = 0
            for b in self.bubbles:
                b["dead"] = True

    def _water_vertical(self, en) -> None:
        """Alt alone is the swim stroke: holding it rises at a constant
        SWIM_UP, releasing it lets the reduced water gravity sink Mario.  No
        Up/Down key is needed.  Runs before check() so head collisions still
        see the stroke (and a bumped '?' block still works underwater)."""
        if not self.in_water or en.cd_lift != 0:
            self.swim_hover = False
            return
        active = self.key_alt and self.swim_hold >= SWIM_INPUT_DELAY
        if self.swim_blocked > 0:
            self.swim_blocked -= 1
        if not active:
            self.swim_hover = False
            if self.status != ST_ON_THE_GROUND and self.yvel >= 0:
                self.status = ST_FALLING
            return
        if self.swim_blocked == 0:
            self.status = ST_JUMPING
            self.yvel = -SWIM_UP
            self.swim_hover = False
        else:
            self.swim_hover = False
            if self.status != ST_ON_THE_GROUND:
                self.status = ST_FALLING

    def _water_after_check(self, en) -> None:
        """Re-assert the constant swim speed after Check()'s gravity/head
        handling, and keep the body from leaving the water upwards."""
        if not self.in_water or en.cd_lift != 0:
            return
        active = self.key_alt and self.swim_hold >= SWIM_INPUT_DELAY
        if active and self.swim_blocked == 0:
            if self.status == ST_JUMPING:
                self.yvel = -SWIM_UP
            elif self.yvel == 0:
                # a ceiling/block stopped the stroke; do not repeat-bump
                self.swim_blocked = 4
        if self.water_y >= 0:
            top = self.y + (H if self._is_small() else 0)
            if top < self.water_y:
                self.y = self.water_y - (H if self._is_small() else 0)
                if self.yvel < 0:
                    self.yvel = 0
                self.swim_blocked = 4

    # ------------------------------------------------------------------ #
    # MovePlayer
    # ------------------------------------------------------------------ #
    def move_player(self, keyboard) -> None:
        """One frame of PLAYERS.MovePlayer.  ``keyboard`` supplies the
        kbLeft/kbRight/... booleans (engine.input.Keyboard or a test double)."""
        en = self.enemies
        hk = self.hooks
        w = self.world

        if self.in_pipe:
            if w.get(self.map_x, self.map_y + 1) == CH_0:
                self.start_demo(DM_UP_OUT_OF_PIPE)
            elif w.get(self.map_x, self.map_y - 1) == CH_0:
                self.start_demo(DM_DOWN_OUT_OF_PIPE)
            elif (w.get(self.map_x - 1, self.map_y) == CH_HP_MOUTH_B
                    and w.get(self.map_x - 1, self.map_y - 1) == CH_HP_MOUTH_T):
                self.start_demo(DM_RIGHT_OUT_OF_PIPE)
            elif (w.get(self.map_x + 1, self.map_y) == CH_HP_MOUTH_B
                    and w.get(self.map_x + 1, self.map_y - 1) == CH_HP_MOUTH_T):
                self.start_demo(DM_LEFT_OUT_OF_PIPE)
            return

        if en.cd_champ != 0:
            if self._mode() == MD_SMALL:
                self.data.Mode[self.player] = MD_LARGE
                self.growing = True
                self.grow_counter = 0
            hk.start_music("grow")
            en.cd_champ = 0
        if en.cd_life != 0:
            en.cd_life = 0
            hk.add_life()
        if en.cd_flower != 0:
            self.data.Mode[self.player] = MD_FIRE
            self.fired = True
            self.fire_counter = 0
            hk.start_music("grow")
            self.growing = True
            self.grow_counter = 0
            en.cd_flower = 0

        if not self.blinking and not en.star and not self.growing:
            if en.cd_hit != 0:
                if self._mode() == MD_SMALL:
                    self.blink_counter = 0
                    self.blinking = True
                    self.start_demo(DM_DEAD)
                    hk.start_music("dead")
                    return
                self.data.Mode[self.player] = MD_SMALL
                self.blink_counter = 0
                self.blinking = True
                hk.start_music("hit")
            en.cd_hit = 0
        else:
            en.cd_hit = 0

        if self.blinking:
            self.blink_counter += 1
            if self.blink_counter >= BLINK_TIME:
                self.blinking = False

        if en.cd_star != 0:
            hk.start_music("star")
            self.star_counter = 0
            en.star = True

        # Test switch: hold the original star state without its timer, so
        # the player stays immune (and still kills on contact) indefinitely.
        if self.invincible and not en.star:
            en.star = True
            self.star_counter = 0

        if en.star:
            self.star_counter += 1
            if self.star_counter >= STAR_TIME and not self.invincible:
                en.star = False
            if self.star_counter % 3 == 0:
                small = int(self._mode() == MD_SMALL)
                hk.start_glitter(self.x, self.y + 11 * small, W,
                                 H + 3 + 11 * (1 - small))
            en.cd_star = 0

        if self.growing:
            self.grow_counter += 1
            if self.grow_counter > GROW_TIME:
                self.growing = False

        self.counter += 1
        if self.xvel == 0 and self.yvel == 0:
            self.counter = 0

        old_dir = self.direction
        old_xvel = self.xvel

        last_left = self.key_left
        last_right = self.key_right
        self.key_left = bool(keyboard.left)
        self.key_right = bool(keyboard.right)
        self.key_up = bool(keyboard.up)
        self.key_down = bool(keyboard.down)
        self.key_alt = bool(keyboard.alt)
        self.key_ctrl = bool(keyboard.ctrl)
        self.key_space = bool(keyboard.space)

        # water state and input latency (port-only)
        self.swim_hold = self.swim_hold + 1 if self.key_alt else 0
        self._update_water_state()
        check_x = (self.counter % (SWIM_ACCEL_DELAY if self.in_water else SLIP)
                   == 0)

        if self.key_right and (not last_right) and self.direction == DIR_LEFT:
            old_dir = DIR_RIGHT
            old_xvel = -self.xvel
        if self.key_left and (not last_left) and self.direction == DIR_RIGHT:
            old_dir = DIR_LEFT
            old_xvel = -self.xvel

        if self.fired and not self.key_space:
            self.fired = False

        if (self.key_space and not self.fired
                and self._mode() == MD_FIRE):
            self.fire_counter = 0
            en.new_enemy(
                TP_FIREBALL, 0,
                idiv(self.x, W) + self.direction,
                idiv(self.y + H, H),
                10 * (-1 + 2 * self.direction),
                3 + 3 * (int(self.key_down) - int(self.key_up)),
                2)
            self.fired = True

        if en.cd_lift != 0:
            # MoveEnemies wrote the lift's box/velocity into the shared
            # PlayerY1/PlayerXVel/PlayerYVel globals.  Here those live on the
            # Enemies object, so take them from there and mirror them into
            # this unit's copies: the rest of MovePlayer (MaxSpeed, the walk
            # reset, the view blocking) reads them as the globals the
            # original does.
            self.player_y1 = en.player_y1
            self.player_y2 = en.player_y2
            self.player_xvel = en.player_xvel
            self.player_yvel = en.player_yvel
            self.y = self.player_y1
            self.xvel = self.player_xvel
            self.yvel = self.player_yvel
            self.status = ST_ON_THE_GROUND
        if en.cd_stop_jump != 0:
            self.jumped = True
            en.cd_stop_jump = 0

        if self.jumped and not self.key_alt:
            self.jumped = False

        if self.in_water:
            # no running underwater: Ctrl/turbo are ignored, only a lift may
            # carry a faster step
            boost = 0
            lift = abs(en.cd_lift * self.player_xvel)
            max_speed = SWIM_MAX_SPEED + lift
            min_speed = -SWIM_MAX_SPEED - lift
        else:
            boost = int(self.key_ctrl) + int(en.turbo)
            lift = abs(en.cd_lift * self.player_xvel)
            max_speed = MAX_SPEED - 1 + boost + lift
            min_speed = -(MAX_SPEED - 1) - boost - lift

        if self.key_left:
            if self.xvel > min_speed:
                if check_x or en.cd_lift != 0:
                    self.xvel -= 1 + int((en.cd_lift != 0) and self.key_ctrl)
            else:
                self.xvel = min_speed
            self.direction = int(self.xvel > 0)
            if self.x + self.xvel < 0:
                self.xvel = -self.x
        elif self.xvel < 0 and check_x and en.cd_lift == 0:
            self.xvel += 1

        if self.key_right:
            if self.xvel < max_speed:
                if check_x or en.cd_lift != 0:
                    self.xvel += 1 + int((en.cd_lift != 0) and self.key_ctrl)
            else:
                self.xvel = max_speed
            self.direction = int(self.xvel >= 0)
        elif self.xvel > 0 and check_x and en.cd_lift == 0:
            self.xvel -= 1

        if self.key_left and self.key_right:
            self.direction = old_dir
            self.xvel = old_xvel

        if self.y + self.yvel >= self.world.rows * H:
            self.game_done = True
            hk.start_music("dead")

        if self.status == ST_ON_THE_GROUND:
            self.hit_enemy = False

        self._water_vertical(en)
        self.check()
        self._water_after_check(en)

        if self.in_water:
            swimming = self.key_alt and self.swim_hold >= SWIM_INPUT_DELAY
            moving = (self.xvel != 0 or swimming
                      or (self.status != ST_ON_THE_GROUND
                          and self.yvel != 0))
            if moving:
                self.swim_counter += 1
                self.swim_frame = 1 + (self.swim_counter // SWIM_ANIM_DIV) % 3
            else:
                self.swim_frame = 0
            self.water_tick += 1
            if swimming:
                self.bubble_timer += 1
                if self.bubble_timer % SWIM_BUBBLE_PERIOD == 0:
                    self._spawn_bubble()
            self._move_bubbles()

        if self.status == ST_ON_THE_GROUND and self.yvel == 0:
            if (self.xvel == 0
                    or (en.cd_lift != 0 and self.xvel == self.player_xvel)):
                self.walking_mode = 0
                self.walk_count = 0
            else:
                self.walk_count = (self.walk_count + 1) & 0xF
                self.walking_mode = int(self.walk_count < 8)
        elif self.yvel < 0:
            self.walking_mode = 2
        else:
            self.walking_mode = 3

        if self.key_down:
            self.check_pipe_below()
        elif self.key_left:
            self.check_pipe_side(DIR_LEFT)
        elif self.key_right:
            self.check_pipe_side(DIR_RIGHT)

        if (self.in_water and self.status == ST_ON_THE_GROUND
                and self.xvel != 0 and en.cd_lift == 0):
            # bottom walking is slower than the ground's 1 px/frame: two of
            # every three whole steps
            self.water_walk_phase = (self.water_walk_phase + 1) % 3
            if self.water_walk_phase != 2:
                self.x += self.xvel
        else:
            self.water_walk_phase = 0
            self.x += self.xvel
        self.y += self.yvel

        # ---- scrolling ------------------------------------------------- #
        old_x_view = self.x_view
        self.x_view = (self.x_view - int(bool(keyboard.left_shift))
                       + int(bool(keyboard.right_shift)))
        if self.x + W + SCROLL_AT > self.x_view + 320:
            self.x_view = self.x + W + SCROLL_AT - 320
        if self.x < self.x_view + SCROLL_AT:
            self.x_view = self.x - SCROLL_AT
        limit = MAX_SPEED + int(en.turbo)
        if self.x_view - old_x_view > limit:
            self.x_view = old_x_view + limit
        if self.x_view - old_x_view < -limit:
            self.x_view = old_x_view - limit
        if self.x_view < 0:
            self.x_view = 0
            if self.x < 0:
                self.x = 0
        if self.x_view > (self.options.XSize - NH) * W:
            self.x_view = (self.options.XSize - NH) * W
        if self.x_view < old_x_view:
            col = idiv(self.x_view, W)
            if w.get(col, self.world.rows) == CH_LEFT_MARK:
                if w.get(col, pascal_round_div(self.player_y1, H)) != CH_SPACE:
                    self.x_view = old_x_view
        if self.x_view > old_x_view:
            col = idiv(self.x_view - 1, W) + NH
            if w.get(col, self.world.rows) == CH_RIGHT_MARK:
                if w.get(col, pascal_round_div(self.player_y1, H)) != CH_SPACE:
                    self.x_view = old_x_view

        self._update_view_y()

        self.player_x1 = self.x + self.xvel
        self.player_x2 = self.player_x1 + W - 1
        self.player_y1 = self.y
        if self._mode() == MD_SMALL:
            self.player_y1 = self.y + H
        self.player_y2 = self.y + 2 * H - 1
        self.player_xvel = self.xvel
        self.player_yvel = self.yvel

        if en.cd_lift != 0:
            self.player_yvel += 2 - self.yvel
            en.cd_lift = 0

        self.sync_enemies()
