"""Port of PLAY.PAS — the level loop.

``PlayWorld`` owns one attempt at a level: it reads the two level maps
(``Map1``/``Map2``, the second one behind the sub-level pipes), builds and
redraws both video pages, then runs PLAY.PAS's frame loop:

    hide all objects -> move -> redraw scrolled tiles -> draw all objects
    -> ShowPage -> palette animation -> music

Music is reached through ``MusicHooks`` (empty by default) so ``music.py`` can
be dropped in later without touching the loop.

Known port-only detail: pages store RGB, while the original animated the DAC.
Palette changes therefore mark the frame dirty and the static layer is drawn
again on the next frame (``_repaint``), which reproduces what the DAC did.
"""
from __future__ import annotations

import random

import pygame

from ..engine.input import Keyboard
from ..engine.vga import (MAX_PAGE, VIR_SCREEN_HEIGHT, VGAScreen, VgaConfig)
from .assets import load_assets
from .backgr import BackGr
from .blocks import Blocks
from .buffers import (MD_FIRE, MD_LARGE, MD_SMALL, NH, NV, PLAYER_NAME, TILE_H,
                      TILE_W, GameData, WorldMap, WorldOptions)
from .enemies import (TP_BLOCK_LIFT, TP_CHAMP,
                      TP_CHIBIBO, TP_FIREBALL, TP_FIRE_PLANT,
                      TP_FIRE_PLANT_AIM, TP_FLOWER, TP_FLY_GREEN, TP_FLY_RED,
                      TP_KOOPA, TP_LIFE, TP_PLANT_FIRE, TP_RED,
                      TP_RISING_CHAMP, TP_RISING_FLOWER, TP_RISING_LIFE,
                      TP_RISING_STAR, TP_RUNNING_KOOPA, TP_SLEEPING_KOOPA,
                      TP_STAR, TP_VERT_FISH, TP_VERT_FIREBALL, TP_VERT_PLANT,
                      TP_WAKING_KOOPA, Enemies)
from .figures import Figures
from .glitter import Glitter
from .music import Music, MusicHooks
from .palettes import (PE_BLACK_WHITE, PE_EGA_MODE, PE_NO_EFFECT, Palettes)
from .players import (CH_ALPHA, CH_COIN, CH_SS, CH_TAU, DM_NO_DEMO, Player,
                      PlayerHooks, VISION_BASE)
from .stars import Stars
from .status import Status
from .tmpobj import TP_COIN, TempObjects
from . import enemies as enemies_unit
from .txt import BOLD, SHADOW, Txt
from .webangelo import AngeloObjects

W = TILE_W
H = TILE_H

TIME_START = 301 * 32
# Port-only switch (user 2026-10-25): all levels are untimed for now.  The
# original countdown/TIME UP code stays intact; set this to False to get it
# back.  Test rooms and WEBANGELO already carry ``no_time`` in their JSON.
NO_TIME_LIMIT = True

# Things that carry their own light in a Vision level, so they stay visible one
# tile *outside* Mario's sight circle: the power-ups, the star, his own thrown
# fireballs, and the fireball that spits out of the lava (TP_VERT_FIREBALL).
# The flame shot by the fire-spitting plant (TP_PLANT_FIRE) is here too: fire
# must read as lit, not as a black blob crossing the room (user, 2026-10-25).
# The piranha-plant family -- the *bodies* -- is deliberately NOT here; the
# user wants them read as ordinary enemies, dark like the pipes they sit in.
#
# ENEMY TYPES ONLY.  TMPOBJ.PAS numbers its own pool from 1 as well, and the two
# namespaces overlap hard (TP_COIN == TP_CHIBIBO == 2, TP_HIT == TP_FLAT_CHIBIBO
# == 3, TP_FIRE == TP_DEAD_CHIBIBO == 4, TP_NOTE == TP_RISING_CHAMP == 5), so a
# temp-object constant in this set silently lights an unrelated walker -- which
# is exactly how the plain Chibibo ended up with a halo.  The collected coin is
# picked up from the temp pool in _vision_lights() instead.
VISION_GLOW_TYPES = frozenset({
    TP_LIFE, TP_RISING_LIFE, TP_CHAMP, TP_RISING_CHAMP,
    TP_FLOWER, TP_RISING_FLOWER, TP_STAR, TP_RISING_STAR,
    TP_FIREBALL, TP_VERT_FIREBALL, TP_PLANT_FIRE,
})
# Guard the namespace mix-up rather than trusting comments: TMPOBJ numbers its
# own pool from 1 too, so importing one of its constants here would silently
# light whatever enemy happens to share the number.  The invariant is "every
# member is a type enemies.py actually defines" -- note that the reverse is not
# checkable, because a legitimate enemy number can coincide with a TMPOBJ one
# (TP_RISING_CHAMP == TP_NOTE == 5) without doing any harm: the temp pool is
# matched against TP_COIN only, in _vision_lights().
_ENEMY_TP = {v for k, v in vars(enemies_unit).items()
             if k.startswith("TP_") and isinstance(v, int)}
assert VISION_GLOW_TYPES <= _ENEMY_TP, "the glow set holds non-enemy types"
assert TP_COIN not in VISION_GLOW_TYPES, "the coin lights from the temp pool"

# A light's own radius, in tiles.  r = 1 means the diamond joins the centroids
# of the four tiles edge-adjacent to the light's own tile.  This is a CONSTANT,
# not Mario's radius plus one: he grows when he collects a coin, and deriving
# the glow from his live radius made every torch, coin and mechanism in the room
# brighten with him, which is absurd (the user's words).  The rotating firebar
# applies this to each ball separately and unions them.
VISION_GLOW_RADIUS = 1

# Where each light's own centre sits inside its 20x14 box, measured from the
# opaque pixels of the sprite actually drawn (see _vision_lights).  The light
# must use Mario's standard -- the diamond is centred on the thing itself -- so
# an object whose art does not fill the box cannot just take (10, 7).
#   CHAMP000 / LIFE000 / FLOWER000 / STAR000 / FISH001 / COIN000: 20x14 art,
#     solid centre (9.5, 6.5) -- the default is within a pixel, so they keep it.
#   FIRE000 / FIRE001 (Mario's fireball, and the plant's shot which reuses the
#     same art): the art is only 12x7 and its solid pixels span x 0..8, y 0..6,
#     centre (4, 3).  The default put the light 6 px right and 4 px below the
#     flame.
VISION_LIGHT_CENTRE = {
    TP_FIREBALL: (4, 3),
    TP_PLANT_FIRE: (4, 3),
}
VISION_LIGHT_DEFAULT = (W // 2, H // 2)
SCREEN_WIDTH = 320

# Enemy types that count as "still alive" for a battle_room target.  Dying,
# flat and dead types are deliberately absent.
BATTLE_ALIVE_TPS = frozenset({
    TP_CHIBIBO, TP_RED, TP_KOOPA, TP_SLEEPING_KOOPA, TP_WAKING_KOOPA,
    TP_RUNNING_KOOPA, TP_VERT_FISH, TP_VERT_PLANT, TP_FLY_GREEN, TP_FLY_RED,
})

# unit-level state (PLAY.PAS interface variables)
STAT = True             # Play.Stat, set from MARIO.CFG (SLine); default on
SHOW_RETRACE = False
CHEATS_USED = 0
GOD_MODE = False        # test switch: keep the star state permanently on
# Test switch for the combat states that do NOT need the star: FORCE_MODE pins
# Data.Mode at every level start (--big = large, no fireballs; --fire = the
# fireball form).  None leaves the level's own start_fire rule in charge.
FORCE_MODE = None


class GameHooks(PlayerHooks):
    """Routes PLAYERS/ENEMIES/TMPOBJ call-outs to the ported units."""

    def __init__(self, play: "PlayWorld") -> None:
        self.play = play

    def add_score(self, n: int) -> None:
        self.play.level_score += n

    def add_life(self) -> None:
        self.play.tmp.add_life()

    def beep(self, freq: int) -> None:
        self.play.music.beep(freq)

    def start_music(self, name: str) -> None:
        self.play.music.start_music(name)

    def new_temp_obj(self, tp, x, y, xv, yv, wid, ht) -> None:
        self.play.tmp.new_temp_obj(tp, x, y, xv, yv, wid, ht)

    def break_block(self, map_x: int, map_y: int) -> None:
        self.play.tmp.break_block(map_x, map_y)

    def hit_coin(self, x: int, y: int, throw_up: bool) -> None:
        self.play.tmp.hit_coin(x, y, throw_up)

    def remove(self, x: int, y: int, w: int, h: int, new_img: int) -> None:
        self.play.tmp.remove(x, y, w, h, new_img)

    def bump_block(self, x: int, y: int) -> None:
        self.play.blocks.bump_block(x, y)

    def new_glitter(self, x: int, y: int, attr: int, duration: int) -> None:
        self.play.glitter.new_glitter(x, y, attr, duration)

    def new_star(self, x: int, y: int, attr: int, duration: int) -> None:
        self.play.glitter.new_star(x, y, attr, duration)

    def start_glitter(self, x: int, y: int, w: int, h: int) -> None:
        self.play.glitter.start_glitter(x, y, w, h)

    def coin_glitter(self, x: int, y: int) -> None:
        self.play.glitter.coin_glitter(x, y)


class LevelData:
    def __init__(self, name: str, level: dict, turbo: bool) -> None:
        self.name = name
        self.world = WorldMap()
        self.options = WorldOptions.from_dict(
            level.get("opt2") if turbo and level.get("opt2") is not None
            else level.get("options"))
        self.rows = int(level.get("rows") or NV)
        self.vertical = bool(level.get("vertical", self.rows > NV))
        self.world.load_columns(level.get("map") or [], self.options,
                                rows=self.rows, vertical=self.vertical)
        # Port-only level flags (WEBANGELO test level, TRAP TEST pack).
        self.no_time = NO_TIME_LIMIT or bool(level.get("no_time"))
        self.start_fire = bool(level.get("start_fire"))
        self.webangelo = list(level.get("webangelo") or [])
        self.battle_room = level.get("battle_room")
        self.hint = str(level.get("hint") or "")
        # Port-only long platforms; see enemies._spawn_platform for the schema.
        self.platforms = list(level.get("platforms") or [])
        # Port-only per-marker walking direction, written by the level editor:
        # [{"x": col, "y": src_row, "dir": -1|1}, ...].  Markers not listed
        # keep the original "walk toward the player" rule.
        self.enemy_dirs = {
            (int(e["x"]), int(e["y"])): int(e["dir"])
            for e in (level.get("enemy_dirs") or [])
            if isinstance(e, dict) and "x" in e and "y" in e and "dir" in e
        }


class PlayWorld:
    """One PlayWorld(...) call: runs a level (and its sub-level) to the end."""

    def __init__(self, assets, screen, keyboard: Keyboard, level_a: str,
                 level_b: str | None = None, player: int = 0,
                 world_number: str = "1-1", turbo: bool = False,
                 stat: bool | None = None, music=None, data: GameData | None = None,
                 seed: int | None = None, only_draw: bool = False,
                 start_level: int = 0) -> None:
        self.assets = assets
        self.screen = screen
        self.keyboard = keyboard
        self.player_index = player
        self.world_number = world_number
        self.turbo = turbo
        self.stat = STAT if stat is None else stat
        self.data = data if data is not None else GameData()
        self.only_draw = only_draw
        self.music = music if music is not None else MusicHooks()
        self.rng = random.Random(seed)

        self.levels = [
            LevelData(level_a, assets.level(level_a), turbo),
            LevelData(level_b, assets.level(level_b), turbo)
            if level_b else None,
        ]
        self.level_index = 1 if (level_b and start_level == 1) else 0

        self.time = TIME_START
        self.level_score = 0
        self.waiting = False
        self.show_objects = True
        self.show_score = False
        self.counting_score = False
        self.text_status = False
        self.quit_game = False
        self.result = False
        self.playing_macro = False
        self.last_x_view = [0] * (MAX_PAGE + 1)
        self.total_back_addr = [0] * (MAX_PAGE + 1)
        self._repaint = True
        self._time_up_done = False

        # --- units ------------------------------------------------------- #
        self.figures = Figures(assets)
        self.backgr = BackGr(self.figures, assets)
        self.figures.backgr = self.backgr
        self.txt = Txt(screen, self.figures, assets)
        self.pal = Palettes(self.figures, screen)
        screen.set_palette(self.figures.palette)

        self.hooks = GameHooks(self)
        lv0 = self.levels[0]
        self.world = lv0.world
        self.options = lv0.options
        self.figures.options = lv0.options

        self.enemies = Enemies(self.figures, assets, lv0.world, lv0.options,
                               self.hooks, seed=seed,
                               pack=lv0.name.split("/")[0] if "/" in lv0.name
                               else "")
        self.enemies.turbo = turbo
        self.enemies.set_direction_overrides(lv0.enemy_dirs)
        self.angelo = AngeloObjects(lv0, self.hooks, random.Random(seed))
        self.no_time = lv0.no_time
        self.player = Player(screen, lv0.world, lv0.options, self.figures,
                             self.enemies, self.data, player, self.hooks)
        self.tmp = TempObjects(screen, self.figures, self.backgr, lv0.world,
                               lv0.options, self.data, player, self.hooks,
                               view_x=lambda: self.player.x_view)
        self.blocks = Blocks(screen, self.backgr)
        self.glitter = Glitter(screen, self.figures,
                               view_x=lambda: self.player.x_view,
                               rows=lambda: self.world.rows)
        self.stars = Stars(screen, self.figures, lv0.options, rng=None)
        self.status = Status(screen, self.txt, self.data, player)
        self.status.time_disabled = self.no_time
        self._battle_open_timer = 0
        self._reset_battle()

    # ------------------------------------------------------------------ #
    # level binding
    # ------------------------------------------------------------------ #
    def bind(self, index: int) -> None:
        lv = self.levels[index]
        self.level_index = index
        self.world = lv.world
        self.options = lv.options
        self.figures.options = lv.options
        for obj in (self.player, self.enemies, self.tmp):
            obj.world = lv.world
            obj.options = lv.options
        self.enemies.set_platforms(lv.platforms)
        self.enemies.set_direction_overrides(lv.enemy_dirs)
        self.stars.options = lv.options
        self.angelo.bind(lv)
        self.no_time = lv.no_time
        self.status.time_disabled = self.no_time
        self._reset_battle()

    def _reset_battle(self) -> None:
        """Arm the port-only battle_room switch of the bound level."""
        spec = self.levels[self.level_index].battle_room
        self._battle = None
        self.enemies.protect = None
        self._battle_open_timer = 0
        if not spec:
            return
        targets = {tuple(int(v) for v in cell)
                   for cell in spec.get("targets", [])}
        barrier = spec.get("barrier") or {}
        cols = [int(v) for v in barrier.get("cols", [])]
        rows = [int(v) for v in barrier.get("rows", [])]
        if targets and cols and rows:
            room = spec.get("room")
            if room:
                x1, x2 = int(room[0]), int(room[1])
            else:
                xs = [cell[0] for cell in targets] + cols
                x1, x2 = min(xs), max(xs)
            self._battle = {
                "targets": targets,
                "seen": set(),
                "unlocked": False,
                "cols": cols,
                "rows": rows,
                "room": (x1, x2),
            }
            # The camera is locked while the room is sealed; without this,
            # an enemy parked at the far edge can cross the forget boundary
            # and be recycled, which looks exactly like a defeat.
            self.enemies.protect = (x1, x2)

    def _update_battle(self) -> None:
        """Clear the barrier once every target has been seen and defeated.

        The barrier itself is an ordinary solid column with the original
        scroll marker underneath it, so clearing the tiles releases both
        Mario and the camera -- no extra camera code is needed.
        """
        b = self._battle
        if b is None or b["unlocked"]:
            return
        alive = set()
        for idx in self.enemies.active:
            e = self.enemies.pool[idx]
            if e.tp in BATTLE_ALIVE_TPS:
                alive.add((e.map_x, e.map_y))
        b["seen"] |= b["targets"] & alive
        if not b["targets"] <= b["seen"]:
            return
        if b["targets"] & alive:
            return
        b["unlocked"] = True
        for x in b["cols"]:
            for y in b["rows"]:
                self.world.set(x, y, ord(" "))
        self._repaint = True
        self._battle_open_timer = 120
        self.music.beep(880)

    def _draw_hint(self) -> None:
        p = self.player
        if p.passed or p.game_done or self.waiting:
            return
        text = ""
        if self._battle is not None:
            if not self._battle["unlocked"]:
                text = (self.levels[self.level_index].hint
                        or "DEFEAT THE ENEMIES!")
            elif self._battle_open_timer > 0:
                text = "GATE OPEN!"
                self._battle_open_timer -= 1
        else:
            text = self.levels[self.level_index].hint
        if text:
            self.txt.set_font(0, BOLD | SHADOW)
            self.txt.center_text(p.view_y + 45, text, 31, p.x_view)

    def swap_levels(self) -> None:
        """BUFFERS.Swap: switch between Map1/Map2 (a/b) and their options."""
        if self.levels[1] is not None:
            self.bind(1 - self.level_index)
            self.player.sync_enemies()

    # ------------------------------------------------------------------ #
    # setup (BuildLevel / Restart)
    # ------------------------------------------------------------------ #
    def build_level(self) -> None:
        o = self.options
        f = self.figures
        f.init_sky(o.SkyType)
        f.init_walls(o.WallType1, o.WallType2, o.WallType3, o)
        f.init_pipes(o.PipeColor)
        self.backgr.init_backgr(o.BackGrType, o.Clouds)
        self.backgr.init_moon_sky(o.MoonSky)
        if o.Stars != 0:
            self.stars.init_stars()
        f.build_world(self.world, o)
        # PLAY.PAS only calls SetSkyPalette/InitGrass after Restart, which is
        # harmless on the DOS side because the page holds palette *indices*.
        # The port bakes RGB, so the very first page (and BackGr.read_color_map,
        # which maps those pixels back to indices) would otherwise be built with
        # $E0 still black -- that is the "black flash" a collected coin leaves
        # behind on the flat-sky levels.
        f.set_sky_palette()
        f.init_grass()

    def _show_page(self) -> None:
        self.screen.show_page()
        self.screen.swap_pages()

    def _apply_vision(self) -> None:
        """Install the mask for the picture that is about to be displayed.

        The mask lives on the screen and only ``show_page`` reads it, so a
        frame displayed while the mask is empty shows the room fully lit.
        ``restart`` displays 1 + MAX_PAGE+1 entry pages and ``fade_up`` 64
        more, all of them before the frame loop runs -- which is the user's
        "进入关卡时瞬间全亮再黑".  Every display path must come through here.
        """
        p = self.player
        if p.options.Vision:
            # Ghost-house sight.  Purely a display effect: the page keeps
            # every pixel, so a blacked-out tile still blocks and still
            # hurts -- only the light circle and the glowing things move.
            # p.x/p.y are already page pixels (that is what draw_player
            # blits at); adding x_view here would double the camera and
            # the circle would crawl backwards as soon as it scrolled.
            # The anchor is the CENTRE OF THE TILE MARIO'S HEAD IS IN:
            # player_y1 is the top of his body (p.y + H small, p.y big), so
            # both sizes get the same rule -- the head tile, one tile up /
            # left / right / down from it are lit, and a big Mario's feet
            # land on the lower edge of that diamond.
            head_row = p.player_y1 // H
            self.screen.set_vision(p.x + W // 2,
                                   head_row * H + H // 2,
                                   p.vision_r,
                                   self._vision_lights())
        else:
            # The mask lives on the screen, which outlives the level: a
            # horizontal pipe swaps levels through swap_levels() without
            # ever calling restart(), so without this the next level keeps
            # the previous one's circle -- parked off-screen, i.e. a black
            # room with only the HUD lit (the 8-5 -> 9-1 report).
            self.screen.clear_vision()
            p.vision_r = 0

    def _vision_lights(self) -> list[tuple[int, int, float]]:
        """(page x, page y, radius in tiles) of everything that lights itself.

        Every light carries the shared constant radius, never Mario's live one.
        """
        en = self.enemies
        r = VISION_GLOW_RADIUS
        out: list[tuple[int, int, float]] = []
        for i in en.active:
            e = en.pool[i]
            if e.tp in VISION_GLOW_TYPES:
                # drawn as nothing at all -- a lava fireball parked back in the
                # floor, or a plant inside its pipe -- must not light the room
                # either (the user's rule, same as the hit box)
                if en._hidden(e):
                    continue
                dx, dy = VISION_LIGHT_CENTRE.get(e.tp, VISION_LIGHT_DEFAULT)
                out.append((e.x_pos + dx, e.y_pos + dy, r))
        for t in self.tmp.pool:
            if t.alive and t.tp == TP_COIN:
                dx, dy = VISION_LIGHT_CENTRE.get(TP_COIN, VISION_LIGHT_DEFAULT)
                out.append((t.x_pos + dx, t.y_pos + dy, r))
        # Coins that still sit in the map light their own tile.  HitCoin blanks
        # the cell, so the glow goes away the moment he eats it.
        w = self.world
        x0 = self.player.x_view // W - 1
        for col in range(x0, x0 + NH + 3):
            for row in range(w.rows):
                if w.get(col, row) == CH_COIN:
                    out.append((col * W + W // 2, row * H + H // 2, r))
        # The firebar is a WEBANGELO object, not an enemy-pool type, so it has
        # to be collected separately: its hub and every rotating ball.  Their
        # positions are already true sprite centres, so no offset applies.
        for bar in self.angelo.bars:
            out.append((int(bar.cx), int(bar.cy), r))
            out.extend((int(sx), int(sy), r) for sx, sy in bar.pos)
        return out

    def _boss_defeated(self) -> None:
        """WEBANGELO Boss death -> the same STAGE CLEAR path as an exit."""
        p = self.player
        if not p.passed:
            p.passed = True
            p.text_counter = 0
            # CH_TAU is the end-of-level pipe code; the run loop shows STAGE
            # CLEAR, then _pipe_transition() sets result=True and exits.
            p.in_pipe = True
            p.pipe_code = [CH_TAU, ord(" ")]

    def _render_page_initial(self) -> None:
        p = self.player
        o = self.options
        xv = p.x_view
        page = self.screen.page
        rows = self.world.rows
        self.figures.draw_sky(page, xv, 0, NH * W, rows * H)
        if o.MoonSky:
            # port-only screen-fixed backdrop (one-screen levels)
            self.backgr.draw_moon_sky(page, xv)
        for x in range(xv // W - 1, xv // W + NH + 1):
            for y in range(rows):
                self.figures.redraw(page, self.world, x, y)
        self.backgr.draw_backgr(page, xv, self.last_x_view[self.screen.current],
                                True)
        # clouds sit on top of the tile/sky layer (the original StartClouds
        # painted them before the tiles, where Redraw erased them again)
        self.backgr.draw_clouds(page, xv)
        self.backgr.read_color_map(page, xv)
        # after the colour map: the map is what run_remove restores to, and a
        # plant pixel in it would smear onto the sky when a coin is taken
        self.figures.draw_decorations(page, self.world,
                                      *self.figures.decoration_columns(xv, NH))
        if o.Stars != 0:
            self.stars.show_stars(xv)
        self.enemies.show_enemies(self.screen)
        if not self.only_draw:
            p.draw_player()

    def restart(self) -> None:
        p = self.player
        self.screen.reset_stack()
        self.text_status = False
        self.status.init_status()
        self.blocks.init_blocks()
        self.tmp.init_temp_obj()
        self.glitter.clear_glitter()
        self.enemies.clear_enemies()
        self.angelo.reset()
        if self.levels[self.level_index].start_fire:
            self.data.Mode[self.player_index] = MD_FIRE
        if FORCE_MODE is not None:
            self.data.Mode[self.player_index] = FORCE_MODE
        # The sight must be armed before the first frame is *displayed*:
        # _show_page below and the MAX_PAGE+1 entry pages after it are real
        # screens, and a mask-less one in a Vision level is the fully lit
        # flash the user reported on entry.
        p.begin_vision()
        self._apply_vision()
        self._show_page()
        p.game_done = False
        p.passed = False
        self.enemies.start_platforms()      # drawn behind enemies
        for i in range(-self.enemies.start_at,
                       NH + self.enemies.start_at + 1):
            j = p.x_view // W + i
            self.enemies.start_enemies(j, -1 if j > p.map_x else 1)
        # the enemies are only lit once they exist, so re-collect the lights
        self._apply_vision()
        for _ in range(MAX_PAGE + 1):
            self._render_page_initial()
            # the original RestartLevel leaves the entities already painted
            # when the fade-up reveals the page; without this the death
            # restart fades up a bright, empty level (user's report)
            if self.show_objects:
                self.blocks.draw_blocks()
                self.enemies.show_enemies(self.screen)
                self.angelo.show(self.screen)
                if not p.in_pipe:
                    p.draw_player()
            self._render_page_tail()
            self._show_page()

    def _render_page_tail(self) -> None:
        """Frame-start bookkeeping that the first Hide/erase pass will undo:
        remember the view so move_screen does not scroll on frame 1."""
        self.last_x_view[self.screen.current] = self.player.x_view

    def _after_restart(self) -> None:
        """The palette block that follows Restart (also re-run on pipe warps)."""
        p = self.player
        p.demo = DM_NO_DEMO
        self.waiting = False
        self.pal.new_palette()
        for _ in range(100):            # waterfalls
            self.pal.blink_palette()
        self.figures.set_sky_palette()
        self.backgr.draw_pal_backgr(p.x_view)
        self.pal.init_grass()
        if self.only_draw:
            return
        self.pal.unlock_pal()
        self.pal.fade_up(64)
        self.pal.read_palette()
        self.text_status = bool(self.stat) and not self.playing_macro
        self._repaint = True

    # ------------------------------------------------------------------ #
    # MoveScreen
    # ------------------------------------------------------------------ #
    def move_screen(self) -> None:
        p = self.player
        page_idx = self.screen.current
        last = self.last_x_view[page_idx]
        scroll = p.x_view - last

        max_y = max(0, self.world.rows * H - VIR_SCREEN_HEIGHT)
        if not p.earth_quake:
            self.screen.set_view(p.x_view, p.view_y)
        else:
            p.earth_quake_counter += 1
            if p.earth_quake_counter > 0:
                p.earth_quake = False
            dy = self.rng.randrange(2) - self.rng.randrange(2)
            # Original 13-row levels shake around view_y=0; tall levels must
            # keep the shaken view inside the level.
            y = (max(0, min(max_y, p.view_y + dy))
                 if self.world.vertical else dy)
            self.screen.set_view(p.x_view, y)

        if scroll < 0:
            self.enemies.start_enemies(
                p.x_view // W - self.enemies.start_at, 1)
        elif scroll > 0:
            self.enemies.start_enemies(
                p.x_view // W + NH + self.enemies.start_at, -1)

        # Horizon + GetYOffset - YBase; YOffset is always YBase here.
        self.backgr.draw_backgr(self.screen.page, p.x_view, last)

        page = self.screen.page
        rows = self.world.rows
        if scroll > 0:
            last_col = None
            for j in range(last, p.x_view + 1):
                i = j - W - W
                if i >= 0:
                    self.screen.put_pixel(i, 0, 0)
                i = W - j % W - 1
                col = j // W + NH + 1
                self.figures.redraw(page, self.world, col, i)
                # The original's rolling i (0..W-1) covers rows 0..19; taller
                # levels need the rest of the entering column as well.
                if col != last_col:
                    for y in range(W, rows):
                        self.figures.redraw(page, self.world, col, y)
                    last_col = col
        elif scroll < 0:
            last_col = None
            for j in range(last, p.x_view - 1, -1):
                i = W - j % W - 1
                col = j // W - 1
                self.figures.redraw(page, self.world, col, i)
                if col != last_col:
                    for y in range(W, rows):
                        self.figures.redraw(page, self.world, col, y)
                    last_col = col
        # the cloud layer must go last: redrawing an entering column paints
        # sky over it, and the original StartClouds call sat on the wrong side
        self.backgr.draw_clouds(page, p.x_view)
        # ... and the same for the scenery: the entering column was just
        # repainted from the tiles, which wipes whatever a plant from the
        # neighbouring column overhangs into it
        self.figures.draw_decorations(
            page, self.world, *self.figures.decoration_columns(p.x_view, NH))

    def _repaint_static(self) -> None:
        """Draw the visible tile/sky layer with the current palette."""
        xv = self.player.x_view
        page = self.screen.page
        rows = self.world.rows
        self.figures.draw_sky(page, xv, 0, NH * W, rows * H)
        if self.options.MoonSky:
            # port-only screen-fixed backdrop (one-screen levels)
            self.backgr.draw_moon_sky(page, xv)
        for x in range(xv // W - 1, xv // W + NH + 1):
            for y in range(rows):
                self.figures.redraw(page, self.world, x, y)
        self.figures.draw_decorations(page, self.world,
                                      *self.figures.decoration_columns(xv, NH))

    def find_pipe_exit(self) -> None:
        p = self.player
        o = self.options
        if p.pipe_side:
            # horizontal pipes store their code pair vertically: pipe_code[0]
            # in the feet row, pipe_code[1] one row above, beside the mouth
            for i in range(o.XSize):
                for j in range(1, NH):
                    if i == p.map_x and j == p.map_y:
                        continue
                    if 0xE0 <= self.world.get(i, j) <= 0xEF and \
                            self.world.get(i, j - 1) == p.pipe_code[1]:
                        p.map_x = i
                        p.map_y = j
                        p.x_view = (i - NH // 2 + 1) * W
                        p.x_view = max(0, min(p.x_view, (o.XSize - NH) * W))
                        return
            return
        for i in range(o.XSize - 1):
            for j in range(NH):
                if i == p.map_x and j == p.map_y:
                    continue
                if 0xE0 <= self.world.get(i, j) <= 0xEF and \
                        self.world.get(i + 1, j) == p.pipe_code[1]:
                    p.map_x = i
                    p.map_y = j
                    p.x_view = (i - NH // 2 + 1) * W
                    p.x_view = max(0, min(p.x_view, (o.XSize - NH) * W))
                    return

    # ------------------------------------------------------------------ #
    # score display
    # ------------------------------------------------------------------ #
    def _write_total_score(self) -> None:
        self.txt.set_font(0, BOLD | SHADOW)
        s = str(self.data.Score[self.player_index]).rjust(11)
        s = "".join("0" if (i >= 3 and c == " ") else c
                    for i, c in enumerate(s))
        self.txt.center_text(self.player.view_y + 120,
                             "TOTAL SCORE:" + s, 31, self.player.x_view)

    def _show_total_back(self) -> None:
        if self.player.passed and self.counting_score:
            self.music.beep(4 * 880)
        self.total_back_addr[self.screen.current] = self.screen.push_backgr(
            self.player.x_view + 160, self.player.view_y + 120, 120, 8)
        if self.player.passed and self.counting_score:
            self.music.beep(2 * 880)
        self._write_total_score()
        if self.player.passed and self.counting_score:
            self.music.beep(0)

    def _hide_total_back(self) -> None:
        page = self.screen.current
        if self.total_back_addr[page] != 0:
            self.screen.pop_backgr(self.total_back_addr[page])
        self.total_back_addr[page] = 0

    # ------------------------------------------------------------------ #
    # input helpers
    # ------------------------------------------------------------------ #
    def _pump(self) -> bool:
        self.keyboard.pump()
        if self.keyboard.quit:
            self.quit_game = True
            return True
        return False

    def _tapped(self, key: int) -> bool:
        return any(k == key for k, _ in self.keyboard.taps)

    def _wait_release(self, keys) -> None:
        while any(self.keyboard.is_down(k) for k in keys):
            if self._pump():
                return

    # ------------------------------------------------------------------ #
    # Pause + cheats
    # ------------------------------------------------------------------ #
    def _pause(self) -> None:
        """PLAY.PAS Pause.

        Hold P to stay paused.  Releasing P does *not* resume: the original
        then spins in `while Chr (bKey - $80) = kbP do ;`, and the BIOS
        scancode byte only changes when the next key event arrives, so the
        next key press resumes (Tab there opens the cheat prompt instead).
        """
        p = self.player
        self.music.pause_music()
        self.screen.set_view(p.x_view, p.view_y)
        self.pal.fade_down(8)
        self.screen.swap_pages()
        pause_back = self.screen.push_backgr(p.x_view + 120, p.view_y + 85,
                                             80, 10)
        self._draw_pause_text("PAUSED")

        tab = self._tapped(pygame.K_TAB)
        while self.keyboard.is_down(pygame.K_p):
            if self._pump():
                break
            if self._tapped(pygame.K_TAB):
                tab = True

        if not tab:
            # wait for the next key event -- that is what resumes the game
            while not self.keyboard.taps:
                if self._pump():
                    break
            if self._tapped(pygame.K_TAB):
                tab = True

        if tab and not self.quit_game:
            self._cheat_loop(pause_back)

        if pause_back != 0:
            self.screen.pop_backgr(pause_back)
        self.screen.swap_pages()
        self.pal.fade_up(8)
        self.pal.apply()

    def _draw_pause_text(self, text: str) -> None:
        self.pal.out_palette(0x0F, 63, 63, 63)
        self.pal.apply()
        self.txt.set_font(0, BOLD | SHADOW)
        self.txt.center_text(self.player.view_y + 85, text, 0x0F,
                             self.player.x_view)
        self.screen.show_page()

    def _cheat_loop(self, pause_back: int) -> None:
        """Read a cheat string until it matches or a non-ASCII key is hit."""
        buf = ""
        while True:
            if self._pump():
                return
            for _key, ch in self.keyboard.taps:
                if not ch or not ch.isprintable():
                    return
                buf += ch.upper()
                entry = _CHEATS.get(buf)
                if entry is None:
                    continue
                action, ends = entry
                new_back = action(self, pause_back)
                if new_back is not None:
                    pause_back = new_back
                if ends:
                    return

    def cheat_add_life(self) -> None:
        self.tmp.add_life()

    def cheat_lives(self) -> None:
        self.data.Lives[self.player_index] = 10000

    def cheat_star(self) -> None:
        self.enemies.cd_star = 1

    def cheat_champ(self) -> None:
        self.enemies.cd_champ = 1

    def cheat_flower(self) -> None:
        self.enemies.cd_flower = 1

    def cheat_turbo(self) -> None:
        self.enemies.turbo = not self.enemies.turbo

    def cheat_god(self) -> None:
        global GOD_MODE
        GOD_MODE = not GOD_MODE
        if GOD_MODE:
            self.music.start_music("star")

    def cheat_one_up(self) -> None:
        global CHEATS_USED
        p = self.player
        if CHEATS_USED & 1 == 0:
            self.enemies.new_enemy(TP_LIFE, 0, p.x_view // W, -1, 2, 0, 2)
            CHEATS_USED |= 1
        else:
            self.enemies.new_enemy(TP_CHAMP, 1, (p.x_view + self.rng.randrange(100)) // W,
                                   -1, 2 - self.rng.randrange(2), 0, 2)
            if self.rng.randrange(10) == 0:
                CHEATS_USED &= ~1

    def cheat_skip(self) -> None:
        p = self.player
        p.passed = True
        self.waiting = True
        p.text_counter = 200
        p.pipe_code[0] = CH_TAU
        p.in_pipe = True

    def cheat_sub(self) -> None:
        p = self.player
        p.passed = True
        self.waiting = False
        p.text_counter = 200
        p.pipe_code = [CH_SS, CH_ALPHA]
        p.in_pipe = True

    def cheat_effect(self, effect: int) -> None:
        self.pal.set_effect(effect)
        self.pal.apply()
        self._repaint = True

    def cheat_spawn(self, tp: int, x=None, y=-1, xv=2, yv=0,
                    delay=2) -> None:
        p = self.player
        if x is None:
            x = p.x_view // W
        self.enemies.new_enemy(tp, 0, x, y, xv, yv, delay)

    def cheat_credit(self, text: str, pause_back: int) -> int:
        if pause_back != 0:
            self.screen.pop_backgr(pause_back)
        pause_back = self.screen.push_backgr(self.player.x_view + 20,
                                             self.player.view_y + 85, 280, 10)
        self.pal.out_palette(0x0F, 63, 63, 63)
        self.pal.apply()
        self.txt.set_font(0, BOLD | SHADOW)
        self.txt.center_text(self.player.view_y + 85, text, 0x0F,
                             self.player.x_view)
        self.screen.show_page()
        return pause_back

    def cheat_record(self) -> None:
        return None              # KEYBOARD.PAS macro recorder not ported

    def cheat_play_macro(self) -> None:
        return None

    def cheat_save_macro(self) -> None:
        return None

    # ------------------------------------------------------------------ #
    # the main loop
    # ------------------------------------------------------------------ #
    def run(self, max_frames: int | None = None, clock=None, fps: int = 60,
            on_frame=None) -> bool:
        global STAT
        p = self.player

        # --- PlayWorld prologue ----------------------------------------- #
        self.level_score = 0
        self.show_objects = True
        self.show_score = False
        self._time_up_done = False
        self.total_back_addr = [0] * (MAX_PAGE + 1)
        p.text_counter = 0
        p.in_pipe = False
        p.pipe_code = [ord(" "), ord(" ")]
        p.demo = DM_NO_DEMO
        self.bind(self.level_index)
        self.time = 0 if self.no_time else TIME_START
        self.status.time = self.time
        self.status.time_disabled = self.no_time
        o = self.options
        p.init_player(o.InitX, o.InitY, self.player_index)
        if self.levels[self.level_index].start_fire:
            self.data.Mode[self.player_index] = MD_FIRE
        if FORCE_MODE is not None:
            self.data.Mode[self.player_index] = FORCE_MODE
        p.map_x = o.InitX
        p.map_y = o.InitY
        p.x_view = 0
        p.sync_enemies()
        self.last_x_view = [0] * (MAX_PAGE + 1)
        self.screen.set_view(0, p.view_y)
        self.pal.clear_palette()
        self.pal.lock_pal()
        self.screen.clear()

        self.build_level()
        self.restart()
        self._after_restart()
        if self.only_draw:
            return False

        # --- frame loop -------------------------------------------------- #
        frames = 0
        while True:
            if self._pump():
                break
            key = self.keyboard.key
            o = self.options

            if not self.playing_macro:
                if self._tapped(pygame.K_s):
                    self.stat = not self.stat
                    STAT = self.stat          # Play.Stat persists across levels
                    self.text_status = self.stat
                if self._tapped(pygame.K_q):
                    self.music.beeper_sound = not self.music.beeper_sound
                    if self.music.beeper_sound:
                        self.music.beep(80)
                if self._tapped(pygame.K_PAUSE):
                    self.music.pause_music()
                    self._wait_release((pygame.K_PAUSE,))
            else:
                if key != 0:
                    p.game_done = True
                    p.passed = True

            if 40 <= p.text_counter <= 40 + MAX_PAGE:
                self.show_objects = False

            # ---- hide ------------------------------------------------ #
            self.glitter.hide_glitter()
            if o.Stars != 0:
                self.stars.hide_stars()
            if self.show_objects:
                self.tmp.hide_temp_obj()
            self.status.hide_status()
            if self.show_score:
                self._hide_total_back()
            p.erase_player()
            if self.show_objects:
                self.angelo.hide(self.screen)
                self.enemies.hide_enemies(self.screen)
                self.blocks.erase_blocks()

            if self._repaint:
                # the original animates the palette through the DAC, which
                # never repaints the map; the port's RGB emulation repaints
                # the whole page, and during the death/score freeze (objects
                # no longer shown) that would wipe every entity off the
                # frozen scene (user's "monsters must stay on screen").
                if self.show_objects:
                    self._repaint_static()
                self._repaint = False

            # ---- update ---------------------------------------------- #
            p.invincible = GOD_MODE
            self.figures.lava_counter += 1
            if not self.waiting:
                if p.demo == DM_NO_DEMO:
                    self.enemies.move_enemies()
                    p.move_player(self.keyboard)
                    self.angelo.move(self.world, self.enemies,
                                     self._boss_defeated)
                    self._update_battle()
                else:
                    p.do_demo()

            if not self.no_time:
                if self.time == 100 * 32 and not p.passed:
                    self.music.start_music("time")
                if self.time > 0 and not p.passed:
                    self.time -= 1
                if self.time <= 0 and not self._time_up_done:
                    # The original re-drew TIME UP every frame; Pause itself
                    # waits for the next key event, so the port shows it once
                    # per attempt.
                    self.txt.set_font(70, BOLD | SHADOW)
                    self.txt.center_text(p.view_y + 90, "TIME UP!", 31,
                                         p.x_view)
                    self._pause()
                    self._time_up_done = True
                    p.game_done = True
                    p.passed = False

            if not self.waiting:
                if p.passed:
                    if p.demo == DM_NO_DEMO or p.in_pipe:
                        self.waiting = True
                        p.text_counter = 0
                    p.text_counter += 1
                    if (not self.show_score
                            and 50 <= p.text_counter <= 50 + MAX_PAGE):
                        self.txt.set_font(0, BOLD | SHADOW)
                        self.txt.center_text(p.view_y + 20,
                                             PLAYER_NAME[self.player_index],
                                             0x1E, p.x_view)
                        self.txt.set_font(1, BOLD | SHADOW)
                        self.txt.center_text(p.view_y + 40, "STAGE CLEAR!",
                                             31, p.x_view)
                        if p.text_counter == 50 + MAX_PAGE:
                            self.show_score = True
                elif p.game_done:
                    self.data.Lives[self.player_index] -= 1
                    self.data.Mode[self.player_index] = MD_SMALL
                    p.text_counter = 0
                    self.data.Score[self.player_index] += self.level_score
                    self.waiting = True
                    p.game_done = False

            if self._tapped(pygame.K_p):
                self._pause()

            if (self.show_score and p.text_counter == 120
                    and self.level_score > 0):
                i = self.level_score - 50
                if i < 0:
                    i = 0
                self.data.Score[self.player_index] += self.level_score - i
                self.level_score = i
                p.text_counter = 119
                self.counting_score = True
            else:
                self.counting_score = False

            if self.waiting:
                p.text_counter += 1
                if self.data.Lives[self.player_index] == 0:
                    if 100 <= p.text_counter <= 100 + MAX_PAGE:
                        self.txt.set_font(0, BOLD | SHADOW)
                        self.txt.center_text(p.view_y + 20,
                                             PLAYER_NAME[self.player_index],
                                             0x1E, p.x_view)
                        self.txt.set_font(1, BOLD | SHADOW)
                        self.txt.center_text(p.view_y + 40, "GAME OVER",
                                             31, p.x_view)
                        self.show_score = True
                    if p.text_counter > 350:
                        p.game_done = True
                elif p.passed:
                    if p.text_counter > 250:
                        self.waiting = False
                else:
                    if p.text_counter > 100:
                        p.game_done = True

            self.tmp.move_temp_obj()
            self.blocks.move_blocks()

            if self._tapped(pygame.K_ESCAPE):
                self.quit_game = True

            self.move_screen()
            self.tmp.run_remove()

            if o.Horizon < self.world.rows:
                j = o.Horizon - 1
                for i in range(0, NH + 1):
                    k = (p.x_view // W
                         + (i + self.figures.lava_counter // 8) % (NH + 1))
                    if self.world.get(k, j) == ord("%"):
                        self.figures.redraw(self.screen.page, self.world, k, j)

            self.screen.reset_stack()

            # ---- draw ------------------------------------------------ #
            if self.show_objects:
                self.blocks.draw_blocks()
                self.enemies.show_enemies(self.screen)
                self.angelo.show(self.screen)
            p.draw_player()

            if self.show_score:
                self._show_total_back()
            if self.text_status:
                self.status.time = self.time
                self.status.show_status(p.x_view, self.level_score,
                                        self.world_number)
            if self.show_objects:
                self.tmp.show_temp_obj()
            if o.Stars != 0:
                self.stars.show_stars(p.x_view)
            self.glitter.show_glitter()
            self._draw_hint()

            self.last_x_view[self.screen.current] = p.x_view

            if o.Vision:
                p.tick_vision()
            self._apply_vision()

            # PLAY's ShowRetrace cheat flashes palette entry 0 around
            # ShowPage; the port paints RGB pixels, so it has no effect.
            self._show_page()

            self.backgr.draw_pal_backgr(p.x_view)
            self.pal.blink_palette()
            if self.pal.apply():
                self._repaint = True
            self.music.play_music()

            if p.in_pipe and self.playing_macro:
                p.game_done = True

            if p.in_pipe and (not p.game_done) and (not self.waiting):
                self._pipe_transition()

            frames += 1
            if on_frame is not None:
                on_frame(self)
            if self.quit_game or p.game_done:
                break
            if max_frames is not None and frames >= max_frames:
                break
            if clock is not None:
                clock.tick(fps)

        # --- PlayWorld epilogue ----------------------------------------- #
        self.enemies.clear_enemies()
        self.glitter.clear_glitter()
        self.pal.fade_down(64)
        self.pal.clear_palette()
        self.screen.clear()
        self.music.stop_music()
        return self.result

    def _pipe_transition(self) -> None:
        p = self.player
        self.enemies.stop_enemies()
        self.glitter.clear_glitter()
        self.pal.fade_down(64)
        self.pal.clear_palette()
        self.pal.lock_pal()
        self.screen.clear()

        code = p.pipe_code[0]
        if code == CH_ALPHA:
            self.find_pipe_exit()
            pygame.time.delay(100)
        elif code == CH_SS:
            self.swap_levels()
            self.find_pipe_exit()
        elif code == CH_TAU:
            p.game_done = True
            self.result = True

        p.init_player(p.map_x * W + W // 2, (p.map_y - 1) * H,
                      self.player_index)
        # a pipe travel always puts the sight back to its base radius
        p.begin_vision()
        self.screen.set_view(p.x_view, p.view_y)
        for i in range(MAX_PAGE + 1):
            self.last_x_view[i] = p.x_view

        if code == CH_ALPHA:
            self.restart()
            self._after_restart()
        elif code == CH_SS:
            self.build_level()
            self.restart()
            self._after_restart()


def _cheat_retrace(play: PlayWorld, pause_back: int) -> None:
    global SHOW_RETRACE
    SHOW_RETRACE = not SHOW_RETRACE


def _cheat_music_bounce(play: PlayWorld, pause_back: int) -> None:
    play.music.start_music("bounce")


def _cheat_credit(text: str):
    def run(play: PlayWorld, pause_back: int) -> int:
        return play.cheat_credit(text, pause_back)
    return run


# (action, ends_pause).  The strings are KEYBOARD.PAS scancodes written as the
# plain letters they correspond to (case does not matter, like the original).
_CHEATS: dict[str, tuple] = {
    "TEST": (_cheat_retrace, True),
    "0044": (_cheat_retrace, True),
    "03E8": (lambda pl, pb: pl.cheat_add_life(), True),
    "B172": (lambda pl, pb: pl.cheat_lives(), True),
    "9C32": (lambda pl, pb: pl.cheat_star(), True),
    "F1F2": (lambda pl, pb: pl.cheat_champ(), True),
    "FFB5": (lambda pl, pb: pl.cheat_flower(), True),
    "D235": (lambda pl, pb: pl.cheat_turbo(), True),
    "76DD": (lambda pl, pb: pl.cheat_record(), True),
    "C7B4": (lambda pl, pb: pl.cheat_play_macro(), True),
    "208D": (lambda pl, pb: pl.cheat_save_macro(), True),
    "1UP": (lambda pl, pb: pl.cheat_one_up(), True),
    "SKIP": (lambda pl, pb: pl.cheat_skip(), True),
    "SUB": (lambda pl, pb: pl.cheat_sub(), True),
    "GOD": (lambda pl, pb: pl.cheat_god(), True),
    "INVINCIBLE": (lambda pl, pb: pl.cheat_god(), True),
    "MONO": (lambda pl, pb: pl.cheat_effect(PE_BLACK_WHITE), True),
    "EGAMODE": (lambda pl, pb: pl.cheat_effect(PE_EGA_MODE), True),
    "VGAMODE": (lambda pl, pb: pl.cheat_effect(PE_NO_EFFECT), True),
    "COLOR": (lambda pl, pb: pl.cheat_effect(PE_NO_EFFECT), True),
    "CREDIT": (_cheat_credit("PROGRAMMED BY MIKE WIERING"), False),
    "MODBY": (_cheat_credit("MODIFIED BY SPENSER WHALEY"), False),
    "KOOPA": (lambda pl, pb: pl.cheat_spawn(TP_KOOPA), True),
    "GOOMBA": (lambda pl, pb: pl.cheat_spawn(TP_CHIBIBO), True),
    "SPINY": (lambda pl, pb: pl.cheat_spawn(TP_RED, None, 5, 4, 3, 4), True),
    "FIRE": (lambda pl, pb: pl.cheat_spawn(TP_FIREBALL), True),
    "FISH": (lambda pl, pb: pl.cheat_spawn(TP_VERT_FISH), True),
    "JUMPB": (lambda pl, pb: pl.cheat_spawn(TP_VERT_FIREBALL), True),
    "LIFT": (lambda pl, pb: pl.cheat_spawn(TP_BLOCK_LIFT), True),
    "BOUNCE": (_cheat_music_bounce, True),
}


def page_width(assets, level_a: str, level_b: str | None = None) -> int:
    """Width of the world-wide video pages needed by a level pair."""
    names = [n for n in (level_a, level_b) if n]
    widest = max((assets.level(n).get("width") or 0) for n in names)
    return max(340, (widest + 1) * W)


def main() -> int:
    pygame.init()
    assets = load_assets()
    screen = VGAScreen(VgaConfig(width=page_width(assets, "1a", "1b"),
                                 height=VIR_SCREEN_HEIGHT, scale=2))
    play = PlayWorld(assets, screen, Keyboard(), "1a", "1b", player=0,
                     world_number="1-1", music=Music())
    play.run(clock=pygame.time.Clock())
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
