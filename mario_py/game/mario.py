"""Port of MARIO.PAS — config, title/menu and the level-progression loop.

Differences from the DOS program, all deliberate:

* MARIO.CFG is JSON instead of Turbo Pascal's typed binary record.  A binary
  CONFIG from the original is ignored (defaults are used) and the file is
  rewritten as JSON on exit.
* Joystick calibration (/J, CalibrateJoystick) is not ported.
* The attract-mode demo used DEMOKEYS.OBJ (a recorded key macro).  The macro
  recorder is not ported, so the menu timeout plays Intro_1 with no input
  until a key is pressed or 900 frames pass.
* Info/Help (/I, /?) print to stdout instead of a text-mode page.

Everything else follows MARIO.PAS: the title screen, the menu state machine
(ST_MENU/START/OPTIONS/NUMPLAYERS/LOAD/ERASE), ShowPlayerName, the 24 level
pairs and the two-player turn order.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from ..engine.input import Keyboard
from ..engine.vga import VIR_SCREEN_HEIGHT, VGAScreen, VgaConfig
from . import play as play_unit
from .assets import load_assets
from .buffers import MD_SMALL, PL_MARIO, PL_LUIGI, TILE_H, GameData
from .figures import Figures
from .music import Music
from . import palettes as palettes_unit
from .play import PlayWorld
from .txt import BOLD, SHADOW

NUM_LEV = 25              # MARIO.PAS const; note only 24 pairs exist (case 0..23)
LAST_LEV = 2 * NUM_LEV - 1
MAX_SAVE = 3
WAIT_BEFORE_DEMO = 550

LEVEL_A = [c + "a" for c in "123456789ABCDEFGHIJKLMNO"]        # 24
LEVEL_B = ([c + "b" for c in "123456789"] + ["Intro_0"]
           + [c + "b" for c in "BCDEFGHIJKLMNO"])
# (N1, N2, level_a, level_b), in Progress order (MARIO.PAS case labels)
LEVEL_PAIRS = [((i // 4) + 1, (i % 4) + 1, LEVEL_A[i], LEVEL_B[i])
               for i in range(len(LEVEL_A))]


def mod_level_pairs(assets) -> dict[str, list[tuple[str, str, str | None]]]:
    """(world, Map1, Map2) per mod pack, already in Progress order.

    levels_mod.json stores the packs in the order of each mod's
    `case Progress mod NUM_LEV of`, so the list index is the Progress value.
    """
    return {pack: [(w["world"], w["a"], w["b"])
                   for w in meta.get("worlds", [])]
            for pack, meta in assets.mod_packs.items()}

# menu states
ST_MENU = "MENU"
ST_START = "START"
ST_LOAD = "LOAD"
ST_ERASE = "ERASE"
ST_OPTIONS = "OPTIONS"
ST_NUMPLAYERS = "NUMPLAYERS"
ST_MODPACKS = "MODPACKS"
ST_MODSLOT = "MODSLOT"

# Esc / edge Up-Down returns to the parent state (MARIO.PAS LastStatus)
MENU_PARENT = {
    ST_MENU: ST_MENU,
    ST_START: ST_MENU,
    ST_OPTIONS: ST_MENU,
    ST_LOAD: ST_START,
    ST_ERASE: ST_START,
    ST_NUMPLAYERS: ST_START,
    ST_MODPACKS: ST_START,
    ST_MODSLOT: ST_MODPACKS,
}


# ---------------------------------------------------------------------- #
# config file (JSON replacement for ConfigFile)
# ---------------------------------------------------------------------- #
def gamedata_to_dict(g: GameData) -> dict:
    return {
        "NumPlayers": int(g.NumPlayers),
        "Progress": [int(v) for v in g.Progress],
        "Lives": [int(v) for v in g.Lives],
        "Coins": [int(v) for v in g.Coins],
        "Score": [int(v) for v in g.Score],
        "Mode": [int(v) for v in g.Mode],
        "ModPack": g.ModPack,
        "ModProgress": {pack: [int(v) for v in vals]
                        for pack, vals in g.ModProgress.items()},
    }


def gamedata_from_dict(d: dict | None) -> GameData:
    g = GameData()
    if not isinstance(d, dict):
        return g
    g.NumPlayers = int(d.get("NumPlayers", 1))
    for name in ("Progress", "Lives", "Coins", "Score", "Mode"):
        vals = d.get(name)
        if isinstance(vals, list) and len(vals) == 2:
            setattr(g, name, [int(vals[0]), int(vals[1])])
    # Port-only fields: absent in configs written before the mod packs existed.
    pack = d.get("ModPack")
    if isinstance(pack, str):
        g.ModPack = pack
    progress = d.get("ModProgress")
    if isinstance(progress, dict):
        for pack, vals in progress.items():
            if isinstance(vals, list) and len(vals) == 2:
                g.ModProgress[str(pack)] = [int(vals[0]), int(vals[1])]
    return g


@dataclass
class Config:
    sound: bool = True
    sline: bool = True
    use_js: bool = False
    games: list[GameData] = field(
        default_factory=lambda: [GameData() for _ in range(MAX_SAVE)])

    @classmethod
    def load(cls, path: Path) -> "Config":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        cfg = cls()
        cfg.sound = bool(raw.get("Sound", True))
        cfg.sline = bool(raw.get("SLine", True))
        cfg.use_js = bool(raw.get("UseJS", False))
        games = raw.get("Games")
        if isinstance(games, list):
            cfg.games = [gamedata_from_dict(g) for g in games[:MAX_SAVE]]
            while len(cfg.games) < MAX_SAVE:
                cfg.games.append(GameData())
        return cfg

    def save(self, path: Path) -> None:
        data = {
            "format": 1,
            "Sound": self.sound,
            "SLine": self.sline,
            "UseJS": self.use_js,
            "Games": [gamedata_to_dict(g) for g in self.games],
        }
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------- #
# game
# ---------------------------------------------------------------------- #
class MarioGame:
    def __init__(self, assets, screen: VGAScreen, keyboard: Keyboard,
                 clock=None, config_path: Path | None = None,
                 argv: list[str] | None = None) -> None:
        self.assets = assets
        self.screen = screen
        self.keyboard = keyboard
        self.clock = clock
        self.config_path = config_path or Path.cwd() / "MARIO.CFG"
        self.argv = list(argv or [])

        self.config = Config.load(self.config_path)
        # keep the synth alive even when muted: the menu can turn sound on
        self.music = Music()
        self.music.beeper_sound = self.config.sound
        play_unit.STAT = self.config.sline

        self.figures = Figures(assets)
        self.data = GameData()
        self.game_number = -1
        self.next_num_players = 1
        self.quit = False
        self.end_game = False
        # mod level packs: Progress index -> (world, Map1, Map2)
        self.LEVEL_PAIRS_MOD = mod_level_pairs(assets)
        # test switches (not part of MARIO.PAS)
        self.debug_progress: int | None = None
        self.debug_start_level = 0
        self.debug_pack = ""
        self.pending_pack = ""       # pack chosen in MOD LEVELS, until NUMPLAYERS

    # ------------------------------------------------------------------ #
    # command line (ReadCmdLine)
    # ------------------------------------------------------------------ #
    def parse_cmdline(self) -> None:
        args = []
        level_text = None
        i = 0
        while i < len(self.argv):
            token = self.argv[i]
            if token == "--level" and i + 1 < len(self.argv):
                level_text = self.argv[i + 1]
                i += 2
                continue
            if token == "--pack" and i + 1 < len(self.argv):
                self._set_debug_pack(self.argv[i + 1])
                i += 2
                continue
            if token.startswith("--level="):
                level_text = token.split("=", 1)[1]
            elif token.startswith("--pack="):
                self._set_debug_pack(token.split("=", 1)[1])
            elif token in ("--invincible", "--god"):
                play_unit.GOD_MODE = True
            elif token == "--big":
                play_unit.FORCE_MODE = play_unit.MD_LARGE
            elif token == "--fire":
                play_unit.FORCE_MODE = play_unit.MD_FIRE
            elif token == "--small":
                play_unit.FORCE_MODE = play_unit.MD_SMALL
            elif token == "--anim-jitter" or token.startswith("--anim-jitter="):
                palettes_unit.ANIM_JITTER = True
                if "=" in token:
                    palettes_unit.JITTER_SPAN = float(token.split("=", 1)[1])
            else:
                args.append(token)
            i += 1
        if level_text is not None:
            self._set_debug_level(level_text)
        for token in args:
            s = token
            while s:
                if len(s) >= 2 and s[0] in "/-":
                    flag = s[1].upper()
                    if flag == "S":
                        play_unit.STAT = True
                        self.config.sline = True
                    elif flag == "Q":
                        self.music.beeper_sound = False
                        self.config.sound = False
                    elif flag == "J":
                        print("joystick calibration is not ported")
                    elif flag == "?":
                        self.print_help()
                    elif flag == "I":
                        self.print_info()
                    s = s[2:]
                else:
                    s = s[1:]

    def _set_debug_pack(self, text: str) -> None:
        pack = text.strip()
        if pack in self.assets.mod_packs:
            self.debug_pack = pack
        else:
            print(f"bad --pack value {text!r} "
                  f"(expected one of: {', '.join(self.assets.mod_packs)})")

    def _set_debug_level(self, text: str) -> None:
        """--level N-M[a|b] or --level <map name>: throw-away 1P session.

        ``6-4`` is the world/level pair, ``6-4b`` starts on Map2 (normally only
        reachable through a pipe), and a bare map name does the same thing in
        one step: ``4a`` == ``1-4``, ``Ob`` == ``6-4b``.  With ``--pack`` the
        same syntax selects a mod world, and a full name (``8g34/1-1a``) also
        sets the pack.  Nothing is written to the save slots (game_number stays
        -1) and lives/score reset as usual.
        """
        s = text.strip()
        if "/" in s:
            prefix, _, rest = s.partition("/")
            if prefix not in self.assets.mod_packs:
                print(f"bad --level value {text!r} (unknown pack)")
                return
            self.debug_pack = prefix
            s = rest
        if self.debug_pack:
            # mod packs are flat lists of worlds, and TRAP TEST has grown past
            # nine of them, so the world number is not limited to one digit
            m = re.fullmatch(r"(\d{1,2})\s*-\s*(\d{1,2})\s*([abAB])?", s)
            if not m:
                print(f"bad --level value {text!r} (pack {self.debug_pack}: "
                      "expected N-M or N-Ma/N-Mb)")
                return
            world = f"{int(m.group(1))}-{int(m.group(2))}"
            pairs = self.LEVEL_PAIRS_MOD[self.debug_pack]
            idx = next((i for i, p in enumerate(pairs) if p[0] == world), -1)
            if idx < 0:
                print(f"bad --level value {text!r} (pack {self.debug_pack} "
                      f"has no world {world})")
                return
            if (m.group(3) or "a").lower() == "b":
                if pairs[idx][2] is None:
                    print(f"bad --level value {text!r} "
                          f"(world {world} has no sub-level)")
                    return
                self.debug_start_level = 1
            else:
                self.debug_start_level = 0
            self.debug_progress = idx
            return
        m = re.fullmatch(r"([0-9A-Oa-o])([abAB])", s)
        if m:
            map_name = m.group(1).upper() + m.group(2).lower()
            table = LEVEL_A if map_name[1] == "a" else LEVEL_B
            if map_name in table:
                self.debug_progress = table.index(map_name)
                self.debug_start_level = 1 if map_name[1] == "b" else 0
            else:
                print(f"bad --level value {text!r} (no such map)")
            return
        m = re.fullmatch(r"(\d)\s*-\s*(\d)\s*([abAB])?", s)
        if not m:
            print(f"bad --level value {text!r} "
                  "(expected N-M or a map name, e.g. 6-4, 6-4b, Ob)")
            return
        n1, n2 = int(m.group(1)), int(m.group(2))
        if not (1 <= n1 <= 6 and 1 <= n2 <= 4):
            print(f"bad --level value {text!r} (world 1..6, level 1..4)")
            return
        self.debug_progress = (n1 - 1) * 4 + (n2 - 1)
        self.debug_start_level = 1 if (m.group(3) or "a").lower() == "b" else 0

    @staticmethod
    def print_info() -> None:
        print("This game has 24 levels and 3 new graphic tiles.")
        print("Levels by Spenser Whaley; original game by Mike Wiering.")

    @staticmethod
    def print_help() -> None:
        print("Mario & Luigi V2.0")
        print("Left/Right move (A/D), ALT/Z jump, CTRL/X run, SPACE fire, "
              "Up/Down (W/S) aim fireballs and Down enters pipes, "
              "Shift scrolls, S status line, Q sound, P pause.")
        print("/S = status on, /Q = sound off, /I = info, /? = this help.")
        print("--level 6-4 | 6-4b | Ob (map name) = start there, "
              "--pack <id> --level N-M = mod world, --invincible = star on, "
              "--big = large without fireballs, --fire = fireball form, "
              "--small = small (each of those pins the state at every level start), "
              "--anim-jitter[=0.35] = hold each grass/coin/star frame for a random "
              "time instead of exactly the same number of frames.")

    # ------------------------------------------------------------------ #
    # top level
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.parse_cmdline()
        if self.debug_progress is not None:
            self.run_debug()
        else:
            while True:
                self.intro()
                if self.quit:
                    break
                self.play_session()
                if self.quit:
                    break
                # END/Esc in the menu set quit inside intro(); a finished game
                # (all lives gone) falls through to the title again, like MARIO.
        self.config.sound = self.music.beeper_sound
        self.config.sline = play_unit.STAT
        try:
            self.config.save(self.config_path)
        except OSError as exc:
            print(f"could not write {self.config_path}: {exc}")

    def _new_play_world(self, level_a: str, level_b: str, player: int,
                        world_number: str, turbo: bool,
                        data: GameData | None = None) -> PlayWorld:
        """Seam for tests; the real game uses PlayWorld."""
        start_level = self.debug_start_level
        self.debug_start_level = 0        # only the first level uses --level
        return PlayWorld(self.assets, self.screen, self.keyboard,
                         level_a, level_b, player=player,
                         world_number=world_number, turbo=turbo,
                         stat=play_unit.STAT, music=self.music,
                         data=data if data is not None else self.data,
                         start_level=start_level)

    def run_debug(self) -> None:
        """--level: skip the title and play the chosen pair once."""
        self.game_number = -1             # throw-away session, never saved
        self.data.NumPlayers = 1
        prog = self.debug_progress
        if self.debug_pack:
            self.data.ModPack = self.debug_pack
            self.data.ModProgress[self.debug_pack] = [prog, prog]
        else:
            self.data.Progress = [prog, prog]
        self.play_session()

    def _show_page(self) -> None:
        self.screen.show_page()
        self.screen.swap_pages()

    def _taps(self) -> list[tuple[int, str]]:
        return list(self.keyboard.taps)

    def _pump(self) -> bool:
        self.keyboard.pump()
        if self.keyboard.quit:
            self.quit = True
            return True
        return False

    def _tick(self) -> None:
        if self.clock is not None:
            self.clock.tick(60)

    # ------------------------------------------------------------------ #
    # Intro: title screen + menu state machine
    # ------------------------------------------------------------------ #
    def intro(self) -> None:
        self.game_number = -1
        self.quit = False
        self.pending_pack = ""
        self.data = GameData()
        self.data.NumPlayers = self.next_num_players
        next_players = self.next_num_players

        pw = PlayWorld(self.assets, self.screen, self.keyboard,
                       "Intro_0", "Intro_0", player=0, world_number="",
                       stat=False, music=self.music, only_draw=True)
        pw.run()
        self._draw_title_static(pw)
        base = [p.subsurface((0, 0, 320, VIR_SCREEN_HEIGHT)).copy()
                for p in self.screen.pages]
        title_world = pw

        status = ST_MENU
        selected = 1
        counter = 1
        intro_done = False

        while not intro_done and not self.quit:
            if self._pump():
                return
            taps = self._taps()
            old_status = status

            # MARIO.PAS sees one scancode per loop; honour the last tap only.
            for key, _ch in taps[-1:]:
                if key == pygame.K_ESCAPE:
                    if status == ST_MENU:
                        self.quit = True
                        break
                    status = MENU_PARENT[status]
                    if status == ST_START:
                        self.pending_pack = ""
                elif key == pygame.K_UP:
                    if selected == 1:
                        if status == ST_MENU:
                            selected = self._menu_num(status)
                        else:
                            status = MENU_PARENT[status]
                    else:
                        selected -= 1
                elif key == pygame.K_DOWN:
                    num = self._menu_num(status)
                    if selected == num:
                        if status == ST_MENU:
                            selected = 1
                        else:
                            status = MENU_PARENT[status]
                    else:
                        selected += 1
                elif key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                             pygame.K_SPACE):
                    action = self._menu_activate(status, selected)
                    if action is not None:
                        new_status, intro_done, next_players = action
                        if new_status is not None:
                            status = new_status
                if self.quit or intro_done:
                    break

            if status != old_status:
                selected = self._default_selected(status)
            if taps:
                counter = 0

            # menu page: restore the title snapshot, then the menu text
            page = self.screen.page
            page.blit(base[self.screen.current], (0, 0))
            self._draw_menu(title_world, status, selected, counter)
            self._show_page()
            title_world.pal.blink_palette()
            title_world.pal.apply()
            counter += 1
            if counter >= WAIT_BEFORE_DEMO and not intro_done:
                self._demo()
                counter = 0
            self._tick()

        if self.quit:
            return
        if self.game_number >= 0:
            self.data = copy.deepcopy(self.config.games[self.game_number])
        if self.pending_pack:
            self.data.ModPack = self.pending_pack
            self.data.ModProgress.setdefault(self.pending_pack, [0, 0])
        self.data.NumPlayers = next_players
        self.next_num_players = next_players
        self.pending_pack = ""

    def _default_selected(self, status: str) -> int:
        if status == ST_NUMPLAYERS:
            return self.data.NumPlayers
        return 1

    def _menu_num(self, status: str) -> int:
        return {ST_MENU: 3, ST_OPTIONS: 2, ST_START: 4,
                ST_NUMPLAYERS: 2, ST_LOAD: 3, ST_ERASE: 3,
                ST_MODPACKS: max(1, len(self.LEVEL_PAIRS_MOD)),
                ST_MODSLOT: 3}[status]

    def _menu_activate(self, status: str, selected: int):
        """Execute Enter/Space.  Returns (new_status, intro_done, players)."""
        if status == ST_MENU:
            if selected == 1:
                return ST_START, False, self.next_num_players
            if selected == 2:
                return ST_OPTIONS, False, self.next_num_players
            self.quit = True
            return None
        if status == ST_START:
            if selected == 1:
                self.pending_pack = ""
                return ST_LOAD, False, self.next_num_players
            if selected == 2:
                self.pending_pack = ""
                return ST_NUMPLAYERS, False, self.next_num_players
            if selected == 3:
                return ST_MODPACKS, False, self.next_num_players
            self.pending_pack = ""
            return ST_ERASE, False, self.next_num_players
        if status == ST_MODPACKS:
            packs = list(self.assets.mod_packs)
            if selected <= len(packs):
                self.pending_pack = packs[selected - 1]
                return ST_MODSLOT, False, self.next_num_players
            return None
        if status == ST_MODSLOT:
            self.game_number = selected - 1
            self.config.games[self.game_number].NumPlayers = 1
            return ST_NUMPLAYERS, False, self.next_num_players
        if status == ST_OPTIONS:
            if selected == 1:
                self.music.beeper_sound = not self.music.beeper_sound
                self.config.sound = self.music.beeper_sound
            else:
                play_unit.STAT = not play_unit.STAT
                self.config.sline = play_unit.STAT
            return None
        if status == ST_NUMPLAYERS:
            self.next_num_players = 1 if selected == 1 else 2
            return None, True, self.next_num_players
        if status == ST_LOAD:
            self.game_number = selected - 1
            slot = self.config.games[self.game_number]
            slot.NumPlayers = 1
            if slot.ModPack in self.assets.mod_packs:
                self.next_num_players = slot.NumPlayers
                return None, True, self.next_num_players
            if slot.Progress[PL_MARIO] == 0 and slot.Progress[PL_LUIGI] == 0:
                return ST_NUMPLAYERS, False, self.next_num_players
            self.next_num_players = slot.NumPlayers
            return None, True, self.next_num_players
        if status == ST_ERASE:
            self.config.games[selected - 1] = GameData()
            self.config.games[selected - 1].NumPlayers = 1
            self.game_number = -1
            return None
        return None

    def _menu_strings(self, status: str) -> list[str]:
        if status == ST_MENU:
            return ["START", "OPTIONS", "END", "", ""]
        if status == ST_OPTIONS:
            return [
                "SOUND ON " if self.music.beeper_sound else "SOUND OFF",
                "STATUSLINE ON " if play_unit.STAT else "STATUSLINE OFF",
                "", "", ""]
        if status == ST_START:
            return ["LOAD A GAME", "NO SAVE", "MOD LEVELS", "ERASE", ""]
        if status == ST_NUMPLAYERS:
            return ["ONE PLAYER", "TWO PLAYERS", "", "", ""]
        if status == ST_MODPACKS:
            menu = [f"{pack.upper()} - "
                    f"{len(self.LEVEL_PAIRS_MOD[pack])} "
                    f"WORLD{'S' if len(self.LEVEL_PAIRS_MOD[pack]) != 1 else ''}"
                    for pack in self.assets.mod_packs]
            return (menu + ["", "", ""])[:5]
        if status == ST_MODSLOT:
            return ([self._mod_slot_string(i, self.pending_pack)
                     for i in range(3)] + ["", ""])
        return [self._slot_string(i) for i in range(3)] + ["", ""]

    def _slot_string(self, index: int) -> str:
        slot = self.config.games[index]
        s = f"GAME {index + 1} {chr(7)} "
        pack = slot.ModPack
        if pack in self.LEVEL_PAIRS_MOD:
            j = max(slot.ModProgress.get(pack, [0, 0]))
            turbo = j >= len(self.LEVEL_PAIRS_MOD[pack])
            s += "IN USE" + ("* " if turbo else f"{chr(7)} ")
            return s + f"{pack.upper()} {slot.NumPlayers}P"
        if slot.Progress[PL_MARIO] == 0 and slot.Progress[PL_LUIGI] == 0:
            return s + "EMPTY"
        j = slot.Progress[PL_MARIO]
        k = int(j >= NUM_LEV)
        if k:
            j -= NUM_LEV
        if slot.Progress[PL_LUIGI] > j:
            j = slot.Progress[PL_LUIGI]
        s += "IN USE" + ("* " if k else f"{chr(7)} ")
        return s + f"{slot.NumPlayers}P"

    def _mod_slot_string(self, index: int, pack: str) -> str:
        slot = self.config.games[index]
        s = f"GAME {index + 1} {chr(7)} "
        if pack not in self.LEVEL_PAIRS_MOD:
            return s + "EMPTY"
        prog = slot.ModProgress.get(pack)
        if not prog or (prog[0] == 0 and prog[1] == 0):
            return s + "EMPTY"
        turbo = max(prog) >= len(self.LEVEL_PAIRS_MOD[pack])
        s += "IN USE" + ("* " if turbo else f"{chr(7)} ")
        return s + f"{slot.NumPlayers}P"

    def _draw_menu(self, pw: PlayWorld, status: str,
                   selected: int, counter: int) -> None:
        menu = self._menu_strings(status)
        widths = [pw.txt.text_width(s) for s in menu]
        widest = max(widths) if widths else 0
        xp = ((320 - widest) // 2) // 4 * 4
        pw.figures.change_palette(14, 63, 61, 31)
        pw.pal.apply()
        pw.txt.set_font(0, BOLD | SHADOW)
        for k in range(1, 6):
            s = menu[k - 1]
            if not s:
                continue
            y = 56 + 14 * k
            if k == selected:
                pw.txt.write_text(xp - 12, y, chr(16), 5)
            attr = 15
            if len(s) > 19 and s[18] == "*":
                attr = 14 + (counter & 1)
            pw.txt.write_text(xp + 8, y, s, attr)

    # ------------------------------------------------------------------ #
    # title art
    # ------------------------------------------------------------------ #
    def _draw_title_static(self, pw: PlayWorld) -> None:
        """PlayWorld(only_draw) for Intro_0, then the MARIO title page."""
        pal = pw.pal
        pal.out_palette(0xA0, 35, 45, 50)
        pal.out_palette(0xA1, 45, 55, 60)
        pal.out_palette(0xEF, 30, 40, 30)
        pal.out_palette(0x18, 10, 15, 25)
        pal.out_palette(0x8D, 28, 38, 50)
        pal.out_palette(0x8F, 40, 50, 63)
        for _ in range(50):
            pal.blink_palette()
        pal.apply()

        # Re-render the Intro_0 background with the final palette (the DOS
        # version relies on the DAC changing under already-drawn pixels).
        for _ in range(2):
            pw._repaint_static()
            pw.backgr.draw_backgr(self.screen.page, 0, 0, True)
            self.screen.swap_pages()

        # InitBackGr(3) happens after the background was drawn (MARIO.PAS);
        # the new height map is only used by DrawBackGrMap below.
        pw.backgr.init_backgr(3, 0)
        intro = ("INTRO000", "INTRO001", "INTRO002")
        wood = pw.figures.sprite_surf("WOOD000")
        for _ in range(2):
            page = self.screen.page
            for i in (1, 0):
                for j in (1, 0):
                    for k in (1, 0):
                        page.blit(pw.figures.sprite_surf(intro[0]),
                                  (38 + i + j, 29 + i + k))
                        page.blit(pw.figures.sprite_surf(intro[1]),
                                  (159 + i + j, 29 + i + k))
                        page.blit(pw.figures.sprite_surf(intro[2]),
                                  (198 + i + j, 29 + i + k))
            for shift, color in ((54, 0xA0), (55, 0xA1), (53, 0xA1)):
                pw.backgr.draw_backgr_map(page, 10 * 14 + 6, 11 * 14 - 1,
                                          shift, color)
            for x in range(16):
                for y in range(13):
                    if x in (0, 15) or y in (0, 12):
                        page.blit(wood, (x * 20, y * 14))
            pw.player.draw_player()
            self._show_page()

        pal.unlock_pal()
        pal.apply()
        pal.fade_up(64)
        self.screen.reset_stack()

    # ------------------------------------------------------------------ #
    # demo (menu timeout); macro playback is not ported
    # ------------------------------------------------------------------ #
    def _demo(self) -> None:
        demo_data = GameData()
        demo_data.Progress[PL_MARIO] = 5
        pw = PlayWorld(self.assets, self.screen, self.keyboard,
                       "Intro_1", "Intro_1", player=PL_MARIO,
                       world_number="1-0", turbo=False,
                       stat=play_unit.STAT, music=self.music,
                       data=demo_data)

        def abort_on_key(p: PlayWorld) -> None:
            if self.keyboard.taps:
                p.quit_game = True

        pw.run(max_frames=900, clock=self.clock, on_frame=abort_on_key)
        if self.keyboard.quit:
            self.quit = True

    # ------------------------------------------------------------------ #
    # ShowPlayerName
    # ------------------------------------------------------------------ #
    def show_player_name(self, player: int) -> bool:
        f = self.figures
        f.reset_palette()
        self.screen.set_palette(f.palette)
        self.screen.set_view(0, 0)
        self.screen.clear()
        name = "START000" if player == PL_MARIO else "START001"
        w = 116 if player == PL_MARIO else 108
        surf = f.sprite_surf(name)
        for _ in range(2):
            self.screen.page.blit(surf, (160 - w // 2, 85 - 13 // 2))
            self._show_page()
        f.reset_palette()
        self.screen.set_palette(f.palette)
        for _ in range(100):
            if self._pump():
                return False
            self.screen.show_page()
            self.screen.swap_pages()
            self._tick()
        self.screen.clear()
        return True

    # ------------------------------------------------------------------ #
    # one session: reset lives/scores, then play levels until all lives end
    # ------------------------------------------------------------------ #
    def play_session(self) -> None:
        data = self.data
        pack = data.ModPack if data.ModPack in self.LEVEL_PAIRS_MOD else ""
        if pack:
            pairs = self.LEVEL_PAIRS_MOD[pack]
            progress = data.ModProgress.setdefault(pack, [0, 0])
        else:
            pairs = LEVEL_PAIRS
            progress = data.Progress
        num_lev = len(pairs) if pack else NUM_LEV
        last_lev = 2 * num_lev - 1

        if data.NumPlayers == 2:
            if progress[PL_MARIO] > progress[PL_LUIGI]:
                progress[PL_LUIGI] = progress[PL_MARIO]
            else:
                progress[PL_MARIO] = progress[PL_LUIGI]
        data.Lives = [5, 5]
        data.Coins = [0, 0]
        data.Score = [0, 0]
        data.Mode = [MD_SMALL, MD_SMALL]

        self.end_game = False
        while not (self.end_game or self.quit
                   or sum(data.Lives) == 0):
            for cur in range(data.NumPlayers):
                if self.end_game or self.quit:
                    break
                if data.NumPlayers == 1:
                    data.Lives[PL_LUIGI] = 0
                if data.Lives[cur] < 1:
                    continue

                prog = progress[cur]
                turbo = prog >= num_lev
                if prog > last_lev:
                    prog = num_lev
                    progress[cur] = prog
                if self.debug_progress is None:
                    if not self.show_player_name(cur):
                        self._save_slot()
                        return

                idx = prog % num_lev
                if idx >= len(pairs):
                    self.end_game = True
                    break
                if pack:
                    world, level_a, level_b = pairs[idx]
                else:
                    n1, n2, level_a, level_b = pairs[idx]
                    world = f"{n1}-{n2}"
                pw = self._new_play_world(level_a, level_b, cur, world, turbo)
                passed = pw.run(clock=self.clock)
                self.config.sound = self.music.beeper_sound
                self.config.sline = play_unit.STAT
                if passed:
                    progress[cur] += 1
                if getattr(pw, "quit_game", False):
                    self.end_game = True
                    break

        self._save_slot()

    def _save_slot(self) -> None:
        if self.game_number >= 0:
            self.config.games[self.game_number] = copy.deepcopy(self.data)


# ---------------------------------------------------------------------- #
# entry point
# ---------------------------------------------------------------------- #
def screen_width(assets) -> int:
    widest = max((lv.get("width") or 0)
                 for lv in (*assets.levels.values(), *assets.mod_levels.values()))
    return max(340, (widest + 1) * 20)


def screen_height(assets) -> int:
    """Page height: the tallest level, but never below the 13-row viewport."""
    tallest = max((int(lv.get("rows") or 13)
                   for lv in (*assets.levels.values(), *assets.mod_levels.values())),
                  default=13)
    return max(VIR_SCREEN_HEIGHT, tallest * TILE_H)


def main(argv: list[str] | None = None) -> int:
    pygame.init()
    assets = load_assets()
    screen = VGAScreen(VgaConfig(width=screen_width(assets),
                                 height=screen_height(assets), scale=2))
    game = MarioGame(assets, screen, Keyboard(), clock=pygame.time.Clock(),
                     argv=list(sys.argv[1:] if argv is None else argv))
    game.run()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
