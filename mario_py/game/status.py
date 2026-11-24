"""Port of STATUS.PAS — the top status band.

ShowStatus saves a 320x20 strip at y=6, draws the player name, lives, score,
timer, coins and world number with F8X8 in bold, and HideStatus restores the
strip.  ``time`` counts down in 1/32-second ticks; the display shows
``time div 32`` (3 digits, clamped 0..999).

Port change (2026-11-18, the user's request): on a no_time level the TIME cell
draws nothing at all — not the digits, not the "TIME" label.  The original
always had a timer, so it has no equivalent branch; drawing "---" there was the
port's earlier compromise and is gone.

The coin icon is two glyphs from the F8X8 control range: character 9 in
colour 13, then character 7 in colour 14 on top.
"""
from __future__ import annotations

from ..engine.vga import MAX_PAGE
from .buffers import PLAYER_NAME
from .txt import BOLD, NORMAL

SCREEN_WIDTH = 320
HEIGHT = 6
BAND_HEIGHT = 20


class Status:
    def __init__(self, screen, txt, data, player: int = 0) -> None:
        self.screen = screen
        self.txt = txt
        self.data = data
        self.player = player
        self.time = 0
        self.time_disabled = False
        self.backgr_addr = [0] * (MAX_PAGE + 1)

    def init_status(self) -> None:
        self.backgr_addr = [0] * (MAX_PAGE + 1)

    @staticmethod
    def _str(value: int, width: int) -> str:
        return str(value).rjust(width)

    def show_status(self, x_view: int, level_score: int,
                    world_number: str) -> None:
        page = self.screen.current
        vy = self.screen.view_y
        self.backgr_addr[page] = self.screen.push_backgr(
            x_view, vy + HEIGHT, SCREEN_WIDTH, BAND_HEIGHT)
        t = self.txt
        t.set_font(0, BOLD)
        t.write_text(x_view + 14, vy + HEIGHT, PLAYER_NAME[self.player], 31)
        lives = min(self.data.Lives[self.player], 99)
        t.write_text(x_view + 58, vy + HEIGHT, self._str(lives, 2), 31)

        s = self._str(level_score, 9)
        s = "".join("0" if (i >= 2 and c == " ") else c
                    for i, c in enumerate(s))
        t.write_text(x_view, vy + HEIGHT + 11, s, 31)

        if self.time_disabled:
            # 屏蔽：不计时就不画这一格（连 TIME 标题一起省掉）
            pass
        else:
            i = self.time // 32
            i = 999 if i > 999 else (0 if i < 0 else i)
            t.write_text(x_view + 281, vy + HEIGHT + 11, self._str(i, 3), 31)
            t.write_text(x_view + 278, vy + HEIGHT, "TIME", 31)

        t.write_text(x_view + 79, vy + HEIGHT + 11, chr(9), 13)
        t.write_text(x_view + 79, vy + HEIGHT + 11, chr(7), 14)
        t.write_text(x_view + 98, vy + HEIGHT + 11,
                     self._str(self.data.Coins[self.player], 2), 31)

        t.write_text(x_view + 170, vy + HEIGHT, "WORLD ", 31)
        t.write_text(x_view + 180, vy + HEIGHT + 11, world_number, 31)

        t.set_font(0, NORMAL)
        t.write_text(x_view + 50, vy + HEIGHT, "x", 31)
        t.write_text(x_view + 89, vy + HEIGHT + 11, "x", 31)

    def hide_status(self) -> None:
        page = self.screen.current
        if self.backgr_addr[page] != 0:
            self.screen.pop_backgr(self.backgr_addr[page])
        self.backgr_addr[page] = 0
