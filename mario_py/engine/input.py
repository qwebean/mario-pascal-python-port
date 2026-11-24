"""Keyboard input — replaces KEYBOARD.PAS.

The original installs a real-mode INT 09h handler and exposes boolean helpers
(kbLeft, kbAlt, ...) plus a `Key` global holding the last scancode. We mirror
that API on top of pygame's event queue and key state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pygame

# scancode constants used throughout the game (KEYBOARD.PAS)
KB_ESC = 1
KB_TAB = 15
KB_ENTER = 28
KB_SPACE = 57

SHOT_DIR = Path("screenshots")


def save_screenshot() -> str:
    """F12 / PrintScreen hotkey: dump the window to screenshots/<time>.png.
    Runs inside the event pump, so it works in the menu and in a level."""
    surf = pygame.display.get_surface()
    if surf is None:
        return ""
    SHOT_DIR.mkdir(exist_ok=True)
    name = SHOT_DIR / f"mario_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    pygame.image.save(surf, str(name))
    print(f"screenshot -> {name}")
    return str(name)

kbLeft = pygame.K_LEFT
kbRight = pygame.K_RIGHT
kbUp = pygame.K_UP
kbDown = pygame.K_DOWN


@dataclass
class InputState:
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    alt: bool = False       # jump (original: ALT)
    ctrl: bool = False      # run  (original: CTRL)
    shift: bool = False
    left_shift: bool = False
    right_shift: bool = False
    space: bool = False     # fire ball
    key: int = 0            # last scancode-ish code
    key_char: str = ""
    # every KEYDOWN seen during the current pump(): (pygame key, character).
    # PLAY.PAS reads the Key global once per frame, but Pause/cheats need the
    # taps themselves, including several in one frame.
    taps: list[tuple[int, str]] = field(default_factory=list)
    _pressed: set[int] = field(default_factory=set)

    def begin_frame(self) -> None:
        self.key = 0
        self.key_char = ""
        self.taps.clear()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            self.key = event.key
            self.key_char = event.unicode
            self.taps.append((event.key, event.unicode))
            self._pressed.add(event.key)
        elif event.type == pygame.KEYUP:
            self._pressed.discard(event.key)

    def refresh(self) -> None:
        pressed = pygame.key.get_pressed()
        self.left = pressed[kbLeft] or pressed[pygame.K_a]
        self.right = pressed[kbRight] or pressed[pygame.K_d]
        self.up = pressed[kbUp] or pressed[pygame.K_w]
        self.down = pressed[kbDown] or pressed[pygame.K_s]
        # ALT is the original jump key; Z is a modern alias.  Do NOT add
        # Up here: up/down aim fireballs and down enters pipes (PLAYERS.PAS).
        self.alt = pressed[pygame.K_LALT] or pressed[pygame.K_RALT] \
            or pressed[pygame.K_z]
        self.ctrl = pressed[pygame.K_LCTRL] or pressed[pygame.K_RCTRL] \
            or pressed[pygame.K_x]
        self.left_shift = pressed[pygame.K_LSHIFT]
        self.right_shift = pressed[pygame.K_RSHIFT]
        self.shift = self.left_shift or self.right_shift
        self.space = pressed[pygame.K_SPACE]


class Keyboard:
    """Drop-in replacement for the Keyboard unit's global state."""

    def __init__(self) -> None:
        self.state = InputState()
        self.quit = False

    def pump(self) -> None:
        self.state.begin_frame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_F12, pygame.K_PRINTSCREEN):
                save_screenshot()
            self.state.handle_event(event)
        self.state.refresh()

    # --- boolean helpers mirroring KEYBOARD.PAS ---------------------------
    @property
    def left(self) -> bool:
        return self.state.left

    @property
    def right(self) -> bool:
        return self.state.right

    @property
    def up(self) -> bool:
        return self.state.up

    @property
    def down(self) -> bool:
        return self.state.down

    @property
    def alt(self) -> bool:
        return self.state.alt

    @property
    def ctrl(self) -> bool:
        return self.state.ctrl

    @property
    def left_shift(self) -> bool:
        return self.state.left_shift

    @property
    def right_shift(self) -> bool:
        return self.state.right_shift

    @property
    def space(self) -> bool:
        return self.state.space

    @property
    def key(self) -> int:
        return self.state.key

    @property
    def taps(self) -> list[tuple[int, str]]:
        return self.state.taps

    def is_down(self, key: int) -> bool:
        return bool(pygame.key.get_pressed()[key])
