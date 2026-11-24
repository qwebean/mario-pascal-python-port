"""Port of MUSIC.PAS — the PC-speaker tunes.

The original keeps one ShortString per tune.  Every byte is either:

* ``> #1``  a countdown: decrement each frame (a rest / note length), or
* ``#1`` or ``#0``: advance one byte, read the next byte as a note index and
  ``Beep`` it (index 0 means silence), then advance past it.

``PlayMusic`` runs once per frame and starts with ``NoSound``, so a note
sounds for at most one frame; the following countdown bytes are rests.  The
port keeps that exactly and synthesises square waves with numpy.

``BeeperSound`` (BUFFERS.PAS) gates everything; ``play.py``'s Q key toggles
the ``beeper_sound`` attribute directly.

The mixer is optional: if pygame cannot open an audio device the class turns
itself into a silent no-op, so headless tests and machines without sound
still run.
"""
from __future__ import annotations

import numpy as np
import pygame


class MusicHooks:
    """Silent MUSIC.PAS stand-in.  ``Music`` below is the playable version."""

    def __init__(self) -> None:
        self.beeper_sound = True
        self.current = None

    def beep(self, freq: int) -> None:
        return None

    def start_music(self, name: str) -> None:
        self.current = name

    def play_music(self) -> None:
        return None

    def pause_music(self) -> None:
        return None

    def stop_music(self) -> None:
        self.current = None


# MUSIC.PAS note indices
C0, D0, E0, F0, G0, A0, B0 = 1, 3, 5, 6, 8, 10, 12
C1, D1, E1, F1, G1, A1, B1 = 13, 15, 17, 18, 20, 22, 24
C2, D2, E2, F2, G2, A2, B2 = 25, 27, 29, 30, 32, 34, 36
C3, D3, E3, F3, G3, A3, B3 = 37, 39, 41, 42, 44, 46, 48
C4, D4, E4, F4, G4, A4, B4 = 49, 51, 53, 54, 56, 58, 60
C5, D5, E5, F5, G5, A5, B5 = 61, 63, 65, 66, 68, 70, 72
C6, D6, E6, F6, G6, A6, B6 = 73, 75, 77, 78, 80, 82, 84

# the Pascal string literals, byte for byte (without the length prefix)
LIFE = bytes((1, G4, 8, C5, 8, E5, 8, C5, 8, D5, 8, G5, 8, 0))
GROW = bytes((1, C3, 4, G3, 4, C4, 4, 38, 4, 45, 4, 50, 4,
              D3, 4, A3, 4, D4, 4, 0))
COIN = bytes((1, F5, 1, 0))
PIPE = bytes((1, C1, 0, C1, 8, C0, 0, C0, 16,
              1, C1, 0, C1, 8, C0, 0, C0, 16,
              1, C1, 0, C1, 8, C0, 0, C0, 16, 0))
FIRE = bytes((1, E3, 1, A3, 1, 0))
HIT = bytes((1, C2, 2, C1, 3, C0, 4, C2, 1, C1, 2, C0, 3, 0))
DEAD = bytes((1, C2, 3, C1, 4, C0, 6, 0))
NOTE = bytes((1, C0, 3, C1, 2, C2, 1, 0))
STAR = bytes((1, C3, 4, E3, 4, G3, 4,
              C4, 4, E4, 4, G4, 4,
              C5, 4, E5, 4, G5, 4, C6, 4, 0))
TIME = bytes((1, C2, 3, C1, 4, C0, 6, 0,
              1, C0, 3, C1, 2, C2, 1, 0,
              1, C2, 3, C1, 4, C0, 6, 0,
              1, C0, 3, C1, 2, C2, 1, 0,
              1, E3, 1, A3, 1, 0,
              1, E3, 1, A3, 1, 0))
BOUNCE = bytes((9, 9, 9, 0, 0, 9, 9, 9, 0, 0,
                9, 9, 9, 0, 0, 7, 7, 7, 0, 0,
                7, 7, 7, 0, 0, 7, 7, 7, 0, 0,
                5, 5, 5, 0, 0, 5, 5, 5, 0, 0,
                5, 5, 5, 0, 0, 3, 3, 3, 0, 0,
                3, 3, 3, 0, 0, 3, 3, 3, 0, 0,
                2, 2, 2, 0, 0, 2, 2, 2, 0, 0,
                2, 2, 2, 2, 2, 2))

SCORES = {
    "life": LIFE,
    "grow": GROW,
    "coin": COIN,
    "pipe": PIPE,
    "fire": FIRE,
    "hit": HIT,
    "dead": DEAD,
    "note": NOTE,
    "star": STAR,
    "time": TIME,
    "bounce": BOUNCE,
}

HALF_NOTE = 1.059463094
MAX_OCT = 7
MAX_NOTE = MAX_OCT * 12

# aiNote[i] = Round(55 * 1.059463094 ^ i), i = 1..84  (MUSIC.PAS init loop)
NOTES = [0] + [int(55.0 * HALF_NOTE ** i + 0.5)
               for i in range(1, MAX_NOTE + 1)]


class Music(MusicHooks):
    """MUSIC.PAS + a square-wave synth on one mixer channel."""

    def __init__(self, volume: float = 0.22) -> None:
        super().__init__()
        self.volume = volume
        self.score = bytearray()
        self.pos = -1
        self.last_freq = 0
        self._channel = None
        self._sounds: dict[int, pygame.mixer.Sound] = {}
        self._rate = 0
        self._channels = 0
        self.available = self._open_mixer()

    # ------------------------------------------------------------------ #
    # mixer
    # ------------------------------------------------------------------ #
    def _open_mixer(self) -> bool:
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=22050, size=-16, channels=1,
                                  buffer=512)
            info = pygame.mixer.get_init()
            if info is None:
                return False
            self._rate, _fmt, self._channels = info
            pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
            self._channel = pygame.mixer.Channel(0)
            return True
        except pygame.error:
            return False

    def _tone(self, freq: int) -> None:
        if self._channel is None:
            return
        sound = self._sounds.get(freq)
        if sound is None:
            n = max(2, int(self._rate * 0.25))
            t = np.arange(n, dtype=np.int64)
            square = np.where(((t * freq * 2) // self._rate) % 2 == 0,
                              1.0, -1.0)
            samples = (square * (32767 * self.volume)).astype(np.int16)
            if self._channels > 1:
                samples = np.repeat(samples[:, None], self._channels, axis=1)
            sound = pygame.sndarray.make_sound(np.ascontiguousarray(samples))
            self._sounds[freq] = sound
        self._channel.play(sound)

    def no_sound(self) -> None:
        if self._channel is not None:
            self._channel.stop()

    # ------------------------------------------------------------------ #
    # the MUSIC.PAS state machine
    # ------------------------------------------------------------------ #
    def start_music(self, name: str) -> None:
        if not self.beeper_sound:
            return
        super().start_music(name)
        score = SCORES.get(name)
        if score is None:
            return
        self.score = bytearray(score)
        self.pos = 0

    def start_score(self, score: bytes) -> None:
        """StartMusic with a raw tune (the Pascal call signature)."""
        if not self.beeper_sound:
            return
        self.score = bytearray(score)
        self.pos = 0

    def _step(self):
        """One PlayMusic step; returns the frequency to sound (or None)."""
        if self.pos < 0 or self.pos >= len(self.score):
            return None
        c = self.score[self.pos]
        if c > 1:
            self.score[self.pos] = c - 1
            return None
        new_pos = self.pos + 1
        note = self.score[new_pos] if new_pos < len(self.score) else 0
        self.pos = new_pos + 1
        if 0 < note < len(NOTES):
            return NOTES[note]
        return None

    def play_music(self) -> None:
        if not self.beeper_sound:
            self.no_sound()
            return
        self.no_sound()
        freq = self._step()
        self.last_freq = freq or 0
        if freq:
            self.beep(freq)

    def pause_music(self) -> None:
        self.no_sound()

    def stop_music(self) -> None:
        self.no_sound()
        self.score = bytearray()
        self.pos = -1
        super().stop_music()

    # ------------------------------------------------------------------ #
    # Buffers.Beep
    # ------------------------------------------------------------------ #
    def beep(self, freq: int) -> None:
        if not self.beeper_sound:
            self.no_sound()
            return
        if freq <= 0:
            self.no_sound()
            return
        self._tone(int(freq))
