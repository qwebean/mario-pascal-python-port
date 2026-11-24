"""Port of BLOCKS.PAS — the single-block bump animation.

When the player hits a block from below, ``BumpBlock`` snapshots the block and
the strip of background above it, then the block is drawn a few pixels higher
for a few frames (``MoveBlocks``) and the gap below is repainted from the
background (``DrawBlocks``).  Only one block can bump at a time; a second hit
while ``Bumping`` is ignored.

Everything is drawn on the current video page.  ``EraseBlocks`` runs at the
start of every frame and restores the snapshot, so the animation survives the
two-page flip.
"""
from __future__ import annotations

import pygame

from .buffers import TILE_H, TILE_W

W = TILE_W
H = TILE_H

BUMP_HEIGHT = 4
MOVE_DELAY = 0


class Blocks:
    """BLOCKS.PAS globals + procedures."""

    def __init__(self, screen, backgr) -> None:
        self.screen = screen
        self.backgr = backgr
        self.bumping = False
        self.bump_x = 0
        self.bump_y = 0
        self.dy = 0
        self.delay_counter = 0
        self._backgr_surf: pygame.Surface | None = None
        self._backgr_pos = (0, 0)
        self._block_surf: pygame.Surface | None = None

    # ------------------------------------------------------------------ #
    # pixel snapshots (GetImage / PutImage, clipped like the VGA asm)
    # ------------------------------------------------------------------ #
    def _get(self, x: int, y: int, w: int, h: int):
        rect = pygame.Rect(x, y, w, h).clip(self.screen.page.get_rect())
        if rect.width <= 0 or rect.height <= 0:
            return None, (x, y)
        return self.screen.page.subsurface(rect).copy(), (rect.x, rect.y)

    def _put(self, surf: pygame.Surface | None, pos: tuple[int, int]) -> None:
        if surf is not None:
            self.screen.page.blit(surf, pos)

    def _save_bump_backgr(self) -> None:
        self._backgr_surf, self._backgr_pos = self._get(
            self.bump_x, self.bump_y - BUMP_HEIGHT, W, H + BUMP_HEIGHT)

    # ------------------------------------------------------------------ #
    # public procedures (played in PLAY.PAS order)
    # ------------------------------------------------------------------ #
    def init_blocks(self) -> None:
        self.bumping = False

    def bump_block(self, x: int, y: int) -> None:
        if self.bumping:
            return
        self.bump_x = x
        self.bump_y = y
        self.dy = -BUMP_HEIGHT
        self._block_surf, _ = self._get(x, y, W, H)
        self._save_bump_backgr()
        self.bumping = True
        self.delay_counter = 0

    def erase_blocks(self) -> None:
        if self.bumping:
            self._put(self._backgr_surf, self._backgr_pos)

    def draw_blocks(self) -> None:
        if not self.bumping:
            return
        if self.dy < BUMP_HEIGHT:
            self._save_bump_backgr()
            y = self.bump_y - BUMP_HEIGHT + abs(self.dy)
            self._put(self._block_surf, (self.bump_x, y))
            self.backgr.draw_backgr_block(
                self.screen.page, self.bump_x, y + H, W,
                BUMP_HEIGHT - abs(self.dy))
        elif self.delay_counter >= 4:
            self.bumping = False

    def move_blocks(self) -> None:
        if self.bumping:
            self.delay_counter += 1
            if self.delay_counter > MOVE_DELAY and self.dy < BUMP_HEIGHT:
                self.dy += 1
                self.delay_counter = 0
