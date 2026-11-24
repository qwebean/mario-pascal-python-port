"""Mike Wiering's Mario & Luigi — Python port.

Original: Turbo Pascal 5.5, DOS, VGA Mode 13h (320x200, 256 colors).
The main line follows the 2006 23-level mod; several older builds are also
playable as separate level packs.  This package ports the game logic to
Python + pygame.
"""

__version__ = "0.1.0"

# Original display geometry (from BUFFERS.PAS / VGA256.PAS)
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 200

TILE_W = 20          # W  in BUFFERS.PAS
TILE_H = 14          # H  in BUFFERS.PAS
NV = 13              # visible map rows
NH = 16              # visible map columns
YBASE = 9            # top offset of the playfield

MAX_WORLD_SIZE = 236
EX = 1               # extra columns of solid border
EY1 = 8              # extra rows above (sky / out-of-world)
EY2 = 3              # extra rows below

# Direction / mode / player enums (mirroring the Pascal constants)
DIR_LEFT, DIR_RIGHT = 0, 1
MD_SMALL, MD_LARGE, MD_FIRE = 0, 1, 2
PL_MARIO, PL_LUIGI = 0, 1

# Player physics (PLAYERS.PAS)
SCROLL_AT = 112
JUMP_VEL = 4
JUMP_DELAY = 6
MAX_Y_VEL = JUMP_VEL * 2
SLIP = 6
BLINK_TIME = 125
STAR_TIME = 750
GROW_TIME = 24
MAX_SPEED = 2

# Collision sets (BUFFERS.PAS) — tile codes as byte values
CAN_HOLD_YOU = set(range(0, 14)) | set(ord(c) for c in "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CAN_STAND_ON = set(range(14, 17)) | set(ord(c) for c in "abcdef")
HIDDEN = {ord("$")}
