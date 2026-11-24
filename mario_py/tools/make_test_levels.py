"""Generate the port-only TRAP TEST pack (data/levels/levels_test.json).

Four levels, all started from the MOD LEVELS menu or with
``--level testrooms/1-1`` / ``1-2`` / ``1-3`` / ``1-4``:

1-1  scroll-gate room.  The room is sealed on the left by a full wall and on
     the right by a wall with a high opening.  The bottom tile of each wall is
     the original $AE/$AF screen marker (174/175): the camera locks against the
     wall until Mario is at the opening's height, exactly like the DOS levels
     that use these unused codes.

1-2  battle room.  Same door geometry, but the $AF wall is solid from floor to
     ceiling.  The level carries a ``battle_room`` block; when every listed
     Chibibo is defeated the port clears the wall cells, which also releases
     the camera (the marker check sees empty space at Mario's row).

1-3  vertical test tower.  26 rows (two screens), a zig-zag climb and an exit
     pipe near the top.  Exercises the dynamic ``rows`` / ``vertical`` fields,
     the vertical dead-zone camera, fixed-screen HUD and enemy spawning above
     the original 13-row band.

1-4  flying-Koopa playground.  Green flyers hop along the ground; red flyers
     patrol a straight line (horizontal or vertical) and turn back.  Stomping
     either one turns it into an ordinary green/red Koopa.

Map rows here are SOURCE rows: row 0 is the bottom of the level, row
``rows-1`` the top.  The converter's ``map`` format uses the same order.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mario_py" / "data" / "levels"
NV = 13
TILE_H = 14

CH_CHIBIBO = 0x80
CH_FLY_GREEN = 0x8B
CH_FLY_RED_H = 0x8C
CH_FLY_RED_V = 0x8D
CH_FLY_BLACK = 0x8E
CH_SPINY_BLACK = 0x8F
CH_FLY_SPINY = 0x90
CH_FLY_SPINY_BLACK = 0x91
CH_FLY_FISH = 0x92            # fish spout, never consumed
CH_FLY_CHIBIBO = 0x93         # winged Chibibo (SMB3 Paragoomba)
CH_BUZZY = 0x94               # SMB3 Buzzy: slow, fireproof, star/shell-killable
CH_FLY_BUZZY = 0x95           # winged Buzzy: cruises flat, no hop
CH_DRY_BONES = 0x96           # Dry Bones: stomp collapses it, then it revives
CH_HAMMER_BRO = 0x97          # Hammer Bro: patrols left of its marker, throws
CH_BOO = 0x98                 # Boo: chases through terrain when unseen
CH_BOMB_STAND = 0x99          # bomb stand: fires a bomb every 3 s on screen
CH_THWOMP = 0x9A              # stone shield: slams, cannot be killed
CH_LAKITU = 0x9B              # Lakitu switch: the cloud then rides the camera
CH_FIRE_PLANT = 0x9C          # red-green piranha that spits while chomping
CH_FIRE_PLANT_AIM = 0x9D      # same plant, aims the flame at Mario
CH_FAST_PLANT = 0x9E          # XP's red piranha: in and out in 1.25 s
CH_SIDE_PLANT = 0x9F          # SMB3's sideways piranha, reaches on x
CH_BLACK_PLANT = 0xA0         # black piranha: green plant's cadence, rank-3 body
CH_PLANT_GREEN = 0x85         # the original plain green plant marker
CH_PLANT1 = CH_PLANT_GREEN      # same code, clearer name for the zoo level
CH_BLACK_KOOPA = 0x8A      # koopa marker index 2
CH_GHOST_CHIBIBO = 0xA1    # the black Chibibo that leaves a blue shadow
CH_VERT_FISH = 0x81        # the original fish marker
CH_VERT_FIREBALL = 0x82    # the fireball that spits straight up out of lava
CH_CHIBIBO_B = 0x83        # brown Chibibo, SubTp 1
CH_PLANT0 = 0x84           # the original vertical piranha markers
CH_PLANT2 = 0x86
CH_RED = 0x87              # Spiny
CH_KOOPA0 = 0x88           # green Koopa
CH_KOOPA1 = 0x89           # red Koopa
CH_EXIT = 0xE7          # end-of-level pipe code
GATE_LEFT = 174         # '«' -> row-`rows` marker 254 (blocks left scroll)
GATE_RIGHT = 175        # '¤' -> row-`rows` marker 255 (blocks right scroll)

GROUND = ord("A")
GREY = ord("I")
SPACE = ord(" ")

# Design 1 decor (FIGURES.redraw): grass top and the four tall-palm pieces.
CH_GRASS_TOP = 247
CH_PALM_TOP = 246
CH_PALM_SIDE = 249
CH_PALM_MID = 250

# Charlie scenery (FIGURES.draw_decorations); same values as the CH_CHAR_*
# constants there.  Non-solid on purpose.
CH_CHAR_GRASS = ord("g")
CH_CHAR_FLOWER = ord("o")
CH_CHAR_CATTAIL = ord("t")
CH_CHAR_PALM = ord("p")


def blank(width: int, rows: int = NV) -> list[list[int]]:
    return [[SPACE] * width for _ in range(rows)]


def ground(g: list[list[int]]) -> None:
    for x in range(len(g[0])):
        g[0][x] = GROUND
        g[1][x] = GROUND


def wall(g: list[list[int]], x: int, y0: int, y1: int,
         tile: int = GROUND) -> None:
    for y in range(y0, y1 + 1):
        g[y][x] = tile


def exit_pipe(g: list[list[int]], x: int, base: int = 1) -> None:
    """A two-tile-high pipe whose body sits in row ``base`` (the ground row
    for a floor-level pipe); the ending code goes in the row Mario's body
    occupies when he stands on the top."""
    g[base][x] = ord("2")
    g[base][x + 1] = ord("3")
    g[base + 1][x] = ord("0")
    g[base + 1][x + 1] = ord("1")
    g[base + 2][x] = CH_EXIT
    g[base + 2][x + 1] = CH_EXIT


def columns(g: list[list[int]]) -> list[list[int]]:
    width = len(g[0])
    rows = len(g)
    return [[g[y][x] for y in range(rows)] for x in range(width)]


def level(width: int, *, init_x: int, hint: str,
          rows: int = NV, battle_room: dict | None = None) -> dict:
    options = json.loads(
        (DATA / "levels.json").read_text(encoding="utf-8"))["1a"]["options"]
    options = dict(options, InitX=init_x, InitY=(rows - 4) * TILE_H)
    if rows > NV:
        # Keep the horizon three tiles above the bottom, like the 13-row
        # levels (1a: 140 == 13*14 - 42).
        options["Horizon"] = rows * TILE_H - 42
    entry = {
        "width": width,
        "rows": rows,
        "vertical": rows > NV,
        "map": None,
        "options": options,
        "no_time": True,
        "hint": hint,
    }
    if battle_room is not None:
        entry["battle_room"] = battle_room
    return entry


def gate_room() -> dict:
    width = 48
    g = blank(width)
    ground(g)

    # Left boundary (never passable, marker only for completeness).
    wall(g, 1, 0, 12)
    g[0][1] = GATE_LEFT

    # Right boundary: a visible frame column (35) plus the real marked column
    # (36).  The camera stops with column 35 still on screen, so the player
    # sees the gate instead of hitting an invisible screen edge.
    wall(g, 35, 0, 6)
    wall(g, 36, 0, 6)
    g[0][36] = GATE_RIGHT

    # Staircase, one tile per step.
    for x, y in ((26, 2), (27, 2), (28, 3), (29, 3),
                 (30, 4), (31, 4), (32, 5), (33, 5), (34, 6)):
        g[y][x] = GREY

    # Coins over the steps and a question block in the room.
    for x, y in ((26, 3), (28, 4), (30, 5), (32, 6)):
        g[y][x] = ord("*")
    g[6][18] = ord("?")

    # One enemy for flavour; it is not required for the gate.
    g[2][20] = CH_CHIBIBO

    exit_pipe(g, 42)

    entry = level(width, init_x=5 * 20,
                  hint="CLIMB OVER THE GATE!")
    entry["map"] = columns(g)
    return entry


def battle_room_level() -> dict:
    width = 44
    g = blank(width)
    ground(g)

    # Fully sealed room.  Column 24 is the visible face of the gate; column
    # 25 carries the scroll marker and stops the camera with column 24 on
    # screen.  Both are cleared when the targets are defeated.
    wall(g, 2, 0, 12)
    g[0][2] = GATE_LEFT
    wall(g, 24, 0, 12, tile=GREY)
    wall(g, 25, 0, 12, tile=GREY)
    g[0][25] = GATE_RIGHT

    # Targets: two Chibibos standing on the floor (source row 2 -> world 10).
    g[2][13] = CH_CHIBIBO
    g[2][19] = CH_CHIBIBO

    g[6][16] = ord("?")
    for x in (10, 22):
        g[7][x] = ord("*")

    exit_pipe(g, 38)

    entry = level(width, init_x=5 * 20,
                  hint="DEFEAT THE ENEMIES!",
                  battle_room={
                      # EnemyRec coordinates (world rows): source row 2 == 10.
                      "targets": [[13, 10], [19, 10]],
                      # Interior columns (left wall 2, gate 24/25).  Enemies
                      # touching this range are never horizontally forgotten.
                      "room": [3, 23],
                      # Clear the wall above the floor; keep world rows 11/12
                      # (the ground) so there is no hole to fall through.
                      "barrier": {"cols": [24, 25],
                                  "rows": list(range(0, 11))},
                  })
    entry["map"] = columns(g)
    return entry


def tower_level() -> dict:
    """26-row vertical test: zig-zag platforms and an exit pipe at the top."""
    rows = 26
    width = 20
    g = blank(width, rows)
    ground(g)
    wall(g, 0, 0, rows - 1)
    wall(g, width - 1, 0, rows - 1)

    # (source row, x0, x1): three-row climb per platform, alternating sides.
    for y, x0, x1 in ((4, 4, 9), (7, 9, 14), (10, 4, 9),
                      (13, 9, 14), (16, 4, 9), (19, 9, 14),
                      (20, 15, 19)):
        for x in range(x0, x1 + 1):
            g[y][x] = GREY

    for x, y in ((5, 5), (10, 8), (5, 11), (10, 14),
                 (5, 17), (11, 20), (18, 23)):
        g[y][x] = ord("*")

    # Chibibos on three of the platforms (marker = platform row + 1).
    g[5][6] = CH_CHIBIBO
    g[11][6] = CH_CHIBIBO
    g[20][11] = CH_CHIBIBO

    # Keep the exit two rows below the level top so Mario does not stand
    # behind the fixed-screen HUD.
    exit_pipe(g, 16, base=20)

    entry = level(width, init_x=3 * 20, rows=rows,
                  hint="CLIMB THE TOWER!")
    entry["map"] = columns(g)
    return entry


def flying_level() -> dict:
    """Flying-Koopa playground: hopping green flyers and patrolling red ones."""
    width = 56
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    # Green flyers hop along the floor (marker row 2 = standing on the ground).
    g[2][10] = CH_FLY_GREEN
    g[2][22] = CH_FLY_GREEN

    # Winged Chibibos hop the same way; a stomp strips the wings, a second
    # stomp flattens them (unlike the Spinies they die to fire/star/shells).
    g[2][7] = CH_FLY_CHIBIBO
    g[5][35] = CH_FLY_CHIBIBO

    # Buzzy Beetles: slower than the turtles, fireballs bounce off, but the
    # star and every coloured shell kill them.
    g[2][9] = CH_BUZZY
    g[2][11] = CH_BUZZY

    # Winged Buzzy: it does not hop, it cruises flat along its marker row at
    # 0.25 px/f (slower than the green flyer).  A stomp drops the wings and
    # leaves a walking Buzzy, so fireproof / star- and shell-killable carry over.
    g[6][30] = CH_FLY_BUZZY
    g[8][44] = CH_FLY_BUZZY

    # Black Koopas: fireballs cannot hurt them; shells have their own rules.
    # A green walker is provided so a coloured shell can be kicked into them.
    g[2][13] = 0x88            # green walking Koopa
    g[2][16] = CH_BLACK_KOOPA
    g[2][19] = CH_BLACK_KOOPA
    g[2][27] = CH_FLY_BLACK

    # Spinies: a black one on the floor plus both winged variants.
    g[2][33] = CH_SPINY_BLACK
    g[5][40] = CH_FLY_SPINY
    g[5][48] = CH_FLY_SPINY_BLACK

    # Red flyers: source row 5 -> world row 7, i.e. three tiles above the
    # floor; source row 6 -> world row 6 for the vertical patroller.
    g[5][30] = CH_FLY_RED_H
    g[5][44] = CH_FLY_RED_H
    g[6][37] = CH_FLY_RED_V

    for x, y in ((14, 4), (18, 4), (26, 6), (34, 6), (41, 4), (48, 4)):
        g[y][x] = ord("*")
    g[6][26] = ord("?")

    exit_pipe(g, 50)

    entry = level(width, init_x=5 * 20,
                  hint="BLACK RESISTS FIRE! STAR = STOMP ONLY!")
    entry["map"] = columns(g)
    return entry


def fish_level() -> dict:
    """Flying-fish run: spouts keep launching fish at Mario, and every fish
    that is killed or leaves the world comes back after the next period."""
    width = 64
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    # Spouts sit on the floor (source row 2 = standing height).  The marker is
    # never consumed, so the water keeps launching.
    for x in (10, 17, 24, 31, 38, 45):
        g[2][x] = CH_FLY_FISH

    # Two platforms to duck behind, and a coin trail above the pools.
    for x in range(13, 17):
        g[6][x] = ord("B")
    for x in range(27, 31):
        g[5][x] = ord("B")
    for x in (12, 20, 28, 36, 44):
        g[4][x] = ord("*")

    exit_pipe(g, 56)

    entry = level(width, init_x=4 * 20, hint="FLYING FISH!")
    entry["map"] = columns(g)
    return entry


def dry_bones_level() -> dict:
    """Dry Bones: walks like a Koopa, a stomp only knocks it apart and it gets
    up again ~1.5 s later, for ever.  Fireballs do nothing; the star and any
    shell kill it for good.  A green Koopa is placed so a shell is at hand."""
    width = 56
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    for x in (8, 14, 20, 26):
        g[2][x] = CH_DRY_BONES
    # two on ledges: the pile art is 31 px wide, so check it does not look
    # clipped against the platform edge
    for x in (34, 40):
        g[5][x] = CH_DRY_BONES
        g[4][x - 1] = GREY
        g[4][x] = GREY
        g[4][x + 1] = GREY

    g[2][30] = 0x88                    # green Koopa: kick its shell into them
    for x, y in ((11, 4), (17, 4), (23, 6)):
        g[y][x] = ord("*")
    g[6][35] = ord("?")                # flower -> proves the fire immunity
    exit_pipe(g, 50)
    entry = level(width, init_x=5 * 20,
                  hint="STOMP = FALLS APART, THEN GETS UP. FIRE RESISTS!")
    entry["map"] = columns(g)
    return entry


def hammer_bro_level() -> dict:
    """Hammer Bros: patrol the 43 px band left of their marker, hop every ~2 s
    (small or big), and throw a hammer on the 3rd animation frame.  Stomp,
    fireball and shell all kill."""
    width = 56
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    for x in (12, 20, 28):
        g[2][x] = CH_HAMMER_BRO
    # on platforms: the band is measured from the marker column, so each one
    # paces between its own column and two tiles to the left
    for x in (36, 44):
        g[5][x] = CH_HAMMER_BRO
        for dx in (-2, -1, 0, 1):
            g[4][x + dx] = GREY

    g[2][8] = 0x88                     # a shell to test shell-vs-Hammer Bro
    for x, y in ((15, 4), (24, 6), (32, 4)):
        g[y][x] = ord("*")
    exit_pipe(g, 52)
    entry = level(width, init_x=5 * 20,
                  hint="THEY THROW. STOMP / FIRE / SHELL ALL KILL")
    entry["map"] = columns(g)
    return entry


def boo_level() -> dict:
    """Boos: they only move while you face away, they walk straight through
    terrain, and landing on one hurts.  Fireballs do nothing; star and shell
    kill.  The wall at column 26 proves the terrain pass-through."""
    width = 56
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    g[2][12] = CH_BOO
    g[4][22] = CH_BOO
    g[6][32] = CH_BOO
    wall(g, 26, 2, 6)                  # the Boo ignores this
    for x in (38, 44):
        g[2][x] = CH_BOO
    g[2][16] = 0x88                    # shell kill test
    exit_pipe(g, 52)
    entry = level(width, init_x=5 * 20,
                  hint="BOO MOVES ONLY WHEN YOUR BACK IS TURNED. IT PASSES WALLS")
    entry["map"] = columns(g)
    return entry


def bomb_level() -> dict:
    """Bomb stands at four different heights, each on its own platform, so the
    launch line can be watched while Mario stands on the stand itself: a bomb
    that dies the instant it spawns is stomped by his own feet.

    Stands fire every 3 s while on screen, towards the player.  The stand itself
    is scenery -- it hurts nobody and nothing can destroy it.  Bombs fly dead
    straight, resist fireballs, die to a stomp, and vanish off view."""
    width = 60
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    # (platform row, stand column): the stand sits one row above its platform,
    # so the platform row is stand row - 1.  Heights: ground, +3, +6, +9 rows.
    for stand_row, x in ((2, 12), (5, 22), (8, 32), (11, 42)):
        g[stand_row][x] = CH_BOMB_STAND
        for px in range(x - 2, x + 3):
            g[stand_row - 1][px] = GREY
        # a step so each platform is reachable without a shell jump
        if stand_row > 2:
            g[stand_row - 4][x - 5] = GREY

    g[2][16] = 0x88                    # a shell: shells kill the bombs
    exit_pipe(g, 56)
    entry = level(width, init_x=5 * 20,
                  hint="STANDS ARE SCENERY. BOMBS RESIST FIRE, STOMP KILLS")
    entry["map"] = columns(g)
    return entry


def bomb_line_level(stand_row: int, hint: str) -> dict:
    """One bomb-stand line at a fixed height, so the flight line can be watched
    against the top of Mario's head with nothing else in the corridor.

    ``stand_row`` is the source row of the stand (row 0 is the bottom).  The
    stand's own origin is world y ``(12 - stand_row) * 14`` and the bomb it
    fires takes the box ``[y - 1, y + 12]`` (the -1 is the pitfall-8 fix), so
    measured against the floor -- feet at y=153, box top y=140 small / y=126
    big -- the two heights the user asked for come out like this:

        stand_row 3  bomb box 125..138  = 2 rows above the floor
            small Mario clears it by 2 px, big Mario's head is inside it
        stand_row 4  bomb box 111..124  = 3 rows above the floor
            big Mario clears it by 2 px

    Try each with ``--small`` and ``--big`` (FORCE_MODE).
    """
    width = 44
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    for x in (10, 26):                     # two stands: fire from both sides
        g[stand_row][x] = CH_BOMB_STAND
        for y in range(2, stand_row):      # the pillar it stands on
            g[y][x] = GREY
    exit_pipe(g, 40)
    entry = level(width, init_x=5 * 20, hint=hint)
    entry["map"] = columns(g)
    return entry


def thwomp_level() -> dict:
    """Stone shields: slam down at 1.875 px/f, creep the last pixel, back up at
    0.469 px/f, pause 0.5 s under a ceiling.  They cannot be killed at all, and
    a shield that starts inside terrain pushes itself free."""
    width = 64
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    g[4][10] = CH_THWOMP               # free fall onto the floor
    g[6][16] = CH_THWOMP               # only one tile to fall
    wall(g, 22, 8, 10)                 # a ceiling block
    g[7][22] = CH_THWOMP               # reverses against it
    for y in (2, 3):
        g[y][30] = GREY                # buried shield: must unstick upwards
    g[5][30] = CH_THWOMP
    g[4][40] = CH_THWOMP
    g[4][48] = CH_THWOMP
    exit_pipe(g, 58)
    entry = level(width, init_x=5 * 20,
                  hint="STONE SHIELDS CANNOT BE KILLED. TIME THE CROSSING")
    entry["map"] = columns(g)
    return entry


def lakitu_level() -> dict:
    """Vertical cloud test: 26 rows, so the camera scrolls up and down while the
    Lakitu switch is live.  The cloud is re-placed every frame at
    ``view_y + yPoint``, so the climb must carry it along the top row -- a cloud
    that stays at its spawn height is exactly what this room shows off.
    The marker is a switch, not a spawn: once the view passes its column the
    cloud rides the camera, drops a Spiny every cycle, and returns 8 s after
    every kill."""
    rows = 26
    width = 20
    g = blank(width, rows)
    ground(g)
    wall(g, 0, 0, rows - 1)
    wall(g, width - 1, 0, rows - 1)

    g[2][3] = CH_LAKITU                # the switch column, on the ground floor
    for y, x0, x1 in ((5, 4, 9), (9, 9, 15), (13, 3, 9),
                      (17, 9, 15), (21, 3, 9)):
        for x in range(x0, x1 + 1):
            g[y][x] = GREY

    g[2][7] = 0x88                     # a shell: shells kill the cloud too
    g[5][6] = 0x87                     # a walking Spiny to compare with drops

    # keep the exit two rows below the top so Mario does not stand behind the
    # fixed-screen HUD
    exit_pipe(g, 16, base=22)

    entry = level(width, init_x=3 * 20, rows=rows,
                  hint="CLOUD RIDES THE TOP ROW. CLIMB AND WATCH IT")
    entry["map"] = columns(g)
    return entry


def platform_level() -> dict:
    """Long-platform playground: all four behaviours, one per board."""
    width = 64
    g = blank(width)
    ground(g)
    wall(g, 0, 0, NV - 1)
    wall(g, width - 1, 0, NV - 1)

    # Something to jump from / a visible height reference.
    for x, y in ((8, 6), (30, 6), (44, 6)):
        g[y][x] = ord("I")
    for x, y in ((8, 3), (9, 3), (10, 3)):
        g[y][x] = ord("*")

    entry = level(width, init_x=4 * 20,
                  hint="RIDE THE FOUR PLATFORMS!")
    entry["map"] = columns(g)
    # x/y are map column / SOURCE row (row 0 = bottom), like the map.
    # len is the number of 20 px segments; speed is px/frame (fall = gravity
    # in px/frame^2).  osc uses SMB3's $41-frame hold; patrol reverses at
    # `range` px; path waypoints are [column, source row].
    entry["platforms"] = [
        {"x": 6, "y": 4, "len": 4, "type": "patrol",
         "speed": 0.8, "range": 80},
        {"x": 24, "y": 5, "len": 5, "type": "osc", "speed": 1.0},
        {"x": 38, "y": 4, "len": 3, "type": "fall", "speed": 0.0625},
        {"x": 46, "y": 5, "len": 3, "type": "path", "speed": 0.8,
         "path": [[46, 5], [50, 5], [50, 2], [55, 2], [55, 6]]},
    ]
    return entry


def forced_route_level() -> dict:
    """3-2a: no floor between the two ledges -- the platforms are the route.

    The pit is bottomless; walking off a platform drops Mario below the world
    (PLAY's normal death, then the level restarts at InitX/InitY).  The boards
    are spaced so every jump is ~20-40 px and the moving ones overlap; the
    board after the falling one is deliberately close (the user's playtest).
    """
    width = 64
    g = blank(width)

    # Start ledge (cols 0..7) and end ledge with the exit pipe (cols 56..63).
    for x in range(0, 8):
        for y in range(0, 3):
            g[y][x] = GROUND
    for x in range(56, 64):
        for y in range(0, 3):
            g[y][x] = GROUND
    exit_pipe(g, 59, base=2)

    entry = level(width, init_x=3 * 20,
                  hint="NO FLOOR. RIDE THE PLATFORMS OR DIE.")
    # The ledge is three source rows tall, so the spawn is two tiles above its
    # top (the player's collision box is 2*H tall): (rows-5)*H, not (rows-4)*H.
    entry["options"]["InitY"] = (13 - 5) * TILE_H
    entry["map"] = columns(g)
    entry["platforms"] = [
        # start: gentle patrol, its left end reaches the ledge
        {"x": 10, "y": 4, "len": 4, "type": "patrol",
         "speed": 0.8, "range": 40},
        # SMB3 oscillating board, overlaps the patrol at both extremes
        {"x": 18, "y": 5, "len": 5, "type": "osc",
         "speed": 1.0, "range": 60},
        # drops the moment it is stood on: cross it, do not wait
        {"x": 27, "y": 4, "len": 4, "type": "fall", "speed": 0.0625},
        # The user's 2026-10-25 playtest: the board after the falling one was
        # still out of reach.  It now starts one row *below* P3 (a sinking
        # board must never need an upward jump), overlaps P3's right end, and
        # loops back to that start instead of parking at its far end.
        {"x": 31, "y": 3, "len": 4, "type": "path", "speed": 0.8,
         "path": [[31, 3], [35, 3], [35, 5], [39, 5]], "loop": 1},
        # last two boards, both reachable from the path's high end
        {"x": 42, "y": 4, "len": 4, "type": "patrol",
         "speed": 1.2, "range": 40},
        {"x": 50, "y": 4, "len": 3, "type": "osc",
         "speed": 1.0, "range": 40},
    ]
    return entry


def black_room_level() -> dict:
    """4-1a: pure-black interior (SkyType 3 + BackGrType 0).

    BackGrType 0 is the "flat fill" branch of Figures.draw_sky; with SkyType 3
    the fill colour is the near-black 0xE0 entry.  Design 5 draws '#' as a solid
    red block and '%' as the five-frame LAVA2001..2005 animation; PLAY re-draws
    the '%' row at Options.Horizon - 1 every frame, so keep Horizon=12 (the
    castle-level value) to make the top of the lava pool live.
    """
    width = 64
    g = blank(width)
    ground(g)

    # Stone pillars and ledges against the black background.  Every step is at
    # most 3 rows up and at most 4 columns across: the measured jump envelope
    # (measured jump envelope) rises 60 px walking / 70 px running and spans
    # 54 px walking / 112 px running at the same level.
    for x, top in ((8, 4), (12, 5), (18, 4), (22, 5), (27, 4), (31, 5),
                   (37, 4), (41, 5), (46, 4), (51, 5)):
        wall(g, x, 2, top)
        wall(g, x + 1, 2, top)
    for x in (24, 25, 26):
        g[4][x] = GREY
    for x in (43, 44, 45):
        g[4][x] = GREY

    # Two-tile lava pits: solid body on the bottom row, animated surface on the
    # row PLAY keeps redrawing (world row Horizon-1 == source row 1).  The
    # first one is on the opening screen so the animation is visible at once.
    for x0 in (14, 33):
        for x in (x0, x0 + 1):
            g[0][x] = ord("#")
            g[1][x] = ord("%")

    g[5][10] = ord("?")
    for x in (6, 20, 36, 50):
        g[3][x] = ord("*")

    g[2][20] = CH_DRY_BONES
    g[2][36] = CH_BOO
    g[2][54] = 0x88                    # green Koopa: a shell to kick around
    g[5][30] = CH_FLY_GREEN

    exit_pipe(g, 58)

    entry = level(width, init_x=4 * 20,
                  hint="BLACK ROOM. THE LAVA SURFACE IS ANIMATED")
    entry["options"].update(
        SkyType=3, BackGrType=0, Design=5, Stars=0, Clouds=0, Horizon=12,
        WallType1=102, GroundColor1=104, BrickColor=24, WoodColor=24,
        XBlockColor=104, PipeColor=24,
    )
    entry["map"] = columns(g)
    return entry


def night_grove_level() -> dict:
    """4-2a: night sky over the normal grass/palm theme.

    SkyType 3 is the unused near-black palette, BackGrType 1 puts the mountain
    silhouette over it and Stars=1 enables STARS.PAS (one twinkling star per
    screen column, ~30% kept).  Design stays 1, so the grass and coconut palms
    keep their normal colours -- only the sky changes.
    """
    width = 72
    g = blank(width)
    ground(g)                          # source rows 0/1
    for x in range(width):
        g[2][x] = CH_GRASS_TOP

    # A hill and a plateau, each three source rows above the floor.
    for x in range(14, 23):
        g[2][x] = GROUND
        g[3][x] = GROUND
        g[4][x] = CH_GRASS_TOP
    for x in range(42, 51):
        g[2][x] = GROUND
        g[3][x] = GROUND
        g[4][x] = CH_GRASS_TOP

    # Two-column waterfall off the plateau's left edge ('#'+'%' is the 1a
    # arrangement; Design 1 draws FALL000/FALL001, animated by palette 7..11).
    for y in range(2, 5):
        g[y][40] = ord("#")
        g[y][41] = ord("%")

    # Tall palms: base = the top ground row they stand on.
    for x, base in ((6, 1), (12, 1), (18, 3), (30, 1),
                    (46, 3), (56, 1), (64, 1)):
        g[base + 1][x] = CH_PALM_TOP
        g[base + 2][x] = CH_PALM_TOP
        g[base + 3][x] = CH_PALM_SIDE
        g[base + 4][x] = CH_PALM_MID

    g[6][26] = ord("?")
    for x, y in ((9, 4), (21, 6), (35, 5), (48, 7), (60, 5)):
        g[y][x] = ord("*")

    g[2][10] = CH_CHIBIBO
    g[2][26] = 0x88                    # green Koopa walking by
    g[2][34] = CH_FLY_GREEN
    g[5][52] = CH_FLY_RED_H
    g[2][62] = CH_FLY_FISH             # spout keeps launching fish

    exit_pipe(g, 68)

    entry = level(width, init_x=4 * 20,
                  hint="NIGHT SKY. PLANTS UNCHANGED")
    entry["options"].update(
        SkyType=3, BackGrType=1, Design=1, Stars=1, Clouds=0, Horizon=140,
    )
    entry["map"] = columns(g)
    return entry


def background_test_level(hint: str, options: dict, *,
                          width: int = 80,
                          tall_walls: bool = False) -> dict:
    """Shared layout for the unused-sky / unused-background showcase levels.

    Flat grassy floor, a few two-column stone towers to give the sky a
    silhouette, a coin row and an exit pipe.  Only the level options differ,
    so the background is the thing under test.
    """
    g = blank(width)
    ground(g)
    for x in range(width):
        g[2][x] = CH_GRASS_TOP

    for i, x in enumerate((12, 20, 28, 36, 44, 52)):
        height = (4 if tall_walls else 2) + (i % 3)
        for y in range(2, 2 + height):
            g[y][x] = GROUND
            g[y][x + 1] = GROUND

    for x in range(width - 20, width - 16):
        g[5][x] = ord("J")
        g[6][x] = ord("*")

    exit_pipe(g, width - 8)

    entry = level(width, init_x=4 * 20, hint=hint)
    entry["options"].update(options)
    entry["map"] = columns(g)
    return entry


def large_bricks_level() -> dict:
    return background_test_level(
        "LARGE BRICKS SKY (BACKGR 5). MOVES ONLY WITH CAMERA",
        dict(SkyType=8, BackGrType=5))


def windows_level() -> dict:
    return background_test_level(
        "WINDOWS SKY (BACKGR 7). MOVES ONLY WITH CAMERA",
        dict(SkyType=8, BackGrType=7))


def grass_wall_level() -> dict:
    return background_test_level(
        "GRASS WALL (WALLTYPE 4)", dict(WallType1=4), tall_walls=True)


def sky_test_level(sky: int) -> dict:
    return background_test_level(
        f"UNUSED SKYTYPE {sky}", dict(SkyType=sky, BackGrType=1))


def backgr_test_level(bgr: int) -> dict:
    # 8 is the "draw nothing" branch, and BACKGR.PAS only dispatches for the
    # textured sky types, so it needs SkyType 8 to be observable at all.
    sky = 8 if bgr == 8 else 0
    return background_test_level(
        f"UNUSED BACKGRTYPE {bgr}", dict(SkyType=sky, BackGrType=bgr))


def cloud_test_level() -> dict:
    # 0 = flat $E0 sky (clouds only paint over $E0); 15 = white.
    return background_test_level(
        "CLOUDS PARALLAX 1/4. MOVES ONLY WITH CAMERA",
        dict(SkyType=0, BackGrType=1, Clouds=15), width=128)


def moon_sky_level() -> dict:
    """7-1a: the port-only XP moon backdrop (Options.MoonSky).

    One screen wide (16 columns) because the backdrop is screen-fixed, and a
    single ground row so the XP hills stay visible above it.  The backdrop is
    blitted as the page base and Figures.draw_sky is skipped, so the normal
    sky/mountain fill never covers it.
    """
    width = 16
    g = blank(width)
    for x in range(width):
        g[0][x] = GROUND

    for x in (3, 8):
        g[1][x] = CH_PALM_TOP
        g[2][x] = CH_PALM_TOP
        g[3][x] = CH_PALM_SIDE
        g[4][x] = CH_PALM_MID

    g[1][5] = CH_CHIBIBO
    g[1][10] = 0x88                    # green Koopa walking under the moon

    exit_pipe(g, 13, base=0)

    entry = level(width, init_x=4 * 20,
                  hint="XP MOON NIGHT SKY. ONE SCREEN WIDE")
    entry["options"].update(
        SkyType=3, BackGrType=0, Design=1, Stars=0, Clouds=0,
        MoonSky=1, Horizon=168,
    )
    entry["map"] = columns(g)
    return entry


def fire_plant_level() -> dict:
    """8-1a: the port-only fire-spitting Piranha Plant (marker 0x9C).

    The marker goes two rows above the pipe mouth, which is how the original
    1a places its plant markers (pipe body rows 2..4, mouth row 5, marker row
    7).  A plain green plant shares the row so the two arts can be compared
    side by side, and walkers stand in the flame's path: the fireball passes
    through them and through terrain, and only ever hurts Mario.
    """
    width = 56
    g = blank(width)
    ground(g)

    def plant_pipe(x, marker):
        for y in (2, 3, 4):
            g[y][x] = ord("2")
            g[y][x + 1] = ord("3")
        g[5][x] = ord("0")
        g[5][x + 1] = ord("1")
        g[7][x] = marker

    plant_pipe(10, CH_FIRE_PLANT)
    plant_pipe(20, CH_PLANT_GREEN)
    plant_pipe(31, CH_FIRE_PLANT_AIM)
    plant_pipe(42, CH_FAST_PLANT)

    g[2][14] = CH_CHIBIBO
    g[2][25] = CH_CHIBIBO
    g[2][36] = 0x88                    # green Koopa: a shell to kick at it
    g[2][46] = CH_CHIBIBO

    for x in (16, 27, 38):
        g[4][x] = ord("*")
    g[6][22] = ord("?")

    exit_pipe(g, 52)

    entry = level(width, init_x=4 * 20,
                  hint="PLANTS: SPIT LEFT / GREEN / AIMED / FAST 1.25 S CYCLE")
    entry["options"].update(Design=1, Horizon=168)
    entry["map"] = columns(g)
    return entry


def plant_retract_level() -> dict:
    """8-2a: floating plants with no pipe at all, to test the retraction rule.

    The original hid the sprite but kept the hit box, so Mario still lost
    health walking through an empty pipe mouth and a fireball or a shell could
    kill a plant nobody could see.  The user's rule is that once a plant has
    fully cleared the box it extends into it is inert.  Every plant here hangs
    two or three rows above the ground with nothing around it, so the only
    thing between Mario and the cell is the plant itself.

    The sideways pair needs its direction written in the level's
    ``enemy_dirs``: +1 reaches right, -1 reaches left.
    """
    width = 48
    g = blank(width)
    ground(g)

    # vertical plants, home cell rows 4 and 5 (markers two rows lower)
    for x, row in ((8, 2), (14, 3), (20, 2), (26, 3)):
        g[row][x] = CH_FIRE_PLANT if x % 3 else CH_FAST_PLANT
    # sideways plants float at rows 3 and 4 (their marker is their own cell)
    g[3][34] = CH_SIDE_PLANT
    g[4][40] = CH_SIDE_PLANT
    g[3][44] = CH_SIDE_PLANT

    # a step to stand on so Mario can walk through the cells at head height
    for x in range(6, 28):
        if x % 6 != 2:
            g[2][x] = ord("A")

    g[1][11] = CH_CHIBIBO
    g[2][31] = ord("*")
    exit_pipe(g, 46)

    entry = level(width, init_x=3 * 20,
                  hint="FLOATING PLANTS. RETRACTED = NO DAMAGE, UNCLEARABLE")
    entry["options"].update(Design=1, Horizon=168)
    entry["enemy_dirs"] = [{"x": 34, "y": 3, "dir": 1},
                           {"x": 40, "y": 4, "dir": -1},
                           {"x": 44, "y": 3, "dir": 1}]
    entry["map"] = columns(g)
    return entry


def side_pipe_run(g: list[list[int]], mouth: int, body: int,
                  facing: str) -> None:
    """A floor-level horizontal pipe: mouth column ``mouth``, ``body`` columns
    of pipe behind it.  facing='left' opens the mouth to the left."""
    cols = ([mouth + k for k in range(1, body + 1)] if facing == "left"
            else [mouth - k for k in range(1, body + 1)])
    g[3][mouth] = ord("4")
    g[2][mouth] = ord("5")
    for x in cols:
        g[3][x] = ord("6")
        g[2][x] = ord("7")


def side_pipe_level() -> dict:
    """9-1a: horizontal pipe warps, both directions.

    Pipe A: walk RIGHT into a left-facing mouth at column 15 and come out of
    the right-facing mouth at column 30, sliding out to the right.  Pipe B
    mirrors the pair (walk LEFT at column 46 into mouth 45, come out LEFT of
    mouth 52 standing at column 51).  Codes: A uses 0xE6 above 0xE0, B uses
    0xE5; the exit scan only looks at the upper code, so the two warps must
    not cross.  The classic end pipe sits at column 68.
    """
    width = 72
    g = blank(width)
    ground(g)

    side_pipe_run(g, 15, 3, "left")            # A entry, walk right into it
    g[2][14] = 0xE0                            # Mario stands left of mouth 15
    g[3][14] = 0xE6
    side_pipe_run(g, 30, 3, "right")           # A exit, Mario slides out right
    g[2][31] = 0xE0                            # mouth 30 is left of cell 31
    g[3][31] = 0xE6

    side_pipe_run(g, 45, 3, "right")           # B entry, walk left into it
    g[2][46] = 0xE0
    g[3][46] = 0xE5
    side_pipe_run(g, 52, 3, "left")            # B exit, Mario slides out right
    g[2][51] = 0xE0
    g[3][51] = 0xE5

    g[2][22] = CH_CHIBIBO
    g[2][38] = CH_CHIBIBO
    for x in (18, 25, 35, 42):
        g[4][x] = ord("*")
    exit_pipe(g, 68)

    entry = level(width, init_x=4 * 20,
                  hint="SIDE PIPES: WALK INTO THE MOUTH AND KEEP PUSHING")
    entry["options"].update(Design=1, Horizon=168)
    entry["map"] = columns(g)
    return entry


def side_pipe_color_level(color: int, name_hint: str) -> dict:
    """9-2a / 9-3a: a decor strip so the recoloured horizontal tiles can be
    compared with the vertical pipe of the same PipeColor."""
    width = 40
    g = blank(width)
    ground(g)
    side_pipe_run(g, 6, 2, "left")
    side_pipe_run(g, 16, 4, "right")
    side_pipe_run(g, 30, 2, "left")
    for y in (2, 3, 4):
        g[y][24] = ord("2")
        g[y][25] = ord("3")
    g[5][24] = ord("0")
    g[5][25] = ord("1")
    exit_pipe(g, 36)          # without this the pack cannot advance to world 10
    entry = level(width, init_x=2 * 20, hint=name_hint)
    entry["options"].update(Design=1, Horizon=168, PipeColor=color)
    entry["map"] = columns(g)
    return entry


def black_plant_level() -> dict:
    """8-3a: the black piranha plant (marker 0xA0), a black creature.

    Same rise cycle as the plain green plant on purpose -- only the art and the
    duel rank differ.  A green and a black Koopa stand on the floor so the
    tester can produce both shell kinds: a kicked coloured shell is destroyed
    against the plant, a kicked black shell kills it.  Fireballs and the star do
    nothing to it, and while it is back in its pipe it is not there at all.
    """
    width = 60
    g = blank(width)
    ground(g)

    def plant_pipe(x, marker):
        for y in (2, 3, 4):
            g[y][x] = ord("2")
            g[y][x + 1] = ord("3")
        g[5][x] = ord("0")
        g[5][x + 1] = ord("1")
        g[7][x] = marker

    plant_pipe(10, CH_BLACK_PLANT)
    plant_pipe(21, CH_PLANT_GREEN)      # side-by-side comparison
    plant_pipe(32, CH_BLACK_PLANT)
    plant_pipe(44, CH_BLACK_PLANT)

    g[2][15] = 0x88                     # green Koopa -> coloured shell
    g[2][26] = CH_BLACK_KOOPA           # black Koopa -> the only shell that wins
    g[2][38] = CH_CHIBIBO

    for x in (17, 28, 40):
        g[4][x] = ord("*")

    exit_pipe(g, 56)
    entry = level(width, init_x=4 * 20,
                  hint="BLACK PLANT: ONLY A MOVING BLACK SHELL KILLS IT")
    entry["options"].update(Design=1, Horizon=168)
    entry["map"] = columns(g)
    return entry


def ghost_chibibo_level() -> dict:
    """8-4a: the ghost Chibibo (marker 0xA1) and the blue shadow it leaves.

    A black-painted Chibibo walks exactly like the brown one -- same speed,
    same squash when stomped -- but every death (stomp, fireball, star, shell,
    bumped block) releases one blue shadow.  It moves like the Boo at 2-x (it
    only closes in while Mario's back is turned) and goes straight through the
    two-tall wall at 34, which Mario has to jump.  Nothing destroys it and it
    is never recycled off-screen either: it just stops following once it is two
    screens (640 px) away, and starts again the moment you come back.
    """
    width = 64
    g = blank(width)
    ground(g)

    g[2][8] = CH_GHOST_CHIBIBO
    g[2][16] = CH_GHOST_CHIBIBO
    g[4][20] = ord("X")                    # bump the block: it dies, ghost out
    g[2][21] = CH_GHOST_CHIBIBO
    g[2][26] = CH_CHIBIBO                  # the brown one for comparison
    g[2][30] = 0x88                        # green Koopa -> a kickable shell
    g[2][40] = CH_GHOST_CHIBIBO
    g[2][46] = CH_GHOST_CHIBIBO

    # a two-tall wall: Mario jumps it, the shadow flies straight through it.
    # Three rows is the port's jump ceiling (RULES §5.13: run-jump 4.7 rows with
    # no slack), so a 3-tall wall here would simply stop the player.
    for y in range(2, 4):
        g[y][34] = ord("@")
    for x in (12, 18, 24, 36, 44):
        g[4][x] = ord("*")

    exit_pipe(g, 60)
    entry = level(width, init_x=4 * 20,
                  hint="SHADOWS NEVER DIE AND NEVER GET RECYCLED: THEY GIVE "
                       "UP 2 SCREENS BEHIND YOU")
    entry["options"].update(Design=1, Horizon=168)
    entry["map"] = columns(g)
    return entry


def ghost_house_level() -> dict:
    """8-5a: the black room with a sight circle (Options.Vision).

    Same art as 4-1a (SkyType 3 + BackGrType 0 = the near-black 0xE0 fill), but
    the room is now painted black except for a Manhattan circle of
    VISION_BASE=2 tiles around Mario's head.  A coin buys one more tile
    (stackable to 6) and one tile falls off every 2 s.  What carries its own
    light -- coins, the power-ups, the star, Mario's fireballs, the lava
    fireball and the firebar -- stays visible one tile further out; the HUD
    band is always lit.  The piranha plants are deliberately NOT lit: they read
    as ordinary enemies and sit in the dark like their pipes.  It is a
    display-only effect -- the blacked-out pillars, lava and pits are still
    there and still hurt.

    Kill the ghost Chibibos to see the blue shadows: they are not on the glow
    list, so they vanish the moment they drift out of the circle and reappear
    when they come back, which is the whole ghost-house trick in one object.
    """
    width = 64
    g = blank(width)
    ground(g)

    for x, top in ((8, 4), (12, 5), (18, 4), (22, 5), (27, 4), (31, 5),
                   (37, 4), (41, 5), (46, 4), (51, 5)):
        wall(g, x, 2, top)
        wall(g, x + 1, 2, top)
    for x0 in (14, 33):
        for x in (x0, x0 + 1):
            g[0][x] = ord("#")
            g[1][x] = ord("%")

    # coins inside the circle, then one every two tiles to grow it
    for x in (6, 9, 11, 13, 15, 17, 19, 21, 23, 25):
        g[3][x] = ord("*")
    g[5][28] = ord("?")                 # a power-up, and it glows too

    def plant_pipe(x):
        for y in (2, 3, 4):
            g[y][x] = ord("2")
            g[y][x + 1] = ord("3")
        g[5][x] = ord("0")
        g[5][x + 1] = ord("1")
        g[7][x] = CH_FIRE_PLANT

    plant_pipe(36)                       # NOT lit: an ordinary enemy
    plant_pipe(44)

    # the fireball that spits up out of the lava pit at col 33: it is a light
    # source, so it is visible from well outside Mario's own circle.
    g[3][34] = CH_VERT_FIREBALL

    g[2][30] = CH_GHOST_CHIBIBO
    g[2][39] = CH_GHOST_CHIBIBO
    g[2][49] = CH_GHOST_CHIBIBO
    g[2][53] = 0x88                      # green Koopa -> a kickable shell
    exit_pipe(g, 60)

    entry = level(width, init_x=4 * 20,
                  hint="THE ROOM IS BLACK: COINS BUY SIGHT, IT FADES IN 2 s")
    entry["options"].update(
        SkyType=3, BackGrType=0, Design=5, Stars=0, Clouds=0, Horizon=12,
        WallType1=102, GroundColor1=104, BrickColor=24, WoodColor=24,
        XBlockColor=104, PipeColor=24, Vision=1,
    )
    # The firebar is a WEBANGELO mechanism, not an enemy-pool type, so it comes
    # from the level's own object list.  x/y are the pivot in page pixels.
    entry["webangelo"] = [{"type": "firebar", "x": 21 * 20, "y": 5 * 14}]
    entry["map"] = columns(g)
    return entry


def glow_zoo_level() -> dict:
    """8-6a: a flat shop-window of everything, to read the glow behaviour.

    Pure black room (Vision=1) on completely flat ground: no pits, no lava, no
    pillars, so Mario can never fall out and the only variable is what lights.
    One of every implemented enemy, spaced three columns apart, then the
    power-ups in ? blocks -- 'alpha' gives a mushroom while small and a flower
    once big, 'beta' the green 1-up, 'gamma' the star, 'phi' the poison
    mushroom.  Coins are laid in a line so the tile glow and the fact that it
    vanishes when eaten are both visible.

    Nothing here glows except Mario's own sight, the coins, the power-ups, his
    fireballs, the lava fireball and the firebar: every walker, plant and
    mechanism is deliberately dark so the contrast can be judged side by side.
    """
    width = 152
    g = blank(width)
    ground(g)

    # ---- every walking / flying enemy, one each ------------------------ #
    walkers = [
        CH_CHIBIBO, CH_CHIBIBO_B, CH_RED, CH_KOOPA0, CH_KOOPA1,
        CH_BLACK_KOOPA, CH_BUZZY, CH_DRY_BONES, CH_GHOST_CHIBIBO,
        CH_SPINY_BLACK, CH_FLY_BLACK,
        CH_FLY_GREEN, CH_FLY_RED_H, CH_FLY_RED_V, CH_FLY_SPINY,
        CH_FLY_SPINY_BLACK, CH_FLY_BUZZY, CH_FLY_CHIBIBO,
        CH_HAMMER_BRO, CH_BOO, CH_BOMB_STAND, CH_THWOMP, CH_LAKITU,
        CH_VERT_FIREBALL,
    ]
    for i, marker in enumerate(walkers):
        col = 4 + i * 3
        # flyers and mechanisms want the row above the walkers' row
        row = 2 if i < 11 else 4
        g[row][col] = marker

    # ---- the piranha family, each in its own pipe ---------------------- #
    plants = [CH_PLANT0, CH_PLANT1, CH_PLANT2, CH_FIRE_PLANT,
              CH_FIRE_PLANT_AIM, CH_FAST_PLANT, CH_SIDE_PLANT,
              CH_BLACK_PLANT]
    base = 4 + len(walkers) * 3 + 2
    for i, marker in enumerate(plants):
        x = base + i * 4
        # one body row + one mouth row = two solid rows. Source row 0 is the
        # BOTTOM (WorldMap.load_columns flips it), and a three-solid-row pipe
        # is exactly the wall the measured jump envelope says Mario cannot
        # clear with slack -- this level has to be walkable end to end.
        g[2][x] = ord("2")
        g[2][x + 1] = ord("3")
        g[3][x] = ord("0")
        g[3][x + 1] = ord("1")
        g[5][x] = marker

    # ---- coins: the tile itself glows, and stops when eaten ------------ #
    coin_x = base + len(plants) * 4 + 2
    for i in range(6):
        g[2][coin_x + i] = ord("*")

    # ---- the power-ups, in ? blocks at jump height --------------------- #
    block_x = coin_x + 8
    for i, code in enumerate((0xE0, 0xE0, 0xE1, 0xE2, 0xED)):
        # 'α' mushroom-then-flower, 'ß' green 1-up, 'Γ' star, 'φ' poison
        g[5][block_x + i * 3] = code
        g[5][block_x + i * 3 - 1] = ord("?")

    # one firebar, its own r=1 halo on every ball
    entry = level(width, init_x=2 * 20,
                  hint="EVERYTHING ON FLAT GROUND: ONLY COINS, POWER-UPS, "
                       "FIRE AND THE BAR GLOW")
    entry["options"].update(
        SkyType=3, BackGrType=0, Design=1, Stars=0, Clouds=0, Horizon=168,
        WallType1=102, GroundColor1=104, BrickColor=24, WoodColor=24,
        XBlockColor=104, PipeColor=24, Vision=1,
    )
    entry["webangelo"] = [{"type": "firebar",
                           "x": (block_x + 18) * 20, "y": 5 * 14}]
    exit_pipe(g, width - 4)
    entry["map"] = columns(g)
    return entry


def charlie_scenery_level() -> dict:
    """10-1a: Charlie the Duck decoration next to the original Mario decor.

    Left half is the Design-1 grass top ('~' 247) and tall palm the DOS levels
    use; the right half is the Charlie port codes from FIGURES.draw_decorations
    (g grass tuft, o flower, t cattail stack, p palm).  Both halves run on the
    same two-state grass clock, so the alternation can be compared side by side.
    None of the four codes is solid -- Mario walks straight through them.
    """
    width = 76
    g = blank(width)
    ground(g)

    # --- original Mario grass + tall palm -------------------------------- #
    for x in range(2, 14):
        g[2][x] = CH_GRASS_TOP
    g[3][8] = CH_PALM_TOP
    g[4][8] = CH_PALM_TOP
    g[5][8] = CH_PALM_SIDE
    g[6][8] = CH_PALM_MID

    # --- Charlie grass + palm over the same stretch ---------------------- #
    for x in range(16, 28):
        g[2][x] = CH_CHAR_GRASS
    g[2][23] = CH_CHAR_PALM

    # --- palms on a stacked trunk: 2 and 4 trunk cells ------------------- #
    for col, cells in ((29, 2), (46, 4)):
        for row in range(2, 2 + cells):
            g[row][col] = CH_CHAR_PALM

    # --- cattail stacks: 1, 2, 3, 4 and 5 cells tall --------------------- #
    for col, height in ((31, 1), (34, 2), (37, 3), (40, 4), (44, 5)):
        for row in range(2, 2 + height):
            g[row][col] = CH_CHAR_CATTAIL

    # --- grass with the flower in the cell above it ---------------------- #
    for x in range(48, 58):
        g[2][x] = CH_CHAR_GRASS
    g[3][50] = CH_CHAR_FLOWER
    g[3][54] = CH_CHAR_FLOWER

    # --- the same decoration on a raised plateau ------------------------- #
    for x in range(60, 72):
        g[2][x] = GROUND
        g[3][x] = GROUND
        g[4][x] = CH_CHAR_GRASS
    for row in range(5, 9):
        g[row][63] = CH_CHAR_CATTAIL
    g[4][67] = CH_CHAR_PALM
    g[5][70] = CH_CHAR_FLOWER

    exit_pipe(g, 74)
    entry = level(width, init_x=2 * 20,
                  hint="LEFT = MARIO GRASS/PALM. RIGHT = CHARLIE. ALL WALK-THROUGH")
    entry["options"].update(Design=1)
    entry["map"] = columns(g)
    return entry


def water_level(style: int, water_top: int, hint: str) -> dict:
    """11-1a / 11-2a: the port-only underwater test rooms.

    WaterTop is a WORLD row (0 = top of the map): 0 means a whole-level dive,
    N means the first N rows are dry sky.  Both rooms deliberately use a
    plain sand floor and no grass/palm decor, so the background preview shows
    the water itself.
    """
    width = 40
    g = blank(width)
    ground(g)

    # a raised sand shelf on the left and a low one on the right
    for x in range(7, 13):
        g[2][x] = GROUND
    for x in range(24, 29):
        g[2][x] = GROUND

    # things to bump from below while swimming up
    g[7][10] = ord("?")
    g[8][10] = 0xE0                  # hidden champ/flower above it
    g[8][18] = ord("?")
    g[6][26] = ord("?")

    # coins and a couple of walkers
    for x, y in ((15, 7), (16, 7), (20, 8), (21, 8), (33, 7)):
        g[y][x] = ord("*")
    g[2][17] = CH_CHIBIBO
    g[2][22] = CH_KOOPA0

    # fish markers below the waterline patrol horizontally
    g[7][14] = CH_VERT_FISH
    g[9][21] = CH_VERT_FISH

    # a vertical pipe and the exit pipe on the floor
    g[2][30] = ord("2")
    g[2][31] = ord("3")
    g[3][30] = ord("0")
    g[3][31] = ord("1")
    exit_pipe(g, 36)

    entry = level(width, init_x=4 * 20, hint=hint)
    entry["options"].update(
        SkyType=10, WallType1=1, BackGrType=0, Design=0,
        Horizon=(water_top + 2) * TILE_H,
        WaterTop=water_top, WaterStyle=style,
    )
    entry["options"]["InitY"] = 9 * TILE_H
    entry["map"] = columns(g)
    return entry


def main() -> None:
    levels = {
        "testrooms/1-1a": gate_room(),
        "testrooms/1-2a": battle_room_level(),
        "testrooms/1-3a": tower_level(),
        "testrooms/1-4a": flying_level(),
        "testrooms/1-5a": fish_level(),
        "testrooms/2-1a": dry_bones_level(),
        "testrooms/2-2a": hammer_bro_level(),
        "testrooms/2-3a": boo_level(),
        "testrooms/2-4a": bomb_level(),
        "testrooms/2-5a": thwomp_level(),
        "testrooms/2-6a": lakitu_level(),
        "testrooms/2-7a": bomb_line_level(
            3, "BOMBS FLY 2 ROWS UP. SMALL CLEARS BY 2PX, BIG GETS HIT"),
        "testrooms/2-8a": bomb_line_level(
            4, "BOMBS FLY 3 ROWS UP. BIG MARIO CLEARS BY 2PX"),
        "testrooms/3-1a": platform_level(),
        "testrooms/3-2a": forced_route_level(),
        "testrooms/4-1a": black_room_level(),
        "testrooms/4-2a": night_grove_level(),
        "testrooms/4-3a": large_bricks_level(),
        "testrooms/4-4a": windows_level(),
        "testrooms/4-5a": grass_wall_level(),
        "testrooms/5-1a": sky_test_level(0),
        "testrooms/5-2a": sky_test_level(1),
        "testrooms/5-3a": sky_test_level(4),
        "testrooms/5-4a": sky_test_level(5),
        "testrooms/5-5a": sky_test_level(9),
        "testrooms/6-1a": backgr_test_level(2),
        "testrooms/6-2a": backgr_test_level(8),
        "testrooms/6-3a": backgr_test_level(11),
        "testrooms/6-4a": cloud_test_level(),
        "testrooms/7-1a": moon_sky_level(),
        "testrooms/8-1a": fire_plant_level(),
        "testrooms/8-2a": plant_retract_level(),
        "testrooms/8-3a": black_plant_level(),
        "testrooms/8-4a": ghost_chibibo_level(),
        "testrooms/8-5a": ghost_house_level(),
        "testrooms/8-6a": glow_zoo_level(),
        "testrooms/9-1a": side_pipe_level(),
        "testrooms/9-2a": side_pipe_color_level(48, "SIDE PIPES RED"),
        "testrooms/9-3a": side_pipe_color_level(24, "SIDE PIPES GREY"),
        "testrooms/10-1a": charlie_scenery_level(),
        "testrooms/11-1a": water_level(
            1, 0, "91898 WATER + SEAWEED. HOLD ALT TO SWIM UP, RELEASE TO SINK"),
        "testrooms/11-2a": water_level(
            2, 0, "DOS WATER. HOLD ALT TO SWIM UP"),
    }
    worlds = [
        ("1-1", "testrooms/1-1a"),
        ("1-2", "testrooms/1-2a"),
        ("1-3", "testrooms/1-3a"),
        ("1-4", "testrooms/1-4a"),
        ("1-5", "testrooms/1-5a"),
        ("2-1", "testrooms/2-1a"),
        ("2-2", "testrooms/2-2a"),
        ("2-3", "testrooms/2-3a"),
        ("2-4", "testrooms/2-4a"),
        ("2-5", "testrooms/2-5a"),
        ("2-6", "testrooms/2-6a"),
        ("2-7", "testrooms/2-7a"),
        ("2-8", "testrooms/2-8a"),
        ("3-1", "testrooms/3-1a"),
        ("3-2", "testrooms/3-2a"),
        ("4-1", "testrooms/4-1a"),
        ("4-2", "testrooms/4-2a"),
        ("4-3", "testrooms/4-3a"),
        ("4-4", "testrooms/4-4a"),
        ("4-5", "testrooms/4-5a"),
        ("5-1", "testrooms/5-1a"),
        ("5-2", "testrooms/5-2a"),
        ("5-3", "testrooms/5-3a"),
        ("5-4", "testrooms/5-4a"),
        ("5-5", "testrooms/5-5a"),
        ("6-1", "testrooms/6-1a"),
        ("6-2", "testrooms/6-2a"),
        ("6-3", "testrooms/6-3a"),
        ("6-4", "testrooms/6-4a"),
        ("7-1", "testrooms/7-1a"),
        ("8-1", "testrooms/8-1a"),
        ("8-2", "testrooms/8-2a"),
        ("8-3", "testrooms/8-3a"),
        ("8-4", "testrooms/8-4a"),
        ("8-5", "testrooms/8-5a"),
        ("8-6", "testrooms/8-6a"),
        ("9-1", "testrooms/9-1a"),
        ("9-2", "testrooms/9-2a"),
        ("9-3", "testrooms/9-3a"),
        ("10-1", "testrooms/10-1a"),
        ("11-1", "testrooms/11-1a"),
        ("11-2", "testrooms/11-2a"),
    ]
    out = {
        "packs": {
            "testrooms": {
                "title": "TRAP TEST",
                "num_lev": len(worlds),
                "worlds": [{"world": w, "a": a, "b": None}
                           for w, a in worlds],
            },
        },
        "levels": levels,
    }
    path = DATA / "levels_test.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, entry in levels.items():
        if "battle_room" in entry:
            kind = "battle_room"
        elif entry.get("vertical"):
            kind = "vertical"
        elif name.endswith("1-4a"):
            kind = "flying koopas"
        elif name.endswith("1-5a"):
            kind = "flying fish"
        elif name.endswith("4-1a"):
            kind = "black room"
        elif name.endswith("8-5a"):
            kind = "ghost-house sight"
        elif name.endswith("8-6a"):
            kind = "glow zoo (flat)"
        elif name.endswith("4-2a"):
            kind = "night sky"
        else:
            kind = "scroll gate"
        print(f"{name}: {entry['width']} cols x {entry['rows']} rows, {kind}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
