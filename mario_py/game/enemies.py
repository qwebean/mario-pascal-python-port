"""Port of ENEMIES.PAS — the enemy / object list of the game.

Everything that moves on its own besides the player lives here: walking
Chibibos and Reds, Koopas (green/red, sleeping/waking/running), the vertical
fish / fireballs / piranha plants, the block lifts and donut platforms, the
power-ups that rise out of bumped blocks (champ, life, flower, star) and the
player's fireballs.

The original keeps a fixed pool of ``MaxEnemiesAtOnce`` (25) EnemyRec records
plus a string of the active pool indices; enemies that leave the range
[XView - ForgetEnemiesAt*W, XView + (NH+ForgetEnemiesAt)*W] are "forgotten"
and their map marker is written back, so StartEnemies() recreates them when
the view returns.  This port keeps that structure, including the per-frame
``LastPos + DelayCounter*Vel div (MoveDelay+1)`` interpolation that produces
the original's slightly uneven gait.

The 25-slot cap is lifted (see POOL_LIMIT): it is a DOS-era memory
constraint, and because block lifts / donut platforms occupy the same slots,
the cap silently deletes monsters and mechanisms in the busier levels.

Coordinate convention (identical to figures.py / the original): x_pos / y_pos
are world pixels, and everything is drawn onto the world-wide page surface;
the visible 320x200 window is cut out of it by VGAScreen.

The units ENEMIES.PAS uses but which are not ported yet (Glitter, TmpObj,
Music, Blocks and the player state in PLAYERS.PAS) are reached through the
EnemyHooks callbacks and the shared attributes below, so this module can be
exercised headlessly with a no-op hook set.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame

from ..engine.vga import VIR_SCREEN_HEIGHT
from .buffers import (CAN_HOLD_YOU, CAN_STAND_ON, DIR_LEFT, DIR_RIGHT, EY1,
                      MD_SMALL, NH, TILE_H, TILE_W, WorldMap, WorldOptions)

# --------------------------------------------------------------------------- #
# Enemy types (ENEMIES.PAS const block)
# --------------------------------------------------------------------------- #
TP_DEAD = 0
TP_DYING = 1
TP_CHIBIBO = 2
TP_FLAT_CHIBIBO = 3
TP_DEAD_CHIBIBO = 4
TP_RISING_CHAMP = 5
TP_CHAMP = 6
TP_RISING_LIFE = 7
TP_LIFE = 8
TP_RISING_FLOWER = 9
TP_FLOWER = 10
TP_RISING_STAR = 11
TP_STAR = 12
TP_FIREBALL = 13
TP_DYING_FIREBALL = 14
TP_VERT_FISH = 15
TP_DEAD_VERT_FISH = 16
TP_VERT_FIREBALL = 17
TP_VERT_PLANT = 18
TP_DEAD_VERT_PLANT = 19
TP_RED = 20
TP_DEAD_RED = 21

TP_KOOPA = 50
TP_SLEEPING_KOOPA = 51
TP_WAKING_KOOPA = 52
TP_RUNNING_KOOPA = 53
TP_DYING_KOOPA = 54
TP_DEAD_KOOPA = 55

# Port-only enemies: winged Koopas.  TpFlyRed uses SubTp as the patrol mode
# (0 = horizontal, 1 = vertical); a stomped flyer becomes a normal Koopa.
# The winged Spinies hop exactly like the green flyer (SubTp unused).
TP_FLY_GREEN = 70
TP_FLY_RED = 71
TP_FLY_BLACK = 72
TP_FLY_SPINY = 73
TP_FLY_SPINY_BLACK = 74
# Port-only flying fish (see FLY_* below): 75 arcs out of a spout, 76 is the
# flipped fish after a stomp / fireball / shell.
TP_FLY_FISH = 75
TP_DEAD_FLY_FISH = 76
# Port-only winged Chibibo (SMB3 "Paragoomba"): hops like the green flyer,
# loses its wings when stomped.  Not black -- fireballs and shells kill it.
TP_FLY_CHIBIBO = 77
# Port-only winged Buzzy.  Unlike every other flyer it does not hop: it cruises
# horizontally at a constant speed.  Combat-wise it IS a Buzzy (fireproof, not
# star-immune, any colour shell kills it); a stomp drops the wings and leaves a
# walking Buzzy.
TP_FLY_BUZZY = 78

# Flying-Koopa motion, scaled from the Mario XP reference.
#   Mario XP: walk 1.6 px/f, max (high) jump 90 px.
#   obj_koopa green paratroopa: walk 0.5, hop impulse -4 + gravity 0.2
#       -> apex 38 px over 39 frames.
#   obj_paratroopa_path "green" (horizontal): accel 0.025, vmax 1.8
#       -> one leg 131 px over 145 frames (0 -> 1.8 -> 0, then turn).
#   obj_paratroopa_path "red" (vertical): accel 0.015, vmax 1.2
#       -> one leg 97 px over 161 frames.
# This port's Mario walks 1 px/f and jumps 60 px, so every XP value is
# converted with the same three ratios:
#   speed  x (1 / 1.6)          = 0.625
#   length x (60 / 90)          = 2/3
#   accel  x 0.625^2 / (2/3)    = 0.5859375
# which keeps speed/walk, distance/jump and hop/jump identical to XP.
# Fractions survive the integer EnemyRec fields as unit counts over
# FLY_X_DEN (horizontal) and FLY_Y_DEN (green hop).
FLY_X_DEN = 1024
FLY_Y_DEN = 128
FLY_GREEN_SPEED = 320       # 0.3125 px/f   = 0.5    * 0.625
FLY_GREEN_JUMP = -320       # -2.5 px/f     = -4     * 0.625
FLY_GRAVITY = 15            # 0.1171875 px/f^2 = 0.2 * 0.5859375
FLY_RED_H_SPEED = 1152      # 1.125 px/f    = 1.8    * 0.625
FLY_RED_H_ACCEL = 15        # 0.0146484 px/f^2 = 0.025 * 0.5859375
FLY_RED_V_SPEED = 768       # 0.75 px/f     = 1.2    * 0.625
FLY_RED_V_ACCEL = 9         # 0.0087891 px/f^2 = 0.015 * 0.5859375
# The winged Buzzy has no hop to scale: it cruises flat.  The user asked for it
# to be "a bit slower" than the other flyers, so it takes 80% of the green
# flyer's 320 (= 0.3125 px/f) -> 256 = 0.25 px/f.
FLY_BUZZY_SPEED = 256       # 0.25 px/f
# Wing animation: frames per wing state (OPEN/FOLDED).  Mario XP's
# paratroopa sprite runs 2 frames at playbackSpeed 8 fps -> 8 at 60 fps.
FLY_FLAP_DIV = 15

TP_LIFT_START = 60
TP_BLOCK_LIFT = 60
TP_DONUT = 61
TP_LIFT_END = 69

# --------------------------------------------------------------------------- #
# Port-only flying fish, from Mario XP's obj_cheep_cheep.
#
# Reference standards (the same pair the winged Koopas use):
#   Mario XP   walk 1.6 px/f, high jump peaks at 90 px
#   this port  walk 1.0 px/f, max jump peaks at 60 px
# so every XP value is converted with
#   speed  x 1/1.6          = 0.625
#   length x 60/90          = 2/3
#   accel  x 0.625^2/(2/3)  = 0.5859375
#   time   x (2/3)/(1/1.6)  = 1.0666667
# XP (obj_stage_manager) launches one fish every `cheep_cheep_timeout` seconds
# (default 2.5 s, 2 s in stage_2_3/2_5) at a random x along the bottom of the
# view.  The port keeps that period -- 2.5 s * 1.0667 = 160 frames -- but
# launches from a placed marker so a level author decides where the water is,
# and the marker is never consumed: a fish that is killed, flies away or drops
# out of the world simply comes back when the timer runs out again.
FLY_FISH_PERIOD = 160           # frames between two launches from one spout
FLY_FISH_NEAR = 67              # XP's 100 px "player is close" radius * 2/3
FLY_FISH_GRAV = 60              # XP 0.1 * 0.5859375, in 1/FLY_X_DEN px/f^2
FLY_FISH_DEAD_GRAV = 120        # XP 0.2 * 0.5859375
# (close-min, close-max, far-min, far-max) in 1/FLY_X_DEN px/frame:
FLY_FISH_XSPEED = (640, 800, 1280, 1920)         # XP 1..1.25 / 2..3 * 0.625
FLY_FISH_YSPEED = (-4480, -3840, -4480, -3584)   # XP -7..-6 / -7..-5.6 * 0.625
FLY_FISH_DEAD_POP = -2048       # XP ySpeed = -2 when a killed fish pops up

# --------------------------------------------------------------------------- #
# Port-only enemy batch 2 (2026-10-23): Dry Bones, Hammer Bro (+ its hammer),
# Boo, the bomb stand (+ its bomb), the stone shield (Thwomp), the Lakitu cloud
# (+ the Spiny it drops).  None of these exist in ENEMIES.PAS, so the whole
# 80..94 block is port-only and the art comes from Mario XP (see
# tools/build_new_enemy_sprites.py / install_new_enemy_sprites.py).
#
# Motion is still scaled from Mario XP with the same three ratios as §5.1.2:
#   speed x 0.625, length x 2/3, accel x 0.5859375, time x 1.0666667
# All sub-pixel values below are counts over FLY_X_DEN (1/1024 px or 1/1024
# px/frame).
# --------------------------------------------------------------------------- #
TP_DRY_BONES = 80           # walking skeleton Koopa
TP_DRY_BONES_FALL = 81      # stomped: collapsed pile, waits to revive
TP_DEAD_DRY_BONES = 82      # killed for good (star / shell), falls flipped
TP_HAMMER_BRO = 83
TP_HAMMER = 84              # its thrown projectile
TP_DEAD_HAMMER_BRO = 85     # stomped / hit: flies off
TP_BOO = 86
TP_DEAD_BOO = 87
TP_BOMB_STAND = 88          # mechanism: fires a bomb every 3 s on screen
TP_BOMB = 89
TP_DEAD_BOMB = 90           # stomped: flipped, falls with gravity
TP_THWOMP = 91              # mechanism: slams down, backs up, cannot be killed
TP_LAKITU = 92              # the cloud, rides the top of the camera
TP_DEAD_LAKITU = 93
TP_SPINY_FALL = 94          # the Spiny the cloud drops; turns into TP_RED
TP_DEAD_THWOMP = 95         # smashed by the star: flies off head down (95 is
                            # port-only too -- XP has no Thwomp death at all)

# --------------------------------------------------------------------------- #
# Port-only long platforms (2026-10-24).  One object is a whole board made of
# 10x8 sprite segments (20x16 at the port's 2x tile scale); ``plat_len`` is
# the number of segments, so the board is plat_len*20 px wide.  The level's
# ``platforms`` list is the source of truth (type/len/speed/range/path), not a
# map marker: a path needs waypoints, which do not fit in one byte per tile.
#
# Behaviour references:
#   osc     SMB3 ObjNorm_OscillatingH: +-1/16 px/f per 2 frames, peak +-1 px/f,
#           hold at the peak for $41 frames, then decelerate/reverse.
#   patrol  Mario XP obj_moving_platform: constant speed, reverse at a range.
#   fall    SMB3 ObjNorm_WoodenPlatform: inert until Mario stands on it, then
#           it descends and eventually resets (the port's off-screen rule).
#   path    constant-speed waypoint follower (tile coordinates in JSON).
#
# All speeds are px/frame at 60 Hz.  Platforms ignore terrain, like the SMB3
# floating logs, and reset to their home after leaving the view by
# PLATFORM_RESET_MARGIN_X/Y.
TP_PLATFORM_PATROL = 96
TP_PLATFORM_OSC = 97
TP_PLATFORM_FALL = 98
TP_PLATFORM_PATH = 99
_PLATFORM_TYPES = (TP_PLATFORM_PATROL, TP_PLATFORM_OSC,
                   TP_PLATFORM_FALL, TP_PLATFORM_PATH)

PLATFORM_SPEED_DEN = 1024        # platform sub-pixel unit (1/1024 px, px/f)
PLATFORM_SEG_W = 20              # 10 px sprite * 2
PLATFORM_SEG_H = 16              # 8 px sprite * 2
# One screen + the enemy forget band (FORGET_ENEMIES_AT=7 tiles = 140 px).
# A falling board used to reset after only one screen (182 px); the user's
# 2026-10-25 report: it came back too soon.  Use the same distance as the
# horizontal enemy forget boundary (320 + 140 = 460 px) on both axes.
PLATFORM_RESET_MARGIN_X = 460
PLATFORM_RESET_MARGIN_Y = 460
PLATFORM_OSC_ACCEL = 64          # 1/16 px/f, applied every 2nd frame
PLATFORM_OSC_HOLD = 0x41         # SMB3 OscTimerSets[0] = 65 frames
PLATFORM_FALL_CAP = 8 * PLATFORM_SPEED_DEN


def _is_platform(tp: int) -> bool:
    return tp in _PLATFORM_TYPES


# --------------------------------------------------------------------------- #
# Port-only fire-spitting Piranha Plant (2026-10-25, user request).
#
# Reference: SMB3's fire plants, object $A4/$A5 (green) and $A6/$A7 "Venus Fire
# Trap" (tall red) -- PRG/prg005.asm.  Facts worth keeping:
#   * fire-ness is a flag, not an object id: Piranha_Style bit 7 -> Var2.
#   * the rise cycle is Var4 = 0 hide / 1 emerge / 2 attack / 3 retract, and
#     "fully out" is exactly state 2 -- the same four states as this port's
#     TP_VERT_PLANT status 0..4.
#   * the ROM reloads Objects_Timer to $60 (96 frames) for a fire plant vs $30
#     (48) for a plain one, and spits when the countdown hits $40 and $10, i.e.
#     twice per extension.  It never checks the mouth: the chomp is a separate
#     counter (Frame = (Var3 >> 3) & 1, toggling every 8 frames, twice as fast
#     while fully out).
#   * the user's rule is the stricter one this port follows: **only while fully
#     out AND the mouth is opening**.  The 60-frame cooldown keeps the ROM's
#     roughly-two-shots-per-extension rate.
#   * the projectile is special object $05, not the player's fireball.  It gets
#     a constant velocity ($0B.$0E = 0.74 px/f at 60 Hz), no gravity, and the
#     solid-check is skipped, so it passes straight through blocks; it is only
#     ever removed off-screen, and its sole collision call is PlayerCollide ->
#     GetHurt.  That matches the user's "hurts the player, touches nothing
#     else", which is also how the thrown hammer was done.
# --------------------------------------------------------------------------- #
TP_FIRE_PLANT = 100
TP_PLANT_FIRE = 101
TP_FIRE_PLANT_AIM = 102            # same plant, aims the flame at Mario
TP_FAST_PLANT = 103                # XP's red piranha: in and out in a flash
TP_SIDE_PLANT = 104                # SMB3's $56/$57: the chew cycle on x
TP_BLACK_PLANT = 105               # black creature, green plant's own cadence
CH_FIRE_PLANT = 0x9C               # port-only map markers
CH_FIRE_PLANT_AIM = 0x9D
CH_FAST_PLANT = 0x9E
CH_SIDE_PLANT = 0x9F
CH_BLACK_PLANT = 0xA0

# --------------------------------------------------------------------------- #
# The ghost Chibibo (2026-10-25, user request): a Chibibo painted black that
# releases a blue shadow when it dies.  Reference: Mario XP's obj_goomba with
# ``has_undead`` -- every death path (stomp / fireball / hammer / cross / block
# bump / shell) creates an obj_boo with ``skin = "goomba"``, i.e. the goomba's
# own silhouette as a Boo.  The port keeps the same rule for all its deaths.
#
# The walker is a plain Chibibo in behaviour and physics -- it is NOT a black
# creature (see _is_black_creature), so fireballs, the star and every shell
# still kill it, which is exactly what makes the shadow releasable.
#
# The shadow moves like the port's Boo (same _move_boo: it only closes in while
# Mario's back is turned, straight through terrain) and cannot be destroyed by
# anything: it stays out of both enemy-collision tables, it is in _IMMORTAL,
# and touching it costs a life from any side.
# --------------------------------------------------------------------------- #
TP_GHOST_CHIBIBO = 106             # walking black Chibibo
TP_GHOST_FLAT = 107                # stomped black Chibibo
TP_GHOST_DEAD = 108                # black Chibibo corpse, thrown head-down
TP_GHOST_SHADE = 109               # the blue shadow it releases
CH_GHOST_CHIBIBO = 0xA1            # map marker (0xA1.. unused in every pack)

PLANT_TPS = (TP_VERT_PLANT, TP_FIRE_PLANT, TP_FIRE_PLANT_AIM, TP_FAST_PLANT,
             TP_SIDE_PLANT, TP_BLACK_PLANT)
PLANT_FIRE_COOLDOWN = 60           # frames between two spits
PLANT_FIRE_SPEED = 758             # SMB3 $0B.$0E in 1/FLY_X_DEN px/frame
PLANT_MOUTH_DIV = 32               # the port's chomp period in frames
PLANT_MOUTH_OPEN = 16              # frames the mouth stays open per chomp
PLANT_ART_H = 20                   # the piranha sprite is 24x20

# --- the fast plant, straight out of Mario XP ---------------------------- #
# objects/obj_piranha_plant: two types.  green = piranhaSpeed 0.5 and the
# direction flips every 3 s; red = piranhaSpeed **2** and flips every **1.25 s**
# (piranhaTimeout -= delta_time/1e6, so those are seconds).  It travels between
# y and y-31 and is clamped at both ends, so with speed 2 it is out of the pipe
# after ~16 steps and simply waits there for the rest of the window -- which is
# what makes it read as "shoots out, chews, snaps back".
# Converted with the port's standing ratios (speed x0.625, time x1.0666667):
#   2 px/step  -> 1.25 px/frame,  1.25 s -> 75 steps -> 80 frames per flip.
# The rise stays at the port's own 19 px because that is where the 24x20 art
# stops looking like it is clipping into the pipe.
FAST_PLANT_SPEED = 1280            # 1/FLY_X_DEN px/frame = 1.25 px/f
FAST_PLANT_FLIP = 80               # frames per direction flip (XP 1.25 s)
FAST_PLANT_RISE = 19               # px it clears the pipe by
FAST_PLANT_FIRST = 120             # XP's initial piranhaTimeout = 2 s

# --- the sideways plant --------------------------------------------------- #
# SMB3 has one (ObjNorm_PiranhaSideways, objects $56/$57) and it never shoots.
# The port keeps the Pascal plant's cadence so it reads as a sibling of the
# vertical ones: 1 px per 3 frames (the vertical plant's own pace), 200 frames
# fully out, then a 100..200 frame wait back in the pipe.
SIDE_PLANT_RISE = 19               # px it clears the pipe mouth by
SIDE_PLANT_HOLD = 200              # frames fully extended, chewing
SIDE_PLANT_WAIT = (100, 200)       # random rest before coming out again


def _is_plant(tp: int) -> bool:
    return tp in PLANT_TPS


def _is_fire_plant(tp: int) -> bool:
    return tp in (TP_FIRE_PLANT, TP_FIRE_PLANT_AIM)


def _plant_visible(e: EnemyRec) -> int:
    """How many pixels of this plant actually stick out of its pipe.

    This is the one number the hit box, the damage test and the "cannot be
    killed" test all read, so they can never disagree.
    """
    if not _is_plant(e.tp):
        return 0
    if e.tp == TP_SIDE_PLANT:
        return max(0, (e.x_pos - e.map_x * W) * e.dir_sign)
    return max(0, min(PLANT_ART_H, e.map_y * H - e.y_pos))


def _plant_retracted(e: EnemyRec) -> bool:
    """Is this plant back inside its pipe, i.e. drawn as nothing at all?

    The original only hid the sprite: the hit box stayed where it was, so
    walking through an empty pipe mouth still cost a life and a fireball or a
    shell could still kill a plant nobody could see.  The user's rule
    (2026-10-25) is that the hidden part does not exist -- no damage, and
    nothing can kill it.  Half out still hurts, half out can still be killed,
    and the box is only the part that is out.

    Only plants have a pipe to be inside, so anything else is never retracted.
    """
    return _is_plant(e.tp) and _plant_visible(e) <= 0


def _fireball_underground(e: EnemyRec, rows: int) -> bool:
    """Has the lava fireball dropped back out of sight into the floor?

    Same rule as a retracted plant (the user, 2026-10-25): while it is drawn as
    nothing it must do nothing either -- no hit box, no damage, and no light of
    its own in a Vision level.  It waits at the world floor between spits, so
    that is exactly the moment it is invisible.  It is invulnerable anyway, so
    unlike the plant it never needs a "cannot be killed" branch.
    """
    return e.tp == TP_VERT_FIREBALL and e.y_pos + TILE_H >= rows * TILE_H

# -- Dry Bones -------------------------------------------------------------- #
# Reference: SMB3's own ObjNorm_DryBones / ObjHit_DryBones (smb3-master,
# PRG/prg002.asm).  SMB3 and this port both run their logic at 60 Hz, so the
# ROM's frame counts are used as-is -- no ratio conversion.
#   walk       Objects_XVel = $08 in SMB3's 4.4 fixed point = 0.5 px/f.
#              The port walker moves x_vel/(move_delay+1) per frame.
#   animation  frame toggles on Counter_1 & $04 (walk mask 4) only while on
#              the ground; the whole walk cycle is 8 frames (7.5 Hz).
#   stomp      Var5 = 9, X velocity = 0, player bounces, 100+ pts; the pile is
#              INERT (ObjHit returns while Var5 <> 0) -- no damage at all.
#   reassemble SMB3 runs 8 stages, each with its own timer, 66 frames in all.
#              The user watched the animation again on 2026-10-25 and overrode
#              that: the bones have to *lie there* a long time and then pull
#              themselves together quickly, so the schedule is now
#              **3 s dormant + 0.5 s reassembly = 210 frames**, walking again
#              on frame 211.  The dormant hold is a leading scatter stage, so
#              every stage/box/bob table below stays one shared list.
DRY_WALK_DELAY = 1                # 1 / (1+1) = 0.5 px/f, exactly SMB3's $08
DRY_ANIM_MASK = 4                 # SMB3's DryBones_AnimCntrMask for flat ground
# (pose, frames held).  **These are the numbers to tune.**  DRY_DORMANT_FRAMES
# is the wait, and the eight entries after it are the gather (30 frames = 0.5 s
# as shipped); every total below is derived, so nothing has to be edited twice.
# The user's 2026-10-24 collapse rule: the standing art is sliced into its 23
# rows, squeezed to 10 px, and spread on the ground with one empty pixel
# between slices (45 px long).  Reassembly pulls the slices back together:
# scatter -> half-gather -> gather -> squat -> rise -> stand.
DRY_DORMANT_FRAMES = 180          # 3 s at 60 Hz: the pile just lies there
DRY_REASSEMBLE = ((0, DRY_DORMANT_FRAMES), (0, 4), (1, 3), (2, 3), (3, 3),
                  (3, 3), (3, 3), (4, 5), (5, 6))
DRY_REBUILD_FRAMES = sum(d for _, d in DRY_REASSEMBLE[1:])   # 30 = 0.5 s
DRY_REVIVE_FRAMES = sum(d for _, d in DRY_REASSEMBLE)        # 210
DRY_STAND_STAGE = len(DRY_REASSEMBLE) - 1               # stage that draws standing
# pose per stage; DRY_FALL_BOXES below carries the matching canvases
DRY_FALL_POSE = ("scatter", "scatter", "half", "gather", "squat", "squat",
                 "squat", "rise", "walk")
# DryBones_HeadYOff indexed by the same stage (non-zero only on the squat)
DRY_HEAD_BOB = (0, 0, 0, 0, -1, -2, -1, 0, 0)
# -- Hammer Bro ------------------------------------------------------------- #
HBRO_SPEED = 640                # XP patrol 1 px/f        * 0.625
HBRO_BAND = 43                  # XP patrol band 64 px    * 2/3
HBRO_GRAVITY = 120              # XP 0.2                  * 0.5859375
HBRO_HOP = -1408                # XP small jump -2.2      * 0.625  (apex 8 px)
# The big leap: the user (2026-10-25) wants the Hammer Bro to reach what Mario
# reaches -- five rows, 14 * 5 = 70 px.  With the XP-scaled gravity that is
# v = sqrt(2 * 0.1171875 * 70) = 4.0505 px/frame, and the port's integer
# _slide loses about a pixel of the arc, so 4230/1024 = 4.1309 px/frame is what
# actually measures 70 px (verify_enemies R prints the number it measured).
HBRO_LEAP = -4230               # apex 70 px = 5 rows, Mario's own height
HBRO_JUMP_FRAMES = 128          # XP jumps every 2 s      * 1.0666667
HBRO_ANIM_DIV = 6               # XP walk cycle runs at 10 fps
HBRO_THROW_FRAME = 2            # XP throws between image_index 2..3 of the 6
HBRO_THROW_DX = 5               # XP x + 8*aim   * 2/3
HBRO_THROW_DY = -11             # XP y - 16      * 2/3
HAMMER_SPEED = (384, 1920)      # XP random_range(0.6, 3) * 0.625
HAMMER_GRAVITY = 132            # XP 0.22                 * 0.5859375
HBRO_DEFEATED_POP = -3328       # XP defeatedYSpeed -5.2  * 0.625
# -- Boo -------------------------------------------------------------------- #
BOO_SPEED = 320                 # XP speed 0.5            * 0.625
BOO_AIM_UP = 5                  # XP aims at player.y - 8 * 2/3
# The ghost shadow is never recycled (user 2026-10-25), so what keeps it honest
# is a leash instead: two screens of straight-line distance to the player
# (320 px * 2 = 640 px, the user's number).  Beyond that it parks where it is
# and starts closing in again the moment the player comes back into range.
GHOST_SHADE_LEASH = 640
# -- bomb stand / bomb ------------------------------------------------------ #
BOMB_STAND_FRAMES = 192         # XP 3 s                  * 1.0666667
BOMB_SPEED = 960                # XP xSpeed 1.5           * 0.625
# The muzzle line only: where the bomb is born relative to the stand's box top.
# It does NOT protect the fresh bomb from a stomp -- Mario's collision box is
# projected by his fall speed (up to players.MAX_Y_VEL = 8), so no 1-2 px
# margin can keep him out of the bomb's box.  That is done positionally by
# _in_muzzle(), the port's copy of XP's obj_bullet_bill.initialOrder guard.
BOMB_LAUNCH_DY = 0
BOMB_DEAD_POP = -2048           # XP ySpeed -2            * 0.625
BOMB_DEAD_GRAVITY = 120         # XP 0.2                  * 0.5859375
# -- stone shield (Thwomp) -------------------------------------------------- #
THWOMP_DOWN = 1920              # XP 3 px/f               * 0.625
THWOMP_DOWN_STEP = 640          # XP's final 1 px/f creep * 0.625
THWOMP_UP = 480                 # XP 0.75 px/f            * 0.625
THWOMP_HOLD = 53                # XP pauses 0.5 s at the top * 1.0666667
# The smashed block rides the pop up still upright and only turns over near the
# top of the arc (the pop apex is ~17 frames), so the flip is something the
# player actually sees instead of a corpse that starts already upside down.
THWOMP_TUMBLE_FRAMES = 12
# -- Lakitu and its Spiny --------------------------------------------------- #
LAKITU_RISE = 320               # XP yPoint += 0.5/f      * 0.625
LAKITU_TOP = -11                # XP starts 24 px above the view top * 2/3
# It sinks in and then rests XP-high, 37 px below the top of the view (its own
# canvas is 24 tall, so the cloud hangs just under Mario's head).  What matters
# is that it is re-placed at view_y + yPoint every frame, so a vertical-scroll
# level (anything taller than 13 rows) carries it along instead of stranding it
# at the world height where it first appeared.
LAKITU_MAX = 37                 # XP yPointMax 56 * 2/3 -- the user's pick
LAKITU_EDGE = 11                # XP screenLimit 16 px    * 2/3
LAKITU_CHASE = 1280             # XP 2 px/f when matching the player's dir
LAKITU_EVADE = 320              # XP 0.5 px/f against it
LAKITU_DRIFT = 640              # XP 1 px/f while the player stands still
LAKITU_ANIM_DIV = 8             # XP runs the 12 frames at 10 fps (1.2 s/cycle)
LAKITU_THROW_FRAME = 10         # XP throws at image_index 10..12
LAKITU_PERIOD = 512             # XP manager respawns the cloud every 8 s
SPINYF_POP = -2240              # XP currentY -3.5        * 0.625
SPINYF_GRAVITY = 102            # XP 0.17                 * 0.5859375
SPINYF_SIT_DY = 5               # XP spawns the walker 8 px lower * 2/3

# -- what each enemy IS ---------------------------------------------------- #
# Six membership lists used to be hand-written at six different spots in this
# file -- NEW_BOX, _NO_TILES, _IMMORTAL, _NEVER_SLEEPING, and two inline lists
# inside _uses_subpixel() and _fireproof().  Adding one enemy meant editing six
# places, and reading up what an enemy does meant the same trip in reverse.
# They are one table now and the old names are derived from it below, so every
# call site keeps working unchanged.  Write only what differs from the default.
#   box           (dx, dy, w, h) relative to (x_pos, y_pos), the 20x14 tile box
#                 a walker occupies.  The new art is wider/taller than that box,
#                 so both the blit and the background save rectangle need it.
#                 None = a Pictures-frame enemy, which needs no override.
#   no_tiles      the mover integrates its own sub-pixel motion and never
#                 touches the walker tile physics in Check().
#   subpixel      new_enemy() has to initialise dir_sign / fly_*.
#   immortal      no stomp kill, no fireball, no star, no shell duel.
#   fireproof     fireballs fizzle; the star and a shell still work.  This is
#                 only the explicit per-type half -- black creatures are
#                 fireproof by colour, see _is_black_creature.
#   never_sleeps  it must keep running outside the vertical dormancy band,
#                 usually because it only ever leaves: a corpse has to keep
#                 falling until the world-bottom rule in MoveEnemies() frees its
#                 pool slot, or a tall level leaks one record per kill.
#   in_pipe       MoveEnemies() gives it one step of the pipe clock per frame
#                 (_move_from_pipe).  TP_FAST_PLANT is not one of these: it is
#                 driven from _move_new() and has no pipe cycle.
#   gravity       the free-fall MoveEnemies() adds outside the walker tile
#                 physics, and which of the two families it belongs to:
#                 GRAV_PIPE  a thing spouting from the floor -- +1 y_vel every
#                            third tick until it reaches the world bottom;
#                 GRAV_THROWN a body thrown up by a stomp -- +1 y_vel whenever
#                            its x is a multiple of 6, and it runs NO tile
#                            check (_check is for things that can still walk).
#                 The dead spouting kinds (TP_DEAD_VERT_FISH,
#                 TP_DEAD_VERT_PLANT) keep GRAV_PIPE: a shot fish or a killed
#                 plant carries on rising the way it was rising.
GRAV_NONE = 0
GRAV_PIPE = 1
GRAV_THROWN = 2


@dataclass(frozen=True)
class EnemySpec:
    box: tuple[int, int, int, int] | None = None
    no_tiles: bool = False
    subpixel: bool = False
    immortal: bool = False
    fireproof: bool = False
    never_sleeps: bool = False
    # The player-contact chain and the enemy-vs-enemy test used to keep their
    # membership in three long tuples buried inside move_enemies and _check,
    # so adding an enemy meant editing two of the biggest functions as well as
    # this table.  These three flags are that information, per enemy.
    touched: bool = False      # the player can make contact with it
    starts_duel: bool = False  # it initiates the enemy-vs-enemy test
    duel_victim: bool = False  # it can be the other side of that test
    in_pipe: bool = False      # MoveEnemies runs the pipe clock for it
    gravity: int = GRAV_NONE   # GRAV_PIPE / GRAV_THROWN, see above
    # The three flyer families.  These were hand-written tuples beside the
    # movers, so a new flyer had to be registered twice -- once here and once in
    # each list -- and forgetting the second copy only shows up as the enemy
    # quietly not flying, not hopping, or falling through the floor.
    flyer: bool = False        # it has wings (see also _WINGED_FORM)
    hops: bool = False         # it hops like the green flyer, not patrols
    fly_fish: bool = False     # the port-only flying fish, alive or flipped


ENEMY_SPEC: dict[int, EnemySpec] = {
    # --- Dry Bones: walks, crumbles, reassembles; fire cannot hurt it ------ #
    # 2026-10-24 the user repainted the skeleton at its own size -- the walker is
    # as tall as a Koopa (24 px canvas drawn 10 px above the box, exactly like
    # GRKOOPA's) and the pile is 25 wide.  Standing poses keep dy + h == 14 (feet
    # on the box bottom); the pile is centred with dx = -(w - 20) // 2.
    TP_DRY_BONES: EnemySpec((-0, -10, 20, 24), fireproof=True, touched=True, starts_duel=True, duel_victim=True),
    TP_DRY_BONES_FALL: EnemySpec((-1, 4, 23, 10), fireproof=True, touched=True, starts_duel=True, duel_victim=True),
    TP_DEAD_DRY_BONES: EnemySpec((-2, 1, 25, 13), fireproof=True,
                                 never_sleeps=True, gravity=GRAV_THROWN),
    # --- Hammer Bro and its hammers ---------------------------------------- #
    TP_HAMMER_BRO: EnemySpec((0, -17, 20, 31), no_tiles=True, subpixel=True, touched=True, starts_duel=True, duel_victim=True),
    TP_DEAD_HAMMER_BRO: EnemySpec((0, -8, 20, 23), no_tiles=True,
                                  never_sleeps=True),
    TP_HAMMER: EnemySpec((0, 0, 11, 13), no_tiles=True, subpixel=True, touched=True),
    # --- Boo: freezes when looked at, passes through walls ----------------- #
    TP_BOO: EnemySpec((0, 0, 20, 14), no_tiles=True, subpixel=True,
                      fireproof=True, touched=True, duel_victim=True),
    TP_DEAD_BOO: EnemySpec((0, 0, 20, 14), no_tiles=True, fireproof=True,
                           never_sleeps=True),
    # --- Bomb stand / bomb: the stand is a mechanism, nothing kills it ------ #
    TP_BOMB_STAND: EnemySpec((0, 0, 20, 14), no_tiles=True, immortal=True, touched=True),
    TP_BOMB: EnemySpec((0, 1, 20, 12), no_tiles=True, subpixel=True,
                       fireproof=True, touched=True),
    TP_DEAD_BOMB: EnemySpec((0, 1, 20, 12), no_tiles=True, fireproof=True,
                            never_sleeps=True),
    # --- Big Stone / Thwomp ------------------------------------------------ #
    TP_THWOMP: EnemySpec((-5, 0, 30, 32), no_tiles=True, subpixel=True, touched=True),
    TP_DEAD_THWOMP: EnemySpec((-5, 0, 30, 32), no_tiles=True, subpixel=True,
                              never_sleeps=True),
    # --- Lakitu on the cloud, and the Spiny it drops ----------------------- #
    TP_LAKITU: EnemySpec((0, 0, 20, 24), no_tiles=True, subpixel=True, touched=True, starts_duel=True),
    TP_DEAD_LAKITU: EnemySpec((0, 0, 20, 24), no_tiles=True,
                              never_sleeps=True),
    TP_SPINY_FALL: EnemySpec((0, -1, 20, 15), no_tiles=True, subpixel=True, touched=True, starts_duel=True, duel_victim=True),
    # --- projectiles and the port-only movers ------------------------------ #
    # SMB3's fireball art is 12x7.  The lava fireball is invincible by the
    # user's rule (2026-10-25), so even the star only costs a life.
    TP_PLANT_FIRE: EnemySpec((0, 0, 12, 7), no_tiles=True, subpixel=True, touched=True),
    TP_VERT_FIREBALL: EnemySpec(immortal=True, touched=True, in_pipe=True,
                                gravity=GRAV_PIPE),
    TP_FAST_PLANT: EnemySpec(no_tiles=True, touched=True, starts_duel=True),
    # The ghost shadow is unkillable and no position rule of MoveEnemies() ever
    # takes it off the level either (the user's rule: it must not vanish), so it
    # has to keep running to notice the player coming back into its leash.
    TP_GHOST_SHADE: EnemySpec(no_tiles=True, subpixel=True, immortal=True,
                              never_sleeps=True, touched=True),
    # --- the original Pascal corpses: no canvas, they just must not sleep --- #
    TP_DYING: EnemySpec(never_sleeps=True),
    TP_DYING_FIREBALL: EnemySpec(never_sleeps=True),
    TP_DYING_KOOPA: EnemySpec(never_sleeps=True),
    TP_FLAT_CHIBIBO: EnemySpec(never_sleeps=True, starts_duel=True, duel_victim=True),
    TP_DEAD_CHIBIBO: EnemySpec(never_sleeps=True, gravity=GRAV_THROWN),
    TP_DEAD_RED: EnemySpec(never_sleeps=True, gravity=GRAV_THROWN),
    TP_DEAD_KOOPA: EnemySpec(never_sleeps=True, gravity=GRAV_THROWN),
    TP_DEAD_VERT_FISH: EnemySpec(never_sleeps=True, gravity=GRAV_PIPE),
    TP_DEAD_VERT_PLANT: EnemySpec(never_sleeps=True, starts_duel=True,
                                  gravity=GRAV_PIPE),
    TP_DEAD_FLY_FISH: EnemySpec(never_sleeps=True, fly_fish=True),
    TP_GHOST_FLAT: EnemySpec(never_sleeps=True, starts_duel=True, duel_victim=True),
    TP_GHOST_DEAD: EnemySpec(never_sleeps=True, gravity=GRAV_THROWN),
    # --- the original Pascal types: only the contact flags live here ----- #
    # Their boxes and physics stay in the walker / flyer tables they have
    # always used.  These rows exist so that "who can be touched, who starts
    # a duel, who can be duelled" is one line per enemy.  The lift slots
    # 62..69 have no name of their own and the board kinds are a family --
    # both are covered by the helpers next to the derived sets below.
    TP_CHIBIBO: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
    TP_CHAMP: EnemySpec(touched=True),
    TP_LIFE: EnemySpec(touched=True),
    TP_FLOWER: EnemySpec(touched=True),
    TP_STAR: EnemySpec(touched=True),
    TP_VERT_FISH: EnemySpec(touched=True, starts_duel=True, in_pipe=True,
                            gravity=GRAV_PIPE),
    TP_VERT_PLANT: EnemySpec(touched=True, starts_duel=True, in_pipe=True),
    TP_RED: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
    TP_KOOPA: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
    TP_SLEEPING_KOOPA: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
    TP_WAKING_KOOPA: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
    TP_RUNNING_KOOPA: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
    TP_LIFT_START: EnemySpec(touched=True),
    TP_DONUT: EnemySpec(touched=True),
    TP_LIFT_END: EnemySpec(touched=True),
    TP_FLY_GREEN: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                            flyer=True, hops=True),
    TP_FLY_RED: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                          flyer=True),
    TP_FLY_BLACK: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                            flyer=True, hops=True),
    TP_FLY_SPINY: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                            flyer=True, hops=True),
    TP_FLY_SPINY_BLACK: EnemySpec(touched=True, starts_duel=True,
                                  duel_victim=True, flyer=True, hops=True),
    TP_FLY_FISH: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                           fly_fish=True),
    TP_FLY_CHIBIBO: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                              flyer=True, hops=True),
    TP_FLY_BUZZY: EnemySpec(touched=True, starts_duel=True, duel_victim=True,
                            flyer=True),
    TP_PLATFORM_PATROL: EnemySpec(touched=True),
    TP_PLATFORM_OSC: EnemySpec(touched=True),
    TP_PLATFORM_FALL: EnemySpec(touched=True),
    TP_PLATFORM_PATH: EnemySpec(touched=True),
    TP_FIRE_PLANT: EnemySpec(touched=True, starts_duel=True, in_pipe=True),
    TP_FIRE_PLANT_AIM: EnemySpec(touched=True, starts_duel=True, in_pipe=True),
    TP_SIDE_PLANT: EnemySpec(touched=True, starts_duel=True, in_pipe=True),
    TP_BLACK_PLANT: EnemySpec(touched=True, starts_duel=True, in_pipe=True),
    TP_GHOST_CHIBIBO: EnemySpec(touched=True, starts_duel=True, duel_victim=True),
}

# The old table names, derived.  Every one of them is only ever tested with
# `in` / keyed by tp, so the order here carries no meaning.
NEW_BOX: dict[int, tuple[int, int, int, int]] = {
    tp: s.box for tp, s in ENEMY_SPEC.items() if s.box is not None}
_NO_TILES = tuple(tp for tp, s in ENEMY_SPEC.items() if s.no_tiles)
_IMMORTAL = tuple(tp for tp, s in ENEMY_SPEC.items() if s.immortal)
_NEVER_SLEEPING = tuple(tp for tp, s in ENEMY_SPEC.items() if s.never_sleeps)
_SUBPIXEL_TPS = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.subpixel)
_FIREPROOF_TPS = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.fireproof)
# Contact and duel membership: the rows say it per enemy, and the two families
# whose slots are not individually registered (block lifts, rideable boards) are
# added by the helpers, which are the only place that still names a range.
_TOUCHED = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.touched)
_DUEL_STARTERS = frozenset(tp for tp, s in ENEMY_SPEC.items()
                           if s.starts_duel)
_DUEL_VICTIMS = frozenset(tp for tp, s in ENEMY_SPEC.items()
                          if s.duel_victim)
# The pipe families whose one per-frame job in MoveEnemies is a step of the pipe
# clock, and the two free-fall rules.  Both were hand-written tuples inline in
# that loop.
_IN_PIPE = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.in_pipe)
_FLYERS = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.flyer)
_HOPPERS = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.hops)
_FLY_FISH_TPS = frozenset(tp for tp, s in ENEMY_SPEC.items() if s.fly_fish)
# The block-lift slots are the one family the table cannot name: 62..68 have no
# type constant of their own, so this is the only place that writes the range.
_LIFT_TPS = frozenset(range(TP_LIFT_START, TP_LIFT_END + 1))
# The three transitional types: a death animation whose only future is to free
# its pool slot.  _forget() puts records into them, this is where they come out.
_DYING_STAGES = (TP_DYING, TP_DYING_FIREBALL, TP_DYING_KOOPA)


def _gravity(tp: int) -> int:
    """Which free-fall MoveEnemies adds for this type -- see the GRAV_* notes."""
    s = ENEMY_SPEC.get(tp)
    return s.gravity if s is not None else GRAV_NONE


def _is_lift(tp: int) -> bool:
    """One of the block-lift slots 60..69.  These have no type constant per
    slot, so this is the only place that names the range -- five call sites used
    to write it out."""
    return tp in _LIFT_TPS


def _touched(tp: int) -> bool:
    """Can the player make contact with this type at all?

    The rows say it per enemy.  Block lifts are the exception: their slots 62..69
    have no name of their own, so ``_is_lift`` adds them -- one place, instead of
    the 40-item tuple this used to be inside MoveEnemies.
    """
    return tp in _TOUCHED or _is_lift(tp) or _is_platform(tp)


def _starts_duel(tp: int) -> bool:
    """Does this type initiate the enemy-vs-enemy test?

    TP_BOO, TP_BOMB and TP_HAMMER are deliberately absent: a Boo, a bomb and a
    thrown hammer take no part in enemy collisions (the user's rule).  The Boo
    still dies to a shell that runs into it -- see ``_duel_victim``.
    """
    return tp in _DUEL_STARTERS


def _duel_victim(e: "EnemyRec") -> bool:
    """Can this record be the other side of an enemy-vs-enemy test?

    A resting shell counts only in the black and Buzzy colours -- that is rank 3,
    and the colour is runtime data, so this is the one rule the table cannot hold.
    """
    if e.tp == TP_DEAD_KOOPA:
        return e.sub_tp in (K_BLACK, K_BUZZY)
    return e.tp in _DUEL_VICTIMS
# TP_DRY_BONES / TP_DRY_BONES_FALL are deliberately NOT in _NO_TILES: the
# skeleton walks on the walker path and the pile just stands there.
# The stone shield is deliberately NOT in _IMMORTAL -- the user allowed the
# star to break it (2026-10-23).

# Collapse canvases (dx, dy, w, h) relative to the 20x14 walker box.  Every
# bottom is at +14.  The scatter is exactly 23*2-1 = 45 px long and 10 px tall.
DRY_SCATTER_BOX = (-12, 4, 45, 10)
DRY_HALF_BOX = (-7, 4, 34, 10)
DRY_GATHER_BOX = (-1, 4, 23, 10)
DRY_SQUAT_BOX = (0, 1, 20, 13)
DRY_RISE_BOX = (0, -3, 20, 17)
DRY_FALL_BOXES = (DRY_SCATTER_BOX, DRY_SCATTER_BOX, DRY_HALF_BOX,
                  DRY_GATHER_BOX, DRY_SQUAT_BOX, DRY_SQUAT_BOX,
                  DRY_SQUAT_BOX, DRY_RISE_BOX, NEW_BOX[TP_DRY_BONES])
# The four stage-indexed tables have to stay the same length as the schedule --
# adding a stage to one and not the others is an IndexError at draw time.
assert (len(DRY_REASSEMBLE) == len(DRY_FALL_POSE) == len(DRY_FALL_BOXES)
        == len(DRY_HEAD_BOB)), "the Dry Bones stage tables disagree"
# The dormancy has to be driven by the constant, not by a literal in the table,
# or tuning DRY_DORMANT_FRAMES would silently do nothing.
assert DRY_REASSEMBLE[0] == (0, DRY_DORMANT_FRAMES) \
    and DRY_FALL_POSE[0] == DRY_FALL_POSE[1] == "scatter", \
    "the Dry Bones dormancy is not the scatter pose held for 3 s"


def _dry_stage(counter: int) -> int:
    """Which reassembly stage a crumpled Dry Bones is in (0 = still dormant)."""
    elapsed = 0
    for i, (_, frames) in enumerate(DRY_REASSEMBLE):
        elapsed += frames
        if counter <= elapsed:
            return i
    return DRY_STAND_STAGE


def _dry_art_box(e: "EnemyRec") -> tuple[int, int, int, int]:
    """Dry Bones changes canvas every reassembly stage.

    The head bob is folded into the box (top moved up, height grown), so the
    background save rectangle always covers the shifted art; _draw_new must
    therefore *not* add DRY_HEAD_BOB a second time.
    """
    if e.tp == TP_DRY_BONES_FALL:
        stage = _dry_stage(e.counter)
        dx, dy, w, h = DRY_FALL_BOXES[stage]
        # the table only carries a bob on the squat stages, so no extra gate
        bob = DRY_HEAD_BOB[stage]
        if bob:
            dy += bob
            h -= bob
        return dx, dy, w, h
    return NEW_BOX[e.tp]


def _new_box(e: "EnemyRec") -> tuple[int, int, int, int]:
    if _is_platform(e.tp):
        return (0, 0, max(1, e.plat_len) * PLATFORM_SEG_W, PLATFORM_SEG_H)
    if e.tp in (TP_DRY_BONES, TP_DRY_BONES_FALL):
        return _dry_art_box(e)
    return NEW_BOX[e.tp]


def _collapse_pile(e: "EnemyRec") -> None:
    """Take a walking Dry Bones apart -- a stomp or a bumped block both do it
    (the user's rule) -- and start SMB3's eight-stage reassembly."""
    e.tp = TP_DRY_BONES_FALL
    e.x_vel = 0
    e.y_vel = 0
    e.move_delay = 0            # DRY_REVIVE_FRAMES counts frames
    e.delay_counter = 0
    e.counter = 0


# _NEVER_SLEEPING (the states that must keep running outside the dormancy band)
# is derived from ENEMY_SPEC above.


def _new_mover(tp: int) -> bool:
    return 80 <= tp <= 99 or tp == TP_PLANT_FIRE


def _uses_subpixel(tp: int) -> bool:
    """Types that need dir_sign / fly_* initialised by new_enemy()."""
    return tp in _SUBPIXEL_TPS

# TmpObj types (TMPOBJ.PAS) — ShowStar / ShowFire spawn these.
TP_HIT = 3
TP_FIRE = 4

K_GREEN = 0
K_RED = 1
K_BLACK = 2
K_BUZZY = 3               # port-only SMB3 Buzzy: fireproof, star-killable

# flyer -> (ground form, SubTp, MoveDelay).  The delays are the ones the stomped
# flyers already used, so a de-winged Koopa still walks as fast as it did.
_WINGED_FORM: dict[int, tuple[int, int, int]] = {
    TP_FLY_CHIBIBO: (TP_CHIBIBO, 0, 2),
    TP_FLY_BUZZY: (TP_KOOPA, K_BUZZY, 3),
    TP_FLY_GREEN: (TP_KOOPA, K_GREEN, 0),
    TP_FLY_RED: (TP_KOOPA, K_RED, 0),
    TP_FLY_BLACK: (TP_KOOPA, K_BLACK, 0),
    TP_FLY_SPINY: (TP_RED, 0, 2),
    TP_FLY_SPINY_BLACK: (TP_RED, K_BLACK, 0),
}


def _de_wing(e: "EnemyRec") -> None:
    """Take the wings off a flyer -- a stomp or a bumped block both do it (the
    user's rule), and what is left walks exactly like its ground form."""
    tp, sub, delay = _WINGED_FORM[e.tp]
    e.tp = tp
    e.sub_tp = sub
    e.x_vel = e.dir_sign
    e.y_vel = 0
    e.status = FALLING
    e.move_delay = delay
    e.delay_counter = 0

# The black Koopa reuses the green body art with only the shell ramp swapped
# for the grayscale ramp (the same preview the user approved):
#   green shell 112..119 -> 16,17,18,19,20,22,23,25.
# The black Spiny does the same with the red shell ramp 56..63.  Everything
# else (body, eyes, feet, the 48..55 shadow/highlight tones that the legs also
# use) stays exactly as the original drew it.
_SHELL_BLACK_RAMP = (16, 17, 18, 19, 20, 22, 23, 25)
_SHELL_GREEN_RAMP = (112, 113, 114, 115, 116, 117, 118, 119)


def _swap_ramp(idx: np.ndarray, src: tuple[int, ...]) -> np.ndarray:
    """Recolour one 8-step shell ramp to the grayscale ramp."""
    out = idx.copy()
    for s, d in zip(src, _SHELL_BLACK_RAMP):
        out[idx == s] = d
    return out


def _blacken_shell(idx: np.ndarray) -> np.ndarray:
    return _swap_ramp(idx, _SHELL_GREEN_RAMP)


# The Spiny shell is drawn with 16 red tones, not the Koopa's 8: 48..51 are
# shadows, 56..63 the body ramp and 52..55 the pale spike highlights.  All of
# them have to go or the back still reads red.  They are ranked by luminance
# and compressed into the same 0..129 gray span the Koopa shell uses, so both
# blacks match.  49/53 also appear on the legs (RED000 (13,2)/(12,16),
# RED001 (12,6)), so the swap stops at row 12 -- the shell's own bottom edge
# is row 11 and must be recoloured too.
_SPINY_BLACK_MAP = {56: 16, 48: 16, 57: 16, 58: 17, 49: 17, 59: 18, 50: 19,
                    60: 19, 61: 20, 51: 20, 62: 20, 52: 22, 63: 22, 53: 22,
                    54: 23, 55: 25,
                    # the pink shading on the shell's white rim (124/125/126)
                    # would otherwise stay reddish on a black shell
                    124: 25, 125: 26, 126: 27}
_SHELL_RED_TONES = tuple(_SPINY_BLACK_MAP)
_SPINY_SHELL_ROWS = 12
_SPINY_LUT = np.arange(256, dtype=np.uint8)
for _s, _d in _SPINY_BLACK_MAP.items():
    _SPINY_LUT[_s] = _d
del _s, _d


def _blacken_spiny(idx: np.ndarray) -> np.ndarray:
    out = idx.copy()
    top = slice(0, _SPINY_SHELL_ROWS)
    out[top] = _SPINY_LUT[idx[top]]
    return out


# The black piranha plant: its body is exactly eight tones -- leaves/stem
# 41..45 and head 59..61 -- so, ranked by luminance, they fill the same 8-step
# grayscale ramp the black Koopa shell uses and every black in the game matches.
# The user picked variant B, which also greys the two salmon head highlights
# (52/53); left orange they read as dirt on a black body.  The white teeth (31)
# and the black outline (16) stay.
_PLANT_BLACK_MAP = {41: 16, 42: 17, 43: 18, 59: 19, 44: 20, 45: 22, 60: 23,
                    61: 25, 52: 23, 53: 25}
_PLANT_BLACK_LUT = np.arange(256, dtype=np.uint8)
for _s, _d in _PLANT_BLACK_MAP.items():
    _PLANT_BLACK_LUT[_s] = _d
del _s, _d


def _blacken_plant(idx: np.ndarray) -> np.ndarray:
    return _PLANT_BLACK_LUT[idx]


# The Chibibo is painted with 14 body tones, not 8: the hat/body ramp 73..79,
# the face/underside salmon ramp 50..55 and the two eye-shade greys 107/109.
# Ranked dark -> light and compressed into the same 8-step grayscale the black
# Koopa shell, the black Spiny and the black plant use, so every black in the
# game matches.  The white of the eyes (31) and the black outline (16) stay.
_CHIB_TONES = (73, 74, 50, 75, 51, 107, 76, 77, 53, 78, 54, 109, 55, 79)
_CHIB_BLACK_RAMP = (16, 17, 18, 19, 20, 22, 23, 25)
_CHIB_BLACK_MAP = {tone: _CHIB_BLACK_RAMP[i * (len(_CHIB_BLACK_RAMP) - 1)
                                        // (len(_CHIB_TONES) - 1)]
                   for i, tone in enumerate(_CHIB_TONES)}
_CHIB_BLACK_LUT = np.arange(256, dtype=np.uint8)
for _s, _d in _CHIB_BLACK_MAP.items():
    _CHIB_BLACK_LUT[_s] = _d
del _s, _d


def _blacken_chibibo(idx: np.ndarray) -> np.ndarray:
    return _CHIB_BLACK_LUT[idx]


# The blue shadow the black Chibibo releases.  The user's brief (2026-10-25,
# second pass): the plain "paint it blue" read was too ordinary, so the three
# tones are rotated -- deep blue -> pale blue, white -> deep blue, pale blue ->
# white.  Applied to the two-tone Chibibo silhouette (dark body half and the
# outline were deep blue, the light half pale blue, the eye white kept), that
# leaves a pale-blue rim around a white body with deep blue eyes.
#
# Index 1 is BlinkPalette's star-flash slot, so this art must be baked with the
# master palette (see _shade_surf / fixed_palette) or the shade flickers yellow
# every frame -- the same trap the blue Dry Bones and the Boo fell into.
_CHIB_SHADE_ROTATE = {1: 95, 95: 31, 31: 1}    # deep blue -> pale -> white -> deep
_CHIB_SHADE_MAP = {tone: _CHIB_SHADE_ROTATE[1 if i < len(_CHIB_TONES) // 2
                                           else 95]
                   for i, tone in enumerate(_CHIB_TONES)}
_CHIB_SHADE_MAP[16] = _CHIB_SHADE_ROTATE[1]    # the outline was deep blue
_CHIB_SHADE_MAP[31] = _CHIB_SHADE_ROTATE[31]   # the eye white
_CHIB_SHADE_LUT = np.arange(256, dtype=np.uint8)
for _s, _d in _CHIB_SHADE_MAP.items():
    _CHIB_SHADE_LUT[_s] = _d
del _s, _d


def _shade_chibibo(idx: np.ndarray) -> np.ndarray:
    return _CHIB_SHADE_LUT[idx]


def _is_black_creature(e: "EnemyRec") -> bool:
    """The black creatures, exactly: black Koopa (walker, flyer, any shell) and
    black Spiny (walker, flyer).  The Buzzy is NOT one -- it only shares the
    fireball immunity, it is star-killable and it loses shell duels.

    Two properties, per the user (2026-10-23 correction):
      1. the star and fireballs do nothing to them;
      2. the defining one: a plain Koopa / Buzzy shell that hits them is
         destroyed while they stay unharmed (see _power rank 3)."""
    if e.tp == TP_BLACK_PLANT:
        return True
    if e.tp in (TP_FLY_BLACK, TP_FLY_SPINY_BLACK):
        return True
    if e.tp == TP_RED:
        return e.sub_tp == K_BLACK
    return e.sub_tp == K_BLACK and TP_KOOPA <= e.tp <= TP_DEAD_KOOPA


def _power(e: "EnemyRec") -> int:
    """Combat rank for the shell-vs-enemy duel.  Higher rank wins, equal
    ranks destroy each other.  One strict ladder, no cycles:

        4  moving black shell (TP_RUNNING_KOOPA, black)
        3  any black creature in any other state (walker, flyer, sleeping,
           resting shell -- _is_black_creature)
        2  moving coloured shell (TP_RUNNING_KOOPA, coloured)
        1  everything else (walkers, chibibos, Spinies, fish, plants,
           sleeping/coloured resting shells)

    A duel only ever starts from a moving shell (or a moving shell hitting a
    resting black shell), so equal ranks can only mean shell-vs-shell and the
    mutual kill stays correct; plain walkers touching each other just bounce.
    Fireballs and the star consult _is_black_creature directly -- the "black"
    bit that makes rank >= 3 is exactly the immunity both share."""
    black = _is_black_creature(e)
    if e.tp == TP_RUNNING_KOOPA:
        return 4 if black else 2
    return 3 if black else 1


def _fireproof(e: "EnemyRec") -> bool:
    """Fireballs cannot hurt the black creatures nor the Buzzy in any of its
    Koopa states (walker / shells).  Unlike the black creatures the Buzzy is
    NOT star-immune and loses shell duels to any colour, see _power.

    The 2026-10-23 batch adds three more fireproof families on the user's
    order -- Dry Bones (all states), Boo (all states) and the bomb -- and they
    stay killable by the star and by a shell, i.e. they are NOT black
    creatures and keep _power rank 1.  Those seven states are the explicit
    half of the rule, in ENEMY_SPEC.fireproof."""
    if e.tp in _FIREPROOF_TPS:
        return True
    return (_is_black_creature(e) or e.tp == TP_FLY_BUZZY
            or (e.sub_tp == K_BUZZY and TP_KOOPA <= e.tp <= TP_DEAD_KOOPA))

def _corpse_dir(e: "EnemyRec") -> int:
    """The +/-1 horizontal drift every corpse gets -- the source rule.

    the 2006 mod source and Wiering's original tree agree, and all three
    corpse assignments in
    ``Kill`` are this same expression (ENEMIES.PAS:179 / 188 / 197): the body
    drifts by **which half of its own 20 px tile the victim stood in**, and
    nothing in the source reads where the blow came from.  The jumping fish is
    the exception the source does encode (``XVel := 0``, :206) -- it drops
    straight down.

    Measured and settled (2026-10-25): the user parked Mario with his x midpoint
    in the left half of a tile and then the right half, letting walkers run into
    the star, and the bodies went **both ways, split by that tile half** -- the
    source rule.  An earlier "it flies away from the shot" reading came from a
    single sample, which cannot separate the two rules; that implementation is
    kept in commit c45e745 if it is ever needed again.  The Boo is the one
    port-only exception to this drift: the user asked for it to fall like the
    jumping fish, straight down (see _kill).
    """
    return -1 + 2 * int((e.x_pos + e.x_vel) % W > W // 2)


GROUNDED = 0
FALLING = 1

MAX_ENEMIES = 11            # EnemyPictures[1..MaxEnemies, Left..Right]
MAX_ENEMIES_AT_ONCE = 25    # the original's fixed pool (see POOL_LIMIT)
# ENEMIES.PAS caps the pool at 25 EnemyRec records, and map mechanisms share
# those slots with monsters: a level whose current columns already hold 25
# entries silently loses further spawns, and because StartEnemies() clears the
# marker before NewEnemy() runs, that marker never comes back either.
# The port has no such memory constraint, so the pool grows on demand.
# Set POOL_LIMIT back to MAX_ENEMIES_AT_ONCE to reproduce the DOS behaviour.
POOL_LIMIT: int | None = None
MAX_PAGE = 1                # two video pages (VGAScreen)

START_ENEMIES_AT = 7
FORGET_ENEMIES_AT = 7
# Per-build bands.  The 2006 mod's source text is only true of that build, so
# every pack that came out of a different build carries its own pair, read out
# of that build's compiled game:
#   forget  MoveEnemies' `sub ax,<m*W>` / `add ax,NH*W` + `add ax,<m*W>`:
#           140 at 0x259C8 in the 2006 build, but 100 (= 5 tiles) at 0xF692 in
#           mario4, 0x19269 in 8g34 and 0x191D9 in sunyu.
#   start   ENEMIES.PAS:15.  Wiering's own tree (the same family, whose forget
#           value the three old EXEs match byte for byte) says 2; only the 2006
#           mod tree says 7.  Not yet located in the old binaries.
# start must stay <= forget: the right frontier spawns a record at
# XView + (NH+start)*W while the right bound is XView + (NH+forget)*W, so a
# larger start recycles every enemy on its first step and leaves its marker
# behind the camera, where the frontier never returns for it.
PACK_ENEMY_BANDS: dict[str, tuple[int, int]] = {
    "mario4": (2, 5),
    "8g34": (2, 5),
    "sunyu": (2, 5),
}

# Vertical twin of FORGET_ENEMIES_AT, in map rows (7 * 14 = 98 px, the user's
# number).  Unlike the horizontal rule, leaving this band does NOT release the
# enemy: it goes dormant (see Enemies._sleeping), so a tall level stops
# simulating the 26 rows start_enemies() spawns in one column without anyone
# having to re-read the markers when the camera comes back.
SLEEP_ENEMIES_AT = 7

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 200

# cp865 map markers placed by WORLDS.PAS and consumed by StartEnemies().
CH_CHIBIBO = 0x80           # 'Ç'
CH_VERT_FISH = 0x81         # 'ü'
CH_VERT_FIREBALL = 0x82     # 'é'
CH_CHIBIBO_B = 0x83         # 'â'  (SubTp = 1)
CH_PLANT0 = 0x84            # 'ä'
CH_PLANT1 = 0x85            # 'à'
CH_PLANT2 = 0x86            # 'å'
CH_RED = 0x87               # 'ç'
CH_KOOPA0 = 0x88            # 'ê'
CH_KOOPA1 = 0x89            # 'ë'
CH_KOOPA2 = 0x8A            # 'è'
CH_FLY_GREEN = 0x8B         # 'ø'  port-only: hopping winged Koopa
CH_FLY_RED_H = 0x8C         # '£'  port-only: red flyer, horizontal patrol
CH_FLY_RED_V = 0x8D         # 'Ø'  port-only: red flyer, vertical patrol
CH_FLY_BLACK = 0x8E         # port-only: black hopper (same motion as green)
CH_SPINY_BLACK = 0x8F       # port-only: walking black Spiny (TP_RED SubTp 2)
CH_FLY_SPINY = 0x90         # port-only: winged Spiny, original red
CH_FLY_SPINY_BLACK = 0x91   # port-only: winged black Spiny
CH_FLY_FISH = 0x92          # port-only: fish spout, never consumed
CH_FLY_CHIBIBO = 0x93       # port-only: winged Chibibo (hops like green)
CH_BUZZY = 0x94             # port-only: walking Buzzy Beetle (slower Koopa)
CH_FLY_BUZZY = 0x95         # port-only: winged Buzzy, level horizontal cruise
CH_DRY_BONES = 0x96         # port-only: Dry Bones (revives when stomped)
CH_HAMMER_BRO = 0x97        # port-only: Hammer Bro, patrols left of its marker
CH_BOO = 0x98               # port-only: Boo, chases through terrain
CH_BOMB_STAND = 0x99        # port-only: bomb stand, fires every 3 s on screen
CH_THWOMP = 0x9A            # port-only: stone shield, slams between floor/ceiling
CH_LAKITU = 0x9B            # port-only: Lakitu cloud marker (spawner switch)
CH_BLOCK_LIFT = 0xB0        # '░'
CH_DONUT = 0xB1             # '▒'

# --------------------------------------------------------------------------- #
# Wings: one independent sprite pair reused by every winged enemy.
#
# The art is hand-edited at mario_py/data/refs/wing/wing_up|wing_down.png.
# Grey 130,130,130 is the transparency key -- it is a real palette tone, so it
# is matched *exactly* while every other colour snaps to the nearest MPAL256
# entry.  FLY_SPEC says where each family wears them, so swapping the art or
# giving a new enemy wings is one line, not a retyped bitmap.
WING_DIR = Path(__file__).resolve().parent.parent / "data" / "refs" / "wing"
WING_FRAMES = ("wing_up", "wing_down")
WING_KEY_RGB = (130, 130, 130)

FLY_W, FLY_H = 20, 24       # the wing never leaves the body's 20 columns
FLY_BODY_Y = 0              # the 20x24 Koopa body fills the canvas
FLY_SHORT_BODY_Y = 10       # 20x14 bodies (Spiny, Chibibo) sit at the bottom
FLY_DRAW_DY = -10           # canvas top relative to YPos (body bottom = YPos+14)

# body_y  row of the body inside the canvas
# wing_y  row of the wing; the Koopa's wing top is its head crown (row 1) and
#         the wing's outer column is the shell's rear (col 19)
# over    True  = the wing root presses on the shell (side-facing enemies)
#         False = the body paints over the wing roots (front-facing)
# pair    True  = a mirrored left/right pair, one at each canvas edge
FLY_SPEC = {
    "koopa": dict(body_y=FLY_BODY_Y, wing_y=1, over=True, pair=False),
    "short": dict(body_y=FLY_SHORT_BODY_Y, wing_y=9, over=True, pair=False),
    "chibibo": dict(body_y=FLY_SHORT_BODY_Y, wing_y=1, over=False, pair=True),
}


def load_wings(palette) -> list[np.ndarray]:
    """Read the editable wing PNGs into palette-index arrays."""
    from PIL import Image           # only needed to read the art
    pal = np.asarray(palette, int)
    out = []
    for name in WING_FRAMES:
        rgb = np.array(Image.open(WING_DIR / f"{name}.png")
                       .convert("RGB")).astype(int)
        key = (rgb == np.array(WING_KEY_RGB)).all(axis=2)
        d = np.abs(pal[None, None, :] - rgb[:, :, None]).sum(axis=3)
        idx = d.argmin(axis=2).astype(np.uint8)
        idx[key] = 0
        out.append(idx)
    return out

# sentinel for BackGrAddr[Page] == $FFFF (no background saved this frame)
NO_BACKGR = -1

W = TILE_W
H = TILE_H
SAFE = EY1                  # 8: the world has EY1 rows above row 0
HSAFE = H * SAFE            # 112: offsets WorldMap row indices by SAFE


def idiv(a: int, b: int) -> int:
    """Pascal ``a div b`` — truncation toward zero (Python // floors)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _mirror(arr: np.ndarray) -> np.ndarray:
    """Mirror()/Mirror20x24(): horizontal flip, dest[y, x] = src[y, W-1-x]."""
    return np.ascontiguousarray(arr[:, ::-1])


# Palette-index planes for the port-only batch, produced by
# tools/install_new_enemy_sprites.py.  They are not in data/sprites/index.json
# because they have no Pascal source archive.
NEW_ART_DIR = (Path(__file__).resolve().parent.parent
               / "data" / "newenemies" / "idx")
_new_art_cache: dict[str, np.ndarray] = {}


def _new_art(name: str) -> np.ndarray:
    arr = _new_art_cache.get(name)
    if arr is None:
        arr = np.load(NEW_ART_DIR / f"{name}.npy")
        _new_art_cache[name] = arr
    return arr


def _compose_wing(base: np.ndarray, wing: np.ndarray, spec: dict,
                  right: bool = False) -> np.ndarray:
    """Composite one winged enemy canvas from a body and the shared wing.

    Side-facing enemies wear a single wing at the rear: its outer column is the
    shell's rear column, so it never hangs past the body.  A front-facing
    (``pair``) enemy wears a mirrored pair and the body is painted over the
    roots, exactly like the SMB3 original.
    """
    out = np.zeros((FLY_H, FLY_W), dtype=np.uint8)

    def blit(src: np.ndarray, x: int, y: int) -> None:
        reg = out[y:y + src.shape[0], x:x + src.shape[1]]
        m = src != 0
        reg[m] = src[m]

    body = _mirror(base) if right else base
    if spec["pair"]:
        blit(_mirror(wing), 0, spec["wing_y"])
        blit(wing, FLY_W - wing.shape[1], spec["wing_y"])
        blit(body, 0, spec["body_y"])
        return out
    w = _mirror(wing) if right else wing
    wx = 0 if right else FLY_W - wing.shape[1]
    if spec["over"]:
        blit(body, 0, spec["body_y"])
        blit(w, wx, spec["wing_y"])
    else:
        blit(w, wx, spec["wing_y"])
        blit(body, 0, spec["body_y"])
    return out


def _flying(tp: int) -> bool:
    """The seven winged kinds.  Their other half is ``_WINGED_FORM`` -- the wing
    they lose and the ground form left behind; verify_enemies BB keeps the two
    lists the same length."""
    return tp in _FLYERS


def _hops_like_green(tp: int) -> bool:
    """Green Koopas, black Koopas, both Spinies and the Chibibo use the same
    hop; the red flyer patrols instead (see ``_move_fly_red``)."""
    return tp in _HOPPERS


def _fish(tp: int) -> bool:
    """The port-only flying fish, alive or flipped."""
    return tp in _FLY_FISH_TPS

# ENEMIES.PAS:682-683 keeps five types out of the walker tile path: the things
# born in the floor and the corpse of a plant.  TP_VERT_PLANT is already covered
# by _is_plant(), so this is the rest of the source's set.
# The corpse belongs here because its frames come from the draw pass, which
# counts them in ``status``: the tile path writes GROUNDED/FALLING into the same
# field, so the animation never reaches its end and the record never leaves the
# pool (it sat in the pipe mouth for ever -- fixed 2026-10-26, see verify_enemies
# BA).  The other four are excluded because the pipe clock owns their motion.
_NO_WALKER_TILES = (TP_VERT_FISH, TP_DEAD_VERT_FISH, TP_VERT_FIREBALL,
                    TP_DEAD_VERT_PLANT)


def _walks_on_tiles(tp: int) -> bool:
    """Does this type run Check()'s walker tile physics (turn at walls,
    grounded/falling)?"""
    return not (_flying(tp) or _fish(tp) or _is_plant(tp)
                or tp in _NO_TILES or tp in _NO_WALKER_TILES)


class EnemyHooks:
    """Call-outs into the units that ENEMIES.PAS uses but which are not ported
    yet (Glitter / TmpObj / Music / Blocks / Status).  The default
    implementation is a set of no-ops, which is enough to exercise the enemy
    logic headlessly; play.py will supply a real one."""

    # --- BUFFERS.PAS ----------------------------------------------------- #
    def add_score(self, n: int) -> None:
        """AddScore(N): Inc(LevelScore, N)."""
        return None

    # --- Crt / MUSIC.PAS ------------------------------------------------- #
    def beep(self, freq: int) -> None:
        return None

    def start_music(self, name: str) -> None:
        """StartMusic(FireMusic) etc.; ``name`` is the MUSIC.PAS constant."""
        return None

    # --- TMPOBJ.PAS ------------------------------------------------------ #
    def new_temp_obj(self, tp: int, x: int, y: int, xv: int, yv: int,
                     wid: int, ht: int) -> None:
        return None

    def break_block(self, map_x: int, map_y: int) -> None:
        """BreakBlock(X, Y) — map coordinates."""
        return None

    def hit_coin(self, x: int, y: int, throw_up: bool) -> None:
        """HitCoin(X, Y, ThrowUp) — pixel coordinates."""
        return None

    def remove(self, x: int, y: int, w: int, h: int, new_img: int) -> None:
        """Remove(X, Y, W, H, NewImg) — RunRemove draws NewImg over the area."""
        return None

    # --- GLITTER.PAS ----------------------------------------------------- #
    def new_glitter(self, x: int, y: int, attr: int, duration: int) -> None:
        return None

    def new_star(self, x: int, y: int, attr: int, duration: int) -> None:
        """Glitter.NewStar — the five-pixel sparkle cross."""
        return None

    def start_glitter(self, x: int, y: int, w: int, h: int) -> None:
        return None

    def coin_glitter(self, x: int, y: int) -> None:
        return None


@dataclass
class EnemyRec:
    """EnemyRec from ENEMIES.PAS.  Tp/SubTp are the type/sub-type, Status is
    Grounded/Falling for walkers and the state-machine step for plants."""
    tp: int = TP_DEAD
    sub_tp: int = 0
    x_pos: int = 0
    y_pos: int = 0
    last_x_pos: int = 0
    last_y_pos: int = 0
    map_x: int = 0
    map_y: int = 0
    x_vel: int = 0
    y_vel: int = 0
    move_delay: int = 0
    delay_counter: int = 0
    counter: int = 0
    status: int = 0
    dir_counter: int = 0
    # underwater speed reduction: one skipped step in three (port-only)
    water_skip: int = 0
    # 0 until the enemy has actually been inside the visible window.  The
    # forget boundary on the right is exactly the column StartEnemies() spawns
    # into, so a rightward mover (block lifts!) used to be dropped on the very
    # frame it was created -- and since StartEnemies() already consumed the
    # marker, it never came back.  See the forget branch of move_enemies().
    seen: int = 0
    # Facing of the port-only flying Koopas.  Their horizontal motion is
    # decoupled from XVel (the green one needs per-frame gravity, the walkers
    # use MoveDelay), so XVel alone cannot carry the direction.
    dir_sign: int = 1
    # sub-pixel accumulators for the flying Koopas (see FLY_* constants)
    fly_xrem: int = 0       # 1/FLY_X_DEN px horizontal remainder
    fly_yrem: int = 0       # 1/FLY_Y_DEN px vertical remainder
    fly_vy: int = 0         # green hop velocity, 1/FLY_Y_DEN px/frame
    fly_vdir: int = -1      # vertical red patrol direction
    fly_spd: int = 0        # red path speed, 1/FLY_X_DEN px/frame
    fly_phase: int = 0      # red path: 0 = accelerating, 1 = decelerating
    fire_cooldown: int = 0  # TP_FIRE_PLANT: frames until it may spit again
    # Long-platform state (TP_PLATFORM_*).  plat_* velocities are 1/1024 px/f;
    # x_vel/y_vel are the whole-pixel step of this frame, which is what the
    # player's cd_lift branch inherits.
    plat_len: int = 1
    plat_kind: int = 0
    plat_home_x: int = 0
    plat_home_y: int = 0
    plat_axis: int = 0       # 0 horizontal, 1 vertical
    plat_range: int = 0      # patrol one-way px; 0 = SMB3 osc has no cap
    plat_speed: int = 0      # 1/1024 px/f (fall: gravity, 1/1024 px/f^2)
    plat_vx: int = 0
    plat_vy: int = 0
    plat_xrem: int = 0
    plat_yrem: int = 0
    plat_dir: int = 1
    plat_hold: int = 0
    plat_state: int = 0      # fall: 0 waiting, 1 falling
    plat_spec: int = -1      # index into Enemies.platforms (respawn key)
    plat_parity: int = 0
    plat_path: list = field(default_factory=list)
    plat_idx: int = 0
    plat_loop: bool = False
    # BackGrAddr[0 .. MAX_PAGE]: $FFFF means "not saved", so NO_BACKGR here.
    backgr_addr: list[int] = field(
        default_factory=lambda: [NO_BACKGR] * (MAX_PAGE + 1))


class Enemies:
    """The whole ENEMIES.PAS unit: the picture tables, the enemy pool and the
    per-frame move/show/hide machinery."""

    def __init__(self, figures, assets: "object", world: WorldMap,
                 options: WorldOptions, hooks: EnemyHooks | None = None,
                 seed: int | None = None, pack: str = "") -> None:
        self.figures = figures
        self.assets = assets
        self.world = world
        self.options = options
        self.hooks = hooks if hooks is not None else EnemyHooks()
        # Which build this level came from, and the two bands it was compiled
        # with (main-line/port packs keep the 2006 mod's pair).
        self.pack = pack
        self.start_at, self.forget_at = PACK_ENEMY_BANDS.get(
            pack, (START_ENEMIES_AT, FORGET_ENEMIES_AT))
        # Random() — seedable so the verification suite is reproducible.
        self.rng = random.Random(seed)
        self.turbo = False

        # --- module-level state shared with PLAYERS.PAS ------------------ #
        # PlayerX1..PlayerYVel / Star are globals in the original; the caller
        # copies the player's box in before move_enemies() and reads the
        # cd* flags (and the lift overrides) out afterwards.
        self.view_x = 0
        self.view_y = 0
        # Battle-room safe zone: (first_col, last_col) inclusive.  Enemies
        # touching these columns are not horizontally forgotten/recycled, so a
        # camera-locked room cannot mistake a recycled target for a defeat.
        self.protect: tuple[int, int] | None = None
        # Flying-fish spouts: (world id, column -> cells) plus one timer per
        # cell.  Keyed by id(world) so play.py's `obj.world = lv.world`
        # re-binding invalidates the cache by itself.
        self._fish_cache: tuple[int, dict[int, list[tuple[int, int]]]] | None = None
        self._fish_timers: dict[tuple[int, int], int] = {}
        # Long-platform strips, cached by (sprite variant, segment count).
        self._plat_surf: dict[tuple[int, int], pygame.Surface] = {}
        self._side_plant_cache: dict[tuple[str, bool], pygame.Surface] = {}
        self.platforms: list[dict] = []
        # Boards that have drifted past the forget margin.  Like a forgotten
        # enemy they are NOT teleported back while the player watches: they
        # only come again when the spawn frontier re-crosses their column
        # (see start_enemies).
        self._plat_forgotten: set[int] = set()
        # Port-only per-marker direction overrides from level JSON
        # ``enemy_dirs``; markers not listed walk toward the player as usual.
        self.dir_overrides: dict[tuple[int, int], int] = {}
        # obj_lakitu_manager: one respawn timer for the whole level, plus the
        # cached column list of CH_LAKITU switches (never consumed).
        self._lakitu_cache: tuple[int, list[tuple[int, int]]] | None = None
        self._lakitu_timer = 0
        self.player_x1 = 0
        self.player_y1 = 0
        self.player_x2 = 0
        self.player_y2 = 0
        self.player_xvel = 0
        self.player_yvel = 0
        self.player_in_water = False
        # XP's obj_player.lastHorizontalDirection: the facing the Boo and the
        # Lakitu chase logic read.  ENEMIES.PAS only gets the velocity, so the
        # port latches the last non-zero one.
        self.player_face = 0
        self.star = False
        # Data / Player (BUFFERS.PAS globals); Data.Mode[Player] decides
        # champ vs. flower out of a '?' block.
        self.data = None
        self.player = 0
        # condition flags consumed by the player code each frame
        self.cd_champ = 0
        self.cd_life = 0
        self.cd_flower = 0
        self.cd_star = 0
        self.cd_enemy = 0
        self.cd_hit = 0
        self.cd_lift = 0
        self.cd_stop_jump = 0

        self.time_counter = 0

        # EnemyPictures[1..MaxEnemies, Left..Right] and
        # KoopaList[Left..Right, kGreen..kRed, 0..1] as raw index arrays;
        # surfaces are baked lazily through the Figures palette cache.
        self._pic: list[list[np.ndarray | None]] = [[None, None]
                                                    for _ in range(MAX_ENEMIES + 1)]
        self._koopa: list[list[list[np.ndarray | None]]] = [
            [[None, None] for _ in range(4)] for _ in range(2)]
        # shell pictures [color][dir][frame]: same data as EnemyPictures 8..11
        # for green/red, plus the black-recoloured GRKP pair.
        self._shell: list[list[list[np.ndarray | None]]] = [
            [[None, None] for _ in range(2)] for _ in range(4)]
        # Spinies: [color][dir][frame] and the winged pair
        # [color][dir][frame][wing].  color 0 = the original red, K_BLACK the
        # recoloured one; index 1 (kRed) is unused because SubTp 0 already
        # means "the original Spiny art".
        self._spiny: list[list[list[np.ndarray | None]]] = [
            [[None, None] for _ in range(2)] for _ in range(3)]
        self._fly_spiny: list[list[list[list[np.ndarray | None]]]] = [
            [[[None, None] for _ in range(2)] for _ in range(2)]
            for _ in range(3)]
        # [color][dir][walk frame][wing frame]
        self._fly: list[list[list[list[np.ndarray | None]]]] = [
            [[[None, None] for _ in range(2)] for _ in range(2)]
            for _ in range(3)]
        # flying Chibibo: [dir][wing frame]  (fixed body, no colour axis)
        self._fly_chib: list[list[np.ndarray | None]] = [
            [None, None] for _ in range(2)]
        # flying Buzzy: [dir][walk frame][wing frame]
        self._fly_buzzy: list[list[list[np.ndarray | None]]] = [
            [[None, None] for _ in range(2)] for _ in range(2)]
        self._fireball: list[np.ndarray] = []
        self.init_enemy_figures()

        # the pool: index 1..MaxEnemiesAtOnce (index 0 unused, like Pascal)
        self.pool: list[EnemyRec] = [EnemyRec()
                                     for _ in range(MAX_ENEMIES_AT_ONCE + 1)]
        self.active: list[int] = []

    # ------------------------------------------------------------------ #
    # picture tables (InitEnemyFigures)
    # ------------------------------------------------------------------ #
    def init_enemy_figures(self) -> None:
        """Build EnemyPictures / KoopaList / FireBallList.

        Note the asymmetry of the original: pictures 6/7 (Red000/001) are
        stored left-facing and mirrored to the right, everything else is
        stored right-facing and mirrored to the left; picture 3 (the fish) is
        stored left-facing and mirrored, and the loop skips it."""
        a = self.assets
        pic = self._pic

        def put(n: int, d: int, name: str) -> None:
            pic[n][d] = a.sprite_idx(name)

        put(1, DIR_RIGHT, "CHIBIBO000")
        put(2, DIR_RIGHT, "CHIBIBO001")
        put(4, DIR_RIGHT, "CHIBIBO002")
        put(5, DIR_RIGHT, "CHIBIBO003")
        put(3, DIR_LEFT, "FISH001")
        put(6, DIR_LEFT, "RED000")
        put(7, DIR_LEFT, "RED001")
        put(8, DIR_RIGHT, "GRKP000")
        put(9, DIR_RIGHT, "GRKP001")
        put(10, DIR_RIGHT, "RDKP000")
        put(11, DIR_RIGHT, "RDKP001")

        pic[3][DIR_RIGHT] = _mirror(pic[3][DIR_LEFT])
        for i in range(1, MAX_ENEMIES + 1):
            if i in (6, 7):
                pic[i][DIR_RIGHT] = _mirror(pic[i][DIR_LEFT])
            elif i != 3:
                pic[i][DIR_LEFT] = _mirror(pic[i][DIR_RIGHT])

        koopa = self._koopa
        koopa[DIR_LEFT][K_GREEN][0] = a.sprite_idx("GRKOOPA000")
        koopa[DIR_LEFT][K_GREEN][1] = a.sprite_idx("GRKOOPA001")
        koopa[DIR_LEFT][K_RED][0] = a.sprite_idx("RDKOOPA000")
        koopa[DIR_LEFT][K_RED][1] = a.sprite_idx("RDKOOPA001")
        koopa[DIR_LEFT][K_BLACK][0] = _blacken_shell(a.sprite_idx("GRKOOPA000"))
        koopa[DIR_LEFT][K_BLACK][1] = _blacken_shell(a.sprite_idx("GRKOOPA001"))
        for c in (K_GREEN, K_RED, K_BLACK):
            for f in (0, 1):
                koopa[DIR_RIGHT][c][f] = _mirror(koopa[DIR_LEFT][c][f])

        # port-only Buzzy (SMB3 art, tools/extract_buzzy.py): every state is
        # the 20x14 sleeping-Koopa shell slot, drawn at (XPos, YPos); the
        # resting shells carry NO eye pixels (user 2026-10-23).
        buz = Path(__file__).resolve().parent.parent / "data" / "buzzy"
        for f in (0, 1):
            left = np.load(buz / f"buzzy_walk00{f}.npy")
            koopa[DIR_LEFT][K_BUZZY][f] = left
            koopa[DIR_RIGHT][K_BUZZY][f] = _mirror(left)
        for f, name in enumerate(("buzzy_shell000", "buzzy_shell002")):
            left = np.load(buz / f"{name}.npy")
            self._shell[K_BUZZY][DIR_LEFT][f] = left
            self._shell[K_BUZZY][DIR_RIGHT][f] = _mirror(left)

        # shell-only pictures: green/red reuse EnemyPictures 8..11 verbatim
        for c, base in ((K_GREEN, 8), (K_RED, 10)):
            for f in (0, 1):
                self._shell[c][DIR_LEFT][f] = pic[base + f][DIR_LEFT]
                self._shell[c][DIR_RIGHT][f] = pic[base + f][DIR_RIGHT]
        for f, name in enumerate(("GRKP000", "GRKP001")):
            idx = _blacken_shell(a.sprite_idx(name))
            self._shell[K_BLACK][DIR_RIGHT][f] = idx
            self._shell[K_BLACK][DIR_LEFT][f] = _mirror(idx)

        # Spinies: SubTp 0 keeps the original art, K_BLACK swaps only the red
        # shell ramp.  Red000/001 are stored left-facing (see above).
        for f in (0, 1):
            self._spiny[0][DIR_LEFT][f] = pic[6 + f][DIR_LEFT]
            self._spiny[0][DIR_RIGHT][f] = pic[6 + f][DIR_RIGHT]
            idx = _blacken_spiny(pic[6 + f][DIR_LEFT])
            self._spiny[K_BLACK][DIR_LEFT][f] = idx
            self._spiny[K_BLACK][DIR_RIGHT][f] = _mirror(idx)

        wings = load_wings(a.palette)
        for c in (0, K_BLACK):
            for f in (0, 1):
                base = self._spiny[c][DIR_LEFT][f]
                for w, wing in enumerate(wings):
                    self._fly_spiny[c][DIR_LEFT][f][w] = _compose_wing(
                        base, wing, FLY_SPEC["short"])
                    self._fly_spiny[c][DIR_RIGHT][f][w] = _compose_wing(
                        base, wing, FLY_SPEC["short"], right=True)

        for c, names in ((K_GREEN, ("GRKOOPA000", "GRKOOPA001")),
                         (K_RED, ("RDKOOPA000", "RDKOOPA001")),
                         (K_BLACK, ("GRKOOPA000", "GRKOOPA001"))):
            for f, name in enumerate(names):
                base = a.sprite_idx(name)
                if c == K_BLACK:
                    base = _blacken_shell(base)
                for w, wing in enumerate(wings):
                    self._fly[c][DIR_LEFT][f][w] = _compose_wing(
                        base, wing, FLY_SPEC["koopa"])
                    self._fly[c][DIR_RIGHT][f][w] = _compose_wing(
                        base, wing, FLY_SPEC["koopa"], right=True)

        # flying Buzzy: the 20x14 Buzzy body in the short-body slot, wing over
        for d in (DIR_LEFT, DIR_RIGHT):
            for f in (0, 1):
                base = self._koopa[DIR_LEFT][K_BUZZY][f]
                for w, wing in enumerate(wings):
                    self._fly_buzzy[d][f][w] = _compose_wing(
                        base, wing, FLY_SPEC["short"], right=(d == DIR_RIGHT))

        self._fireball = [a.sprite_idx(f"F00{i}") for i in range(4)]

        # flying Chibibo: the same shared wing, as a mirrored pair with the
        # body painted over the roots.  The body is always CHIBIBO000: walking
        # chibibos never alternate frames (their second index is the direction)
        # and CHIBIBO001 is the *flat* pose, which would read as a vanished
        # body mid-hop.
        for d in (DIR_LEFT, DIR_RIGHT):
            base = self._pic[1][DIR_LEFT]         # CHIBIBO000, left facing
            for w, wing in enumerate(wings):
                self._fly_chib[d][w] = _compose_wing(
                    base, wing, FLY_SPEC["chibibo"], right=(d == DIR_RIGHT))

    def _pic_surf(self, n: int, d: int) -> pygame.Surface:
        return self.figures.cache_surface(f"EN{n}:{d}", self._pic[n][d])

    def _koopa_surf(self, d: int, c: int, f: int) -> pygame.Surface:
        return self.figures.cache_surface(f"ENK{d}{c}{f}",
                                          self._koopa[d][c][f])

    def _shell_surf(self, c: int, f: int, d: int) -> pygame.Surface:
        return self.figures.cache_surface(f"ENS{c}{f}{d}",
                                          self._shell[c][d][f])

    def _spiny_surf(self, c: int, f: int, d: int) -> pygame.Surface:
        return self.figures.cache_surface(f"ENR{c}{f}{d}",
                                          self._spiny[c][d][f])

    def _fly_spiny_surf(self, c: int, d: int, f: int,
                        w: int) -> pygame.Surface:
        return self.figures.cache_surface(f"FLYS{c}{d}{f}{w}",
                                          self._fly_spiny[c][d][f][w])

    def _fly_surf(self, c: int, d: int, f: int, w: int) -> pygame.Surface:
        return self.figures.cache_surface(f"FLY{c}{d}{f}{w}",
                                          self._fly[c][d][f][w])

    def _fly_chib_surf(self, d: int, w: int) -> pygame.Surface:
        return self.figures.cache_surface(f"FLYC{d}{w}",
                                          self._fly_chib[d][w])

    def _fly_buzzy_surf(self, d: int, f: int, w: int) -> pygame.Surface:
        return self.figures.cache_surface(f"FLYB{d}{f}{w}",
                                          self._fly_buzzy[d][f][w])

    def _sprite_surf(self, name: str) -> pygame.Surface:
        """Plain asset sprite through the Figures palette cache."""
        return self.figures.sprite_surf(name)

    def _black_plant_surf(self, open_now: bool) -> pygame.Surface:
        """The black piranha: the red-green pair with its body ramp swapped."""
        name = "PPLANT000" if open_now else "PPLANT001"
        return self.figures.cache_surface(
            f"PLANTBLK:{name}",
            _blacken_plant(self.figures.assets.sprite_idx(name)))

    def _chib_black_surf(self, pose: int, d: int) -> pygame.Surface:
        """The ghost Chibibo's own body: the plain Chibibo art with its 14 body
        tones compressed into the game's shared 8-step black.

        ``pose`` 0 = walking (EnemyPictures slot 1), 1 = the stomped flat pose
        (slot 2); ``d`` is the direction, and the swap is index-level so it
        commutes with the mirrored copy in the table."""
        return self.figures.cache_surface(
            f"CHIBBLK:{pose}:{d}", _blacken_chibibo(self._pic[1 + pose][d]))

    def _shade_surf(self, d: int) -> pygame.Surface:
        """The blue shadow: the same Chibibo silhouette in the Dry Bones blue.

        Baked with the master palette -- its outline is palette 1, the slot
        BlinkPalette rewrites to a random yellow every frame."""
        return self.figures.cache_surface(
            f"CHIBSHADE:{d}", _shade_chibibo(self._pic[1][d]),
            pal=self.figures.base_pal)

    def _new_surf(self, name: str, mirror: bool = False,
                  fixed_palette: bool = False) -> pygame.Surface:
        """Port-only batch art (data/newenemies/idx), optionally flipped.

        ``fixed_palette`` bakes with the master palette instead of the live
        level palette.  The blue Dry Bones needs it: BlinkPalette rewrites
        palette 1 (its outline) to a random yellow every frame.
        """
        arr = _new_art(name)
        if mirror:
            arr = _mirror(arr)
        if fixed_palette:
            return self.figures.cache_surface(
                f"NEW{name}:{int(mirror)}:base", arr,
                pal=self.figures.base_pal)
        return self.figures.cache_surface(f"NEW{name}:{int(mirror)}", arr)

    @staticmethod
    def _draw_part_x(target: pygame.Surface, x: int, y: int, x1: int,
                     x2: int, surf: pygame.Surface) -> None:
        """The horizontal twin of DrawPart: blit only the columns x1..x2, so a
        sideways plant slides out of its pipe instead of floating."""
        w = surf.get_width()
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        if x2 < x1:
            return
        target.blit(surf, (x + x1, y),
                    area=pygame.Rect(x1, 0, x2 - x1 + 1, surf.get_height()))

    def _side_plant_surf(self, name: str, right: bool) -> pygame.Surface:
        """The 24x20 plant turned 90 degrees: head up becomes head right."""
        key = (name, right)
        surf = self._side_plant_cache.get(key)
        if surf is None:
            surf = pygame.transform.rotate(self._sprite_surf(name),
                                           -90 if right else 90)
            self._side_plant_cache[key] = surf
        return surf

    def _platform_surface(self, e: "EnemyRec") -> pygame.Surface:
        """One continuous board: end caps + the centre column repeated.

        The source segment is 10x8 with a black frame and a bevel on every
        side.  Repeating it unchanged makes a row of crates, so for n > 1 only
        the two end caps keep their vertical bevel; the body repeats the
        centre column.  Everything is doubled to the port's 20x14 tile scale
        (20x16).
        """
        n = max(1, e.plat_len)
        variant = 1 if e.sub_tp else 0
        key = (variant, n)
        surf = self._plat_surf.get(key)
        if surf is None:
            seg = _new_art(f"PLATFORM_seg{variant:02d}")
            h, w = seg.shape
            if n == 1:
                strip = seg
            else:
                strip = np.repeat(seg[:, w // 2][:, None], w * n, axis=1)
                strip[:, :2] = seg[:, :2]
                strip[:, -2:] = seg[:, -2:]
                strip[0, :] = seg[0, w // 2]
                strip[-1, :] = seg[-1, w // 2]
            big = np.repeat(np.repeat(strip, 2, axis=0), 2, axis=1)
            surf = self.figures.cache_surface(f"PLAT:{variant}:{n}", big)
            self._plat_surf[key] = surf
        return surf

    def _draw_dry_bones(self, target: pygame.Surface, e: "EnemyRec",
                        x: int, y: int, right: int) -> None:
        """The skeleton's three canvases: walker, collapsing pile, corpse."""
        if e.tp == TP_DRY_BONES:
            # The authored art faces LEFT (head on the left, like the green
            # Koopa source), so mirror it when walking right.  The XP skull
            # it replaced faced right and used the opposite flag.
            self._draw(target, x, y,
                       self._new_surf(f"SKULLB_walk{e.sub_tp:02d}",
                                      right, True))
        elif e.tp == TP_DRY_BONES_FALL:
            # 2026-10-24 collapse: scatter (45x10) -> half-gather -> gather ->
            # squat -> rise -> the standing walker.  The box (and its head bob)
            # comes from _dry_art_box, so no extra y offset here.
            stage = _dry_stage(e.counter)
            pose = DRY_FALL_POSE[stage]
            name = ("SKULLB_walk00" if stage == DRY_STAND_STAGE
                    else f"SKULLB_{pose}")
            self._draw(target, x, y,
                       self._new_surf(name, right, True))
        else:
            self._draw(target, x, y,
                       self._new_surf("SKULLB_dead", fixed_palette=True))

    def _draw_hammer_bro(self, target: pygame.Surface, e: "EnemyRec",
                         x: int, y: int, right: int) -> None:
        """The brother's 6-frame walk and his head-down corpse."""
        if e.tp == TP_HAMMER_BRO:
            f = (e.dir_counter // HBRO_ANIM_DIV) % 6
            self._draw(target, x, y,
                       self._new_surf(f"HAMMER_BRO_walk{f:02d}", not right))
        else:
            # knocked away: head down, like every other corpse in this port
            pose = "defeated" if e.sub_tp else "dead"
            self._upside_down(target, x, y,
                              self._new_surf(f"HAMMER_BRO_{pose}00",
                                             not right))

    def _draw_boo(self, target: pygame.Surface, e: "EnemyRec",
                  x: int, y: int, right: int) -> None:
        """The ghost and its overturned corpse, both baked with the master
        palette: BOO art uses palette 1 (the BlinkPalette star flash) for its
        outline, so a live bake makes it flash yellow every frame."""
        if e.tp == TP_BOO:
            self._draw(target, x, y,
                       self._new_surf("BOO_moving00" if e.status
                                      else "BOO_idle00", not right,
                                      fixed_palette=True))
        else:
            self._upside_down(target, x, y,
                              self._new_surf("BOO_idle00", not right,
                                             fixed_palette=True))

    def _draw_bomb(self, target: pygame.Surface, e: "EnemyRec",
                   x: int, y: int, right: int) -> None:
        """The cannon (never animated), the flying bomb, the corpse."""
        if e.tp == TP_BOMB_STAND:
            # 7/8 are the waterfall animator's entries; without a fixed bake
            # the barrel changes colour with every level's water cycle.
            self._draw(target, x, y,
                       self._new_surf("BOMB_STAND_stand00",
                                      fixed_palette=True))
        elif e.tp == TP_BOMB:
            f = int(e.dir_counter // 8) % 2
            self._draw(target, x, y,
                       self._new_surf(f"BOMB_fly{f:02d}", not right))
        else:
            # same class as the cloud: XP's spr_bulletbill_defeated is already
            # the upside-down body (tail flame on the right, highlight band on
            # the bottom row), and XP only adds image_yscale = -1 for the dead
            # state -- which that art does not need.  Keep the nose along the
            # flight direction with the dir_sign mirror, drop the second flip.
            self._draw(target, x, y,
                       self._new_surf("BOMB_dead00", not right))

    def _draw_thwomp(self, target: pygame.Surface, e: "EnemyRec",
                     x: int, y: int) -> None:
        """The slab and its corpse; the art faces front, so never mirrored."""
        if e.tp == TP_THWOMP:
            # 1 = star flash, 11 = waterfall cycle -> fixed master palette.
            self._draw(target, x, y,
                       self._new_surf("THWOMP_down00" if e.fly_phase == 1
                                      else "THWOMP_up00",
                                      fixed_palette=True))
        else:
            # It rides the pop up upright and only turns over
            # THWOMP_TUMBLE_FRAMES later, so the flip is visible instead of the
            # corpse starting upside down.
            surf = self._new_surf("THWOMP_down00", fixed_palette=True)
            if e.counter < THWOMP_TUMBLE_FRAMES:
                self._draw(target, x, y, surf)
            else:
                self._upside_down(target, x, y, surf)

    def _draw_lakitu(self, target: pygame.Surface, e: "EnemyRec",
                     x: int, y: int, right: int) -> None:
        """Lakitu's 12-frame walk and his corpse.

        XP's spr_lakitu_dead / spr_lakitu_defeated are *already* drawn head
        down -- they are the vertical flip of a main frame (77 of the 480
        pixels differ, and those are the 1 px alpha edges).  XP never touches
        image_yscale/image_angle for the cloud, so the old _upside_down() here
        flipped pre-flipped art and the corpse tumbled head up.  Only the
        left/right mirror is kept, which is what XP's image_xscale =
        entityDirection keeps doing too.
        """
        if e.tp == TP_LAKITU:
            f = (e.dir_counter // LAKITU_ANIM_DIV) % 12
            self._draw(target, x, y,
                       self._new_surf(f"LAKITU_main{f:02d}", not right))
        else:
            pose = "defeated" if e.sub_tp else "dead"
            self._draw(target, x, y,
                       self._new_surf(f"LAKITU_{pose}00", not right))

    def _draw_new(self, target: pygame.Surface, e: "EnemyRec") -> None:
        """Draw one of the 2026-10-23 batch enemies and report its canvas box.

        The art is wider/taller than the 20x14 box a walker owns, so the blit
        position and the background save rectangle both come from NEW_BOX.
        Dry Bones changes canvas mid-state (the pile stands up), hence
        _dry_art_box().
        """
        dx, dy, w, h = _new_box(e)
        x = e.x_pos + dx
        y = e.y_pos + dy
        tp = e.tp
        right = int(e.dir_sign > 0)
        if _is_platform(tp):
            self._draw(target, x, y, self._platform_surface(e))
        elif tp in (TP_DRY_BONES, TP_DRY_BONES_FALL, TP_DEAD_DRY_BONES):
            self._draw_dry_bones(target, e, x, y, right)
        elif tp in (TP_HAMMER_BRO, TP_DEAD_HAMMER_BRO):
            self._draw_hammer_bro(target, e, x, y, right)
        elif tp == TP_HAMMER:
            self._draw(target, x, y, self._new_surf("HAMMER_thrown00", not right))
        elif tp == TP_PLANT_FIRE:
            # SMB3 rotates one static frame by travel direction and never
            # animates the flame; the port's fireball art is a round blob, so
            # there is nothing to mirror either.
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf("FIRE000"))
        elif tp in (TP_BOO, TP_DEAD_BOO):
            self._draw_boo(target, e, x, y, right)
        elif tp in (TP_BOMB_STAND, TP_BOMB, TP_DEAD_BOMB):
            self._draw_bomb(target, e, x, y, right)
        elif tp in (TP_THWOMP, TP_DEAD_THWOMP):
            self._draw_thwomp(target, e, x, y)
        elif tp in (TP_LAKITU, TP_DEAD_LAKITU):
            self._draw_lakitu(target, e, x, y, right)
        elif tp == TP_SPINY_FALL:
            self._draw(target, x, y, self._new_surf("SPINY_FALL_fall00"))
        else:
            raise AssertionError(f"no art branch for tp {tp}")

    # ------------------------------------------------------------------ #
    # list management (ClearEnemies / StopEnemies / NewEnemy)
    # ------------------------------------------------------------------ #
    @property
    def num_enemies(self) -> int:
        return len(self.active)

    def clear_enemies(self) -> None:
        for e in self.pool:
            e.tp = TP_DEAD
        self.active.clear()
        self.cd_champ = 0
        self.cd_life = 0
        self.cd_flower = 0
        self.cd_star = 0
        self.cd_enemy = 0
        self.cd_hit = 0
        self.cd_lift = 0
        self.cd_stop_jump = 0

    def _in_protected_room(self, x_pos: int) -> bool:
        p = self.protect
        if p is None:
            return False
        x1, x2 = p
        return x_pos + W > x1 * W and x_pos < (x2 + 1) * W

    def _fish_zones(self) -> dict[int, list[tuple[int, int]]]:
        """Column -> spout cells, rebuilt whenever the world is re-bound."""
        key = id(self.world)
        if self._fish_cache is None or self._fish_cache[0] != key:
            cols: dict[int, list[tuple[int, int]]] = {}
            for x in range(self.options.XSize):
                cells = [(x, y) for y in range(self.world.rows)
                         if self.world.get(x, y) == CH_FLY_FISH]
                if cells:
                    cols[x] = cells
            self._fish_cache = (key, cols)
            self._fish_timers = {}
        return self._fish_cache[1]

    def _lakitu_switches(self) -> list[tuple[int, int]]:
        """Cells carrying CH_LAKITU; the marker is a switch, not a spawn."""
        key = id(self.world)
        if self._lakitu_cache is None or self._lakitu_cache[0] != key:
            cells = [(x, y) for x in range(self.options.XSize)
                     for y in range(self.world.rows)
                     if self.world.get(x, y) == CH_LAKITU]
            self._lakitu_cache = (key, cells)
            self._lakitu_timer = 0
        return self._lakitu_cache[1]

    def _tick_lakitu(self) -> None:
        """obj_lakitu_manager: once the player has passed a cloud switch, the
        Lakitu rides the camera and comes back LAKITU_PERIOD frames after it is
        killed."""
        cells = self._lakitu_switches()
        if not cells:
            return
        first_col = idiv(self.view_x, W)
        if not any(x <= first_col + 1 for x, _ in cells):
            return                                  # switch not reached yet
        if any(self.pool[i].tp == TP_LAKITU for i in self.active):
            return
        # the first cloud appears as soon as the switch is reached; after that
        # it is XP's 8 s respawn
        if self._lakitu_timer > 0:
            self._lakitu_timer += 1
            if self._lakitu_timer < LAKITU_PERIOD:
                return
        self._lakitu_timer = 1
        e = self._spawn_at(TP_LAKITU, self.view_x + LAKITU_EDGE,
                           self.view_y + LAKITU_TOP, xvel=1)
        if e is not None:
            e.dir_sign = 1
            e.fly_spd = LAKITU_TOP * FLY_X_DEN      # yPoint starts above the view
            e.fly_xrem = 0

    def _tick_fish(self) -> None:
        """One timer per spout that is currently on screen."""
        zones = self._fish_zones()
        if not zones:
            return
        first = idiv(self.view_x, W)
        for x in range(first, first + NH + 1):
            for cell in zones.get(x, ()):
                timer = self._fish_timers.get(cell, 0) + 1
                if timer >= FLY_FISH_PERIOD:
                    timer = 0
                    self._spawn_fish(cell[0], cell[1])
                self._fish_timers[cell] = timer

    def _spawn_fish(self, map_x: int, map_y: int) -> None:
        """XP's obj_cheep_cheep Create: it jumps towards whoever is standing
        on the water, and flatter + faster when they are far away."""
        x = map_x * W + W // 2
        direction = 1 if self.player_x1 >= x else -1
        near = abs(self.player_x1 - x) < FLY_FISH_NEAR
        lo, hi = ((FLY_FISH_XSPEED[0], FLY_FISH_XSPEED[1]) if near
                  else (FLY_FISH_XSPEED[2], FLY_FISH_XSPEED[3]))
        ylo, yhi = ((FLY_FISH_YSPEED[0], FLY_FISH_YSPEED[1]) if near
                    else (FLY_FISH_YSPEED[2], FLY_FISH_YSPEED[3]))
        before = len(self.active)
        self.new_enemy(TP_FLY_FISH, 0, map_x, map_y, 0, 0, 0)
        if len(self.active) == before:
            return                      # pool limit reached, nothing spawned
        e = self.pool[self.active[-1]]
        e.dir_sign = direction
        e.fly_spd = self.rng.randrange(lo, hi + 1)
        e.fly_vy = self.rng.randrange(ylo, yhi + 1)
        e.fly_xrem = 0
        e.fly_yrem = 0

    def _restore_marker(self, e: EnemyRec) -> None:
        """The `case Tp of ...` block shared by StopEnemies() and the
        forget branch of MoveEnemies(): put the map marker back."""
        w = self.world
        if e.tp == TP_CHIBIBO:
            w.set(e.map_x, e.map_y, CH_CHIBIBO)
        elif e.tp == TP_GHOST_CHIBIBO:
            w.set(e.map_x, e.map_y, CH_GHOST_CHIBIBO)
        elif e.tp == TP_VERT_FISH:
            row = (e.map_y if self.options.in_water_row(e.map_y)
                   else e.map_y - 2)
            w.set(e.map_x, row, CH_VERT_FISH)
        elif e.tp == TP_VERT_FIREBALL:
            w.set(e.map_x, e.map_y - 2, CH_VERT_FIREBALL)
        elif _is_plant(e.tp):
            # the sideways plant's marker is its own cell; the vertical ones
            # sit two rows above the pipe they live in
            row = e.map_y if e.tp == TP_SIDE_PLANT else e.map_y - 2
            w.set(e.map_x, row,
                  {TP_FIRE_PLANT: CH_FIRE_PLANT,
                   TP_FIRE_PLANT_AIM: CH_FIRE_PLANT_AIM,
                   TP_FAST_PLANT: CH_FAST_PLANT,
                   TP_SIDE_PLANT: CH_SIDE_PLANT,
                   TP_BLACK_PLANT: CH_BLACK_PLANT}.get(e.tp,
                                                     CH_PLANT0 + e.sub_tp))
        elif e.tp == TP_RED:
            w.set(e.map_x, e.map_y,
                  CH_SPINY_BLACK if e.sub_tp == K_BLACK else CH_RED)
        elif TP_KOOPA <= e.tp <= TP_RUNNING_KOOPA:
            w.set(e.map_x, e.map_y,
                  CH_BUZZY if e.sub_tp == K_BUZZY else CH_KOOPA0 + e.sub_tp)
        elif e.tp == TP_FLY_GREEN:
            w.set(e.map_x, e.map_y, CH_FLY_GREEN)
        elif e.tp == TP_FLY_RED:
            w.set(e.map_x, e.map_y,
                  CH_FLY_RED_H if e.sub_tp == 0 else CH_FLY_RED_V)
        elif e.tp == TP_FLY_BLACK:
            w.set(e.map_x, e.map_y, CH_FLY_BLACK)
        elif e.tp == TP_FLY_SPINY:
            w.set(e.map_x, e.map_y, CH_FLY_SPINY)
        elif e.tp == TP_FLY_SPINY_BLACK:
            w.set(e.map_x, e.map_y, CH_FLY_SPINY_BLACK)
        elif e.tp == TP_FLY_CHIBIBO:
            w.set(e.map_x, e.map_y, CH_FLY_CHIBIBO)
        elif e.tp == TP_FLY_BUZZY:
            w.set(e.map_x, e.map_y, CH_FLY_BUZZY)
        elif e.tp == TP_DRY_BONES or e.tp == TP_DRY_BONES_FALL:
            w.set(e.map_x, e.map_y, CH_DRY_BONES)
        elif e.tp == TP_HAMMER_BRO:
            w.set(e.map_x, e.map_y, CH_HAMMER_BRO)
        elif e.tp == TP_BOO:
            w.set(e.map_x, e.map_y, CH_BOO)
        elif e.tp == TP_BOMB_STAND:
            w.set(e.map_x, e.map_y, CH_BOMB_STAND)
        elif e.tp == TP_THWOMP:
            w.set(e.map_x, e.map_y, CH_THWOMP)
        # TP_LAKITU keeps its marker in the map (it is a switch, not a spawn),
        # and the runtime objects (hammer / bomb / falling Spiny / dead states)
        # have no marker at all.
        elif e.tp == TP_BLOCK_LIFT:
            w.set(e.map_x, e.map_y, CH_BLOCK_LIFT)
        elif e.tp == TP_DONUT:
            w.set(e.map_x, e.map_y, CH_DONUT)

    def stop_enemies(self) -> None:
        """StopEnemies: write every active enemy back into the map, then
        clear the list (used when entering a pipe)."""
        for idx in self.active:
            self._restore_marker(self.pool[idx])
        self.clear_enemies()

    def new_enemy(self, init_type: int, sub_type: int, init_x: int,
                  init_y: int, init_xvel: int, init_yvel: int,
                  init_delay: int) -> None:
        hk = self.hooks
        if self.turbo:
            init_xvel *= 2
            init_yvel *= 2
            init_delay = idiv(init_delay, 2)

        if init_type == TP_FIREBALL:
            n = 0
            for idx in self.active:
                if self.pool[idx].tp == TP_FIREBALL:
                    n += 1
            if n >= 2:
                return
            hk.start_music("fire")

        i = 1
        while i < len(self.pool) and self.pool[i].tp != TP_DEAD:
            i += 1
        if i == len(self.pool):
            # every existing slot is busy
            if POOL_LIMIT is not None and len(self.pool) - 1 >= POOL_LIMIT:
                return              # NewEnemy() exits; the marker stays eaten
            self.pool.append(EnemyRec())

        e = self.pool[i]
        e.tp = init_type
        e.sub_tp = sub_type
        e.map_x = init_x
        e.map_y = init_y
        e.x_pos = init_x * W
        e.y_pos = init_y * H
        e.x_vel = init_xvel
        e.y_vel = init_yvel
        e.move_delay = init_delay
        e.delay_counter = 0
        e.dir_counter = 0
        e.status = GROUNDED
        e.seen = 0
        e.backgr_addr = [NO_BACKGR] * (MAX_PAGE + 1)
        e.counter = 0
        if _is_plant(e.tp):
            e.x_pos += 8
            e.status = 0
            if e.tp == TP_SIDE_PLANT:
                # starts inside the pipe; dir_sign is the side it reaches for.
                # x_vel stays 0: the mover below steps x_pos itself, and
                # leaving it set would make the walker path move it too.
                e.x_vel = 0
                e.x_pos = e.map_x * W
                e.dir_sign = 1 if init_xvel >= 0 else -1
                e.status = 0
                e.counter = 0
                e.move_delay = 2
            elif e.tp == TP_FAST_PLANT:
                # XP starts hidden with piranhaDirection = 1 and a 2 s fuse.
                # move_delay -1 puts the mover on every frame: XP's 1.25 s is
                # a wall-clock window and the 80-frame flip below counts frames.
                e.dir_sign = 1
                e.fly_yrem = 0
                e.counter = FAST_PLANT_FIRST
                e.move_delay = -1
        elif _flying(e.tp):
            e.dir_sign = 1 if init_xvel >= 0 else -1
            e.x_vel = 0
            e.fly_xrem = 0
            e.fly_yrem = 0
            e.fly_spd = 0
            e.fly_phase = 0
            if e.tp == TP_FLY_RED:
                e.fly_vdir = -1
            else:
                e.fly_vy = FLY_GREEN_JUMP
        elif e.tp == TP_FIREBALL:
            # fireballs start at the player's edge, not at MapX*W
            e.x_pos = self.player_x2 if e.x_vel > 0 else self.player_x1
        elif _uses_subpixel(e.tp):
            # the batch-2 movers integrate fly_* themselves; XVel stays 0 so
            # the shared interpolation in MoveEnemies cannot fight them.
            e.dir_sign = -1 if init_xvel < 0 else 1
            e.x_vel = 0
            e.fly_xrem = 0
            e.fly_yrem = 0
            e.fly_spd = 0
            e.fly_vy = 0
            e.fly_phase = 1 if e.tp == TP_THWOMP else 0
            if e.tp == TP_SPINY_FALL:
                e.fly_vy = SPINYF_POP
        e.last_x_pos = e.x_pos
        e.last_y_pos = e.y_pos
        self.active.append(i)
        return i

    # ------------------------------------------------------------------ #
    # Kill / ShowStar / ShowFire
    # ------------------------------------------------------------------ #
    def _release_shade(self, e: EnemyRec) -> None:
        """XP's ``has_undead``: every death of the ghost Chibibo leaves its blue
        shadow behind.  Spawned where the body fell; the shade has no map marker
        of its own, so it is gone for good once it is destroyed or forgotten."""
        self._spawn_at(TP_GHOST_SHADE, e.x_pos, e.y_pos)

    def _hidden(self, e: EnemyRec) -> bool:
        """Is this thing drawn as nothing at all right now?

        One predicate for the two things that hide themselves -- a plant back in
        its pipe and a lava fireball dropped back into the floor.  The hit box,
        the damage test, the kill test and the Vision glow all read this, so
        they can never disagree about what "invisible" means.
        """
        return _plant_retracted(e) or _fireball_underground(e, self.world.rows)

    def _kill_walker(self, e: EnemyRec, dead: int, score: int = 100) -> None:
        """The one corpse every ground walker gets: sideways drift, -4 up.

        This is the source's ``Kill`` body (ENEMIES.PAS:179/188/197); only the
        dead type differs between reds, greens and sleeping/waking/running
        Koopas.  The ghost Chibibo and the winged forms that land as a shell route
        through here too, and so do the two port-only types that die like a walker
        (Dry Bones, 200, and the dropped Spiny) -- the source has no corpse for
        them, this is the port's choice.  See ``_corpse_dir`` for why the drift is
        the tile half.
        """
        e.tp = dead
        e.x_vel = _corpse_dir(e)
        e.y_vel = -4
        e.move_delay = 0
        e.delay_counter = 0
        self.hooks.add_score(score)

    def _kill_hammer_bro(self, e: EnemyRec) -> None:
        """XP's *defeated* pose (``sub_tp`` 1) with his lightest upward pop; the
        corpse is carried by ``fly_vy``, not by ``y_vel``."""
        e.tp = TP_DEAD_HAMMER_BRO
        e.sub_tp = 1
        e.x_vel = _corpse_dir(e)
        e.fly_vy = HBRO_DEFEATED_POP
        e.fly_yrem = 0
        e.move_delay = 0
        e.delay_counter = 0
        self.hooks.add_score(200)

    def _kill_lakitu(self, e: EnemyRec) -> None:
        """The same defeated-pose pop as the Hammer Brother, minus the two
        resets: this branch never zeroed ``move_delay`` / ``delay_counter``, and
        merging them would change how the corpse interpolates on its first
        frames, so the difference is kept as it is."""
        e.tp = TP_DEAD_LAKITU
        e.sub_tp = 1
        e.x_vel = _corpse_dir(e)
        e.fly_vy = HBRO_DEFEATED_POP
        e.fly_yrem = 0
        self.hooks.add_score(400)

    def _kill_bomb(self, e: EnemyRec) -> None:
        """The Bullet Bill corpse: XP's lightest pop (``BOMB_DEAD_POP``) with the
        nose still mirrored by ``dir_sign``."""
        e.tp = TP_DEAD_BOMB
        e.x_vel = _corpse_dir(e)
        e.fly_vy = BOMB_DEAD_POP
        e.fly_yrem = 0
        self.hooks.add_score(200)

    def _kill_thwomp(self, e: EnemyRec) -> None:
        """The slab is the one thing that can be broken (user 2026-10-23).

        XP cannot kill a Thwomp at all, so there is no defeated art to copy; it
        goes the way every other corpse in this port goes -- pops up, turns head
        down, falls out of the level.  XP's lightest defeated pop (the Bullet
        Bill's -2 px/f) because a 30x32 block must not leap like a Boo does.
        """
        e.tp = TP_DEAD_THWOMP
        e.x_vel = _corpse_dir(e)
        e.y_vel = 0
        e.status = 0
        e.counter = 0                   # counts the upright phase
        e.fly_vy = BOMB_DEAD_POP
        e.fly_yrem = 0
        e.delay_counter = 0
        self.hooks.add_score(200)

    def _kill_boo(self, e: EnemyRec) -> None:
        """The Boo drops out of the level like the jumping fish (the user's rule,
        2026-10-25): no sideways drift and no upward pop, it just turns over."""
        e.tp = TP_DEAD_BOO
        e.status = 0
        e.x_vel = 0
        e.y_vel = 0
        e.fly_vy = 0
        e.fly_yrem = 0
        self.hooks.add_score(200)

    def _kill_plant(self, e: EnemyRec) -> None:
        """All six pipe plants die into the one ``TP_DEAD_VERT_PLANT`` clock,
        whose frames are driven from the art branch (it keeps advancing
        ``status`` while drawing)."""
        e.tp = TP_DEAD_VERT_PLANT
        e.delay_counter = 0
        e.y_vel = 0
        self.hooks.add_score(100)

    def _kill_fish(self, e: EnemyRec) -> None:
        """Both fish drop straight -- the one exception the source encodes
        (ENEMIES.PAS:206 ``XVel := 0``), which is why the Boo was later moved to
        this class instead of the walkers'.

        The jumping fish just turns over; the flying one stops, pops up and
        flips (XP), so it also loses its patrol speed and subpixel remainders.
        """
        e.x_vel = 0
        e.y_vel = 0
        e.delay_counter = 0
        if e.tp == TP_FLY_FISH:
            e.tp = TP_DEAD_FLY_FISH
            e.fly_spd = 0
            e.fly_vy = FLY_FISH_DEAD_POP
            e.fly_xrem = 0
            e.fly_yrem = 0
            e.move_delay = 0
        else:
            e.tp = TP_DEAD_VERT_FISH
            e.move_delay = 2
            e.status = FALLING
        self.hooks.add_score(100)

    def _kill_fly_spiny(self, e: EnemyRec) -> None:
        """A flying Spiny comes down as the red ground corpse."""
        if e.x_vel == 0:
            e.x_vel = e.dir_sign
        e.sub_tp = K_BLACK if e.tp == TP_FLY_SPINY_BLACK else 0
        self._kill_walker(e, TP_DEAD_RED)

    def _kill_fly_koopa(self, e: EnemyRec) -> None:
        """A flying Koopa (or Buzzy) becomes the shell of its own colour."""
        if e.x_vel == 0:
            e.x_vel = e.dir_sign
        if e.tp == TP_FLY_GREEN:
            e.sub_tp = K_GREEN
        elif e.tp == TP_FLY_RED:
            e.sub_tp = K_RED
        elif e.tp == TP_FLY_BUZZY:
            e.sub_tp = K_BUZZY
        else:
            e.sub_tp = K_BLACK
        self._kill_walker(e, TP_DEAD_KOOPA)

    def _kill_chibibo(self, e: EnemyRec, ghost: bool) -> None:
        """The Chibibo family corpse: plain, flying, and the ghost's own, which
        also leaves XP's blue shadow where the body fell."""
        self._kill_walker(e, TP_GHOST_DEAD if ghost else TP_DEAD_CHIBIBO)
        if ghost:
            self._release_shade(e)

    def _kill(self, i: int) -> None:
        """Kill(i): the one dispatch table -- every corpse rule lives in its own
        ``_kill_*`` family method, so changing one enemy's death touches one
        place.  The order of the chain is the evaluation order; the branches are
        disjoint, so it is only the plant fallback that has to stay last."""
        e = self.pool[i]
        if self._hidden(e):
            # the user's rule (2026-10-25): a plant that is back in its pipe is
            # not there at all -- nothing can kill it until it comes out again
            return
        if e.tp in (TP_GHOST_CHIBIBO, TP_CHIBIBO, TP_FLY_CHIBIBO):
            self._kill_chibibo(e, e.tp == TP_GHOST_CHIBIBO)
        elif e.tp == TP_RED:
            self._kill_walker(e, TP_DEAD_RED)
        elif TP_KOOPA <= e.tp <= TP_RUNNING_KOOPA:
            self._kill_walker(e, TP_DEAD_KOOPA)
        elif e.tp in (TP_FLY_SPINY, TP_FLY_SPINY_BLACK):
            self._kill_fly_spiny(e)
        elif e.tp in (TP_FLY_GREEN, TP_FLY_RED, TP_FLY_BLACK, TP_FLY_BUZZY):
            self._kill_fly_koopa(e)
        elif e.tp in (TP_FLY_FISH, TP_VERT_FISH):
            self._kill_fish(e)
        elif e.tp in (TP_DRY_BONES, TP_DRY_BONES_FALL):
            # only the star and a shell get here (the user's rule); a stomp
            # collapses the pile instead and it revives.  The dead states get
            # their gravity from `if XPos mod 6 = 0`, so the corpse must drift
            # sideways like the others or it never falls.
            self._kill_walker(e, TP_DEAD_DRY_BONES, 200)
        elif e.tp == TP_HAMMER_BRO:
            self._kill_hammer_bro(e)
        elif e.tp == TP_BOO:
            self._kill_boo(e)
        elif e.tp == TP_BOMB:
            self._kill_bomb(e)
        elif e.tp == TP_LAKITU:
            self._kill_lakitu(e)
        elif e.tp == TP_THWOMP:
            self._kill_thwomp(e)
        elif e.tp == TP_SPINY_FALL:
            # the dropped Spiny dies like a ground one; no new state needed
            e.sub_tp = 0
            self._kill_walker(e, TP_DEAD_RED)
        elif _is_plant(e.tp):
            self._kill_plant(e)

    def _break_stationary_shell(self, i: int) -> None:
        """Rule 3: a moving black shell destroys a resting black shell.
        Kept separate from _kill() so a shell knocked up earlier in the same
        frame is not hit twice."""
        e = self.pool[i]
        e.tp = TP_DYING_KOOPA
        e.x_vel = 0
        e.y_vel = 0
        e.delay_counter = -(MAX_PAGE + 1)
        self.hooks.add_score(100)

    def _show_star(self, x: int, y: int) -> None:
        self.hooks.beep(100)
        if (x + W > self.view_x) and (x < self.view_x + SCREEN_WIDTH):
            self.hooks.new_temp_obj(TP_HIT, x, y, 0, 0, W, H)

    def _show_fire(self, x: int, y: int) -> None:
        self.hooks.beep(50)
        x -= 4
        y -= 4
        if (x + W > self.view_x) and (x < self.view_x + SCREEN_WIDTH):
            self.hooks.new_temp_obj(TP_FIRE, x, y, 0, 0, W, H)

    # ------------------------------------------------------------------ #
    # drawing (ShowEnemies / HideEnemies)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hit_box(e: EnemyRec) -> tuple[int, int, int, int]:
        """The rectangle the player is tested against.

        The original's enemies all live in one 20x14 tile box, so a taller
        sprite simply has dead air above its head.  The batch-2 art is much
        larger than that box (a 20x31 Hammer Bro, a 24-wide bomb, an 11-wide
        hammer), so for those the drawn canvas *is* the hit rectangle --
        otherwise a hammer hurts from 9 px away and a Bullet Bill's nose passes
        straight through Mario.
        """
        if _is_platform(e.tp):
            # Only the top strip is solid; the 16 px art hangs below it.
            # A falling board must still catch the rider when its per-frame
            # step exceeds the player's 2 px collision projection, so extend
            # the box upward by this frame's drop.
            drop = max(0, e.y_vel)
            return (e.x_pos, e.y_pos - drop,
                    max(1, e.plat_len) * PLATFORM_SEG_W, 8 + drop)
        if _is_plant(e.tp):
            # Only the part that is out of the pipe exists (the user's rule).
            # A vertical plant's box runs from its top down to the pipe mouth;
            # a sideways one runs from the mouth outward.  Fully retracted, the
            # box has no width/height, so nothing can overlap it at all.
            out = _plant_visible(e)
            if e.tp == TP_SIDE_PLANT:
                x = e.map_x * W + W if e.dir_sign >= 0 else e.x_pos
                return x, e.map_y * H + H // 2 - 12, out, 24
            return e.x_pos, e.y_pos, W, out
        if _new_mover(e.tp):
            dx, dy, w, h = NEW_BOX[e.tp]
            return e.x_pos + dx, e.y_pos + dy, w, h
        return e.x_pos, e.y_pos, W, H

    def _fly_facing(self, e: EnemyRec) -> int:
        if e.tp == TP_FLY_RED and e.sub_tp == 1:
            return int(self.player_x1 > e.x_pos)
        return int(e.dir_sign > 0)

    def _draw_order(self) -> list[int]:
        """The order ShowEnemies() paints in, which HideEnemies() must undo in
        reverse because the VGA save stack is LIFO.

        The active list is in spawn order, so a bomb fired by a stand that
        already exists would be painted *over* its muzzle.  The bomb is meant
        to slide out from behind the stand, so the stands go last (stable
        sort: everything else keeps its relative order)."""
        return sorted(self.active,
                      key=lambda i: int(self.pool[i].tp == TP_BOMB_STAND))

    def show_enemies(self, screen) -> None:
        """Draw every active enemy onto screen.page, saving the background
        each one covers into the VGA stack.  Mirrors ShowEnemies(): the
        visibility test, the per-type save rectangle and then the big case."""
        page = screen.current
        target = screen.page
        view_x = self.view_x
        for idx in self._draw_order():
            e = self.pool[idx]
            fly = _flying(e.tp)
            if fly:
                x1, x2 = e.x_pos, e.x_pos + FLY_W
            elif _new_mover(e.tp):
                dx, _, w, _ = _new_box(e)
                x1, x2 = e.x_pos + dx, e.x_pos + dx + w
            else:
                x1, x2 = e.x_pos, e.x_pos + W
            if ((x2 < view_x)
                    or (x1 > view_x + SCREEN_WIDTH)
                    or (e.y_pos + 2 * H <= self.view_y)
                    or (e.y_pos >= self.view_y + SCREEN_HEIGHT)):
                e.backgr_addr[page] = NO_BACKGR
                continue

            if e.tp in (TP_FIREBALL, TP_DYING_FIREBALL):
                e.backgr_addr[page] = screen.push_backgr(e.x_pos, e.y_pos,
                                                         W, H // 2)
            elif e.tp in PLANT_TPS or e.tp == TP_DEAD_VERT_PLANT:
                e.backgr_addr[page] = screen.push_backgr(e.x_pos, e.y_pos,
                                                         24, 20)
            elif TP_KOOPA <= e.tp <= TP_DEAD_KOOPA:
                e.backgr_addr[page] = screen.push_backgr(e.x_pos,
                                                         e.y_pos - 10, 24, 24)
            elif fly:
                e.backgr_addr[page] = screen.push_backgr(
                    e.x_pos, e.y_pos + FLY_DRAW_DY, FLY_W, FLY_H)
            elif _new_mover(e.tp):
                dx, dy, w, h = _new_box(e)
                e.backgr_addr[page] = screen.push_backgr(e.x_pos + dx,
                                                         e.y_pos + dy, w, h)
            else:
                e.backgr_addr[page] = screen.push_backgr(e.x_pos, e.y_pos,
                                                         W + 4, H)
            self._draw_enemy(target, e)

    def hide_enemies(self, screen) -> None:
        """HideEnemies: restore what show_enemies() saved, in reverse order
        (the VGA stack is LIFO, so this must undo _draw_order(), not the
        active list itself)."""
        page = screen.current
        for idx in reversed(self._draw_order()):
            e = self.pool[idx]
            if e.backgr_addr[page] != NO_BACKGR:
                screen.pop_backgr(e.backgr_addr[page])

    # -- blit helpers (DrawImage / UpSideDown / DrawPart) ---------------- #
    @staticmethod
    def _draw(target: pygame.Surface, x: int, y: int,
              surf: pygame.Surface) -> None:
        target.blit(surf, (x, y))

    @staticmethod
    def _upside_down(target: pygame.Surface, x: int, y: int,
                     surf: pygame.Surface) -> None:
        target.blit(pygame.transform.flip(surf, False, True), (x, y))

    @staticmethod
    def _draw_part(target: pygame.Surface, x: int, y: int, y1: int, y2: int,
                   surf: pygame.Surface) -> None:
        """DrawPart(XPos, YPos, W, H, Y1, Y2, BitMap): blit only the bitmap
        rows Y1..Y2 (inclusive) at (XPos, YPos+row).  Rows outside the bitmap
        are simply not drawn — the original's asm skips them with
        `cmp dx, Y1; jl / cmp dx, Y2; jg`."""
        h = surf.get_height()
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if y2 < y1:
            return
        target.blit(surf, (x, y + y1),
                    area=pygame.Rect(0, y1, surf.get_width(), y2 - y1 + 1))

    def _draw_enemy(self, target: pygame.Surface, e: EnemyRec) -> None:
        tp = e.tp
        hk = self.hooks
        if tp == TP_CHIBIBO:
            self._draw(target, e.x_pos, e.y_pos,
                       self._pic_surf(1 + 3 * e.sub_tp,
                                      int(e.dir_counter % 32 < 16)))
        elif tp == TP_FLAT_CHIBIBO:
            self._draw(target, e.x_pos, e.y_pos,
                       self._pic_surf(2 + 3 * e.sub_tp,
                                      int(e.dir_counter % 32 < 16)))
        elif tp == TP_DEAD_CHIBIBO:
            self._upside_down(target, e.x_pos, e.y_pos,
                              self._pic_surf(1, DIR_LEFT))
        elif tp == TP_GHOST_CHIBIBO:
            self._draw(target, e.x_pos, e.y_pos,
                       self._chib_black_surf(0, int(e.dir_counter % 32 < 16)))
        elif tp == TP_GHOST_FLAT:
            self._draw(target, e.x_pos, e.y_pos,
                       self._chib_black_surf(1, int(e.dir_counter % 32 < 16)))
        elif tp == TP_GHOST_DEAD:
            self._upside_down(target, e.x_pos, e.y_pos,
                              self._chib_black_surf(0, DIR_LEFT))
        elif tp == TP_GHOST_SHADE:
            # the Boo mirror: the Chibibo art faces right, so it is flipped
            # when the shade hangs to the player's left
            self._draw(target, e.x_pos, e.y_pos,
                       self._shade_surf(int(e.dir_sign > 0)))
        elif tp == TP_RISING_CHAMP:
            if e.y_pos != e.map_y * H:
                name = "CHAMP000" if e.sub_tp == 0 else "POISON000"
                self._draw_part(target, e.x_pos, e.y_pos, 0,
                                H - e.y_pos % H - 1, self._sprite_surf(name))
        elif tp == TP_CHAMP:
            name = "CHAMP000" if e.sub_tp == 0 else "POISON000"
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf(name))
        elif tp == TP_RISING_LIFE:
            if e.y_pos != e.map_y * H:
                self._draw_part(target, e.x_pos, e.y_pos, 0,
                                H - e.y_pos % H - 1, self._sprite_surf("LIFE000"))
        elif tp == TP_LIFE:
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf("LIFE000"))
        elif tp == TP_RISING_FLOWER:
            if e.y_pos != e.map_y * H:
                self._draw_part(target, e.x_pos, e.y_pos, 0,
                                H - e.y_pos % H - 1,
                                self._sprite_surf("FLOWER000"))
        elif tp == TP_FLOWER:
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf("FLOWER000"))
        elif tp == TP_RISING_STAR:
            if e.y_pos != e.map_y * H:
                self._draw_part(target, e.x_pos, e.y_pos, 0,
                                H - e.y_pos % H - 1, self._sprite_surf("STAR000"))
        elif tp == TP_STAR:
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf("STAR000"))
        elif tp == TP_FIREBALL:
            name = "FIRE000" if e.x_pos % 4 < 2 else "FIRE001"
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf(name))
        elif tp == TP_VERT_FISH:
            if self._in_water(e):
                self._draw(target, e.x_pos, e.y_pos,
                           self._pic_surf(3, int(e.dir_sign > 0)))
            elif (e.y_vel != 0) or (e.y_pos < self.world.rows * H - H):
                self._draw(target, e.x_pos, e.y_pos,
                           self._pic_surf(3, int(self.player_x1 > e.x_pos)))
        elif tp == TP_DEAD_VERT_FISH:
            if (e.y_pos < self.world.rows * H - H) or (e.y_vel != 0):
                self._upside_down(target, e.x_pos, e.y_pos,
                                  self._pic_surf(3,
                                                 int(self.player_x1 <= e.x_pos)))
        elif tp == TP_FLY_FISH:
            # same sprite as the jumping fish, facing where it is heading
            self._draw(target, e.x_pos, e.y_pos,
                       self._pic_surf(3, int(e.dir_sign > 0)))
        elif tp == TP_DEAD_FLY_FISH:
            self._upside_down(target, e.x_pos, e.y_pos,
                              self._pic_surf(3, int(e.dir_sign > 0)))
        elif tp == TP_VERT_FIREBALL:
            if abs(e.delay_counter - e.move_delay) <= 1:
                f = self.rng.randrange(4)
                self._draw(target, e.x_pos, e.y_pos,
                           self.figures.cache_surface(f"ENF{f}",
                                                      self._fireball[f]))
                hk.new_glitter(e.x_pos + self.rng.randrange(W),
                               e.y_pos + self.rng.randrange(H),
                               57 + self.rng.randrange(7),
                               14 + self.rng.randrange(20))
                hk.new_star(e.x_pos + self.rng.randrange(W),
                            e.y_pos + self.rng.randrange(H),
                            57 + self.rng.randrange(7),
                            14 + self.rng.randrange(20))
        elif e.tp == TP_SIDE_PLANT:
            # the sideways plant is the red-green pair turned 90 degrees
            open_now = (self.time_counter % PLANT_MOUTH_DIV
                        < PLANT_MOUTH_OPEN)
            name = "PPLANT000" if open_now else "PPLANT001"
            right = e.dir_sign >= 0
            surf = self._side_plant_surf(name, right)
            # the columns still inside the pipe cell stay hidden
            if right:
                x1, x2 = e.map_x * W + W - e.x_pos, surf.get_width() - 1
            else:
                x1, x2 = 0, e.map_x * W - 1 - e.x_pos
            self._draw_part_x(target, e.x_pos, e.map_y * H + H // 2 - 12,
                              x1, x2, surf)
        elif _is_plant(e.tp):
            # PPLANT000/001 are the red-green pair, 002/003 the plain green one
            open_now = (self.time_counter % PLANT_MOUTH_DIV
                        < PLANT_MOUTH_OPEN)
            if e.tp == TP_BLACK_PLANT:
                surf = self._black_plant_surf(open_now)
            elif _is_fire_plant(e.tp) or e.tp == TP_FAST_PLANT or e.sub_tp == 2:
                surf = self._sprite_surf("PPLANT000" if open_now else "PPLANT001")
            else:
                surf = self._sprite_surf("PPLANT002" if open_now else "PPLANT003")
            self._draw_part(target, e.x_pos, e.y_pos, 0,
                            e.map_y * H - e.y_pos - 1, surf)
        elif tp == TP_DEAD_VERT_PLANT:
            e.delay_counter = 0
            e.move_delay = 0
            e.y_vel = 0
            e.status += 1
            if e.status < 12:
                self._draw(target, e.x_pos, e.y_pos, self._sprite_surf("HIT000"))
            elif e.status > 14:
                e.tp = TP_DYING
        elif tp == TP_RED:
            self._draw(target, e.x_pos, e.y_pos,
                       self._spiny_surf(e.sub_tp,
                                        int(e.dir_counter % 16 <= 8),
                                        int(e.x_vel > 0)))
        elif tp == TP_DEAD_RED:
            self._upside_down(target, e.x_pos, e.y_pos,
                              self._spiny_surf(e.sub_tp,
                                               int(e.dir_counter % 16 <= 8),
                                               int(e.x_vel > 0)))
        elif tp == TP_KOOPA:
            # every Buzzy state (walking included) sits in the 20x14
            # sleeping-shell slot and rests on the ground at YPos+H
            dy = 0 if e.sub_tp == K_BUZZY else -10
            self._draw(target, e.x_pos, e.y_pos + dy,
                       self._koopa_surf(int(e.x_vel > 0), e.sub_tp,
                                        int(e.dir_counter % 16 <= 8)))
        elif tp in (TP_WAKING_KOOPA, TP_RUNNING_KOOPA):
            self._draw(target, e.x_pos, e.y_pos,
                       self._shell_surf(e.sub_tp,
                                        1 - int(e.dir_counter % 16 <= 8),
                                        int(e.dir_counter % 32 <= 16)))
        elif tp == TP_SLEEPING_KOOPA:
            self._draw(target, e.x_pos, e.y_pos,
                       self._shell_surf(e.sub_tp, 0, DIR_LEFT))
        elif tp == TP_DEAD_KOOPA:
            self._upside_down(target, e.x_pos, e.y_pos,
                              self._shell_surf(e.sub_tp, 0,
                                               int(e.dir_counter % 16 <= 8)))
        elif tp in (TP_FLY_GREEN, TP_FLY_RED, TP_FLY_BLACK):
            if tp == TP_FLY_GREEN:
                color = K_GREEN
            elif tp == TP_FLY_RED:
                color = K_RED
            else:
                color = K_BLACK
            d = self._fly_facing(e)
            self._draw(target, e.x_pos, e.y_pos + FLY_DRAW_DY,
                       self._fly_surf(color, d,
                                      int(e.dir_counter % 16 <= 8),
                                      int((e.dir_counter // FLY_FLAP_DIV) % 2)))
        elif tp in (TP_FLY_SPINY, TP_FLY_SPINY_BLACK):
            d = self._fly_facing(e)
            self._draw(target, e.x_pos, e.y_pos + FLY_DRAW_DY,
                       self._fly_spiny_surf(
                           K_BLACK if tp == TP_FLY_SPINY_BLACK else 0, d,
                           int(e.dir_counter % 16 <= 8),
                           int((e.dir_counter // FLY_FLAP_DIV) % 2)))
        elif tp == TP_FLY_CHIBIBO:
            # fixed body (CHIBIBO000), wings flap independently
            self._draw(target, e.x_pos, e.y_pos + FLY_DRAW_DY,
                       self._fly_chib_surf(
                           int(e.dir_sign > 0),
                           int((e.dir_counter // FLY_FLAP_DIV) % 2)))
        elif tp == TP_FLY_BUZZY:
            # same body animation as the walking Buzzy, plus the shared wing
            d = self._fly_facing(e)
            self._draw(target, e.x_pos, e.y_pos + FLY_DRAW_DY,
                       self._fly_buzzy_surf(
                           d, int(e.dir_counter % 16 <= 8),
                           int((e.dir_counter // FLY_FLAP_DIV) % 2)))
        elif _new_mover(tp):
            self._draw_new(target, e)
        elif tp == TP_BLOCK_LIFT:
            self._draw(target, e.x_pos, e.y_pos, self._sprite_surf("LIFT1000"))
        elif tp == TP_DONUT:
            if e.status == 0:
                self._draw(target, e.x_pos, e.y_pos,
                           self._sprite_surf("DONUT000"))
                if e.y_vel == 0:
                    e.counter = 0
            else:
                self._draw(target, e.x_pos, e.y_pos,
                           self._sprite_surf("DONUT001"))
                e.status -= 1
            if e.y_vel > 0 and e.counter % 24 == 0:
                e.y_vel += 1
            e.counter += 1

    # ------------------------------------------------------------------ #
    # collision / movement helpers (Check)
    # ------------------------------------------------------------------ #
    def _player_mode(self) -> int:
        if self.data is not None:
            return int(self.data.Mode[self.player])
        return MD_SMALL

    # ------------------------------------------------------------------ #
    # port-only flying Koopas
    # ------------------------------------------------------------------ #
    def _move_fly_green(self, e: EnemyRec) -> None:
        """Green flyer: a continuous hop, scaled from Mario XP.

        XP's green paratroopa walks 0.5 px/f and hops with impulse -4 and
        gravity 0.2 (38 px apex, 39 frames).  This port keeps the same ratios
        to its own Mario (walk 1, jump 60): 0.3125 px/f, impulse -2.5 px/f,
        gravity 0.1171875 px/f^2 -> 25 px apex."""
        world = self.world
        side = (W - 1) if e.dir_sign > 0 else 0
        row = idiv(e.y_pos + HSAFE + H // 2, H) - SAFE
        if world.get(idiv(e.x_pos + side + e.dir_sign, W), row) in CAN_HOLD_YOU:
            e.dir_sign = -e.dir_sign
            e.fly_xrem = 0
        else:
            e.fly_xrem += FLY_GREEN_SPEED * e.dir_sign
            dx = idiv(e.fly_xrem, FLY_X_DEN)
            e.x_pos += dx
            e.fly_xrem -= dx * FLY_X_DEN

        e.fly_vy += FLY_GRAVITY
        e.fly_yrem += e.fly_vy
        dy = idiv(e.fly_yrem, FLY_Y_DEN)
        e.fly_yrem -= dy * FLY_Y_DEN
        e.y_vel = dy
        if dy > 0:
            below = idiv(e.y_pos + 1 + H + dy + HSAFE, H) - SAFE
            x1 = idiv(e.x_pos, W)
            x2 = idiv(e.x_pos + W - 1, W)
            solid = CAN_HOLD_YOU | CAN_STAND_ON
            if (world.get(x1, below) in solid
                    or world.get(x2, below) in solid):
                e.y_pos = (idiv(e.y_pos + dy + 1 + HSAFE, H) - SAFE) * H
                e.y_vel = 0
                e.fly_vy = FLY_GREEN_JUMP
                e.fly_yrem = 0
        elif dy < 0:
            above = idiv(e.y_pos - 10, H)
            if world.get(idiv(e.x_pos + W // 2, W), above) in CAN_HOLD_YOU:
                e.y_vel = 1
                e.fly_vy = 0
                e.fly_yrem = 0

    def _move_fly_red(self, e: EnemyRec) -> None:
        """Red flyer: the XP ``obj_paratroopa_path`` speed profile -- speed
        ramps 0 -> vmax -> 0 over one leg, then the direction turns around.
        SubTp 0 uses the horizontal path values (one leg 86 px), SubTp 1 the
        vertical ones (one leg 64 px).  The marker is the leg's starting end
        (horizontal: left end, vertical: bottom end), as in XP."""
        world = self.world
        if e.sub_tp == 0:
            e.y_vel = 0
            side = (W - 1) if e.dir_sign > 0 else 0
            row = idiv(e.y_pos + HSAFE + 12, H) - SAFE
            wall = world.get(idiv(e.x_pos + side + e.dir_sign, W),
                             row) in CAN_HOLD_YOU
            if wall:
                e.dir_sign = -e.dir_sign
                e.fly_spd = 0
                e.fly_phase = 0
                e.fly_xrem = 0
            else:
                self._fly_red_speed(e, FLY_RED_H_ACCEL, FLY_RED_H_SPEED,
                                    vertical=False)
                e.fly_xrem += e.fly_spd * e.dir_sign
                dx = idiv(e.fly_xrem, FLY_X_DEN)
                e.x_pos += dx
                e.fly_xrem -= dx * FLY_X_DEN
        else:
            e.x_vel = 0
            self._fly_red_speed(e, FLY_RED_V_ACCEL, FLY_RED_V_SPEED,
                                vertical=True)
            e.fly_yrem += e.fly_spd * e.fly_vdir
            dy = idiv(e.fly_yrem, FLY_X_DEN)
            e.y_pos += dy
            e.fly_yrem -= dy * FLY_X_DEN
            e.y_vel = 0
            top = self.world.rows * H - H
            if e.y_pos < 0:
                e.y_pos, e.fly_vdir = 0, 1
                e.fly_spd = 0
                e.fly_phase = 0
                e.fly_yrem = 0
            elif e.y_pos > top:
                e.y_pos, e.fly_vdir = top, -1
                e.fly_spd = 0
                e.fly_phase = 0
                e.fly_yrem = 0

    def _fly_red_speed(self, e: EnemyRec, accel: int, vmax: int,
                       vertical: bool) -> None:
        """One frame of XP's triangular path speed: accelerate to ``vmax``,
        then decelerate back to zero and turn around (at zero the mover has
        travelled one full leg, exactly like ``obj_paratroopa_path``)."""
        if e.fly_phase == 0:
            e.fly_spd += accel
            if e.fly_spd >= vmax:
                e.fly_spd = vmax
                e.fly_phase = 1
        else:
            e.fly_spd -= accel
            if e.fly_spd <= 0:
                e.fly_spd = 0
                e.fly_phase = 0
                if vertical:
                    e.fly_vdir = -e.fly_vdir
                else:
                    e.dir_sign = -e.dir_sign

    def _move_fly_buzzy(self, e: EnemyRec) -> None:
        """Winged Buzzy: a flat cruise, no hop and no gravity.

        It keeps its spawn row, turns around on a wall the same way the green
        flyer does, and moves at a constant FLY_BUZZY_SPEED instead of the
        triangular XP path profile -- the user asked for slow, steady flight.
        """
        world = self.world
        e.x_vel = 0
        e.y_vel = 0
        side = (W - 1) if e.dir_sign > 0 else 0
        row = idiv(e.y_pos + HSAFE + H // 2, H) - SAFE
        if world.get(idiv(e.x_pos + side + e.dir_sign, W), row) in CAN_HOLD_YOU:
            e.dir_sign = -e.dir_sign
            e.fly_xrem = 0
        else:
            e.fly_xrem += FLY_BUZZY_SPEED * e.dir_sign
            dx = idiv(e.fly_xrem, FLY_X_DEN)
            e.x_pos += dx
            e.fly_xrem -= dx * FLY_X_DEN

    def _move_fish(self, e: EnemyRec) -> None:
        """XP's cheep-cheep: a free parabola that ignores tiles completely.
        Horizontal motion stops the moment it is killed."""
        e.x_vel = 0
        e.fly_vy += (FLY_FISH_GRAV if e.tp == TP_FLY_FISH
                     else FLY_FISH_DEAD_GRAV)
        e.fly_yrem += e.fly_vy
        dy = idiv(e.fly_yrem, FLY_X_DEN)
        e.fly_yrem -= dy * FLY_X_DEN
        e.y_vel = dy
        if e.tp == TP_FLY_FISH:
            e.fly_xrem += e.fly_spd * e.dir_sign
            dx = idiv(e.fly_xrem, FLY_X_DEN)
            e.fly_xrem -= dx * FLY_X_DEN
            e.x_pos += dx

    # ------------------------------------------------------------------ #
    # 2026-10-23 batch: Hammer Bro / Boo / bomb stand / bomb / stone shield /
    # Lakitu / falling Spiny.  All of them integrate their own sub-pixel
    # motion over FLY_X_DEN and never enter the walker tile physics.
    # ------------------------------------------------------------------ #
    _SOLID = CAN_HOLD_YOU | CAN_STAND_ON

    # ------------------------------------------------------------------ #
    # water (port-only)
    # ------------------------------------------------------------------ #
    def _in_water(self, e: EnemyRec) -> bool:
        wy = self.options.water_y
        return wy >= 0 and e.y_pos + H > wy

    def _water_slow(self, e: EnemyRec) -> bool:
        """Everything except fish, the pipe families, power-ups and shots
        runs at two thirds speed below the waterline."""
        if not self._in_water(e):
            return False
        if e.tp in (TP_VERT_FISH, TP_DEAD_VERT_FISH,
                    TP_FLY_FISH, TP_DEAD_FLY_FISH):
            return False
        if _is_plant(e.tp):
            return False
        if e.tp in (TP_CHAMP, TP_LIFE, TP_FLOWER, TP_STAR,
                    TP_RISING_CHAMP, TP_RISING_LIFE, TP_RISING_FLOWER,
                    TP_RISING_STAR, TP_FIREBALL, TP_VERT_FIREBALL,
                    TP_PLANT_FIRE):
            return False
        return True

    def _move_water_fish(self, e: EnemyRec) -> None:
        """A fish below the waterline swims horizontally at the green Koopa's
        1 px/frame and turns at walls; the pipe-spout arc belongs to dry
        levels only (user's rule)."""
        world = self.world
        e.y_vel = 0
        if e.x_vel == 0:
            e.x_vel = 1 if e.dir_sign >= 0 else -1
        e.dir_sign = 1 if e.x_vel > 0 else -1
        side = (W - 1) if e.x_vel > 0 else 0
        row = idiv(e.y_pos + H // 2, H)
        ahead = idiv(e.x_pos + side + e.x_vel, W)
        if world.get(ahead, row) in CAN_HOLD_YOU:
            e.x_vel = -e.x_vel
            e.dir_sign = -e.dir_sign

    def _sleeping(self, e: EnemyRec) -> bool:
        """Dormancy: an enemy more than ``SLEEP_ENEMIES_AT`` rows outside the
        top or the bottom of the view is not simulated this frame.

        This is the user's option (1) for the vertical axis (2026-10-24): the
        record is neither released nor re-spawned, so its position, phase,
        timers and animation keep running from exactly where they stopped when
        the camera comes back.  Corpses are exempt -- see _NEVER_SLEEPING.
        """
        if e.tp in _NEVER_SLEEPING:
            return False
        band_top = self.view_y - SLEEP_ENEMIES_AT * H
        band_bottom = self.view_y + VIR_SCREEN_HEIGHT + SLEEP_ENEMIES_AT * H
        return e.y_pos + H <= band_top or e.y_pos >= band_bottom

    def _on_view(self, e: EnemyRec, margin: int = 32) -> bool:
        """XP's onCamera test, in this port's units."""
        return (e.x_pos + W > self.view_x - margin
                and e.x_pos < self.view_x + NH * W + margin
                and e.y_pos + H > self.view_y - margin
                and e.y_pos < self.view_y + SCREEN_HEIGHT + margin)

    def _within_leash(self, e: EnemyRec, leash: int) -> bool:
        """Is the centre of ``e`` within ``leash`` px of the player's centre?
        Straight-line, so a shadow parked above or below a tall level is held
        by the same rule as one trailing behind Mario."""
        cx = e.x_pos + W // 2 - (self.player_x1 + self.player_x2) // 2
        cy = e.y_pos + H // 2 - (self.player_y1 + self.player_y2) // 2
        return cx * cx + cy * cy <= leash * leash

    def _solid_span(self, x1: int, x2: int, y: int) -> bool:
        """Is any tile of the world row containing ``y`` solid under columns
        x1..x2 (pixel coordinates)?"""
        row = idiv(y + HSAFE, H) - SAFE
        for cx in range(idiv(x1, W), idiv(x2, W) + 1):
            if self.world.get(cx, row) in self._SOLID:
                return True
        return False

    def _box_solid(self, x1: int, x2: int, y: int, h: int) -> bool:
        """Same, for every row a ``h`` px tall box overlaps -- used by the
        stone shield, whose 32 px body can straddle a floor tile."""
        r0 = idiv(y + HSAFE, H) - SAFE
        r1 = idiv(y + h - 1 + HSAFE, H) - SAFE
        for row in range(r0, r1 + 1):
            for cx in range(idiv(x1, W), idiv(x2, W) + 1):
                if self.world.get(cx, row) in self._SOLID:
                    return True
        return False

    def _slide(self, e: EnemyRec, v: int) -> int:
        """Accumulate a constant velocity (1/FLY_X_DEN px/frame) and return the
        whole pixels to move this frame; move_enemies applies it via y_vel."""
        e.fly_yrem += v
        dy = idiv(e.fly_yrem, FLY_X_DEN)
        e.fly_yrem -= dy * FLY_X_DEN
        return dy

    def _spawn_at(self, tp: int, x: int, y: int, sub: int = 0,
                  xvel: int = 0) -> EnemyRec | None:
        """new_enemy() takes map cells; runtime spawns need pixel positions, so
        place it in the cell and then overwrite the pixel origin."""
        before = len(self.active)
        self.new_enemy(tp, sub, idiv(x, W), idiv(y, H), xvel, 0, 0)
        if len(self.active) == before:
            return None
        e = self.pool[self.active[-1]]
        e.x_pos = e.last_x_pos = x
        e.y_pos = e.last_y_pos = y
        return e

    def _move_hammer_bro(self, e: EnemyRec) -> None:
        """obj_hammer_bro: patrol the 43 px band left of the marker, hop every
        ~2 s, and throw on animation frame 2 of the 6-frame cycle.

        dir_sign is the aim (the art faces the player, like XP's image_xscale);
        fly_vdir is the patrol direction -- the two are separate in XP too.
        """
        hk = self.hooks
        e.dir_sign = 1 if self.player_x1 >= e.x_pos + W // 2 else -1
        right = e.map_x * W
        left = right - HBRO_BAND
        if e.fly_vdir > 0 and e.x_pos >= right:
            e.fly_vdir = -1
        elif e.fly_vdir < 0 and e.x_pos <= left:
            e.fly_vdir = 1

        e.fly_xrem += HBRO_SPEED * e.fly_vdir
        dx = idiv(e.fly_xrem, FLY_X_DEN)
        if dx:
            side = (W - 1) if e.fly_vdir > 0 else 0
            col = idiv(e.x_pos + side + dx, W)
            r1 = idiv(e.y_pos + HSAFE, H) - SAFE
            r2 = idiv(e.y_pos + HSAFE + H - 1, H) - SAFE
            if (self.world.get(col, r1) in CAN_HOLD_YOU
                    or self.world.get(col, r2) in CAN_HOLD_YOU):
                e.fly_vdir = -e.fly_vdir
                e.fly_xrem = 0
            else:
                e.x_pos += dx
                e.fly_xrem -= dx * FLY_X_DEN

        e.fly_vy += HBRO_GRAVITY
        dy = self._slide(e, e.fly_vy)
        e.y_vel = dy
        x1, x2 = e.x_pos, e.x_pos + W - 1
        if dy > 0 and self._solid_span(x1, x2, e.y_pos + H + dy):
            e.y_pos = (idiv(e.y_pos + dy + 1 + HSAFE, H) - SAFE) * H
            e.y_vel = 0
            e.fly_vy = 0
            e.fly_yrem = 0
            e.status = GROUNDED
        elif dy < 0 and self._solid_span(x1, x2, e.y_pos + dy):
            e.y_vel = 0
            e.fly_vy = 0
            e.fly_yrem = 0

        e.counter += 1
        if e.status == GROUNDED and e.counter >= HBRO_JUMP_FRAMES:
            e.counter = 0
            e.status = FALLING
            e.fly_vy = self.rng.choice((HBRO_HOP, HBRO_LEAP))

        if self._on_view(e):
            # XP triggers inside image_index 2..3 but guards it with a
            # "already thrown this cycle" flag; the port hits the exact frame
            # of the cycle instead, which needs no extra state.
            cycle = (e.dir_counter % (HBRO_ANIM_DIV * 6))
            if cycle == HBRO_THROW_FRAME * HBRO_ANIM_DIV:
                h = self._spawn_at(
                    TP_HAMMER, e.x_pos + W // 2 + HBRO_THROW_DX * e.dir_sign,
                    e.y_pos + HBRO_THROW_DY, xvel=e.dir_sign)
                if h is not None:
                    h.fly_spd = self.rng.randrange(HAMMER_SPEED[0],
                                                   HAMMER_SPEED[1] + 1)
                    h.fly_vy = 0
                hk.beep(400)

    def _move_hammer(self, e: EnemyRec) -> None:
        """obj_hammer_enemy: a thrown hammer, gravity 0.22, destroyed as soon
        as it leaves the view (XP destroys it at view +-30 px)."""
        e.fly_vy += HAMMER_GRAVITY
        e.y_vel = self._slide(e, e.fly_vy)
        e.fly_xrem += e.fly_spd * e.dir_sign
        dx = idiv(e.fly_xrem, FLY_X_DEN)
        e.fly_xrem -= dx * FLY_X_DEN
        e.x_pos += dx
        if not self._on_view(e, 30):
            e.tp = TP_DYING

    def _plant_spit(self, e: EnemyRec, hk) -> None:
        """The user's rule: a fire plant only spits while it is fully out of
        the pipe **and** its mouth is opening.

        SMB3 gates only on "fully out" and spits twice per extension off a
        countdown; the mouth test here is the stricter version asked for, and
        PLANT_FIRE_COOLDOWN keeps the ROM's roughly-two-shots rate.

        Two kinds (the user's split, 2026-10-25): the plain one fires
        **horizontally** at the player's side, the aiming one fires **at
        Mario's body**, so the flame leaves along the real line of sight.
        """
        if not _is_fire_plant(e.tp):
            return
        if e.fire_cooldown:
            e.fire_cooldown -= 1
            return
        if self.time_counter % PLANT_MOUTH_DIV != 0:
            return                        # the mouth opens on frame 0 only
        e.fire_cooldown = PLANT_FIRE_COOLDOWN
        mouth_x = e.x_pos + W // 2 + (6 if self.player_x1 >= e.x_pos else -6)
        mouth_y = e.y_pos + 3
        f = self._spawn_at(TP_PLANT_FIRE, mouth_x, mouth_y, xvel=0)
        if f is None:
            return
        if e.tp == TP_FIRE_PLANT:
            f.dir_sign = 1 if self.player_x1 >= e.x_pos else -1
            f.fly_spd = PLANT_FIRE_SPEED
            f.fly_vy = 0
        else:
            # unit vector from the mouth to the player's chest, at flame speed
            cx = self.player_x1 + (self.player_x2 - self.player_x1) // 2
            cy = self.player_y1 - 6
            dx = cx - mouth_x
            dy = cy - mouth_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1:
                dist = 1.0
            f.fly_spd = 0                 # both axes come from fly_vx/vy
            f.dir_sign = 1 if dx >= 0 else -1
            f.x_vel = 0
            f.fly_vy = int(dy / dist * PLANT_FIRE_SPEED)
            f.fly_phase = int(dx / dist * PLANT_FIRE_SPEED)   # x speed, 1/1024
        hk.beep(600)

    def _move_plant_fire(self, e: EnemyRec) -> None:
        """SMB3's SOBJ_PIRANHAFIREBALL: constant velocity, no gravity, and the
        solid check is skipped, so the flame crosses terrain.  It is removed
        once it leaves the view -- it never hits an enemy (the same exclusion
        the thrown hammer has, which is the rule the user asked for).

        ``fly_phase``/``fly_vy`` hold the x/y velocities in 1/FLY_X_DEN px per
        frame; the horizontal plant keeps fly_vy at 0.
        """
        e.fly_xrem += e.fly_spd * e.dir_sign + e.fly_phase
        e.fly_yrem += e.fly_vy
        dx = idiv(e.fly_xrem, FLY_X_DEN)
        dy = idiv(e.fly_yrem, FLY_X_DEN)
        e.fly_xrem -= dx * FLY_X_DEN
        e.fly_yrem -= dy * FLY_X_DEN
        e.x_pos += dx
        e.y_pos += dy
        if not self._on_view(e, 30):
            e.tp = TP_DYING

    def _move_fast_plant(self, e: EnemyRec) -> None:
        """XP's red piranha: 1.25 s out, 1.25 s back, and it clears the pipe
        in a fraction of that window and waits.  dir_sign follows XP's
        piranhaDirection (1 = sink back into the pipe)."""
        e.counter += 1
        if e.counter >= FAST_PLANT_FLIP:
            e.counter = 0
            e.dir_sign = -e.dir_sign
        e.fly_yrem += FAST_PLANT_SPEED * e.dir_sign
        dy = idiv(e.fly_yrem, FLY_X_DEN)
        e.fly_yrem -= dy * FLY_X_DEN
        e.y_pos += dy
        bottom = e.map_y * H
        top = bottom - FAST_PLANT_RISE
        if e.y_pos >= bottom:
            e.y_pos = bottom
            e.fly_yrem = 0
        elif e.y_pos <= top:
            e.y_pos = top
            e.fly_yrem = 0

    def _move_side_plant(self, e: EnemyRec) -> None:
        """The vertical plant's chew cycle turned onto the x axis.

        Runs on the same ``move_delay`` gate, so it slides out at the same
        1 px per 3 frames.  ``dir_sign`` is the side it reaches for (the
        per-marker direction the editor writes)."""
        home = e.map_x * W
        if e.status == 0:                       # coming out
            e.x_pos += e.dir_sign
            if abs(e.x_pos - home) >= SIDE_PLANT_RISE:
                e.status = 1
                e.counter = 0
        elif e.status == 1:                     # fully out, chewing
            e.counter += 1
            if e.counter > SIDE_PLANT_HOLD:
                e.status = 2
        elif e.status == 2:                     # going back
            e.x_pos -= e.dir_sign
            if (e.x_pos - home) * e.dir_sign <= 0:
                e.x_pos = home
                e.status = 3
                e.counter = 0
        else:                                   # resting inside the pipe
            e.counter += 1
            if e.counter > self.rng.randrange(*SIDE_PLANT_WAIT):
                e.counter = 0
                e.status = 0

    def _move_boo(self, e: EnemyRec) -> None:
        """obj_boo: chases the player at 0.3125 px/f, straight through terrain,
        but only while the player is facing away from it.  Aim is the player's
        chest (XP aims at obj_player.y - 8)."""
        if e.tp == TP_DEAD_BOO:
            e.fly_vy += HBRO_GRAVITY
            e.y_vel = self._slide(e, e.fly_vy)
            e.x_pos += e.x_vel
            if e.y_pos > self.world.rows * H:
                e.tp = TP_DYING
            return
        cx = e.x_pos + W // 2
        px = self.player_x1 + (self.player_x2 - self.player_x1) // 2
        py = self.player_y1 - BOO_AIM_UP
        e.dir_sign = 1 if px >= cx else -1
        # XP's canMove: the player's last horizontal direction points away
        # from the Boo (left of it and walking left, or right and walking right)
        away = ((px < cx and self.player_face < 0)
                or (px > cx and self.player_face > 0))
        # the Boo is only allowed to move while it is on screen; the shadow is
        # never recycled, so its range is the leash, not the view
        near = (self._within_leash(e, GHOST_SHADE_LEASH)
                if e.tp == TP_GHOST_SHADE else self._on_view(e, 30))
        moving = away and near
        e.status = 1 if moving else 0
        if moving:
            dx = px - cx
            dy = py - (e.y_pos + H // 2)
            n = max(1, int(round((dx * dx + dy * dy) ** 0.5)))
            e.fly_xrem += int(BOO_SPEED * dx / n)
            mx = idiv(e.fly_xrem, FLY_X_DEN)
            e.fly_xrem -= mx * FLY_X_DEN
            e.x_pos += mx
            e.y_vel = self._slide(e, int(BOO_SPEED * dy / n))
        else:
            e.y_vel = 0

    def _move_bomb_stand(self, e: EnemyRec) -> None:
        """obj_thrower: counts only while on screen, then fires a bomb towards
        the player every 3 s.  Nothing can destroy it (user's rule)."""
        if not self._on_view(e):
            return
        e.counter += 1
        if e.counter < BOMB_STAND_FRAMES:
            return
        e.counter = 0
        d = 1 if self.player_x1 >= e.x_pos + W // 2 else -1
        # The canvas is the tile box width now, so no horizontal centring; the
        # vertical launch line is BOMB_LAUNCH_DY (see the pitfall-8 comment).
        b = self._spawn_at(TP_BOMB, e.x_pos, e.y_pos + BOMB_LAUNCH_DY, xvel=d)
        if b is not None:
            b.fly_spd = BOMB_SPEED
            b.fly_vy = 0
        self.hooks.beep(200)

    def _in_muzzle(self, e: EnemyRec) -> bool:
        """Is this bomb still inside a bomb stand?  That is XP's
        ``initialOrder`` flag: obj_bullet_bill only allows itself to be stomped
        once ``!place_meeting(x, y, obj_thrower)`` holds."""
        x1, y1, w1, h1 = self._hit_box(e)
        for idx in self.active:
            s = self.pool[idx]
            if s.tp != TP_BOMB_STAND:
                continue
            x2, y2, w2, h2 = self._hit_box(s)
            if (x1 < x2 + w2 and x1 + w1 > x2
                    and y1 < y2 + h2 and y1 + h1 > y2):
                return True
        return False

    def _move_bomb(self, e: EnemyRec) -> None:
        """obj_bullet_bill: dead straight while alive, flipped and falling once
        stomped, and deleted the moment it leaves the camera."""
        if e.tp == TP_DEAD_BOMB:
            e.fly_vy += BOMB_DEAD_GRAVITY
            e.y_vel = self._slide(e, e.fly_vy)
            e.x_pos += e.x_vel
            if e.y_pos > self.world.rows * H:
                e.tp = TP_DYING
            return
        e.fly_xrem += BOMB_SPEED * e.dir_sign
        dx = idiv(e.fly_xrem, FLY_X_DEN)
        e.fly_xrem -= dx * FLY_X_DEN
        e.x_pos += dx
        e.y_vel = 0
        if not self._on_view(e, 32):
            e.tp = TP_DYING

    def _move_thwomp(self, e: EnemyRec) -> None:
        """obj_thwomp: slams down at 1.875 px/f (creeping the last pixel),
        backs up at 0.469 px/f, pauses 0.5 s at the ceiling.  Only moves while
        on screen, and cannot be killed -- except by the star, which sends it
        tumbling down like every other corpse here."""
        hk = self.hooks
        if e.tp == TP_DEAD_THWOMP:
            e.counter += 1
            e.fly_vy += BOMB_DEAD_GRAVITY
            e.y_vel = self._slide(e, e.fly_vy)
            e.x_pos += e.x_vel
            if e.y_pos > self.world.rows * H:
                e.tp = TP_DYING
            return
        _, _, bw, bh = NEW_BOX[TP_THWOMP]
        x1 = e.x_pos + NEW_BOX[TP_THWOMP][0]
        x2 = x1 + bw - 1
        if e.status == 0:
            # XP unsticks a Thwomp that starts inside terrain: while its body
            # overlaps the ground it is pushed up one pixel at a time.
            for _ in range(bh + H):
                if not self._box_solid(x1, x2, e.y_pos, bh):
                    break
                e.y_pos -= 1
            e.status = 1
        e.y_vel = 0
        if e.counter > 0:
            e.counter -= 1
            return
        if not self._on_view(e, 30):
            return
        if e.fly_phase == 1:                       # going down
            if self._solid_span(x1, x2, e.y_pos + bh):
                e.fly_phase = -1
                e.fly_yrem = 0
                hk.beep(60)
            else:
                creep = self._solid_span(x1, x2, e.y_pos + bh + 1)
                e.y_vel = self._slide(e, THWOMP_DOWN_STEP if creep
                                      else THWOMP_DOWN)
        else:                                      # backing up
            # XP only reverses against terrain; this port's levels rarely have
            # a ceiling over a shield, so the top of the view is the backstop.
            if self._solid_span(x1, x2, e.y_pos - 1) or e.y_pos <= self.view_y:
                e.fly_phase = 1
                e.counter = THWOMP_HOLD
                e.fly_yrem = 0
            else:
                e.y_vel = self._slide(e, -THWOMP_UP)

    def _move_lakitu(self, e: EnemyRec) -> None:
        """obj_lakitu: rides the top of the camera, sinks from -11 px to +37 px
        into the view, chases the player's screen side, and drops a Spiny on
        animation frame 10 of the 12-frame cycle."""
        hk = self.hooks
        if e.tp == TP_DEAD_LAKITU:
            e.fly_vy += HBRO_GRAVITY
            e.y_vel = self._slide(e, e.fly_vy)
            e.x_pos += e.x_vel
            if e.y_pos > self.world.rows * H:
                e.tp = TP_DYING
            return
        # fly_spd doubles as the yPoint accumulator (1/FLY_X_DEN px)
        e.fly_spd = min(LAKITU_MAX * FLY_X_DEN, e.fly_spd + LAKITU_RISE)
        e.y_pos = self.view_y + idiv(e.fly_spd, FLY_X_DEN)
        e.y_vel = 0
        left = self.view_x + LAKITU_EDGE
        right = self.view_x + NH * W - LAKITU_EDGE - W
        if e.dir_sign > 0 and e.x_pos >= right:
            e.dir_sign = -1
        elif e.dir_sign < 0 and e.x_pos <= left:
            e.dir_sign = 1
        if self.player_face == e.dir_sign:
            v = LAKITU_CHASE
        elif self.player_face == -e.dir_sign:
            v = LAKITU_EVADE
        else:
            v = LAKITU_DRIFT
        e.fly_xrem += v * e.dir_sign
        dx = idiv(e.fly_xrem, FLY_X_DEN)
        e.fly_xrem -= dx * FLY_X_DEN
        e.x_pos += dx
        if (e.dir_counter % (LAKITU_ANIM_DIV * 12)
                == LAKITU_THROW_FRAME * LAKITU_ANIM_DIV):
            s = self._spawn_at(TP_SPINY_FALL, e.x_pos + (W - 20) // 2,
                               e.y_pos + 20, xvel=e.dir_sign)
            if s is not None:
                s.fly_vy = SPINYF_POP
                s.fly_yrem = 0
            hk.beep(300)

    # enemies a falling Spiny cannot land on: the cloud rides the camera and
    # the Boo has no body to stand on (the user's exclusions, 2026-10-24); the
    # thrown hammer takes part in no enemy collision at all, so it is not a
    # floor either
    _NO_LAND_ON = (TP_LAKITU, TP_DEAD_LAKITU, TP_BOO, TP_DEAD_BOO,
                   TP_SPINY_FALL, TP_HAMMER, TP_GHOST_SHADE)

    def _head_landing(self, e: EnemyRec) -> EnemyRec | None:
        """The enemy whose head ``e`` is dropping onto this frame, if any.

        Landing, not hitting: the Spiny has to be falling and its feet have to
        cross the top of the other hit box, so a sideways touch still just
        bounces the two apart (the user's rule, 2026-10-24)."""
        if e.y_vel <= 0:
            return None
        x1, x2 = e.x_pos, e.x_pos + W - 1
        feet = e.y_pos + H
        for idx in self.active:
            j = self.pool[idx]
            if (j is e or j.tp in self._NO_LAND_ON
                    or j.tp in (TP_DEAD, TP_DYING, TP_DYING_FIREBALL,
                                TP_DYING_KOOPA)):
                continue
            hx, hy, hw, _hh = self._hit_box(j)
            if x2 < hx or x1 > hx + hw - 1:
                continue
            if feet <= hy and feet + e.y_vel > hy:
                return j
        return None

    def _sit_spiny(self, e: EnemyRec, land: int) -> EnemyRec | None:
        """Turn a falling Spiny into an ordinary walking one at ``land``."""
        hk = self.hooks
        d = 1 if self.player_x1 >= e.x_pos else -1
        w = self._spawn_at(TP_RED, e.x_pos, land + SPINYF_SIT_DY, xvel=d)
        if w is None:
            return None
        w.x_pos = e.x_pos
        w.y_pos = land
        w.last_x_pos = w.x_pos
        w.last_y_pos = w.y_pos
        w.x_vel = d
        w.move_delay = 2
        e.tp = TP_DYING
        hk.beep(120)
        return w

    def _move_spiny_fall(self, e: EnemyRec) -> None:
        """obj_spiny_falling: pops up from the cloud, falls with gravity 0.0996
        and becomes an ordinary walking Spiny where it lands -- on the floor or
        on top of another enemy's head."""
        e.fly_vy += SPINYF_GRAVITY
        e.y_vel = self._slide(e, e.fly_vy)
        if e.y_pos > self.world.rows * H:
            e.tp = TP_DYING
            return
        if e.y_vel > 0 and self._solid_span(e.x_pos, e.x_pos + W - 1,
                                            e.y_pos + H + e.y_vel):
            self._sit_spiny(e, (idiv(e.y_pos + e.y_vel + 1 + HSAFE, H)
                                - SAFE) * H)
        elif self._head_landing(e) is not None:
            self._sit_spiny(e, e.y_pos)

    def _move_from_pipe(self, e: EnemyRec, hk) -> None:
        """The clock of everything that lives in a pipe.

        Five states, the original's plant cycle: ``0`` waits inside until the
        player is close enough (``sub_tp`` picks how close: 0 = a screen away,
        1 = one tile, anything else = now); ``1`` rises the 19 px above the pipe
        mouth; ``2`` stays out chewing, and a fire-spitting plant shoots here;
        ``3`` sinks back past its own map row; ``4`` rests for a random
        100..199 frames and returns to ``0``.

        ``TP_SIDE_PLANT`` shares the cycle but turns it onto the x axis (its own
        mover), and the spouting fish / lava fireball only use the
        bottom-of-world half: they wait in the floor and launch upward.
        """
        if e.tp == TP_SIDE_PLANT:
            self._move_side_plant(e)
        elif _is_plant(e.tp):
            if e.status == 0:
                if e.sub_tp == 0:
                    if ((e.x_pos > self.player_x2 + W)
                            or (e.x_pos + 24 + W < self.player_x1)):
                        e.status += 1
                elif e.sub_tp == 1:
                    if ((e.x_pos > self.player_x2)
                            or (e.x_pos + 24 < self.player_x1)):
                        e.status += 1
                else:
                    e.status += 1
                e.y_vel = 0
                e.delay_counter = 0
                e.move_delay = 1
            elif e.status == 1:
                e.y_vel = -1
                e.delay_counter = 0
                e.move_delay = 2
                if e.y_pos + e.y_vel <= e.map_y * H - 19:
                    e.y_vel = 0
                    e.delay_counter = 0
                    e.move_delay = 2
                    e.counter = 0
                    e.status += 1
            elif e.status == 2:
                e.counter += 1
                if e.counter > 200:
                    e.status += 1
                e.move_delay = 0
                e.delay_counter = 0
                self._plant_spit(e, hk)
            elif e.status == 3:
                e.y_vel = 1
                e.delay_counter = 0
                e.move_delay = 2
                if e.y_pos > e.map_y * H:
                    e.status += 1
            else:           # status == 4
                e.y_vel = 0
                e.move_delay = 100 + self.rng.randrange(100)
                e.delay_counter = 0
                e.status = 0
        elif (e.y_pos + H >= self.world.rows * H):
            if e.y_vel > 0:
                e.y_vel = 0
                e.move_delay = 100 + self.rng.randrange(300)
                e.delay_counter = 0
            else:
                e.y_vel = -10
                e.move_delay = 1
                e.delay_counter = 0
                if e.tp == TP_VERT_FIREBALL:
                    hk.beep(100)
                    e.y_vel = -9

    def _move_dry_bones(self, e: EnemyRec) -> None:
        """The skeleton's own clock, kept out of the MoveEnemies loop.

        Dry Bones is deliberately NOT a ``_move_new`` mover: it walks on the
        walker tile path, so it turns at ledges and falls like a Koopa, and the
        pile just stands there.  What it needs from the loop once per frame is
        only this.  The collapse schedule itself lives beside ``DRY_REASSEMBLE``
        with ``_collapse_pile`` / ``_dry_stage`` / ``_dry_art_box``.
        """
        if e.tp == TP_DRY_BONES:
            # the walker path gives x_vel; the art reads dir_sign
            if e.x_vel:
                e.dir_sign = 1 if e.x_vel > 0 else -1
            # SMB3 only ticks the walk cycle on the ground
            if e.status == GROUNDED:
                e.sub_tp = 1 if (self.time_counter & DRY_ANIM_MASK) else 0
            return
        e.counter += 1
        if e.counter > DRY_REVIVE_FRAMES:
            # SMB3: stand up, then face the player and walk at him
            e.tp = TP_DRY_BONES
            e.counter = 0
            e.move_delay = DRY_WALK_DELAY
            e.dir_sign = 1 if self.player_x1 > e.x_pos else -1
            e.x_vel = e.dir_sign

    def _move_new(self, e: EnemyRec) -> None:
        tp = e.tp
        if tp == TP_HAMMER_BRO:
            self._move_hammer_bro(e)
        elif tp == TP_DEAD_HAMMER_BRO:
            e.fly_vy += HBRO_GRAVITY
            e.y_vel = self._slide(e, e.fly_vy)
            e.x_pos += e.x_vel
            if e.y_pos < self.view_y - 64 or e.y_pos > self.world.rows * H:
                e.tp = TP_DYING
        elif tp == TP_HAMMER:
            self._move_hammer(e)
        elif tp == TP_PLANT_FIRE:
            self._move_plant_fire(e)
        elif tp == TP_FAST_PLANT:
            self._move_fast_plant(e)
        elif tp in (TP_BOO, TP_DEAD_BOO, TP_GHOST_SHADE):
            self._move_boo(e)
        elif tp == TP_BOMB_STAND:
            self._move_bomb_stand(e)
        elif tp in (TP_BOMB, TP_DEAD_BOMB):
            self._move_bomb(e)
        elif tp in (TP_THWOMP, TP_DEAD_THWOMP):
            self._move_thwomp(e)
        elif tp in (TP_LAKITU, TP_DEAD_LAKITU):
            self._move_lakitu(e)
        elif tp == TP_SPINY_FALL:
            self._move_spiny_fall(e)

    def _hit_running_koopa(self, i: int, k: int) -> None:
        """Shell duel: compare ``_power``, higher wins, equal ranks destroy each
        other (4 vs 4 or 2 vs 2 -- only shells reach equal ranks)."""
        j = self.pool[k]
        e = self.pool[i]
        pj, pe = _power(j), _power(e)
        if pe >= pj:
            self._show_star(j.x_pos, j.y_pos)
            self._kill(k)
        if pj >= pe:
            self._show_star(e.x_pos, e.y_pos)
            self._kill(i)

    def _hit_stationary_shell(self, i: int, k: int) -> None:
        """Rule 3: a moving shell runs into a resting one.

        Resting black (rank 3) or Buzzy (rank 1) shells get here.  A shell
        knocked up earlier this frame has y_vel != 0 and must not be hit a second
        time.  Higher ``_power`` wins; a coloured shell breaks against a resting
        black one instead of destroying it.
        """
        e = self.pool[i]
        j = self.pool[k]
        if e.tp == TP_RUNNING_KOOPA and j.y_vel == 0:
            if _power(e) > _power(j):
                # a moving black shell wins
                self._show_star(j.x_pos, j.y_pos)
                self._break_stationary_shell(k)
            else:
                # a coloured shell breaks against it
                self._show_star(e.x_pos, e.y_pos)
                self._kill(i)

    def _hit_boo(self, i: int, k: int) -> None:
        """The Boo does not initiate collisions at all, so reaching here means
        something ran into it: only a moving shell hurts it, everything else
        passes straight through."""
        e = self.pool[i]
        if e.tp == TP_RUNNING_KOOPA:
            j = self.pool[k]
            self._show_star(j.x_pos, j.y_pos)
            self._kill(k)

    def _hit_fireball(self, i: int, k: int) -> None:
        """Rule 1: a fireball is extinguished by any contact, but cannot hurt a
        black creature or the Buzzy in any state."""
        e = self.pool[i]
        j = self.pool[k]
        j.tp = TP_DYING_FIREBALL
        j.delay_counter = -(MAX_PAGE + 1)
        self._show_star(e.x_pos, e.y_pos)
        if not _fireproof(e):
            self._kill(i)

    def _bump_pair(self, e: EnemyRec, j: EnemyRec, x: int,
                   new_x1: int) -> None:
        """Two plain walkers meet: both turn around and both swap their vertical
        sign, then the one that was overtaken is pushed back a step."""
        for obj in (e, j):
            if _flying(obj.tp):
                obj.dir_sign = -obj.dir_sign
            else:
                obj.x_vel = -obj.x_vel
        e.y_vel = -e.y_vel
        j.y_vel = -j.y_vel
        if abs(x - new_x1) < W:
            if x > new_x1:
                e.x_pos = e.x_pos - e.x_vel
                e.x_vel = -abs(e.x_vel)
            elif x < new_x1:
                e.x_pos = e.x_pos - e.x_vel
                e.x_vel = abs(e.x_vel)

    def _check_rising(self, e: EnemyRec, world) -> bool:
        """The top of Check: a rising power-up emerges tile by tile.

        True = the frame is over, the item is still inside its bumped block (the
        original beeps on every other pixel of travel on the way up).  False =
        it has just cleared a whole tile, popped out and must keep running
        through the rest of Check like any other enemy.
        """
        if not (((e.y_pos / H) == idiv(e.y_pos, H))
                and (e.y_pos != e.map_y * H)):
            j = e.y_pos % H
            if j % 2 == 0:
                self.hooks.beep(130 - 20 * j)
            return True
        e.x_vel = 1 - 2 * int(world.get(e.map_x + 1, e.map_y - 1)
                              in CAN_HOLD_YOU)
        if e.tp == TP_RISING_CHAMP:
            e.tp = TP_CHAMP
        elif e.tp == TP_RISING_LIFE:
            e.tp = TP_LIFE
            e.x_vel = 2 * e.x_vel
        elif e.tp == TP_RISING_FLOWER:
            e.x_vel = 0
            e.tp = TP_FLOWER
        else:
            e.tp = TP_STAR
            e.x_vel = 2 * e.x_vel
        e.y_vel = -7
        e.move_delay = 1
        e.status = FALLING
        return False

    def _check_fireball(self, e: EnemyRec, world, view_x: int) -> None:
        """Mario's own fireball: tile edge, its little hop, and the four ways it
        dies (stopped, left the view, right of it, or below the floor)."""
        at_x = idiv(e.x_pos + W // 4, W)
        new_x = idiv(e.x_pos + W // 4 + e.x_vel, W)
        if (at_x != new_x) or (self.player_x1 % W == 0):
            y1 = idiv(e.y_pos + H // 4 + HSAFE, H) - SAFE
            if world.get(new_x, y1) in CAN_HOLD_YOU:
                e.x_vel = 0
        new_x = e.x_pos
        new_y = e.y_pos
        at_x = idiv(e.x_pos + W // 4 + e.x_vel, W)
        new_y = idiv(e.y_pos + 2 + H // 4 + e.y_vel + HSAFE, H) - SAFE
        if (e.y_vel > 0) and (world.get(at_x, new_y)
                              in CAN_HOLD_YOU | CAN_STAND_ON):
            e.y_pos = (idiv(e.y_pos + e.y_vel - 5 + HSAFE, H)
                       - SAFE) * H
            e.y_vel = -2
        elif e.x_pos % 3 == 0:
            e.y_vel += 1
        if ((e.x_vel == 0) or (new_x < view_x - W)
                or (new_x > view_x + NH * W + W)
                or (new_y > self.world.rows * H)):
            e.delay_counter = -(MAX_PAGE + 1)
            e.tp = TP_DYING_FIREBALL

    def _check(self, i: int) -> None:
        """Check(i) — tile collision, edge turning, the running-Koopa        block-breaking, enemy-enemy collisions and fireball physics."""
        e = self.pool[i]
        world = self.world
        hk = self.hooks
        view_x = self.view_x

        if e.tp in (TP_RISING_CHAMP, TP_RISING_LIFE, TP_RISING_FLOWER,
                    TP_RISING_STAR):
            if self._check_rising(e, world):
                return

        if e.tp == TP_FIREBALL:
            self._check_fireball(e, world, view_x)
            return

        if e.tp == TP_VERT_FISH and self._in_water(e):
            self._move_water_fish(e)
        elif _fish(e.tp):
            self._move_fish(e)

        if e.tp == TP_STAR:
            hk.start_glitter(e.x_pos, e.y_pos, W, H)

        if _hops_like_green(e.tp):
            self._move_fly_green(e)
        elif e.tp == TP_FLY_RED:
            self._move_fly_red(e)
        elif e.tp == TP_FLY_BUZZY:
            self._move_fly_buzzy(e)
        elif e.tp in _NO_TILES:
            self._move_new(e)

        if _walks_on_tiles(e.tp):
            self._tile_side(e, world, view_x, hk)
            self._tile_ground(e, world)

        self._duel_with_pool(i, e)

    def _tile_side(self, e: EnemyRec, world, view_x: int, hk) -> None:
        """Horizontal: a walker turns round when the column it is entering is
        solid anywhere in the two rows it occupies (not the rows under its feet
        -- see §5.1.1).  A running shell is the one type that does something
        about that column instead of just stopping."""
        side = int(e.x_vel > 0) * (W - 1)
        at_x = idiv(e.x_pos + side, W)
        new_x = idiv(e.x_pos + side + e.x_vel, W)
        if (at_x != new_x) or (e.status == FALLING):
            y1 = idiv(e.y_pos + HSAFE, H) - SAFE
            y2 = idiv(e.y_pos + HSAFE + H - 1, H) - SAFE
            hold1 = world.get(new_x, y1) in CAN_HOLD_YOU
            hold2 = world.get(new_x, y2) in CAN_HOLD_YOU
            if hold1 or hold2:
                if e.tp == TP_RUNNING_KOOPA:
                    self._running_shell_hits(e, world, new_x, view_x, hk)
                e.x_vel = 0

    def _running_shell_hits(self, e: EnemyRec, world, new_x: int,
                            view_x: int, hk) -> None:
        """A running shell into a wall: it always sparkles, and only when it is
        wholly on screen does it smash the brick (``J``) or punch a ``?`` block
        -- coin if something is already above it, otherwise the champ/flower for
        a small player, the extra life for 0xE1 -- and turn it into ``@``."""
        self._show_star(e.x_pos + e.x_vel, e.y_pos)
        l = idiv(e.y_pos + HSAFE + H // 2, H) - SAFE
        ch = world.get(new_x, l)
        if ((e.x_pos >= view_x)
                and (e.x_pos + W <= view_x + NH * W)):
            if ch == ord("J"):
                hk.break_block(new_x, l)
            elif ch == ord("?"):
                above = world.get(new_x, l - 1)
                if above == ord(" "):
                    hk.hit_coin(new_x * W, l * H, True)
                elif above == 0xE0:      # 'α'
                    if self._player_mode() == MD_SMALL:
                        self.new_enemy(TP_RISING_CHAMP, 0,
                                       new_x, l, 0, -1, 1)
                    else:
                        self.new_enemy(TP_RISING_FLOWER, 0,
                                       new_x, l, 0, -1, 1)
                elif above == 0xE1:      # 'ß'
                    self.new_enemy(TP_RISING_LIFE, 0,
                                   new_x, l, 0, -1, 2)
                hk.remove(new_x * W, l * H, W, H, 1)
                world.set(new_x, l, ord("@"))

    def _tile_ground(self, e: EnemyRec, world) -> None:
        """Vertical: the two tiles under the front corners of the walker decide
        its grounded/falling state.  Three families have a rule of their own in
        here -- the block lifts bounce, red Koopas refuse the edge, the star
        keeps falling after it lands -- and each is its own method."""
        at_x = idiv(e.x_pos + e.x_vel, W)
        new_x = idiv(e.x_pos + e.x_vel + W - 1, W)
        new_y = idiv(e.y_pos + 1 + H + e.y_vel + HSAFE, H) - SAFE
        hold1 = world.get(at_x, new_y) in CAN_HOLD_YOU | CAN_STAND_ON
        hold2 = world.get(new_x, new_y) in CAN_HOLD_YOU | CAN_STAND_ON

        if _is_lift(e.tp):
            self._lift_bounce(e, hold1)
        elif e.status == GROUNDED:
            if not (hold1 or hold2):
                e.status = FALLING
                e.y_vel = 1
            self._red_koopa_edge(e, hold1, hold2)
        elif e.status == FALLING:
            self._walker_fall(e, hold1, hold2)

    def _lift_bounce(self, e: EnemyRec, hold1: bool) -> None:
        """A block lift reverses on what it can hold; going up, what it holds is
        the ceiling it was born under, not a tile.  A donut board never bounces
        -- it is the thing that falls away under the player."""
        if (e.y_vel != 0) and (e.tp != TP_DONUT):
            if e.y_vel < 0:
                hold1 = idiv(e.y_pos + e.y_vel, H) < e.map_y
            if hold1:
                e.y_vel = -e.y_vel

    def _red_koopa_edge(self, e: EnemyRec, hold1: bool, hold2: bool) -> None:
        """A red Koopa steps off a platform only when the tile ahead is empty
        AND the one below it is not; otherwise it stops and turns round.  The
        two halves are the same rule read from each side, and the second one
        must see the x_vel the first one may have zeroed."""
        if (e.sub_tp == 1) and (e.tp == TP_KOOPA):
            if (e.x_vel > 0) and (e.x_pos % W in range(11, 20)):
                if (not hold2) and hold1:
                    e.x_vel = 0
            if (e.x_vel < 0) and (e.x_pos % W in range(1, 10)):
                if (not hold1) and hold2:
                    e.x_vel = 0

    def _walker_fall(self, e: EnemyRec, hold1: bool, hold2: bool) -> None:
        """Falling: land flush on the tile grid, or keep accelerating to the
        walker's terminal speed of 4.  The star is the exception that lands and
        stays in the air, bouncing with 2.5 times what it fell."""
        if hold1 or hold2:
            e.status = GROUNDED
            e.y_pos = (idiv(e.y_pos + e.y_vel + 1 + HSAFE, H)
                       - SAFE) * H
            if e.tp == TP_STAR:
                e.y_vel = idiv(-5 * e.y_vel, 2)
                e.status = FALLING
            else:
                e.y_vel = 0
        else:
            e.y_vel += 1
            if e.y_vel > 4:
                e.y_vel = 4

    def _duel_with_pool(self, i: int, e: EnemyRec) -> None:
        """Enemy vs enemy, for every type including the ones outside the tile
        guard: the original runs this for the vertical enemies too
        (ENEMIES.PAS 785-865), which is how a fireball can kill a piranha plant
        or a jumping fish.

        TP_BOO / TP_BOMB / TP_HAMMER are deliberately missing from the
        initiating side (_starts_duel): a Boo, a bomb and a thrown hammer take
        no part in enemy collisions (the user's rule).  The Boo still dies to a
        shell that runs into it -- j.tp == TP_BOO below; the bomb and the hammer
        are killed by nothing here (star and stomp only).  The WEBANGELO fire
        bar is not in this pool at all, so it never collides either.
        """
        new_x1 = e.x_pos + e.x_vel
        new_x2 = new_x1 + W - 1 + 4 * int(_is_plant(e.tp))
        y1 = e.y_pos + e.y_vel
        y2 = y1 + H - 1
        if self._hidden(e):
            # drawn as nothing: it starts no collision and takes no part in one
            # either (the user's rule, 2026-10-25)
            return
        if _starts_duel(e.tp):
            for k in list(self.active):
                j = self.pool[k]
                if k == i:
                    continue
                if _duel_victim(j):
                    x = j.x_pos + j.x_vel
                    y = j.y_pos + j.y_vel
                    if (new_x1 < x + W) and (new_x2 > x) \
                            and (y1 < y + H) and (y2 > y):
                        if j.tp == TP_RUNNING_KOOPA:
                            self._hit_running_koopa(i, k)
                        elif j.tp == TP_DEAD_KOOPA:
                            self._hit_stationary_shell(i, k)
                        elif j.tp == TP_BOO:
                            self._hit_boo(i, k)
                        elif (e.tp != TP_RUNNING_KOOPA
                              and e.tp != TP_LAKITU   # fireball target only
                              and not (_fish(e.tp) or _fish(j.tp))):
                            self._bump_pair(e, j, x, new_x1)
                elif j.tp == TP_FIREBALL:
                    x = j.x_pos + j.x_vel
                    y = j.y_pos + j.y_vel
                    if ((new_x1 <= x + W // 2) and (new_x2 >= x)
                            and (y1 <= y + H // 2) and (y2 >= y)):
                        self._hit_fireball(i, k)

    # ------------------------------------------------------------------ #
    # the per-frame update (MoveEnemies)
    # ------------------------------------------------------------------ #
    def _touch_shell(self, e: EnemyRec) -> None:
        """A sleeping or waking Koopa touched at all becomes a kicked shell,
        sent away from Mario at five pixels a frame."""
        e.tp = TP_RUNNING_KOOPA
        e.x_vel = 5 * (2 * int(e.x_pos > self.player_x1) - 1)
        e.move_delay = 0
        e.delay_counter = 0
        self.hooks.beep(800)
        self.cd_enemy = 1
        self.hooks.add_score(100)

    def _collect_item(self, e: EnemyRec) -> None:
        """A power-up reached Mario: raise his pickup flag, hide the item, scatter
        coins.

        The mushroom row carries two different things and ``sub_tp`` tells them
        apart -- the art is literally ``CHAMP000`` vs ``POISON000``: 0 is the
        normal mushroom (grows Mario *and* scores 1000), 1 is the poison one,
        which hurts him instead (``cd_hit`` shrinks him, and kills him if there is
        nothing left to shrink).  The 1-UP, the flower and the star only set their
        own flag plus the 1000.
        """
        hk = self.hooks
        if e.tp == TP_CHAMP:
            if e.sub_tp == 0:
                self.cd_champ = 1
                hk.add_score(1000)
            else:
                self.cd_hit = 1
        elif e.tp == TP_LIFE:
            self.cd_life = 1
            hk.add_score(1000)
        elif e.tp == TP_FLOWER:
            self.cd_flower = 1
            hk.add_score(1000)
        else:
            self.cd_star = 1
            hk.add_score(1000)
        e.tp = TP_DYING
        e.delay_counter = -(MAX_PAGE + 1)
        hk.coin_glitter(e.x_pos, e.y_pos)

    def _stomp_fish(self, e: EnemyRec, idx: int) -> None:
        """The fish family under Mario's feet.  The spouting one only dies while
        it is still above the floor -- down there it is inside its pipe."""
        hk = self.hooks
        if e.tp == TP_FLY_FISH or e.y_pos + H < self.world.rows * H:
            self._kill(idx)
            hk.beep(800)
            self.cd_enemy = 1

    def _stomp_koopa(self, e: EnemyRec) -> None:
        """A Koopa, running shell or all, stops dead and goes to sleep."""
        hk = self.hooks
        e.tp = TP_SLEEPING_KOOPA
        e.x_vel = 0
        e.counter = 0
        hk.beep(800)
        self.cd_enemy = 1
        hk.add_score(100)

    def _stomp_winged(self, e: EnemyRec) -> None:
        """Any winged form loses its wings under Mario's feet and walks on.

        The Chibibo, the Buzzy (still fireproof afterwards), the three Koopa
        colours, and -- only under the star -- the black Spiny: a walking black
        Spiny is untouched by the star, just as a plain Spiny can never be jumped
        on at all.  The ground form, its colour and its delay all come out of
        ``_WINGED_FORM``.
        """
        hk = self.hooks
        hk.beep(800)
        self.cd_enemy = 1
        _de_wing(e)
        hk.add_score(100)

    def _touch_side(self, e: EnemyRec, idx: int) -> None:
        """Side contact hurts, except for the things the source keeps harmless:
        a fish mid-spout, the lifts, the bomb stand, a collapsed skeleton in any
        stage (SMB3: the pile never hurts, the walker does) and the rideable
        platforms.  Under the star it also kills, and black creatures are the
        exception there too.
        """
        self.cd_hit = 1
        if self.star and not _is_black_creature(e):
            self._kill(idx)

    def _stomp_dry_bones(self, e: EnemyRec) -> None:
        """The user's rule: a stomp only collapses the pile, and it gets up again
        ``DRY_REVIVE_FRAMES`` later -- for ever."""
        hk = self.hooks
        _collapse_pile(e)
        hk.beep(800)
        self.cd_enemy = 1
        hk.add_score(100)

    def _stomp_defeated(self, e: EnemyRec, dead: int, score: int) -> None:
        """The two flyers that have a dedicated *dead* (not "defeated") pose under
        a stomp: ``sub_tp`` 0 picks that pose, and the corpse rides the same
        light upward pop as the Hammer Brother's other death."""
        hk = self.hooks
        e.tp = dead
        e.sub_tp = 0
        e.x_vel = _corpse_dir(e)
        e.fly_vy = HBRO_DEFEATED_POP
        e.fly_yrem = 0
        hk.beep(800)
        self.cd_enemy = 1
        hk.add_score(score)

    def _stomp_bomb(self, e: EnemyRec, idx: int) -> None:
        """A bomb that has not left its stand yet cannot be stomped (XP's
        initialOrder).  Mario landing on the stand carries up to MAX_Y_VEL = 8 px
        of fall, and the collision test projects his feet by that much, so no
        launch offset can keep him out of the fresh bomb's box -- the guard has to
        be positional."""
        hk = self.hooks
        if not self._in_muzzle(e):
            self._kill(idx)
            hk.beep(800)
            self.cd_enemy = 1

    def _ride_bomb_stand(self, e: EnemyRec, pvel_y: int) -> None:
        """The bomb stand is a mechanism, not a monster: you ride it the way you
        ride a block lift, star on or off."""
        top = e.y_pos + NEW_BOX[e.tp][1]
        self.cd_stop_jump = int(pvel_y != 2)
        self.cd_lift = 1
        self.player_y1 = top - 2 * H
        self.player_y2 = top - 1
        self.player_xvel = 0
        self.player_yvel = e.y_vel

    def _ride_platform(self, e: EnemyRec, pvel_y: int) -> None:
        """One-way top, like the block lifts: snap Mario's feet to the board and
        inherit this frame's whole-pixel step."""
        top = e.y_pos
        self.cd_stop_jump = int(pvel_y != 2)
        self.cd_lift = 1
        self.player_y1 = top - 2 * H
        self.player_y2 = top - 1
        self.player_xvel = e.x_vel
        self.player_yvel = e.y_vel
        if e.tp == TP_PLATFORM_FALL:
            e.plat_state = 1     # SMB3 wooden platform: stand -> fall

    def _ride_lift(self, e: EnemyRec, pvel_y: int) -> None:
        """Block lifts and donut boards: the donut starts crumbling under a rider,
        and a lift mid-patrol hands Mario a sub-pixel step instead of a whole one."""
        if e.tp == TP_DONUT:
            e.status = 2
            if (e.counter > 20) and (e.y_vel == 0):
                e.y_vel += 1
        self.cd_stop_jump = int(pvel_y != 2)
        self.cd_lift = 1
        self.player_y1 = e.y_pos - 2 * H
        self.player_y2 = e.y_pos - 1
        self.player_xvel = e.x_vel
        if e.move_delay != 0:
            self.player_xvel = (e.x_vel * e.x_pos) % 2
        self.player_yvel = e.y_vel

    def _stomp_chibibo(self, e: EnemyRec, ghost: bool) -> None:
        """Chibibo and its ghost cousin flatten the same way; the ghost's death
        also releases its shadow (the user's rule: every death path leaves one,
        exactly like XP's has_undead goomba)."""
        hk = self.hooks
        e.tp = TP_GHOST_FLAT if ghost else TP_FLAT_CHIBIBO
        e.x_vel = 0
        e.delay_counter = -2 - 15 * int(e.y_vel == 0)
        hk.beep(800)
        self.cd_enemy = 1
        hk.add_score(100)
        if ghost:
            self._release_shade(e)

    def _run_type_clock(self, e: EnemyRec, hk) -> None:
        """What each family does on its own once MoveDelay has run out.

        These stay as separate ``if``s rather than an elif-chain: the sleeping
        Koopa sets tp = TP_WAKING_KOOPA here and the original then runs the
        waking step in the SAME frame.
        """
        if e.tp in _IN_PIPE and not (e.tp == TP_VERT_FISH
                                     and self._in_water(e)):
            self._move_from_pipe(e, hk)

        if e.tp == TP_SLEEPING_KOOPA:
            e.counter += 1
            if e.counter > 150:
                e.tp = TP_WAKING_KOOPA
                e.x_vel = 1
                e.counter = 0

        if e.tp in (TP_DRY_BONES, TP_DRY_BONES_FALL):
            self._move_dry_bones(e)

        if e.tp == TP_WAKING_KOOPA:
            # it turns on the spot until it has counted itself awake
            e.x_vel = -e.x_vel
            e.move_delay = 1
            e.delay_counter = 0
            e.counter += 1
            if e.counter > 50:
                e.tp = TP_KOOPA
                e.x_vel = 1 if self.player_x1 > e.x_pos else -1

        if (e.x_pos + W > self.view_x
                and e.x_pos < self.view_x + NH * W):
            e.seen = 1

    def _leaves_scene(self, e: EnemyRec, new_x: int) -> bool:
        """Is this record out of the stage for good?

        ENEMIES.PAS:983-990 lists four bounds and one special type, with no
        other condition: a flattened Chibibo is simply gone wherever it
        happens to be, and the ghost shade is exempt from every one of them --
        even falling through the bottom of the world leaves it alive, because
        nothing is allowed to recycle its pool slot (see the ghost-Chibibo note
        in ENEMY_SPEC).  Neither sideways rule applies inside a protected room.

        ``Seen`` is NOT one of the original's conditions (EnemyRec has no such
        field), so it is not one here either -- except for the block-lift
        family: StartEnemies() is fed exactly the frontier column
        ``XView div W + NH + StartEnemiesAt``, which with StartEnemiesAt ==
        ForgetEnemiesAt == 7 is the right forget boundary, so a lift spawned
        there and stepping right is past it on its very first frame; the
        marker is already consumed and the frontier never returns, so the lift
        would be lost forever.  That one family keeps the grace; every other
        enemy is dropped by the boundary as written.
        """
        if e.tp in (TP_FLAT_CHIBIBO, TP_GHOST_FLAT):
            return True
        if e.tp == TP_GHOST_SHADE:
            return False
        if new_x <= -W:
            return True
        protected = self._in_protected_room(new_x)
        if (not protected) and (new_x < self.view_x - self.forget_at * W):
            return True
        if (not protected
                and new_x > self.view_x + NH * W + self.forget_at * W
                and (e.seen or not _is_lift(e.tp))):
            return True
        return e.y_pos + e.y_vel > self.world.rows * H

    def _forget(self, e: EnemyRec) -> None:
        """Drop a record that left: put its map marker back and start the death
        animation, which is what frees the pool slot."""
        self._restore_marker(e)
        if e.tp == TP_KOOPA:
            e.tp = TP_DYING_KOOPA
        elif e.tp != TP_FIREBALL:
            e.tp = TP_DYING
        else:
            e.tp = TP_DYING_FIREBALL
        e.delay_counter = -(MAX_PAGE + 1)

    def _advance_mover(self, e: EnemyRec, idx: int) -> None:
        """The normal frame for a record that is staying: gravity, tile
        physics, then one whole step.

        The thrown bodies (GRAV_THROWN) are the only ones that skip ``_check``:
        a corpse has no tile physics, it just keeps falling until the world
        bottom frees it.  The pipe spouts get their gravity AND the tile check,
        exactly as they did inline.
        """
        e.delay_counter = 0
        old_x_vel = e.x_vel
        grav = _gravity(e.tp)
        if grav == GRAV_PIPE:
            if (not (e.tp == TP_VERT_FISH and self._in_water(e))
                    and e.dir_counter % 3 == 0
                    and e.y_pos + H < self.world.rows * H):
                e.y_vel += 1
        elif grav == GRAV_THROWN:
            if e.x_pos % 6 == 0:
                e.y_vel += 1
        if grav != GRAV_THROWN:
            self._check(idx)
        e.x_pos += e.x_vel
        e.y_pos += e.y_vel
        if e.x_vel == 0:
            e.x_vel = -old_x_vel
            if e.tp == TP_DYING_FIREBALL:
                self._show_fire(e.x_pos, e.y_pos)

    def move_enemies(self) -> None:
        hk = self.hooks
        self.time_counter = (self.time_counter + 1) & 0xFF
        # derive the water status from the box each frame instead of trusting
        # the previous MovePlayer (contact must be right on the entry frame)
        self.player_in_water = self.options.in_water_box(
            self.player_y1, self.player_y2 - self.player_y1 + 1)
        self._tick_fish()
        self._tick_lakitu()
        self._tick_platforms()
        if self.player_xvel:
            self.player_face = 1 if self.player_xvel > 0 else -1

        # snapshot: _check() can call new_enemy(), which appends to
        # self.active -- the original's `for i := 1 to NumEnemies` evaluates
        # the bound once, so new entries are not visited this frame.
        for idx in list(self.active):
            e = self.pool[idx]
            if _is_platform(e.tp):
                continue        # _tick_platforms() already ran this frame
            if self._sleeping(e):
                # Nothing runs for a dormant enemy -- not even delay_counter,
                # whose LastPos interpolation would otherwise slide it down to
                # the edge of the band.
                continue
            e.delay_counter += 1
            new_x = e.x_pos + e.x_vel
            if e.delay_counter > e.move_delay:
                e.x_pos = e.last_x_pos
                e.y_pos = e.last_y_pos
                if self._water_slow(e):
                    e.water_skip = (e.water_skip + 1) % 3
                    if e.water_skip == 2:
                        # drop one whole movement cycle in three
                        e.delay_counter = 0
                        e.last_x_pos = e.x_pos
                        e.last_y_pos = e.y_pos
                        continue
                e.dir_counter = (e.dir_counter + 1) & 0xFF

                self._run_type_clock(e, hk)

                if e.tp in _DYING_STAGES:
                    e.tp = TP_DEAD
                elif self._leaves_scene(e, new_x):
                    self._forget(e)
                else:
                    self._advance_mover(e, idx)
                e.last_x_pos = e.x_pos
                e.last_y_pos = e.y_pos
            elif (e.x_vel != 0) or (e.y_vel != 0):
                # still waiting for MoveDelay: interpolate from the last
                # whole-step position
                e.x_pos = e.last_x_pos + idiv(e.delay_counter * e.x_vel,
                                              e.move_delay + 1)
                e.y_pos = e.last_y_pos + idiv(e.delay_counter * e.y_vel,
                                              e.move_delay + 1)

        # --- player collision (second loop in MoveEnemies) --------------- #
        px1, px2 = self.player_x1, self.player_x2
        py1, py2 = self.player_y1, self.player_y2
        pvel_y = self.player_yvel
        for idx in list(self.active):
            e = self.pool[idx]
            if not _touched(e.tp):
                continue
            if self._hidden(e):
                # drawn as nothing: no hit box at all (the user's rule)
                continue
            hx, hy, hw, hh = self._hit_box(e)
            if not ((px1 < hx + hw) and (px2 > hx)
                    and (py1 + pvel_y < hy + hh)
                    and (py2 + pvel_y > hy)):
                continue

            if (self.star and not _is_lift(e.tp)
                    and not _is_platform(e.tp)
                    and e.tp not in _IMMORTAL
                    and not _is_black_creature(e)
                    # a bomb still inside its stand is untouchable -- the star
                    # standing on the muzzle must not eat every shot
                    and not (e.tp == TP_BOMB and self._in_muzzle(e))):
                # Touch-killing works on everything except lifts, the bomb
                # stand and the black creatures.
                hk.beep(800)
                self._kill(idx)
                self.cd_hit = 1

            if self.player_in_water:
                # no stomp underwater: any contact with a monster hurts
                # (user's rule).  Power-ups, rideable mechanisms and the inert
                # Dry Bones pile keep their own harmless behaviour.
                if e.tp in (TP_CHAMP, TP_LIFE, TP_FLOWER, TP_STAR):
                    self._collect_item(e)
                elif e.tp == TP_VERT_FIREBALL:
                    self.cd_hit = 1
                elif e.tp == TP_BOMB_STAND:
                    self._ride_bomb_stand(e, pvel_y)
                elif _is_platform(e.tp):
                    self._ride_platform(e, pvel_y)
                elif _is_lift(e.tp):
                    self._ride_lift(e, pvel_y)
                elif e.tp == TP_DRY_BONES_FALL:
                    pass
                else:
                    self._touch_side(e, idx)
                continue

            if e.tp in (TP_SLEEPING_KOOPA, TP_WAKING_KOOPA):
                self._touch_shell(e)
            elif e.tp in (TP_CHAMP, TP_LIFE, TP_FLOWER, TP_STAR):
                self._collect_item(e)
            elif e.tp == TP_VERT_FIREBALL:
                self.cd_hit = 1
            elif ((pvel_y > e.y_vel) or (pvel_y > 0)) and (py2 <= hy + hh):
                # landed on top of it
                if e.tp in (TP_CHIBIBO, TP_GHOST_CHIBIBO):
                    self._stomp_chibibo(e, e.tp == TP_GHOST_CHIBIBO)
                elif e.tp in (TP_VERT_FISH, TP_FLY_FISH):
                    self._stomp_fish(e, idx)
                elif e.tp in (TP_KOOPA, TP_RUNNING_KOOPA):
                    self._stomp_koopa(e)
                elif e.tp in (TP_FLY_CHIBIBO, TP_FLY_BUZZY,
                              TP_FLY_GREEN, TP_FLY_RED, TP_FLY_BLACK):
                    self._stomp_winged(e)
                elif self.star and e.tp == TP_FLY_SPINY_BLACK:
                    # Under the star Mario can bounce off a winged black
                    # Spiny: it loses its wings and becomes a walking one.
                    # A walking black Spiny is untouched by the star, just
                    # like a plain Spiny cannot be jumped on at all.
                    self._stomp_winged(e)
                elif e.tp == TP_DRY_BONES:
                    self._stomp_dry_bones(e)
                elif e.tp == TP_HAMMER_BRO:
                    self._stomp_defeated(e, TP_DEAD_HAMMER_BRO, 200)
                elif e.tp == TP_LAKITU:
                    self._stomp_defeated(e, TP_DEAD_LAKITU, 400)
                elif e.tp == TP_BOMB:
                    self._stomp_bomb(e, idx)
                elif e.tp == TP_DRY_BONES_FALL:
                    # SMB3: inert while Var5 <> 0 -- touching the bones does
                    # nothing at all, it cannot even be stomped
                    pass
                elif e.tp in (TP_BOO, TP_HAMMER, TP_THWOMP, TP_PLANT_FIRE,
                              TP_GHOST_SHADE):
                    # No stomp branch for these: touching them from above hurts
                    # just like touching them sideways.  The stone shield is a
                    # threat, not a platform (the user's rule, 2026-10-23).
                    self.cd_hit = 1
                elif e.tp == TP_BOMB_STAND:
                    self._ride_bomb_stand(e, pvel_y)
                elif _is_platform(e.tp):
                    self._ride_platform(e, pvel_y)
                elif _is_lift(e.tp):
                    self._ride_lift(e, pvel_y)
            elif not (((e.tp == TP_VERT_FISH)
                       and (abs(e.delay_counter - e.move_delay) > 1))
                      or _is_lift(e.tp)
                      # the bomb stand is scenery: touching it from the side is
                      # as harmless as standing on it (user 2026-10-23)
                      or e.tp == TP_BOMB_STAND
                      # ... and so is a collapsed skeleton, in every stage
                      # (SMB3: the pile never hurts, the walker does)
                      or e.tp == TP_DRY_BONES_FALL
                      # platforms are rideable scenery, not monsters
                      or _is_platform(e.tp)):
                self._touch_side(e, idx)

        # drop the dead entries (the original does this with a Delete loop)
        i = 0
        while i < len(self.active):
            if self.pool[self.active[i]].tp == TP_DEAD:
                del self.active[i]
            else:
                i += 1

    # ------------------------------------------------------------------ #
    # spawning (StartEnemies) and block bumps (HitAbove)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Long platforms (JSON ``platforms`` list)
    # ------------------------------------------------------------------ #
    def set_platforms(self, platforms: list[dict] | None) -> None:
        self.platforms = list(platforms or [])

    def start_platforms(self) -> None:
        """Create one object per level ``platforms`` entry (after a restart)."""
        self._plat_forgotten.clear()
        for key, spec in enumerate(self.platforms):
            if isinstance(spec, dict):
                self._spawn_platform(spec, key)

    def _respawn_platforms(self, col: int) -> None:
        """Bring back a forgotten board when the spawn frontier reaches it."""
        for key in sorted(self._plat_forgotten):
            spec = self.platforms[key]
            if not isinstance(spec, dict):
                continue
            if int(spec.get("x", 0)) == col:
                self._plat_forgotten.discard(key)
                self._spawn_platform(spec, key)

    def _spawn_platform(self, spec: dict, key: int = -1) -> None:
        kind = {
            "patrol": TP_PLATFORM_PATROL,
            "osc": TP_PLATFORM_OSC,
            "fall": TP_PLATFORM_FALL,
            "path": TP_PLATFORM_PATH,
        }.get(str(spec.get("type", "patrol")).lower(), TP_PLATFORM_PATROL)
        col = int(spec.get("x", 0))
        src_row = int(spec.get("y", 0))
        rows = self.world.rows
        if not (0 <= col < self.options.XSize):
            return
        world_row = rows - 1 - src_row
        n = max(1, min(int(spec.get("len", 3)), self.options.XSize - col))
        variant = 1 if int(spec.get("variant", 0)) else 0

        idx = self.new_enemy(kind, variant, col, world_row, 0, 0, 0)
        if idx is None:
            return
        e = self.pool[idx]
        e.plat_spec = key
        e.plat_len = n
        e.plat_kind = kind
        e.plat_home_x = e.x_pos
        e.plat_home_y = e.y_pos
        e.plat_axis = 0 if str(spec.get("axis", "h")).lower().startswith("h") \
            else 1
        default_speed = {
            TP_PLATFORM_PATROL: 0.8,
            TP_PLATFORM_OSC: 1.0,
            TP_PLATFORM_FALL: 0.0625,      # SMB3: +1/16 px/f per frame
            TP_PLATFORM_PATH: 0.8,
        }[kind]
        speed = float(spec.get("speed", default_speed))
        e.plat_speed = max(1, int(round(speed * PLATFORM_SPEED_DEN)))
        default_range = 80 if kind == TP_PLATFORM_PATROL else 0
        e.plat_range = max(0, int(spec.get("range", default_range)))
        e.plat_loop = bool(spec.get("loop", False))
        e.plat_path = []
        for point in spec.get("path") or []:
            if len(point) >= 2:
                px = int(point[0]) * W
                py = (rows - 1 - int(point[1])) * H
                e.plat_path.append((px, py))
        e.plat_idx = 0
        e.plat_vx = e.plat_vy = 0
        e.plat_xrem = e.plat_yrem = 0
        e.plat_dir = 1
        e.plat_hold = 0
        e.plat_state = 0
        e.plat_parity = 0
        e.x_vel = e.y_vel = 0

    def _forget_platform(self, idx: int, e: "EnemyRec") -> None:
        """Drop the board like a forgotten enemy: it leaves the pool and only
        reappears once the spawn frontier crosses its column again."""
        if e.plat_spec >= 0:
            self._plat_forgotten.add(e.plat_spec)
        e.plat_spec = -1
        e.tp = TP_DEAD
        if idx in self.active:
            self.active.remove(idx)

    @staticmethod
    def _plat_step(e: "EnemyRec", vx: int, vy: int) -> None:
        """Integrate 1/1024 px/f velocities; XVel/YVel carry the whole-pixel
        step to the player's cd_lift branch."""
        e.plat_xrem += vx
        sx = e.plat_xrem // PLATFORM_SPEED_DEN
        e.plat_xrem -= sx * PLATFORM_SPEED_DEN
        e.plat_yrem += vy
        sy = e.plat_yrem // PLATFORM_SPEED_DEN
        e.plat_yrem -= sy * PLATFORM_SPEED_DEN
        e.x_pos += sx
        e.y_pos += sy
        e.x_vel = sx
        e.y_vel = sy

    def _tick_platforms(self) -> None:
        """Move every platform.  A board that gets past the enemy-style forget
        margin (one screen + 7 tiles) is dropped, not teleported back: it
        returns only when the spawn frontier crosses its column again."""
        vx0, vy0 = self.view_x, self.view_y
        right = vx0 + NH * W + PLATFORM_RESET_MARGIN_X
        bottom = vy0 + VIR_SCREEN_HEIGHT + PLATFORM_RESET_MARGIN_Y
        for idx in list(self.active):
            e = self.pool[idx]
            if not _is_platform(e.tp):
                continue
            hw = max(1, e.plat_len) * PLATFORM_SEG_W
            if (e.x_pos + hw < vx0 - PLATFORM_RESET_MARGIN_X
                    or e.x_pos > right
                    or e.y_pos + PLATFORM_SEG_H < vy0 - PLATFORM_RESET_MARGIN_Y
                    or e.y_pos > bottom):
                self._forget_platform(idx, e)
                continue
            if self._in_water(e):
                e.water_skip = (e.water_skip + 1) % 3
                if e.water_skip == 0:
                    continue

            e.x_vel = e.y_vel = 0
            kind = e.plat_kind

            if kind == TP_PLATFORM_PATROL:
                v = e.plat_speed * e.plat_dir
                self._plat_step(e, v if e.plat_axis == 0 else 0,
                                v if e.plat_axis == 1 else 0)
                if e.plat_range > 0:
                    if e.plat_axis == 0:
                        off = e.x_pos - e.plat_home_x
                        if off > e.plat_range:
                            e.x_pos = e.plat_home_x + e.plat_range
                            e.plat_dir = -1
                            e.plat_xrem = 0
                        elif off < -e.plat_range:
                            e.x_pos = e.plat_home_x - e.plat_range
                            e.plat_dir = 1
                            e.plat_xrem = 0
                    else:
                        off = e.y_pos - e.plat_home_y
                        if off > e.plat_range:
                            e.y_pos = e.plat_home_y + e.plat_range
                            e.plat_dir = -1
                            e.plat_yrem = 0
                        elif off < -e.plat_range:
                            e.y_pos = e.plat_home_y - e.plat_range
                            e.plat_dir = 1
                            e.plat_yrem = 0

            elif kind == TP_PLATFORM_OSC:
                peak = max(1, e.plat_speed)
                if e.plat_hold > 0:
                    e.plat_hold -= 1
                else:
                    e.plat_parity ^= 1
                    if e.plat_parity == 0:
                        e.plat_vx += e.plat_dir * PLATFORM_OSC_ACCEL
                if e.plat_dir > 0 and e.plat_vx > peak:
                    e.plat_vx = peak
                elif e.plat_dir < 0 and e.plat_vx < -peak:
                    e.plat_vx = -peak
                if e.plat_vx >= peak and e.plat_dir > 0:
                    e.plat_vx = peak
                    e.plat_dir = -1
                    e.plat_hold = PLATFORM_OSC_HOLD
                elif e.plat_vx <= -peak and e.plat_dir < 0:
                    e.plat_vx = -peak
                    e.plat_dir = 1
                    e.plat_hold = PLATFORM_OSC_HOLD
                self._plat_step(e, e.plat_vx if e.plat_axis == 0 else 0,
                                e.plat_vx if e.plat_axis == 1 else 0)
                if e.plat_range > 0:
                    if e.plat_axis == 0:
                        off = e.x_pos - e.plat_home_x
                        if abs(off) > e.plat_range:
                            e.x_pos = e.plat_home_x + (
                                e.plat_range if off > 0 else -e.plat_range)
                            e.plat_vx = 0
                            e.plat_xrem = 0
                            e.plat_dir = -1 if off > 0 else 1
                            e.plat_hold = PLATFORM_OSC_HOLD
                    else:
                        off = e.y_pos - e.plat_home_y
                        if abs(off) > e.plat_range:
                            e.y_pos = e.plat_home_y + (
                                e.plat_range if off > 0 else -e.plat_range)
                            e.plat_vx = 0
                            e.plat_yrem = 0
                            e.plat_dir = -1 if off > 0 else 1
                            e.plat_hold = PLATFORM_OSC_HOLD

            elif kind == TP_PLATFORM_FALL:
                if e.plat_state:
                    e.plat_vy = min(PLATFORM_FALL_CAP,
                                    e.plat_vy + e.plat_speed)
                    self._plat_step(e, 0, e.plat_vy)

            elif kind == TP_PLATFORM_PATH:
                if e.plat_idx < len(e.plat_path):
                    tx, ty = e.plat_path[e.plat_idx]
                    dx, dy = tx - e.x_pos, ty - e.y_pos
                    dist = math.hypot(dx, dy)
                    step_px = e.plat_speed / PLATFORM_SPEED_DEN
                    if dist <= step_px:
                        e.x_pos, e.y_pos = tx, ty
                        e.plat_xrem = e.plat_yrem = 0
                        e.plat_idx += 1
                        if e.plat_loop and e.plat_idx >= len(e.plat_path):
                            e.plat_idx = 0
                    elif dist > 0:
                        vx = int(dx / dist * e.plat_speed)
                        vy = int(dy / dist * e.plat_speed)
                        self._plat_step(e, vx, vy)

    def set_direction_overrides(self,
                                overrides: dict[tuple[int, int], int]) -> None:
        """Per-marker spawn direction from level JSON ``enemy_dirs``."""
        self.dir_overrides = dict(overrides or {})

    def start_enemies(self, x: int, direction: int) -> None:
        """Convert the enemy markers of map column ``x`` into enemies and
        clear the markers; ``direction`` is the initial walking direction.
        A per-marker override (level JSON ``enemy_dirs``) wins over it."""
        if (x < 0) or (x > self.options.XSize):
            return
        world = self.world
        rng = self.rng
        self._respawn_platforms(x)
        for i in range(self.world.rows):
            d = self.dir_overrides.get((x, i), direction)
            remove = True
            ch = world.get(x, i)
            if ch == CH_CHIBIBO:
                self.new_enemy(TP_CHIBIBO, 0, x, i, 1 * d, 0, 2)
            elif ch == CH_VERT_FISH:
                if self.options.in_water_row(i):
                    # a fish placed below the waterline patrols horizontally
                    # at Koopa speed; the spout arc is for dry levels
                    self.new_enemy(TP_VERT_FISH, 0, x, i, 1 * d, 0, 2)
                else:
                    self.new_enemy(TP_VERT_FISH, 0, x, i + 2, 0, 0,
                                   50 + rng.randrange(100))
            elif ch == CH_VERT_FIREBALL:
                self.new_enemy(TP_VERT_FIREBALL, 0, x, i + 2, 0, 0,
                               50 + rng.randrange(100))
            elif ch == CH_CHIBIBO_B:
                self.new_enemy(TP_CHIBIBO, 1, x, i, 1 * d, 0, 2)
            elif ch == CH_GHOST_CHIBIBO:
                # the ghost Chibibo: same walker, same speed, black paint
                self.new_enemy(TP_GHOST_CHIBIBO, 0, x, i, 1 * d, 0, 2)
            elif ch in (CH_PLANT0, CH_PLANT1, CH_PLANT2):
                self.new_enemy(TP_VERT_PLANT, ch - CH_PLANT0, x, i + 2, 0, 0,
                               20 + rng.randrange(50))
            elif ch == CH_FIRE_PLANT:
                # the red-green plant that spits; same rise cycle, own marker
                self.new_enemy(TP_FIRE_PLANT, 2, x, i + 2, 0, 0,
                               20 + rng.randrange(50))
            elif ch == CH_FIRE_PLANT_AIM:
                self.new_enemy(TP_FIRE_PLANT_AIM, 2, x, i + 2, 0, 0,
                               20 + rng.randrange(50))
            elif ch == CH_FAST_PLANT:
                # XP's red piranha: 1.25 s out, 1.25 s back
                self.new_enemy(TP_FAST_PLANT, 2, x, i + 2, 0, 0, 2)
            elif ch == CH_SIDE_PLANT:
                # reaches sideways; the editor's per-marker direction picks
                # which side (default: toward the player's start)
                self.new_enemy(TP_SIDE_PLANT, 2, x, i, 1 * d, 0, 2)
            elif ch == CH_BLACK_PLANT:
                # black creature, deliberately the plain plant's own cadence
                self.new_enemy(TP_BLACK_PLANT, 2, x, i + 2, 0, 0,
                               20 + rng.randrange(50))
            elif ch == CH_RED:
                self.new_enemy(TP_RED, 0, x, i, 1 * d, 0, 2)
            elif ch in (CH_KOOPA0, CH_KOOPA1, CH_KOOPA2):
                self.new_enemy(TP_KOOPA, ch - CH_KOOPA0, x, i, d, 0, 2)
            elif ch == CH_BUZZY:
                # move_delay 3 (turtle 2): the Buzzy walks slightly slower
                self.new_enemy(TP_KOOPA, K_BUZZY, x, i, d, 0, 3)
            elif ch == CH_FLY_GREEN:
                self.new_enemy(TP_FLY_GREEN, 0, x, i, 1 * d, 0, 0)
            elif ch == CH_FLY_RED_H:
                self.new_enemy(TP_FLY_RED, 0, x, i, 1 * d, 0, 0)
            elif ch == CH_FLY_RED_V:
                idx = self.new_enemy(TP_FLY_RED, 1, x, i, 0, 0, 0)
                if idx is not None and (x, i) in self.dir_overrides:
                    # d = +1 starts upward, -1 starts downward
                    self.pool[idx].fly_vdir = -1 if d >= 0 else 1
            elif ch == CH_FLY_BLACK:
                self.new_enemy(TP_FLY_BLACK, 0, x, i, 1 * d, 0, 0)
            elif ch == CH_SPINY_BLACK:
                self.new_enemy(TP_RED, K_BLACK, x, i, 1 * d, 0, 2)
            elif ch == CH_FLY_SPINY:
                self.new_enemy(TP_FLY_SPINY, 0, x, i, 1 * d, 0, 0)
            elif ch == CH_FLY_SPINY_BLACK:
                self.new_enemy(TP_FLY_SPINY_BLACK, 0, x, i, 1 * d, 0, 0)
            elif ch == CH_FLY_CHIBIBO:
                self.new_enemy(TP_FLY_CHIBIBO, 0, x, i, 1 * d, 0, 0)
            elif ch == CH_FLY_BUZZY:
                self.new_enemy(TP_FLY_BUZZY, K_BUZZY, x, i,
                               1 * d, 0, 0)
            elif ch == CH_DRY_BONES:
                # 0.5 px/f, SMB3's $08 (see DRY_WALK_DELAY)
                self.new_enemy(TP_DRY_BONES, 0, x, i, 1 * d, 0,
                               DRY_WALK_DELAY)
            elif ch == CH_HAMMER_BRO:
                # XP's obj_hammer_bro starts by patrolling left
                self.new_enemy(TP_HAMMER_BRO, 0, x, i,
                               d if (x, i) in self.dir_overrides else -1, 0, 0)
            elif ch == CH_BOO:
                self.new_enemy(TP_BOO, 0, x, i,
                               d if (x, i) in self.dir_overrides else 1, 0, 0)
            elif ch == CH_BOMB_STAND:
                self.new_enemy(TP_BOMB_STAND, 0, x, i, 0, 0, 0)
            elif ch == CH_THWOMP:
                self.new_enemy(TP_THWOMP, 0, x, i, 0, 0, 0)
            # CH_LAKITU is deliberately not consumed: like the fish spout it is
            # a permanent switch read by _tick_lakitu().
            elif ch == CH_BLOCK_LIFT:
                if ((world.get(x - 1, i) in CAN_HOLD_YOU)
                        or (world.get(x + 1, i) in CAN_HOLD_YOU)):
                    self.new_enemy(TP_BLOCK_LIFT, 0, x, i, -direction, 0, 0)
                else:
                    self.new_enemy(TP_BLOCK_LIFT, 0, x, i, 0, -direction, 0)
            elif ch == CH_DONUT:
                self.new_enemy(TP_DONUT, 0, x, i, 0, 0, 0)
            else:
                remove = False
            if remove:
                world.set(x, i, ord(" "))

    def hit_above(self, map_x: int, map_y: int) -> None:
        """HitAbove(MapX, MapY): a block was bumped from below — bounce
        whatever is standing on it, and kill walking enemies."""
        y = map_y * H
        x = map_x * W
        for idx in self.active:
            e = self.pool[idx]
            if e.y_pos != y:
                continue
            if not ((e.x_pos + e.x_vel + W > x) and (e.x_pos + e.x_vel < x + W)):
                continue
            if (e.tp in (TP_CHAMP, TP_LIFE, TP_FLOWER, TP_STAR)
                    or TP_KOOPA <= e.tp <= TP_WAKING_KOOPA):
                if (((e.x_vel > 0) and (e.x_pos + e.x_vel + W // 2 <= x))
                        or ((e.x_vel < 0)
                            and (e.x_pos + e.x_vel + W // 2 >= x))):
                    e.x_vel = -e.x_vel
                e.y_vel = -7
                e.status = FALLING
                if TP_KOOPA <= e.tp <= TP_WAKING_KOOPA:
                    e.tp = TP_SLEEPING_KOOPA
                    e.x_vel = 0
            elif e.tp in _WINGED_FORM:
                # the user's rule (2026-10-24): a bumped block knocks the wings
                # off a flyer -- what is left walks on as its ground form
                _de_wing(e)
            elif e.tp in (TP_CHIBIBO, TP_RED, TP_GHOST_CHIBIBO):
                # a black creature on a bumped block survives (user's rule);
                # the black Koopa already went to sleep in the branch above
                if not _is_black_creature(e):
                    self._kill(idx)
            elif e.tp == TP_DRY_BONES:
                # the user's rule (2026-10-24): a bumped block shakes the
                # skeleton apart just like a stomp does -- it reassembles.
                _collapse_pile(e)
                self.hooks.add_score(100)
            elif e.tp == TP_HAMMER_BRO:
                # ... but it kills a Hammer Bro outright.
                self._kill(idx)
