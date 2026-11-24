# Mario & Luigi — Python port

A Python + pygame port of **Mike Wiering's DOS game *Mario & Luigi*** (Turbo Pascal 5.5,
VGA mode 13h). The main line follows Spenser Whaley's 2006 23-level mod; several older
builds are playable as separate level packs. No original executable or Pascal source is
included — only converted runtime data and this port's code.

Chinese notes: [关卡来源.md](关卡来源.md) describes every level pack, and
[关卡编辑器手册.md](关卡编辑器手册.md) documents the level editor.

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m mario_py.game.mario                      # title screen / main game
python -m mario_py.game.mario --level 1-1a         # a single level
python -m mario_py.game.mario --pack mario4 --level 1-1   # 4-level (2004) build
python -m mario_py.game.mario --level testrooms/8-5a      # port-only test level
```

In game: `START → MOD LEVELS` lists the five mod/test packs. Command-line form is
`--level <pack>/<world><a|b>`; `mario4` has no `1-2b` / `1-3b` (those sub-levels do not
exist in the original build).

## Level editor

```bash
python -m mario_py.tools.level_editor --selftest
python -m mario_py.tools.level_editor
```

The editor can edit every bundled level (tiles, options, enemies, pipe links, platform
paths) and supports F5 playtest for registered level names. See the manual for limits.

## Rendering / tests

```bash
python tests/smoke.py
python -m mario_py.tools.render_level 1-1a --out preview.png
```

See [TESTING.md](TESTING.md) for headless invocations and what the smoke checks cover.

## Credits and licence

See [NOTICE.md](NOTICE.md). The original game and source are freeware by Mike Wiering and
must not be sold. This port and its converted data are for study and free sharing only;
please do not use them commercially.
