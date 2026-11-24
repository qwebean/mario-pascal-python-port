# Testing / 验证

All commands assume the repository root as the working directory and the Python
interpreter with `requirements.txt` installed.

## Headless smoke test

```bash
python tests/smoke.py
```

Loads all converted data, checks all five bundled level packs, builds the world +
figures for representative levels (main line, `mario4`, `testrooms`, water and vision
levels), and re-renders the first page. It exits non-zero on any failure.

On a machine without a display, set the SDL dummy driver first:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python tests/smoke.py
```

Windows PowerShell:

```powershell
$env:SDL_VIDEODRIVER='dummy'; $env:SDL_AUDIODRIVER='dummy'; python tests/smoke.py
```

## Level editor self-test

```bash
SDL_VIDEODRIVER=dummy python -m mario_py.tools.level_editor --selftest
```

Covers paint, platform, options, pipe links, enemy directions, level size, marker art
and undo/redo.

## Manual playtest

```bash
python -m mario_py.game.mario
python -m mario_py.game.mario --pack mario4 --level 1-1
python -m mario_py.game.mario --level testrooms/8-5a
```

Useful debug switches are documented in the in-game help; the port starts with no time
limit, so the TIME cell is intentionally left blank.

## Rendering a still

```bash
python -m mario_py.tools.render_level 1-1a --out preview.png
```
