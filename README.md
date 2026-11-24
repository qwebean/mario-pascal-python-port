# Mario & Luigi（Python 移植）

DOS 游戏的 Python + pygame 移植。

```bash
pip install -r requirements.txt
python -m mario_py.game.mario
```

游戏里 `START → MOD LEVELS` 可切换关卡包；也可命令行指定：

```bash
python -m mario_py.game.mario --pack mario4 --level 1-1
python -m mario_py.game.mario --level testrooms/8-5a
```

关卡来源见 `关卡来源.md`，编辑器用法见 `关卡编辑器手册.md`，冒烟测试 `python tests/smoke.py`。

原版游戏与源码是 Mike Wiering 的 freeware，不得出售；本移植版仅供学习与免费分享。
