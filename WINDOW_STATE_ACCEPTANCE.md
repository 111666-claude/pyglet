# 窗口透明渲染 × 鼠标穿透：独立开关与跨平台验收

本扩展把原先捆绑在 `WINDOW_STYLE_OVERLAY` 中的能力拆成三个可独立启用/关闭、
可在运行时切换、并在窗口生命周期内保持一致的状态：

| 状态 | 构造参数 | 运行时方法 | 属性 |
| --- | --- | --- | --- |
| 逐像素透明渲染 | `transparent=True` | `win.set_transparent(bool)` | `win.transparent` |
| 鼠标穿透（click-through） | `mouse_passthrough=True`（别名 `click_through`） | `win.set_mouse_passthrough(bool)` / `win.set_click_through(bool)` | `win.mouse_passthrough` / `win.click_through` |
| 置顶 | `topmost=True` | `win.set_topmost(bool)` | `win.topmost` |

旧样式语义保持不变：

- `style="transparent"` ⇒ `transparent=True`，穿透/置顶关闭；
- `style="overlay"` ⇒ 三者全开；
- 显式传入的构造参数始终覆盖样式推断，因此可以有“不透明但穿透”
  或“透明但可交互”等任意组合。

## 1. 不支持的组合：明确报错

以下情况抛出 `pyglet.window.WindowStateException`（`WindowException` 的子类），
错误信息包含可执行的修复建议，且**不会**改变当前状态、不会调用原生接口：

1. 窗口创建时没有 alpha 帧缓冲位（`win.transparent_supported is False`），
   却在运行时调用 `set_transparent(True)`。
   - 修复：用 `transparent=True` 重新建窗（会自动请求 `alpha_size=8`），
     或传入 `alpha_size >= 8` 的 `config`。
2. X11 显示服务器不提供 SHAPE 扩展，却调用 `set_mouse_passthrough(True)`。

关闭类操作（传 `False`）永远不会因为能力不足而报错。

## 2. 自动化测试（可注入原生调用）

状态切换逻辑与原生调用完全分离：

- `pyglet/window/state.py`
  - `WindowState`：不可变状态快照；
  - `WindowStateController`：唯一状态权威，负责校验、去重、生命周期
    （`realize()` / `invalidate()` / `teardown()`）；
  - `RecordingNativeState`：记录 `(aspect, value)` 的可注入假后端；
  - `Win32NativeState` / `CocoaNativeState` / `X11NativeState`：
    依赖（user32/dwmapi/gdi32、NSWindow、xlib/xshape）全部由构造注入。

无需显示器、在任意操作系统上运行的注入式单元测试：

```bash
python -m pytest tests/unit/window -v
```

覆盖：独立切换、重复调用不产生原生调用、不支持组合的原子失败、
`realize → invalidate → realize`（模拟全屏重建）状态完整恢复、
关闭时下发 all-off 重置、关闭后拒绝再改、以及三平台驱动的精确调用序列。

带真实（不可见）X11 窗口的生命周期集成测试，无显示时自动 skip：

```bash
python -m pytest tests/integration/window/test_window_state.py -v
```

一键验收脚本（见下节）：

```bash
# 自动轮播 不透明 → 透明可交互 → 透明穿透 → 普通窗口，每 5 秒一切换
python examples/window/state_acceptance.py --demo

# 交互模式：T 透明 / M 穿透（6 秒后自动恢复，--sticky 可保持）/
# A 置顶 / F 全屏切换 / Q 退出
python examples/window/state_acceptance.py
```

## 3. 各平台可重复的人工验收步骤

准备：一个能看到桌面/其他窗口的图形会话；在窗口下方放一个可点击的目标
（桌面图标、记事本、浏览器按钮等）。统一运行
`python examples/window/state_acceptance.py --demo`，或按下表手动切换。

每个平台都要验证的通用检查点：

1. 默认状态创建出的普通窗口与旧版本行为一致（无多余原生调用）。
2. 透明阶段：未绘制区域能透出桌面，绘制的圆/半透明面板颜色正常。
3. 穿透阶段：窗口**任意位置**（含已绘制像素）的点击都穿透到下方目标，窗口本身不获得焦点。这是与上游覆盖层一致的整窗穿透模型（基于 alpha 的逐像素命中穿透不在本次范围内）。
4. 关闭穿透后：窗口恢复接收鼠标/键盘输入。
5. 按 `F`（或调用 `set_fullscreen`）来回切换全屏后，三项状态与切换前一致，
   穿透/透明仍然生效。
6. 触发 `_recreate`（全屏路径）后状态被重新应用到新原生句柄。
7. 关闭窗口后：下方窗口交互正常，不存在“看不见但挡鼠标”的残留区域，
   任务栏/Dock/窗口列表无残留。

### 3.1 Windows 10 / 11（Win32 + DWM）

```powershell
py -m pip install -e .
py examples\window\state_acceptance.py --demo
```

原生机制：`WS_EX_LAYERED` + `WS_EX_TRANSPARENT`（顶层窗口与 `_view_hwnd`
子窗口都会设置穿透位）、`DwmEnableBlurBehindWindow`、
`SetLayeredWindowAttributes(..., 255, LWA_ALPHA)`、`HWND_TOPMOST/NOTOPMOST`。

额外检查：

- 用 Spy++ / Win32 属性查看器确认穿透阶段窗口扩展样式含
  `WS_EX_LAYERED (0x80000)` 与 `WS_EX_TRANSPARENT (0x20)`；
  关闭穿透但仍透明时只剩 `WS_EX_LAYERED`；全部关闭后两者都消失。
- 全屏来回切换后重复上述样式检查。
- 关闭脚本后，用鼠标扫过原窗口区域，确认悬停/点击都落在下方窗口。
- 远程桌面/虚拟机中若 DWM 合成被关闭，逐像素透明可能不可用；
  此时应得到 `WindowStateException` 或不透明回退，而不是崩溃。

### 3.2 macOS（Cocoa）

```bash
python3 -m pip install -e .
python3 examples/window/state_acceptance.py --demo
```

原生机制：`setOpaque:NO` + `backgroundColor = clearColor` +
`hasShadow = NO`；穿透为 `setIgnoresMouseEvents:`；置顶为
`setLevel:NSStatusWindowLevel`（恢复 `NSNormalWindowLevel`）。

额外检查：

- 用“窗口检查器”或 Xcode View Debugging 确认对应 NSWindow 的
  `opaque=NO`、`ignoresMouseEvents=YES/NO`、level 随切换变化。
- 穿透阶段窗口不接收点击，下方 App 收到操作；用 `Cmd+Tab` 或 Dock
  重新选中 pyglet 后按 `M` 可关闭穿透。
- 进入/退出全屏（旧版 `capture_display` 路径）后状态被重新应用。
- 关闭后确认“调度中心 / Mission Control”中无残留窗口。
- 注意：全屏独占捕获模式下透明效果受系统策略影响，属预期平台差异；
  窗口化与普通 Spaces 全屏下应正常。

### 3.3 Linux / X11（含 WSLg、XWayland）

```bash
# 需要 X.Org 与 SHAPE 扩展（绝大多数桌面/WSLg 默认具备）
python3 -m pip install -e .
DISPLAY=:0 python3 examples/window/state_acceptance.py --demo
```

原生机制：透明在创建时选择 32 位 ARGB visual（`XMatchVisualInfo(...,32,
TrueColor,...)`）并将背景像素置 0；穿透通过 SHAPE 扩展把
`ShapeInput` 区域设为空 region（顶层 `_window` 与覆盖客户区的 `_view`
都会设置），关闭时用 `XShapeCombineMask(..., None, ShapeSet)` 恢复；
置顶通过标准 EWMH `_NET_WM_STATE` 增删 `_NET_WM_STATE_ABOVE`。

可重复性检查：

```bash
# 1) 确认 SHAPE 扩展存在：
xdpyinfo | grep -i SHAPE

# 2) 运行注入式 + 真实窗口测试：
python -m pytest tests/unit/window tests/integration/window/test_window_state.py -v

# 3) 交互验收：
DISPLAY=:0 python examples/window/state_acceptance.py
```

额外检查：

- 穿透阶段可用 `xev`/`xwininfo` 观察事件不再落在 pyglet 窗口；
  关闭后 `XShapeCombineMask(None)` 恢复完整输入区。
- 在 GNOME/KDE/i3 等不同 WM 下验证 `_NET_WM_STATE_ABOVE` 置顶与取消。
- Wayland 会话中 pyglet 仍使用 Xlib，窗口运行在 XWayland 下；
  个别合成器对 ARGB / SHAPE / ABOVE 的处理可能不同，以该 WM 的行为为准。
- 无 SHAPE 的精简 X server（如部分 Xvfb 配置）上，
  `mouse_passthrough=True` 会得到明确的 `WindowStateException`。

## 4. 设计说明：为什么“普通窗口”不受影响

- 所有状态默认全为 `False`；驱动首次 `sync` 以 all-off 为差异基线，
  普通窗口不会产生任何额外原生调用。
- 透明所需的 alpha 位只在 `transparent=True`（或旧 transparent/overlay
  样式）时强制写入 config；用户自带 config 且未要求透明时不再被改写为
  `alpha_size=None`。
- 状态机对相同值的重复设置直接返回 `False`，不调用原生层。
- 原生句柄重建时 `invalidate()` 清除驱动差异缓存，`realize()` 把期望状态
  完整推送到新句柄；`teardown()` 在销毁句柄前下发 all-off，且吞掉关闭路径
  上的原生错误，保证 `close()` 总能完成。
