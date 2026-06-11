# AutoAlias 运行文档

本文只说明一件事：怎么启动 GUI、旧网页和新网页。

建议新手优先使用 GUI；需要局域网多人访问时使用新网页；旧网页主要用于对比历史版本或继续使用旧流程。

## 运行前准备

先确认项目路径：

```powershell
F:\430AutoAlias
```

本项目的 Windows 启动脚本会按下面顺序选择 Python：

1. 如果设置了 `AUTOALIAS_PYTHON`，使用这个环境；
2. 如果存在 `F:\ComfyUI\.venv\Scripts\python.exe`，优先使用它；
3. 否则使用系统里的 `python`。

如果你想强制指定 Python，可以在 PowerShell 中运行：

```powershell
$env:AUTOALIAS_PYTHON = "F:\ComfyUI\.venv\Scripts\python.exe"
```

如果依赖没有安装，先在项目根目录执行：

```powershell
pip install -e ".[gui,web]"
```

新网页还需要前端依赖：

```powershell
cd F:\430AutoAlias\webapp
npm install
npm run build
```

## 1. 启动 GUI

GUI 是目前最推荐的本地人工分段工具。

### 方式 A：直接打开 GUI

```powershell
F:\430AutoAlias\scripts\autoalias_gui.cmd
```

打开后在界面里上传图片。

### 方式 B：启动时指定图片

```powershell
F:\430AutoAlias\scripts\autoalias_gui.cmd F:\430AutoAlias\test.png
```

### 默认输出位置

GUI 默认会把工程和导出结果放到：

```text
F:\430AutoAlias\lan_reviews
```

### GUI 适合做什么

- 上传图片；
- 调整图片预处理；
- 提取骨架；
- 手动添加分段点；
- 拖动和删除分段点；
- 保存曲线；
- 导出 IGES / JSON / SVG；
- 保存和打开工程文件。

## 2. 启动旧网页

旧网页有两个入口，按你的使用习惯选择。

### 方式 A：旧版指定图片纠错页

这个入口需要提前指定一张图片：

```powershell
F:\430AutoAlias\scripts\autoalias.cmd review-image F:\430AutoAlias\test.png --out F:\430AutoAlias\corrections
```

默认地址：

```text
http://127.0.0.1:8765/
```

如果 `8765` 被占用，程序会自动向后寻找可用端口，比如 `8766`、`8767`。

如果你想手动指定端口：

```powershell
F:\430AutoAlias\scripts\autoalias.cmd review-image F:\430AutoAlias\test.png --out F:\430AutoAlias\corrections --port 8780
```

然后打开：

```text
http://127.0.0.1:8780/
```

### 方式 B：旧版上传式分段网页

这个入口不需要提前指定图片，可以在网页里上传：

```powershell
F:\430AutoAlias\scripts\autoalias.cmd skeleton-review --out F:\430AutoAlias\lan_reviews --host 0.0.0.0 --port 8780
```

本机访问：

```text
http://127.0.0.1:8780/
```

局域网访问：

```text
http://你的电脑IP:8780/
```

### 旧网页适合做什么

- 对比历史功能；
- 使用旧版 CV 预览、旧版 G2 编辑实验功能；
- 继续打开过去旧流程保存的标注；
- 做旧版逻辑回归测试。

如果你现在只是正常人工分段，建议使用 GUI 或新网页，不建议继续把旧网页作为主工作台。

## 3. 启动新网页

新网页是 React + TypeScript 前端和 FastAPI 后端，适合局域网多人访问。

### 第一次使用前构建前端

```powershell
cd F:\430AutoAlias\webapp
npm install
npm run build
```

### 启动新网页后端

```powershell
F:\430AutoAlias\scripts\autoalias_next_api.cmd
```

默认地址：

```text
http://127.0.0.1:8790/
```

局域网访问：

```text
http://你的电脑IP:8790/
```

### 开发模式

如果你在修改前端代码，希望热更新，可以开两个终端。

终端 1：启动 FastAPI 后端：

```powershell
F:\430AutoAlias\scripts\autoalias_next_api.cmd
```

终端 2：启动 Vite 前端：

```powershell
F:\430AutoAlias\scripts\autoalias_next_frontend.cmd
```

开发地址：

```text
http://127.0.0.1:5173/
```

开发模式下，`/api` 会自动转发到：

```text
http://127.0.0.1:8790/
```

### 新网页默认输出位置

新网页默认输出到：

```text
F:\430AutoAlias\lan_reviews_next
```

如果想改输出位置，可以启动前设置：

```powershell
$env:AUTOALIAS_WEB_OUT = "F:\430AutoAlias\my_web_outputs"
F:\430AutoAlias\scripts\autoalias_next_api.cmd
```

### 新网页和 GUI 的关系

新网页目前和 GUI 的主要编辑逻辑保持一致：

- 手动分段时只做骨架寻路；
- 不在拖点时实时做 CV/NURBS 拟合；
- 最终点击导出时才执行完整拟合；
- 支持图片预处理、骨架显示、手动分段、自动分段、骨架修补、工程保存和导出。

## 4. 如何关闭服务

GUI：直接关闭窗口。

旧网页或新网页：回到启动它的终端，按：

```text
Ctrl + C
```

如果端口被占用，可以换一个端口，或者关闭之前的终端进程。

## 5. 常用选择建议

| 需求 | 推荐入口 |
|---|---|
| 自己本机人工分段 | GUI |
| 公司内网多人使用 | 新网页 `8790` |
| 查看旧版功能 | 旧网页 |
| 调试前端界面 | 新网页开发模式 `5173` |
| 只跑命令行导出 | `autoalias.cmd fit-reviewed` |

## 6. 最常用命令汇总

```powershell
# GUI
F:\430AutoAlias\scripts\autoalias_gui.cmd

# GUI + 指定图片
F:\430AutoAlias\scripts\autoalias_gui.cmd F:\430AutoAlias\test.png

# 旧网页：指定图片
F:\430AutoAlias\scripts\autoalias.cmd review-image F:\430AutoAlias\test.png --out F:\430AutoAlias\corrections

# 旧网页：上传式工作台
F:\430AutoAlias\scripts\autoalias.cmd skeleton-review --out F:\430AutoAlias\lan_reviews --host 0.0.0.0 --port 8780

# 新网页：生产模式
F:\430AutoAlias\scripts\autoalias_next_api.cmd

# 新网页：前端开发模式
F:\430AutoAlias\scripts\autoalias_next_frontend.cmd
```

