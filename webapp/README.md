# AutoAlias Next Web

这是新版网页端，不替换旧版 `skeleton-review` 页面。

## 结构

- 前端：React + TypeScript + Vite
- 画布：Konva / react-konva
- Worker：`src/workers/geometry.worker.ts`
- 后端：`src/autoalias/web_next/api.py`
- 导出：后端复用现有 `fit_reviewed_annotations` 生成 IGES / WIRE / JSON / SVG

## 开发运行

先启动 API：

```powershell
F:\430AutoAlias\scripts\autoalias_next_api.cmd
```

再启动前端：

```powershell
F:\430AutoAlias\scripts\autoalias_next_frontend.cmd
```

浏览器打开：

```text
http://127.0.0.1:5173/
```

## 内网部署运行

构建前端：

```powershell
F:\430AutoAlias\scripts\autoalias_next_build.cmd
```

启动 API：

```powershell
F:\430AutoAlias\scripts\autoalias_next_api.cmd
```

浏览器打开：

```text
http://127.0.0.1:8790/
```

同事在局域网访问：

```text
http://你的电脑内网IP:8790/
```

