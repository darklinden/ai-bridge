# manual-test

三个本地入口的手工 curl 测试脚本，默认以**流式（SSE）**方式调用，用来验证代理的请求/响应转换和常开日志（`[REQ #xxxx]` / `[RESP #xxxx]`）。

| 脚本 | 本地入口 | 端点 |
|---|---|---|
| `test-messages.sh` | Anthropic Messages | `POST /v1/messages` |
| `test-chat.sh` | OpenAI Chat Completions | `POST /v1/chat/completions` |
| `test-responses.sh` | OpenAI Responses | `POST /v1/responses` |

## 用法

```bash
# 1. 先启动 ai-bridge（前台实时看 [REQ]/[RESP] 日志，或 ./run.sh 落盘）
cargo run   # 或 ./run.sh

# 2. 分别请求三个入口
./manual-test/test-messages.sh
./manual-test/test-chat.sh
./manual-test/test-responses.sh

# 非流式
./manual-test/test-chat.sh --no-stream

# 自定义提示词
./manual-test/test-chat.sh "用一句话介绍你支持的 API 格式。"
```

## 配置

- 脚本自动从仓库根 `.env` 读取 `LISTEN_ADDR` / `LISTEN_PORT` / `UPSTREAM_AUTH_KEY`（缺失时兜底默认 `http://127.0.0.1:18650`）。
- 设置环境变量 `AI_BRIDGE_URL` 可整体覆盖 Base URL，例如指向一个 mock 上游：
  ```bash
  AI_BRIDGE_URL=http://127.0.0.1:9999 ./manual-test/test-chat.sh
  ```
- 认证 key 存在时自动附带 `Authorization: Bearer <key>` 头；无需认证时留空即可。

## 关注点

启动 ai-bridge 的终端应看到类似输出：

```
[REQ SYSTEM]: 你是一个乐于助人的助手。
[REQ #0001]: anything 用一句话介绍你自己。

[RESP #0001]: 我是... 一行内实时流式输出...
```

- **流式**：`[RESP #xxxx]: ` 一行应出现，且响应文本**实时 append 到同一行**，完成后换行。
- **错误**：应出现 `[ERR-REQ #xxxx]` / `[ERR-RESP #xxxx]` 标记行。
- 覆盖三种 `UPSTREAM_TYPE`（`anthropic-messages` / `oai-chat` / `oai-responses`）各跑一遍三个脚本，可验证全部转换矩阵；特别是 `UPSTREAM_TYPE=anthropic-messages` 时三入口都应能在 `[RESP]` 行看到流式文本。