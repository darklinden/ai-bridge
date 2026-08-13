# AI Bridge (cc-bridge)

一个 Rust 中继服务，把本地的 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses 三种客户端入口，统一转发到一种配置的上游 API，并在转换过程中记录 AI 消息日志。

## Language

**Upstream**:
服务实际请求的 AI 模型 API，由 `UPSTREAM_TYPE` 指定为 `anthropic-messages`、`oai-chat` 或 `oai-responses` 三者之一。
_Avoid_: 上游, backend, provider

**Local entry**:
本地暴露给客户端（CLI、SDK）的三个 HTTP 入口：`/v1/messages`（Anthropic Messages）、`/v1/chat/completions`（OpenAI Chat）、`/v1/responses`（OpenAI Responses）。三者共享同一个 Upstream。
_Avoid_: 本地接口, endpoint, route

**Request log**:
进程内 stdout 单行流日志，形如 `[REQ #0001]: ...` 与 `[RESP #0001]: ...`，在每个请求/响应的单行前用两个换行分隔，方便终端阅读。非调试工具，不做落盘或持久化。
_Avoid_: 日志服务, log file, telemetry

**Media description**:
当本地请求携带图片、而 Upstream 是纯文本模型时，由配置的视觉模型（`VISION_*`）把图片转成文字描述，替换图片块后再转发。多图一次联合分析，结果按图片指纹做进程内缓存。
_Avoid_: image-to-text, vision, image placeholder

**Upstream type detection**:
由 `UPSTREAM_TYPE` 显式声明的上游格式，决定认证头、格式转换与错误格式。不做 URL 启发式猜测。
_Avoid_: auto-detect, url-based detection

## Decisions

- **Upstream type is explicit** — `UPSTREAM_TYPE` 必填，缺失即报配置错误。不做 URL 启发式（避免把 `anthropic.com/v1/messages` 误判为 chat）。
- **Unified `UPSTREAM_*` naming** — 上游配置统一为 `UPSTREAM_URL` / `UPSTREAM_API_KEY` / `UPSTREAM_MODEL` / `UPSTREAM_AUTH_KEY`，废弃旧的 `OPENAI_COMP_*`。
- **Local model is overridden** — 三个本地入口请求体里的 `model` 一律替换为 `UPSTREAM_MODEL`，客户端写的 model 不生效。
- **Error format follows local entry** — 错误响应的 JSON 结构跟随本地入口（Anthropic 入口返回 Anthropic 风格错误，OpenAI 入口返回 OpenAI 风格错误）。
- **Inbound auth unchanged** — `UPSTREAM_AUTH_KEY` 是入站认证（校验本地客户端访问中转的 token），与上游自身的认证（`UPSTREAM_API_KEY`）分离。
