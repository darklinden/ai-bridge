# AI Bridge (cc-bridge)

一个 Rust 中继服务，把本地的 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses 三种客户端入口，统一转发到一种配置的上游 API，并在转换过程中记录 AI 消息日志。

本文件只是**词表**：只定义本项目特有的概念。功能需求见 `docs/requirements.md`，架构见 `docs/architecture.md`，实现机制见 `docs/implementation.md`，架构决策见 `docs/adr/`。

## Language

**Upstream**:
服务实际请求的 AI 模型 API。其线路格式由所属 profile 的 `upstream_type` 键显式声明为 `anthropic-messages`、`oai-chat` 或 `oai-responses` 三者之一。`[vision]` 表配置的视觉模型端点也是一个 Upstream，同样显式声明格式。
_Avoid_: 上游, backend, provider, zen

**Local entry**:
本地暴露给客户端（CLI、SDK）的三个 HTTP 入口：`/v1/messages`（Anthropic Messages）、`/v1/chat/completions`（OpenAI Chat）、`/v1/responses`（OpenAI Responses）。三者共享同一个 Upstream。
_Avoid_: 本地接口, endpoint, route

**Pass-through**:
Local entry 与 Upstream 线路格式相同时的路径：请求与响应字节原样转发，不做转换，只旁路观察以便记录日志。
_Avoid_: 直通, proxy mode, raw mode

**Profile**:
一份 TOML 配置文件 `~/.ai-bridge/<name>.toml`（ADR-0005），描述一个完整的上游配置。启动时按名字选择，无参数时加载上次记录的 profile（首次为 `default.toml`），`--list` 可列出全部。每次成功启动都会把所用 profile 记入 `~/.ai-bridge/.settings.toml`（点前缀隐藏文件，ai-bridge 自动维护，存 `current_profile` 键）；`--list` 用 `*` 标记当前选择。
_Avoid_: config file, env, 环境变量

**Request log**:
进程内单行流日志，一次请求/响应对应四类行：`[REQ SYSTEM]`（system prompt 变化时才打印）、`[REQ #id]`、`[RESP #id]`（流式文本追加在同一条未闭合的行上）、以及错误时的 `[ERR-REQ #id]` / `[ERR-RESP #id]`。与 `RUST_LOG` 无关，始终开启；正文一律原样记录（含 base64 媒体载荷），不做摘要。默认只写 stdout；给出 `--log <path>` 时同一份内容 **tee** 到该文件（stdout 照写），文件以追加方式打开、由使用者自行轮转（ADR-0012）。
_Avoid_: 日志服务, log file, telemetry, debug log

**Media description**:
当本地请求携带图片、而 Upstream 是纯文本模型时，由 `[vision]` 表配置的视觉模型把图片转成文字描述，替换图片块后再转发。多图一次联合分析，结果按图片指纹做进程内缓存；描述失败降级为 `[Unsupported Image]`，不阻断请求。该功能由顶层 `vision_supplement` 键控制，**默认关闭**（ADR-0004）——因为"上游是纯文本"由模型名清单判定，只是启发式。
_Avoid_: image-to-text, vision, image placeholder

**Upstream type detection**:
由 profile 的 `upstream_type` 键（主上游）与 `[vision].upstream_type` 键（视觉端点）**显式声明**，决定认证头、格式转换与错误格式。不做 URL 启发式猜测，没有例外（ADR-0002）。
_Avoid_: auto-detect, url-based detection
