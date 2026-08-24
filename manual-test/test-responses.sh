#!/usr/bin/env bash
# 本地 OpenAI Responses 入口测试（POST /v1/responses）。
# 默认流式（SSE），用于验证 [REQ]/[RESP] 日志与响应转换。
#
# 用法:
#   ./test-responses.sh [--no-stream] [提示词]
#
# 说明:
#   - 自动从 ~/.ai-bridge/<profile>.toml 读取 listen_addr / listen_port / auth_key
#     （默认 default profile；AI_BRIDGE_PROFILE 可换名字，AI_BRIDGE_CONFIG 可直接指定文件）；
#   - 设置环境变量 AI_BRIDGE_URL 可整体覆盖 Base URL（例如指向 mock 上游的代理）;
#   - 认证 key 存在时自动附带 `Authorization: Bearer <key>`。
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${AI_BRIDGE_CONFIG:-$HOME/.ai-bridge/${AI_BRIDGE_PROFILE:-default}.toml}"

_read_toml() {
    # 只读顶层标量键：遇到第一个 [table] 头即停止，避免误取 [vision] 等小节的键。
    awk -v pat="^[[:space:]]*$1[[:space:]]*=" '
        /^[ \t]*\[/ { exit }
        $0 ~ pat { sub(/^[^=]*=[[:space:]]*/, ""); gsub(/^"|"$/, ""); print; exit }
    ' "$CONFIG_FILE" 2>/dev/null || true
}

LISTEN_ADDR="$(_read_toml listen_addr)"
LISTEN_PORT="$(_read_toml listen_port)"
AUTH_KEY="$(_read_toml auth_key)"

BASE_URL="${AI_BRIDGE_URL:-http://${LISTEN_ADDR:-127.0.0.1}:${LISTEN_PORT:-18650}}"
AUTH_HEADER=()
[ -n "$AUTH_KEY" ] && AUTH_HEADER=(-H "Authorization: Bearer $AUTH_KEY")

STREAM=true
[ "${1:-}" = "--no-stream" ] && { STREAM=false; shift; }
PROMPT="${1:-用一句话介绍你自己。}"

echo "==> POST $BASE_URL/v1/responses  (stream=$STREAM)"
echo "==> prompt: $PROMPT"
curl -N -sS --max-time 120 -X POST "$BASE_URL/v1/responses" \
  "${AUTH_HEADER[@]}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"anything\",\"stream\":$STREAM,\"instructions\":\"你是一个乐于助人的助手。\",\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"$PROMPT\"}]}]}"
echo
echo "==> done"