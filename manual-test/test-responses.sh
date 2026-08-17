#!/usr/bin/env bash
# 本地 OpenAI Responses 入口测试（POST /v1/responses）。
# 默认流式（SSE），用于验证 [REQ]/[RESP] 日志与响应转换。
#
# 用法:
#   ./test-responses.sh [--no-stream] [提示词]
#
# 说明:
#   - 自动从仓库根 .env 读取 LISTEN_ADDR / LISTEN_PORT / UPSTREAM_AUTH_KEY；
#   - 设置环境变量 AI_BRIDGE_URL 可整体覆盖 Base URL（例如指向 mock 上游的代理）;
#   - 认证 key 存在时自动附带 `Authorization: Bearer <key>`。
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$DIR/../.env"

_read_env() {
    grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' || true
}

LISTEN_ADDR="$(_read_env LISTEN_ADDR)"
LISTEN_PORT="$(_read_env LISTEN_PORT)"
AUTH_KEY="$(_read_env UPSTREAM_AUTH_KEY)"

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