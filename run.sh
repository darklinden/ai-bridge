#!/usr/bin/env zsh

BASEDIR=$(dirname "$0")
SCRIPT_DIR="$(realpath "${BASEDIR}")"

# ai-bridge 启动脚本：把 stdout/stderr（[REQ]/[RESP]/[RECV]/[ERR] 全量）落盘，按天轮转。
#
# 用法：
#   前台（终端实时看输出 + 同时落盘）: ./run.sh
#   后台（脱离终端，仅落盘）        : ./run.sh >/dev/null 2>&1 &   （或 nohup ./run.sh >/dev/null 2>&1 &）
#   实时看日志                       : tail -f logs/ai-bridge-$(date +%Y%m%d).log
#
# 说明：
#   - 日志写到 logs/ai-bridge-YYYYMMDD.log，每天一个文件，天然按天轮转。
#   - reqlog.rs 直写 stdout（绕过 tracing），tracing 输出也走 stdout；
#     因此必须重定向整个进程的 stdout/stderr 才能全量捕获。
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p ai-bridge-logs
LOG="ai-bridge-logs/ai-bridge-$(date +%Y%m%d).log"
echo "[run.sh] $(date '+%F %T') starting ai-bridge, log -> $LOG"

# tee 让终端和文件都拿到输出；若想终端更安静，把下面这行换成直接 exec 重定向。
exec $SCRIPT_DIR/ai-bridge 2>&1 | tee -a "$LOG"
