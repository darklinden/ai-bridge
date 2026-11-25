#!/usr/bin/env bash
# 流式吞吐测试：测「首 delta 延迟(TTFT)」与「tok/s」，顺带暴露卡顿（最大 chunk 间隔）。
#
# 两种模式：
#   默认（经本机 ai-bridge）：从 ~/.ai-bridge/<profile>.toml 读 listen_addr / listen_port /
#   auth_key，POST 到 /v1/chat/completions。测的是「客户端 → 代理 → opencode go」全链路。
#   --direct：直连 profile 里的 url，用环境变量 OPENCODE_GO_KEY 认证（不落盘、不回显）。
#   两者对比即可判断慢在代理还是上游。
#
# 用法:
#   ./test-tkps.sh [选项]
#     -n <int>       max_tokens（默认 1200）
#     -m <model>     覆盖 model（直连时生效；经代理时被 profile 覆盖）
#     -p <prompt>    用户提示词
#     --effort <x>   直连时附带 reasoning_effort（high / medium / low）
#     --ctx <chars>  在用户消息前塞入约 chars 个字符的填充，用于测长上下文下的 prefill 延迟
#     --no-usage     不发送 stream_options.include_usage（个别上游不支持）
#     --direct       直连上游
#
# 环境变量:
#   AI_BRIDGE_PROFILE   profile 名（默认取 ~/.ai-bridge/.settings.toml 的 current_profile）
#   AI_BRIDGE_CONFIG    直接指定 profile 文件
#   AI_BRIDGE_URL       覆盖本机 Base URL
#   OPENCODE_GO_KEY     直连上游用的 key（--direct 必填）
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

PROFILE="${AI_BRIDGE_PROFILE:-}"
if [ -z "$PROFILE" ] && [ -z "${AI_BRIDGE_CONFIG:-}" ]; then
    PROFILE="$(awk -F'"' '/^[[:space:]]*current_profile[[:space:]]*=/ { print $2; exit }' \
        "$HOME/.ai-bridge/.settings.toml" 2>/dev/null || true)"
fi
CONFIG_FILE="${AI_BRIDGE_CONFIG:-$HOME/.ai-bridge/${PROFILE:-default}.toml}"

_read_toml() {
    awk -v pat="^[[:space:]]*$1[[:space:]]*=" '
        /^[ \t]*\[/ { exit }
        $0 ~ pat { sub(/^[^=]*=[[:space:]]*/, ""); gsub(/^"|"$/, ""); print; exit }
    ' "$CONFIG_FILE" 2>/dev/null || true
}

MAX_TOKENS=1200
MODEL="$(_read_toml model)"
PROMPT="用大约 300 字的中文解释 TCP 三次握手，并说明为什么需要第三次握手。"
DIRECT=false
EFFORT=""
CTX=0
USE_USAGE=true

while [ $# -gt 0 ]; do
    case "$1" in
        -n) MAX_TOKENS="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        --effort) EFFORT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --no-usage) USE_USAGE=false; shift ;;
        --direct) DIRECT=true; shift ;;
        -h|--help) sed -n '1,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

# ---------- 目标与鉴权 ----------
if $DIRECT; then
    URL="$(_read_toml url)"
    [ -n "$URL" ] || { echo "profile 里没有 url: $CONFIG_FILE" >&2; exit 2; }
    KEY="${OPENCODE_GO_KEY:-}"
    [ -n "$KEY" ] || { echo "--direct 需要环境变量 OPENCODE_GO_KEY" >&2; exit 2; }
    HDR=(-H "Authorization: Bearer $KEY" -H "x-opencode-session: tkps-$(date +%s)-$$")
else
    LISTEN_ADDR="$(_read_toml listen_addr)"
    LISTEN_PORT="$(_read_toml listen_port)"
    BASE_URL="${AI_BRIDGE_URL:-http://${LISTEN_ADDR:-127.0.0.1}:${LISTEN_PORT:-18650}}"
    URL="$BASE_URL/v1/chat/completions"
    KEY="$(_read_toml auth_key)"
    HDR=()
    [ -n "$KEY" ] && HDR=(-H "Authorization: Bearer $KEY")
fi

# ---------- 请求体（最小 JSON 转义） ----------
json_escape() { sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr '\n' ' '; }
if [ "$CTX" -gt 0 ]; then
    FILLER="$(perl -e 'print "以下内容为压测填充文本，与问题无关，可忽略。" x (int($ARGV[0]/24)+1)' "$CTX")"
    PROMPT="$(printf '%s%s' "$FILLER" "$PROMPT")"
fi
PROMPT_JSON="$(printf '%s' "$PROMPT" | json_escape)"

build_body() {
    local usage="" effort=""
    $1 && usage=',"stream_options":{"include_usage":true}'
    [ -n "$EFFORT" ] && $DIRECT && effort=",\"reasoning_effort\":\"$EFFORT\""
    printf '{"model":"%s","stream":true,"max_tokens":%s%s%s,"messages":[{"role":"system","content":"你是一个乐于助人的助手。"},{"role":"user","content":"%s"}]}' \
        "${MODEL:-deepseek-v4.1-flash}" "$MAX_TOKENS" "$usage" "$effort" "$PROMPT_JSON"
}

# ---------- 流式计时/统计 ----------
read -r -d '' PERL <<'PERL' || true
use strict; use warnings; use utf8; use Time::HiRes qw(time);
my $s = 0 + ($ENV{START} // time);
my ($first, $fr, $fc, $last, $http, $tst);
my ($n, $nr, $nc, $rchars, $cchars) = (0, 0, 0, 0, 0);
my ($pt, $ct, $tt) = (0, 0, 0);
my ($maxgap, $gap_at, $gap_len) = (0, 0, 0);
my $maxlen = 0;
my @raw;
while (my $l = <STDIN>) {
    chomp $l;
    if ($l =~ /^__HTTP__(\d+)\s+([\d.]+)/) { $http = $1; $tst = $2; next; }
    if ($l !~ /^data:\s?(.*)$/) { push @raw, $l if $l =~ /\S/ && @raw < 3; next; }
    my $d = $1;
    next if $d eq '[DONE]';
    my $t = time;
    $n++;
    $first = $t unless defined $first;
    if (defined $last) {
        my $g = $t - $last;
        if ($g > $maxgap) { $maxgap = $g; $gap_at = $last - $s; $gap_len = $g; }
    }
    $last = $t;
    if ($d =~ /"(?:reasoning_content|reasoning)"\s*:\s*"((?:[^"\\]|\\.)*)"/) {
        $nr++; $rchars += length($1); $fr = $t unless defined $fr;
    }
    if ($d =~ /"content"\s*:\s*"((?:[^"\\]|\\.)*)"/) {
        my $c = $1;
        $nc++ if length $c;
        $cchars += length $c;
        # 首包常带 "content":""，只认非空正文作为正文起点。
        $fc = $t if length $c && !defined $fc;
    }
    $pt = $1 if $d =~ /"prompt_tokens"\s*:\s*(\d+)/;
    $ct = $1 if $d =~ /"completion_tokens"\s*:\s*(\d+)/;
    $tt = $1 if $d =~ /"total_tokens"\s*:\s*(\d+)/;
}

printf "\n---- 结果 ----\n";
printf "HTTP             : %s\n", defined $http ? $http : '未收到响应';
if (defined $first) {
    printf "TTFT (首 delta)  : %.2fs\n", $first - $s;
    printf "首个思考 delta   : %.2fs\n", $fr - $s if defined $fr;
    printf "首个正文 delta   : %.2fs  (思考阶段 %.2fs)\n", $fc - $s, $fc - ($fr // $first)
        if defined $fc;
    printf "总时长           : %.2fs  (curl time_starttransfer=%s)\n", $last - $s, $tst // '-';
    printf "chunk 数         : %d (思考 %d / 正文 %d)\n", $n, $nr, $nc;
    printf "输出字符         : 思考 %d / 正文 %d\n", $rchars, $cchars;
    printf "最大 chunk 间隔  : %.2fs (发生在 %.2fs 处)\n", $maxgap, $gap_at if $maxgap > 0;
    if ($ct > 0) {
        my $dur = $last - $s;
        printf "usage            : prompt %d / completion %d / total %d\n", $pt, $ct, $tt;
        printf "tok/s (全程)     : %.1f  (= %d tokens / %.2fs)\n", $ct / $dur, $ct, $dur;
        printf "tok/s (首 delta 后): %.1f\n", ($ct - 1) / ($last - $first) if $last > $first;
    } else {
        printf "usage            : 上游未返回 usage，无法给出精确 tok/s\n";
    }
    printf "字符速率(正文)   : %.1f 字/秒\n", $cchars / ($last - $fc)
        if defined $fc && $last - $fc > 0.2 && $cchars > 20;
} else {
    printf "未解析到任何 SSE data 行。原始响应片段:\n%s\n", join("\n", @raw) if @raw;
}
exit(defined $http && $http >= 400 ? 3 : 0);
PERL

run_once() {
    local body="$1"
    START="$(perl -MTime::HiRes=time -e 'printf "%.6f", time')"
    START="$START" curl -N -sS --connect-timeout 10 --max-time 300 \
        -X POST "$URL" "${HDR[@]}" -H 'Content-Type: application/json' \
        -d "$body" -w '\n__HTTP__%{http_code} %{time_starttransfer}\n' \
        | START="$START" perl -CS -e "$PERL"
}

echo "==> POST $URL"
echo "==> model=${MODEL:-deepseek-v4.1-flash} max_tokens=$MAX_TOKENS stream=true profile=$(basename "$CONFIG_FILE")"
echo "==> prompt: $PROMPT"

if ! run_once "$(build_body "$USE_USAGE")"; then
    if $USE_USAGE; then
        echo "==> 请求失败（可能上游不支持 stream_options），去掉 include_usage 重试 …" >&2
        run_once "$(build_body false)"
    else
        exit 1
    fi
fi
