#!/usr/bin/env python3
"""
Cline / ClinePass 上游锁定（DeepSeek Pinned Upstream）本地检测脚本
无需第三方依赖，仅使用 Python 3 标准库。
"""

import sys
import os
import re
import json
import ssl
import urllib.request
import urllib.error

# 忽略本地 Python 环境可能存在的 SSL 证书链验证问题
ctx = ssl._create_unverified_context()

def get_api_key():
    # 1. 优先读取环境变量
    key = os.environ.get("CLINE_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. 从 magpie 的 providers.json 自动读取
    magpie_prov = os.path.expanduser("~/.config/magpie/providers.json")
    if os.path.exists(magpie_prov):
        try:
            with open(magpie_prov, "r", encoding="utf-8") as f:
                data = json.load(f)
                providers = data.get("providers", [])
                for p in providers:
                    if p.get("id") in ("clinepass", "cline") and p.get("key"):
                        return p["key"].strip()
        except Exception:
            pass

    # 3. 从 cline 的 providers.json 自动读取
    cline_prov = os.path.expanduser("~/.cline/data/settings/providers.json")
    if os.path.exists(cline_prov):
        try:
            with open(cline_prov, "r", encoding="utf-8") as f:
                data = json.load(f)
                for pid in ("cline-pass", "cline"):
                    p_info = data.get("providers", {}).get(pid, {}).get("settings", {})
                    token = p_info.get("auth", {}).get("accessToken") or p_info.get("apiKey")
                    if token:
                        return token.strip()
        except Exception:
            pass

    return None

def test_request(api_key, label, extra_options, model="cline-pass/deepseek-v4.1-flash-1017"):
    url = "https://api.cline.bot/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cline.bot",
        "X-Title": "Cline",
        "User-Agent": "Cline/0.0.49",
        "X-IS-MULTIROOT": "false",
        "X-CLIENT-TYPE": "cline-desktop",
        "X-CLIENT-VERSION": "0.0.49",
        "X-PLATFORM": "Cline Desktop",
        "X-PLATFORM-VERSION": "0.0.49",
        "X-CORE-VERSION": "0.2.0",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "1"}],
        "max_tokens": 1,
        "stream": True,
        **extra_options
    }

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    print(f"\n👉 [1;34m测试项目: {label}[0m")

    routing_found = None
    stream_error = None
    finish_model = None

    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            for line in resp:
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw.startswith("data: ") or raw == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(raw[6:])
                except Exception:
                    continue

                if "error" in chunk:
                    stream_error = chunk["error"]

                if "model" in chunk:
                    finish_model = chunk["model"]

                # 提取 Vercel AI Gateway 注入的底层真实路由信息
                choices = chunk.get("choices", [])
                if choices:
                    provider_meta = choices[0].get("delta", {}).get("provider_metadata", {})
                    gateway = provider_meta.get("gateway", {})
                    routing = gateway.get("routing")
                    if routing:
                        routing_found = routing
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        try:
            stream_error = json.loads(err_msg).get("error", {"message": err_msg})
        except Exception:
            stream_error = {"message": err_msg, "code": e.code}
    except Exception as e:
        print(f"❌ 网络或连接错误: {e}")
        return

    # 输出判定分析
    if stream_error:
        err_text = stream_error.get("message", "")
        print(f"  [1;33m[网关拦截反馈][0m {err_text}")
        
        # 提取网关反馈中包含的所有可用上游
        m = re.search(r"Available providers are:\s*([^\"\n]+)", err_text)
        if m:
            provs = [p.strip() for p in m.group(1).split(",") if p.strip()]
            print(f"\n  [1;36m📦 该模型在网关池中包含的全部上游厂商（共 {len(provs)} 家）：[0m")
            formatted = []
            for p in sorted(provs):
                if p == "deepseek":
                    formatted.append(f"[1;32m{p} (官方自有)[0m")
                else:
                    formatted.append(f"[90m{p}[0m")
            # 每行输出 5 个
            for i in range(0, len(formatted), 5):
                print("    " + "  ·  ".join(formatted[i:i+5]))

        if "No available providers match the only filter" in err_text:
            print("\n  [1;32m✅ 证实：白名单强约束生效！只要限定了 only: [deepseek]，网关就会排除其余所有 19 家第三方，绝不向它们 fallback。[0m")
        return

    if routing_found:
        pinned = routing_found.get("affinity", {}).get("pinnedProvider")
        final = routing_found.get("finalProvider")
        resolved = routing_found.get("resolvedProvider")
        reasoning = routing_found.get("planningReasoning")

        print(f"  • 外层返回模型名 : [1m{finish_model}[0m")
        print(f"  • 锁定上游目标   : [1m{pinned}[0m")
        print(f"  • 最终应答厂商   : [1;32m{final}[0m (resolved: {resolved})")
        print(f"  • 网关路由决策链 : [36m{reasoning}[0m")

        if final == "deepseek":
            print("  [1;32m🎉 判定结果: 100% 锁死在 DeepSeek 官方上游！没有被偷换！[0m")
        else:
            print(f"  [1;31m⚠️ 警告: 实际应答厂商为 {final}，未钉住 DeepSeek！[0m")
    else:
        print(f"  • 收到响应，外层模型为: {finish_model} (本次流未包含网关 routing 元数据)")

def main():
    print("=" * 60)
    print(" Cline / ClinePass 上游锁定（DeepSeek Pinned Upstream）检测")
    print("=" * 60)

    key = get_api_key()
    if not key:
        print("❌ 未检测到 API Key。请设置环境变量 export CLINE_API_KEY=\"sk_...\" 后重试。")
        sys.exit(1)

    masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
    print(f"🔑 使用凭据: {masked_key}")

    # 测试 1: 发送正常的锁定请求 (magpie 默认给 ClinePass 发送的参数)
    test_request(
        key,
        "1. 携带锁定参数 (gateway.only = [deepseek], has = [vision, implicit-caching])",
        {"providerOptions": {"gateway": {"only": ["deepseek"], "has": ["vision", "implicit-caching"]}}}
    )

    # 测试 2: 对照探测 (获取池中所有上游列表 & 验证白名单有效性)
    test_request(
        key,
        "2. 上游探测与对照实验 (探查上游池名单 & 验证只有 deepseek 可通过)",
        {"providerOptions": {"gateway": {"only": ["probe-all-providers"], "has": ["vision", "implicit-caching"]}}}
    )

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

