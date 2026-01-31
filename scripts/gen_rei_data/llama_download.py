#!/usr/bin/env python3
"""
llama_download.py
专门用于下载 LLaMA 模型（包括 gated 模型，如 Llama-3.1-8B-Instruct）

功能：
- 自动登录 HuggingFace（可传 token 或从环境变量读取）
- 自动检测访问权限
- 自动下载模型文件（tokenizer + weights）
- 带详细错误提示（例如 401 无权限）
- 断点续传
"""

import argparse
import os
from huggingface_hub import HfApi, snapshot_download, login, HfHubHTTPError, GatedRepoError


def print_bar():
    print("=" * 80)


def llama_download(model_name, token=None, local_dir="llama_models", revision="main"):
    print_bar()
    print(f"🚀 开始下载模型: {model_name}")
    print_bar()

    # -------------------------
    # Step 1: 登录 HuggingFace
    # -------------------------
    if token is None:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)

    if token is None:
        print("❌ 未提供 token，也没有从环境变量找到 HUGGINGFACE_HUB_TOKEN。")
        print("   你可以这样运行：")
        print("       HUGGINGFACE_HUB_TOKEN=xxx python llama_download.py --model meta-llama/Llama-3.1-8B-Instruct")
        return

    try:
        login(token=token)
        print("🔐 HuggingFace 登录成功！")
    except Exception as e:
        print("❌ 登录失败：", e)
        return

    # -------------------------
    # Step 2: 检查权限
    # -------------------------
    api = HfApi()
    try:
        print("🔍 正在检查访问权限…")
        api.model_info(model_name, token=token)
        print("✅ 访问权限正常，可以下载。")
    except GatedRepoError as e:
        print("❌ 你没有权限访问该 gated 模型：")
        print(e)
        print("\n请前往 HF 模型页面申请访问权限：")
        print(f"👉 https://huggingface.co/{model_name}")
        return
    except HfHubHTTPError as e:
        print("❌ 访问 HuggingFace 失败：", e)
        return
    except Exception as e:
        print("❌ 未知错误：", e)
        return

    # -------------------------
    # Step 3: 下载模型
    # -------------------------
    print("📥 开始下载模型文件（支持断点续传）…")
    try:
        snapshot_download(
            repo_id=model_name,
            token=token,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 方便复制
            resume_download=True
        )
        print_bar()
        print(f"🎉 模型已成功下载到: {local_dir}/{model_name}")
        print("你可以直接用 transformers 加载该目录。")
        print_bar()

    except Exception as e:
        print("❌ 下载失败：", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称，例如 meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--token", type=str, default=None,
                        help="可选：HF token，不提供则读取环境变量 HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--out", type=str, default="llama_models",
                        help="下载目录")
    parser.add_argument("--revision", type=str, default="main",
                        help="模型 revision 或分支")
    args = parser.parse_args()

    llama_download(
        model_name=args.model,
        token=args.token,
        local_dir=args.out,
        revision=args.revision
    )


if __name__ == "__main__":
    main()
