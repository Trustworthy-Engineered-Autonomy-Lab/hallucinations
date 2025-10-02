#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.1 T2V-1.3B (Diffusers) —— 本地下载权重 + 单卡推理生成视频脚本
- 默认下载到 ./Wan2.1-T2V-1.3B-Diffusers
- 默认分辨率 832x480，帧数 81，FPS=15
- 自动选择 torch_dtype（优先 bfloat16，其次 float16；CPU 回退 float32）
- 可选 CPU-Offload 与 T5 仅在 CPU 以节省显存
用法示例：
  python wan_t2v_local.py --prompt "A cat walks on the grass, realistic" --offload --t5_cpu
"""

import os
import sys
import argparse
from pathlib import Path

# 禁用 xformers（避免与你机器上的 flash-attn/xformers 引发导入冲突）
os.environ.setdefault("XFORMERS_DISABLED", "1")

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 T2V-1.3B (Diffusers) local inference")
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                   help="Hugging Face repo id")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers",
                   help="本地权重保存/读取目录")
    p.add_argument("--prompt", default="A cat walks on the grass, realistic",
                   help="正向提示词")
    p.add_argument("--negative_prompt", default=("Bright tones, overexposed, static, blurred details, subtitles, "
                                                 "style, works, paintings, images, static, overall gray, worst quality, "
                                                 "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                                                 "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                                                 "misshapen limbs, fused fingers, still picture, messy background, "
                                                 "three legs, many people in the background, walking backwards"),
                   help="反向提示词")
    p.add_argument("--width", type=int, default=832, help="生成宽度（需为32的倍数）")
    p.add_argument("--height", type=int, default=480, help="生成高度（需为32的倍数）")
    p.add_argument("--num_frames", type=int, default=81, help="视频帧数（例如81对应约5.4秒@15fps）")
    p.add_argument("--fps", type=int, default=15, help="导出视频帧率")
    p.add_argument("--guidance_scale", type=float, default=5.0, help="CFG 引导系数（官方示例 5~6）")
    p.add_argument("--out", default="output.mp4", help="输出视频文件名")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    p.add_argument("--offload", action="store_true", help="启用模型 CPU-Offload 以省显存")
    p.add_argument("--t5_cpu", action="store_true", help="将 T5 文本编码器固定在 CPU 以省显存")
    p.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    return p.parse_args()

def main():
    args = parse_args()

    # 基础检查
    try:
        import torch
    except Exception:
        print("❌ 需要安装 PyTorch（建议 torch>=2.4.0）：pip install torch --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        raise

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ 未检测到可用 GPU，将改用 CPU（速度会很慢）")
        args.device = "cpu"

    # 选择 dtype：优先 bfloat16（Ampere+），否则 float16；CPU 回退 float32
    if args.device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # 下载模型到本地
    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"⬇️ 正在下载权重到本地：{local_dir}")
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            print("➡️ 安装 huggingface_hub： pip install -U 'huggingface_hub[cli]'", file=sys.stderr)
            raise
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,   # 直接落地完整文件，便于离线
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],  # 可选：略过不必要文件
        )
    else:
        print(f"✅ 已检测到本地权重目录：{local_dir}")

    # 载入 Diffusers 管线
    print("🚀 正在加载 Diffusers 管线 …")
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    # 单独加载 VAE 子目录（官方示例）
    vae = AutoencoderKLWan.from_pretrained(
        str(local_dir),
        subfolder="vae",
        torch_dtype=(torch.float32 if args.device == "cpu" else torch.float16)
    )

    pipe = WanPipeline.from_pretrained(
        str(local_dir),
        vae=vae,
        torch_dtype=dtype
    )

    # 放到设备
    pipe.to(args.device)

    # 省显存选项
    if args.offload:
        # 两种 offload 方案：按需生效
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    # T5 文本编码器放 CPU（进一步省显存；单 4090 常用）
    if args.t5_cpu:
        try:
            pipe.text_encoder.to("cpu")
        except Exception:
            print("ℹ️ 当前管线不含 text_encoder 或已在 CPU。")

    # 设置随机种子（可复现）
    if args.seed is not None:
        import random
        import numpy as np
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 参数校验
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("width/height 必须为 32 的倍数（例如 832x480）。")

    # 生成
    print("🎬 开始生成视频 …（首次运行会慢一些）")
    output_frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
    ).frames[0]

    # 导出视频
    out_path = Path(args.out).resolve()
    export_to_video(output_frames, str(out_path), fps=args.fps)
    print(f"✅ 已保存视频到：{out_path}")

if __name__ == "__main__":
    main()

