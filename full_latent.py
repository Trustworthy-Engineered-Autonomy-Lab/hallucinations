#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.1 T2V-1.3B (Diffusers)
- 分辨率: 832x480
- 帧数:   81
- 保存 DiT 输入/输出 latent（按 forward 调用顺序）
- 自动本地缓存权重
- 可选 CPU-Offload 与 T5 仅在 CPU 以省显存

用法示例：
  python wan_t2v_capture_dit_full.py --offload --t5_cpu \
    --prompt "A cat walks on the grass, realistic"

输出目录结构：
  wan_dit_dump/run-YYYYMMDD-HHMMSS/
    ├── output.mp4
    ├── dit_inputs.pt      # list of tuples (each is the forward inputs)
    ├── dit_outputs.pt     # list of Tensors (forward output)
    ├── metadata.json
    └── step_map.json      # 按调用顺序的启发式步数映射（CFG 粗略标注）
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# 避免 xformers/flash-attn 导入冲突（可选）
os.environ.setdefault("XFORMERS_DISABLED", "1")

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 T2V-1.3B: full video + capture DiT latents")
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HF repo id")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers", help="本地权重目录")

    # 正常长度与尺寸
    p.add_argument("--width", type=int, default=832, help="宽（需为32的倍数）")
    p.add_argument("--height", type=int, default=480, help="高（需为32的倍数）")
    p.add_argument("--num_frames", type=int, default=81, help="帧数（默认 81 ≈ 5.4s @15fps）")
    p.add_argument("--fps", type=int, default=15, help="导出视频帧率")

    # 采样/CFG
    p.add_argument("--guidance_scale", type=float, default=5.5, help="CFG 引导系数（推荐 5~6）")
    p.add_argument("--steps", type=int, default=30, help="去噪步数（Diffusers 的 num_inference_steps）")

    # 提示词
    p.add_argument("--prompt", default="A cat walks on the grass, realistic", help="正向提示词")
    p.add_argument("--negative_prompt", default=("Bright tones, overexposed, static, blurred details, subtitles, "
                                                 "style, works, paintings, images, static, overall gray, worst quality, "
                                                 "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                                                 "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                                                 "misshapen limbs, fused fingers, still picture, messy background, "
                                                 "three legs, many people in the background, walking backwards"),
                   help="反向提示词")

    # 设备/内存
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    p.add_argument("--offload", action="store_true", help="启用 CPU-Offload 省显存")
    p.add_argument("--t5_cpu", action="store_true", help="T5 文本编码器固定在 CPU")
    p.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")

    p.add_argument("--save_root", default="./wan_dit_dump", help="保存输出与 latent 的根目录")
    return p.parse_args()

def ensure_model(local_dir_str, repo_id):
    local_dir = Path(local_dir_str).expanduser().resolve()
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"⬇️ 正在下载权重到：{local_dir}")
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            print("➡️ 请先安装 huggingface_hub：pip install -U 'huggingface_hub[cli]'", file=sys.stderr)
            raise
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    else:
        print(f"✅ 使用本地权重目录：{local_dir}")
    return local_dir

def main():
    args = parse_args()

    # PyTorch & 设备
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ 未检测到 GPU，自动切换到 CPU（会很慢）")
        args.device = "cpu"

    # dtype：GPU 优先 bfloat16 -> float16，CPU 用 float32
    if args.device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # 基本校验
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("width/height 必须为 32 的倍数，例如 832x480")

    # 运行目录
    save_root = Path(args.save_root).expanduser().resolve()
    ts_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = save_root / f"run-{ts_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 权重
    local_dir = ensure_model(args.local_dir, args.model_id)

    # 加载管线
    print("🚀 加载 Diffusers WanPipeline …")
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    vae = AutoencoderKLWan.from_pretrained(
        str(local_dir),
        subfolder="vae",
        torch_dtype=(torch.float32 if args.device == "cpu" else torch.float16)
    )
    pipe = WanPipeline.from_pretrained(
        str(local_dir),
        vae=vae,
        torch_dtype=dtype
    ).to(args.device)

    # 省显存
    if args.offload:
        try: pipe.enable_sequential_cpu_offload()
        except Exception: pass
        try: pipe.enable_model_cpu_offload()
        except Exception: pass

    if args.t5_cpu:
        try:
            pipe.text_encoder.to("cpu")
        except Exception:
            print("ℹ️ 当前管线不含 text_encoder 或已在 CPU。")

    # 随机种子
    if args.seed is not None:
        import random, numpy as np
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    # 抓取 DiT（transformer）latent
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("未在 WanPipeline 中找到 transformer（DiT）模块，无法挂钩捕获 latent。")
    transformer = pipe.transformer

    dit_inputs, dit_outputs = [], []

    def _to_cpu_detached(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu")
        if isinstance(x, (list, tuple)):
            return type(x)(_to_cpu_detached(xx) for xx in x)
        if isinstance(x, dict):
            return {k: _to_cpu_detached(v) for k, v in x.items()}
        return x

    def pre_hook(mod, inputs):
        # inputs 是 tuple；通常第一个是 latent
        dit_inputs.append(_to_cpu_detached(inputs))

    def fwd_hook(mod, inputs, output):
        # output 可能是 tensor 或 tuple
        if isinstance(output, (list, tuple)):
            out = output[0]
        else:
            out = output
        dit_outputs.append(_to_cpu_detached(out))

    pre_h = transformer.register_forward_pre_hook(pre_hook, with_kwargs=False)
    fwd_h = transformer.register_forward_hook(fwd_hook, with_kwargs=False)

    print("🎬 开始生成 832x480, 81 帧 的视频，并捕获 DiT latent …")
    try:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
        )
    finally:
        # 移除 hook，避免重复注册与显存泄漏
        try: pre_h.remove()
        except Exception: pass
        try: fwd_h.remove()
        except Exception: pass

    frames = result.frames[0]

    # 导出视频
    out_video = run_dir / "output.mp4"
    export_to_video(frames, str(out_video), fps=args.fps)
    print(f"✅ 已保存视频：{out_video}")

    # 保存 latent
    import torch as _torch
    _torch.save(dit_inputs,  run_dir / "dit_inputs.pt")
    _torch.save(dit_outputs, run_dir / "dit_outputs.pt")

    # 元信息
    meta = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.steps,
        "dtype": str(dtype),
        "device": args.device,
        "model_dir": str(local_dir),
        "save_dir": str(run_dir),
        "seed": args.seed,
        "dit_calls": len(dit_inputs),
        "notes": "DiT 输入/输出按 forward 调用顺序保存；使用 CFG 时通常每步两次（无条件/有条件）。",
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 启发式步数映射（不保证严格准确；便于后处理）
    approx_calls_per_step = 2 if args.guidance_scale and args.guidance_scale > 1.0 else 1
    call_map = []
    for i in range(len(dit_inputs)):
        step_idx = i // approx_calls_per_step
        is_cfg_negative = (i % approx_calls_per_step == 0 and approx_calls_per_step == 2)
        call_map.append({"call_idx": i, "approx_step": int(step_idx), "is_cfg_negative": int(is_cfg_negative)})
    with open(run_dir / "step_map.json", "w", encoding="utf-8") as f:
        json.dump({"calls": call_map}, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 DiT latent 与元信息到：{run_dir}")
    print("   - dit_inputs.pt / dit_outputs.pt / metadata.json / step_map.json / output.mp4")

if __name__ == "__main__":
    main()

