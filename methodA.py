#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法 A：多随机种子生成 + 捕获并保存 DiT 输入/输出 latent（Diffusers / Wan2.1 T2V-1.3B）

每个 seed 会单独生成：
  seed{SEED}/
    ├── output.mp4
    ├── dit_inputs.pt         # list[tuple(...)]，所有张量已 cpu().detach()
    ├── dit_outputs.pt        # list[tensor]（或代表性第 0 个 tensor），已 cpu().detach()
    ├── metadata.json
    └── step_map.json         # 启发式映射：call_idx -> step/uncond-cond

可选：
  └── dit_inputs_kwargs.json  # forward kwargs 的结构与张量形状摘要（不保存实际巨大张量）
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# 避免 xformers / flash-attn 导入冲突
os.environ.setdefault("XFORMERS_DISABLED", "1")

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 multi-seed with DiT latent capture")
    # 模型
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers")
    # 存储
    p.add_argument("--save_root", default="./wan_multi_runs")
    # 视频设置
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=15)
    # 采样与 CFG
    p.add_argument("--guidance_scale", type=float, default=5.5)
    p.add_argument("--steps", type=int, default=30)
    # 提示词
    p.add_argument("--prompt", default="A cat walks on the grass, realistic")
    p.add_argument("--negative_prompt", default=("Bright tones, overexposed, static, blurred details, subtitles, "
                                                 "style, works, paintings, images, static, overall gray, worst quality, "
                                                 "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                                                 "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                                                 "misshapen limbs, fused fingers, still picture, messy background, "
                                                 "three legs, many people in the background, walking backwards"))
    # 多 seed
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    # 设备/内存
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--offload", action="store_true", help="启用 CPU-Offload 以省显存")
    p.add_argument("--t5_cpu", action="store_true", help="把 T5 编码器放到 CPU")
    # 可选：保存 kwargs 摘要
    p.add_argument("--save_kwargs_summary", action="store_true",
                   help="保存每次 forward 的 kwargs 结构摘要到 dit_inputs_kwargs.json（不含大张量）")
    return p.parse_args()

def ensure_model(local_dir_str, repo_id):
    local_dir = Path(local_dir_str).expanduser().resolve()
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"⬇️ 正在下载模型到：{local_dir}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    else:
        print(f"✅ 使用本地模型目录：{local_dir}")
    return local_dir

def to_cpu_detached(obj):
    """把嵌套结构里所有 Tensor → cpu().detach()；其它类型保持。"""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu_detached(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu_detached(v) for k, v in obj.items()}
    else:
        return obj

def summarize_shapes(obj):
    """只摘要结构：把 kwargs 中的张量替换为 {'_tensor': True, 'shape': ..., 'dtype': ...}。"""
    import torch
    if isinstance(obj, torch.Tensor):
        return {"_tensor": True, "shape": tuple(obj.shape), "dtype": str(obj.dtype)}
    elif isinstance(obj, (list, tuple)):
        return [summarize_shapes(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: summarize_shapes(v) for k, v in obj.items()}
    else:
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return {"_type": type(obj).__name__}

def main():
    args = parse_args()
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    # 设备 & dtype
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ 未检测到 GPU，切换到 CPU（速度较慢）")
        args.device = "cpu"

    if args.device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # 尺寸校验
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("width/height 必须是 32 的倍数（例如 832x480）")

    # 模型
    local_dir = ensure_model(args.local_dir, args.model_id)
    print("🚀 正在加载管线 …")
    vae = AutoencoderKLWan.from_pretrained(str(local_dir), subfolder="vae",
                                           torch_dtype=(torch.float32 if args.device == "cpu" else torch.float16))
    pipe = WanPipeline.from_pretrained(str(local_dir), vae=vae, torch_dtype=dtype).to(args.device)

    if args.offload:
        try: pipe.enable_sequential_cpu_offload()
        except Exception: pass
        try: pipe.enable_model_cpu_offload()
        except Exception: pass
    if args.t5_cpu:
        try: pipe.text_encoder.to("cpu")
        except Exception: pass

    # transformer 句柄
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("未在管线中找到 transformer（DiT）模块。")
    transformer = pipe.transformer

    # 根目录
    save_root = Path(args.save_root).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    # 多 seed 循环
    for seed in args.seeds:
        print(f"\n🎬 生成 seed={seed} 的结果并捕获 latent …")
        run_dir = save_root / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 存放捕获
        dit_inputs_args = []   # list[tuple(...)]（所有张量已转CPU）
        dit_outputs = []       # list[tensor]     （代表性输出，已转CPU）
        dit_inputs_kwargs_summary = []  # 结构摘要（可选）

        # hooks
        def pre_hook(mod, inputs, kwargs=None):
            # 位置参数：真正保存张量（cpu/detach）
            dit_inputs_args.append(tuple(to_cpu_detached(inputs)))
            # 关键字参数：根据需要只保存结构摘要，避免冗余
            if args.save_kwargs_summary:
                dit_inputs_kwargs_summary.append(summarize_shapes(kwargs or {}))

        def fwd_hook(mod, inputs, output):
            # 仅保存代表性输出（若是 list/tuple 取第一个张量）
            if isinstance(output, (list, tuple)):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                dit_outputs.append(out.detach().to("cpu"))
            else:
                # 极少情况：非张量输出，保存占位
                dit_outputs.append(torch.tensor(0))

        # 兼容不同 PyTorch 版本：优先 with_kwargs=True
        try:
            pre_h = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        except TypeError:
            def pre_hook_no_kwargs(mod, inputs):
                pre_hook(mod, inputs, kwargs=None)
            pre_h = transformer.register_forward_pre_hook(pre_hook_no_kwargs)

        fwd_h = transformer.register_forward_hook(fwd_hook, with_kwargs=False)

        # 为该 seed 构造独立 generator（位于目标设备）
        gen_device = "cuda" if args.device == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        try:
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=generator,
            )
        finally:
            # 清理 hooks
            try: pre_h.remove()
            except Exception: pass
            try: fwd_h.remove()
            except Exception: pass

        # 导出视频
        frames = result.frames[0]
        out_video = run_dir / "output.mp4"
        export_to_video(frames, str(out_video), fps=args.fps)
        print(f"✅ 视频已保存：{out_video}")

        # 保存 latent
        torch.save(dit_inputs_args, run_dir / "dit_inputs.pt")
        torch.save(dit_outputs,    run_dir / "dit_outputs.pt")
        if args.save_kwargs_summary:
            with open(run_dir / "dit_inputs_kwargs.json", "w", encoding="utf-8") as f:
                json.dump(dit_inputs_kwargs_summary, f, ensure_ascii=False, indent=2)

        # 元信息
        meta = {
            "seed": int(seed),
            "width": args.width, "height": args.height, "num_frames": args.num_frames, "fps": args.fps,
            "guidance_scale": args.guidance_scale, "num_inference_steps": args.steps,
            "dtype": str(dtype), "device": args.device,
            "model_dir": str(local_dir), "save_dir": str(run_dir),
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 启发式 step map（CFG 时每步 2 次调用）
        approx_calls_per_step = 2 if args.guidance_scale and args.guidance_scale > 1.0 else 1
        call_map = []
        for i in range(len(dit_inputs_args)):
            step_idx = i // approx_calls_per_step
            is_cfg_negative = (i % approx_calls_per_step == 0 and approx_calls_per_step == 2)
            call_map.append({"call_idx": i, "approx_step": int(step_idx), "is_cfg_negative": int(is_cfg_negative)})
        with open(run_dir / "step_map.json", "w", encoding="utf-8") as f:
            json.dump({"calls": call_map}, f, ensure_ascii=False, indent=2)

        # 简要检查
        print(f"🔎 捕获检查：inputs={len(dit_inputs_args)}, outputs={len(dit_outputs)} "
              f"(期望 ~ {args.steps * approx_calls_per_step})")

    print("\n🎉 全部种子完成。输出位于：", save_root)

if __name__ == "__main__":
    main()

