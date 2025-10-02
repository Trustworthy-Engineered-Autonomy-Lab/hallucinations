#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法 B：开启 dropout 的多样本生成 + 捕获并保存 DiT 输入/输出 latent（Diffusers / Wan2.1 T2V-1.3B）

每个样本（sampleK）会单独生成目录，包含：
  sampleK/
    ├── output.mp4
    ├── dit_inputs.pt          # list[tuple(...)]，所有张量已 cpu().detach()
    ├── dit_outputs.pt         # list[tensor]（或代表性第 0 个 tensor），已 cpu().detach()
    ├── metadata.json
    └── step_map.json
  （可选）dit_inputs_kwargs.json  # forward kwargs 的结构/张量形状摘要（不含大张量）

关键点：
- 强制开启 DiT 的 dropout：pipe.transformer.train()
- 可选也开启文本编码器的 dropout：--text_dropout
- --same_noise：所有样本共享同一初始噪声；差异来源于 dropout mask
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")  # 避免 xformers/flash-attn 导入冲突

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 dropout multi-sample with DiT latent capture")
    # 模型
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers")
    # 存储
    p.add_argument("--save_root", default="./wan_dropout_runs")
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
    # 多样本 & 随机性
    p.add_argument("--n_samples", type=int, default=3, help="要生成的样本数量")
    p.add_argument("--base_seed", type=int, default=1234, help="基准种子")
    p.add_argument("--same_noise", action="store_true",
                   help="让所有样本使用相同初始噪声（区别仅来自 dropout）")
    # 设备/内存
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--offload", action="store_true", help="启用 CPU-Offload 以省显存")
    p.add_argument("--t5_cpu", action="store_true", help="把 T5 编码器放到 CPU")
    # dropout 控制
    p.add_argument("--text_dropout", action="store_true", help="同时开启文本编码器的 dropout（可带来更多随机性）")
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
        print("⚠️ 未检测到 GPU，切换到 CPU（较慢）")
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

    # 开启 dropout —— 关键步骤
    # DiT（transformer）启用 train() 以激活其中的 Dropout
    pipe.transformer.train()
    # 可选：文本编码器也开启 dropout（随机性更强）
    if args.text_dropout and hasattr(pipe, "text_encoder"):
        try:
            pipe.text_encoder.train()
        except Exception:
            pass

    # 根目录
    save_root = Path(args.save_root).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d-%H%M%S")
    session_dir = save_root / f"session-{tag}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 生成 n 个样本
    gen_device = "cuda" if args.device == "cuda" else "cpu"

    # 如果 same_noise=True，我们给所有样本用同一个 generator（相同初始噪声）
    fixed_noise_gen = None
    if args.same_noise:
        fixed_noise_gen = torch.Generator(device=gen_device).manual_seed(int(args.base_seed))

    for k in range(args.n_samples):
        print(f"\n🎬 生成样本 #{k} 并捕获 latent …")
        sample_dir = session_dir / f"sample{k:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 构建 generator：
        # - same_noise=True：所有样本共享 fixed_noise_gen（同一初始噪声）
        # - same_noise=False：每个样本都有不同初始噪声（base_seed + k）
        if args.same_noise:
            generator = fixed_noise_gen
        else:
            generator = torch.Generator(device=gen_device).manual_seed(int(args.base_seed) + k)

        # 为了让 dropout 掩码在每个样本间不同，即使 same_noise=True，
        # 我们也显式扰动 PyTorch 的全局 RNG（影响 dropout）
        torch.manual_seed(int(args.base_seed) + 100000 + k)
        if args.device == "cuda":
            torch.cuda.manual_seed(int(args.base_seed) + 200000 + k)

        # 捕获容器
        dit_inputs_args = []
        dit_outputs = []
        dit_inputs_kwargs_summary = []

        # hooks
        def pre_hook(mod, inputs, kwargs=None):
            dit_inputs_args.append(tuple(to_cpu_detached(inputs)))
            if args.save_kwargs_summary:
                dit_inputs_kwargs_summary.append(summarize_shapes(kwargs or {}))

        def fwd_hook(mod, inputs, output):
            if isinstance(output, (list, tuple)):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                dit_outputs.append(out.detach().to("cpu"))
            else:
                dit_outputs.append(torch.tensor(0))

        # 注册 hooks（优先 with_kwargs）
        try:
            pre_h = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        except TypeError:
            def pre_hook_no_kwargs(mod, inputs):
                pre_hook(mod, inputs, kwargs=None)
            pre_h = pipe.transformer.register_forward_pre_hook(pre_hook_no_kwargs)
        fwd_h = pipe.transformer.register_forward_hook(fwd_hook, with_kwargs=False)

        try:
            # 重要：在开启 train() 但仍保持推理图不追踪梯度
            with torch.no_grad():
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
            try: pre_h.remove()
            except Exception: pass
            try: fwd_h.remove()
            except Exception: pass

        # 导出视频
        from diffusers.utils import export_to_video
        frames = result.frames[0]
        out_video = sample_dir / "output.mp4"
        export_to_video(frames, str(out_video), fps=args.fps)
        print(f"✅ 已保存视频：{out_video}")

        # 保存 latent
        import torch as _torch
        _torch.save(dit_inputs_args, sample_dir / "dit_inputs.pt")
        _torch.save(dit_outputs,    sample_dir / "dit_outputs.pt")
        if args.save_kwargs_summary:
            with open(sample_dir / "dit_inputs_kwargs.json", "w", encoding="utf-8") as f:
                json.dump(dit_inputs_kwargs_summary, f, ensure_ascii=False, indent=2)

        # 元信息
        meta = {
            "sample_index": k,
            "width": args.width, "height": args.height, "num_frames": args.num_frames, "fps": args.fps,
            "guidance_scale": args.guidance_scale, "num_inference_steps": args.steps,
            "dtype": str(dtype), "device": args.device,
            "model_dir": str(local_dir), "save_dir": str(sample_dir),
            "base_seed": int(args.base_seed),
            "same_noise": bool(args.same_noise),
            "dropout": True,
            "text_dropout": bool(args.text_dropout),
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 启发式 step map（CFG 时每步 2 次调用）
        approx_calls_per_step = 2 if args.guidance_scale and args.guidance_scale > 1.0 else 1
        call_map = []
        for i in range(len(dit_inputs_args)):
            step_idx = i // approx_calls_per_step
            is_cfg_negative = (i % approx_calls_per_step == 0 and approx_calls_per_step == 2)
            call_map.append({"call_idx": i, "approx_step": int(step_idx), "is_cfg_negative": int(is_cfg_negative)})
        with open(sample_dir / "step_map.json", "w", encoding="utf-8") as f:
            json.dump({"calls": call_map}, f, ensure_ascii=False, indent=2)

        print(f"🔎 捕获检查：inputs={len(dit_inputs_args)}, outputs={len(dit_outputs)} "
              f"(期望 ~ {args.steps * approx_calls_per_step})")

    print("\n🎉 完成。所有输出位于：", session_dir)

if __name__ == "__main__":
    main()

