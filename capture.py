#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.1 T2V-1.3B (Diffusers) —— 捕获 DiT 输入/输出（修正版）
- 修复：pre-hook 捕获到的 inputs 为空 () 的问题；现在会把所有 Tensor 转成 cpu + detach 后保存
- 同时捕获 kwargs（以 JSON 摘要形式保存其张量形状，避免重复保存大张量）
- 正常分辨率与时长：832x480, 81 帧；可通过参数修改
"""

import os
import sys
import json
import time
from pathlib import Path
import argparse

os.environ.setdefault("XFORMERS_DISABLED", "1")  # 规避 xformers/flash-attn 导入冲突

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 T2V-1.3B capture DiT inputs/outputs (fixed)")
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HF repo id")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers", help="本地权重目录")
    p.add_argument("--save_root", default="./wan_dit_dump", help="输出根目录")

    # 正常长度与尺寸
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=15)

    # 采样/CFG
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

    # 设备/内存
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--offload", action="store_true", help="启用 CPU-Offload 省显存")
    p.add_argument("--t5_cpu", action="store_true", help="T5 文本编码器固定在 CPU")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def ensure_model(local_dir_str, repo_id):
    local_dir = Path(local_dir_str).expanduser().resolve()
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"⬇️ 正在下载权重到：{local_dir}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    else:
        print(f"✅ 使用本地权重目录：{local_dir}")
    return local_dir

def to_cpu_detached(obj):
    """把任意嵌套结构里的 Tensor -> cpu().detach()；其它类型保持引用或做轻量转换。"""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu_detached(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu_detached(v) for k, v in obj.items()}
    else:
        return obj  # int/float/str/None 等原样返回

def summarize_shapes(obj):
    """仅摘要 kwargs 结构中的张量: 返回同结构、位置上用形状/ dtype 字符串代替实际张量。"""
    import torch
    if isinstance(obj, torch.Tensor):
        return {"_tensor": True, "shape": tuple(obj.shape), "dtype": str(obj.dtype)}
    elif isinstance(obj, (list, tuple)):
        return [summarize_shapes(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: summarize_shapes(v) for k, v in obj.items()}
    else:
        # 基本类型直接返回；复杂对象返回类型名
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return {"_type": type(obj).__name__}

def main():
    args = parse_args()

    # Torch & 设备
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ 未检测到 GPU，改用 CPU（较慢）")
        args.device = "cpu"

    dtype = torch.bfloat16 if (args.device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if args.device == "cuda" else torch.float32)

    # 校验尺寸
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("width/height 必须为 32 的倍数（例如 832x480）")

    # 目录
    save_root = Path(args.save_root).expanduser().resolve()
    run_dir = save_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    local_dir = ensure_model(args.local_dir, args.model_id)

    print("🚀 加载 Diffusers WanPipeline …")
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

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

    # 随机种子
    if args.seed is not None:
        import random, numpy as np
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    # 找 transformer
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("未找到 pipe.transformer（DiT 模块）。")

    transformer = pipe.transformer

    # 容器
    dit_inputs_args = []   # list[tuple(...)]  —— 已转换到 cpu
    dit_inputs_kwargs = [] # list[dict ...]    —— 仅摘要结构和张量形状
    dit_outputs = []       # list[tensor]      —— 已转换到 cpu

    # Hooks
    def pre_hook_args_kwargs(mod, inputs, kwargs=None):
        # inputs: tuple; kwargs: dict or None
        args_cpu = to_cpu_detached(inputs)                    # 把位置参数的 Tensor 拷到 CPU
        kwargs_summary = summarize_shapes(kwargs or {})       # 关键字参数只保存形状摘要
        dit_inputs_args.append(tuple(args_cpu))
        dit_inputs_kwargs.append(kwargs_summary)

    def fwd_hook(mod, inputs, output):
        # 仅保存代表性输出（若是 tuple/list 取第一个 Tensor）
        if isinstance(output, (list, tuple)):
            out = output[0]
        else:
            out = output
        if isinstance(out, torch.Tensor):
            dit_outputs.append(out.detach().to("cpu"))
        else:
            # 不常见分支：非张量输出，保存占位信息
            import numpy as _np
            dit_outputs.append(torch.tensor(0))  # 占位，避免长度不齐

    # 兼容不同 PyTorch 版本：优先 with_kwargs=True，否则回退
    try:
        pre_h = transformer.register_forward_pre_hook(pre_hook_args_kwargs, with_kwargs=True)
    except TypeError:
        # 老版本回退：没有 kwargs
        def pre_hook_only_args(mod, inputs):
            pre_hook_args_kwargs(mod, inputs, kwargs=None)
        pre_h = transformer.register_forward_pre_hook(pre_hook_only_args)

    fwd_h = transformer.register_forward_hook(fwd_hook, with_kwargs=False)

    print("🎬 生成视频并捕获 DiT inputs/outputs …")
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
        try: pre_h.remove()
        except Exception: pass
        try: fwd_h.remove()
        except Exception: pass

    frames = result.frames[0]
    out_video = run_dir / "output.mp4"
    from diffusers.utils import export_to_video
    export_to_video(frames, str(out_video), fps=args.fps)
    print(f"✅ 已保存视频：{out_video}")

    # 保存
    torch.save(dit_inputs_args, run_dir / "dit_inputs.pt")
    torch.save(dit_outputs,    run_dir / "dit_outputs.pt")
    with open(run_dir / "dit_inputs_kwargs.json", "w", encoding="utf-8") as f:
        json.dump(dit_inputs_kwargs, f, ensure_ascii=False, indent=2)

    # 元信息
    meta = {
        "width": args.width, "height": args.height, "num_frames": args.num_frames, "fps": args.fps,
        "guidance_scale": args.guidance_scale, "num_inference_steps": args.steps,
        "dtype": str(dtype), "device": args.device, "seed": args.seed,
        "model_dir": str(local_dir), "save_dir": str(run_dir)
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 启发式 step map（CFG 情况下每步 2 次调用）
    approx_calls_per_step = 2 if args.guidance_scale and args.guidance_scale > 1.0 else 1
    call_map = []
    for i in range(len(dit_inputs_args)):
        step_idx = i // approx_calls_per_step
        is_cfg_negative = (i % approx_calls_per_step == 0 and approx_calls_per_step == 2)
        call_map.append({"call_idx": i, "approx_step": int(step_idx), "is_cfg_negative": int(is_cfg_negative)})
    with open(run_dir / "step_map.json", "w", encoding="utf-8") as f:
        json.dump({"calls": call_map}, f, ensure_ascii=False, indent=2)

    print(f"✅ 捕获完成，文件保存在：{run_dir}")
    print("   - dit_inputs.pt（位置参数，含 latent/timestep/text_embeds 等，Tensor 已转 CPU）")
    print("   - dit_inputs_kwargs.json（关键字参数的结构/形状摘要）")
    print("   - dit_outputs.pt（输出 Tensor，CPU）")
    print("   - metadata.json / step_map.json / output.mp4")

if __name__ == "__main__":
    main()

