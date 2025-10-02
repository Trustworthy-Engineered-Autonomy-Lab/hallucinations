#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法 C：单次运行批量生成（Batch）+ 捕获并保存 DiT 输入/输出 latent（Diffusers / Wan2.1 T2V-1.3B）

输出结构：
  wan_batch_runs/session-YYYYMMDD-HHMMSS/
    ├── batched/
    │   ├── dit_inputs_batched.pt    # list[tuple(...)]，其中张量的 batch 维度=B
    │   └── dit_outputs_batched.pt   # list[tensor]，shape=(B, C, T, H, W)
    └── sample00/
        ├── output.mp4
        ├── dit_inputs.pt            # 已按样本切分后的 list[tuple(...)]（各张量 batch 维已切成单样本）
        ├── dit_outputs.pt           # list[tensor]（各张量去掉 batch 维 -> (1, C, T, H, W) 或 (C,T,H,W)）
        ├── metadata.json
        └── step_map.json
       sample01/...
       ...

注意：
- 启用 CFG (guidance_scale>1) 时，每步有 2 次 DiT 调用；若 num_inference_steps=30，则总 calls ≈ 60（每个调用返回一个 (B,C,T,H,W) 的输出）。
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")  # 避免 xformers/flash-attn 导入冲突

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 batch generation with DiT latent capture")
    # 模型
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers")
    # 存储
    p.add_argument("--save_root", default="./wan_batch_runs")
    # 视频设置
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=15)
    # 采样与 CFG
    p.add_argument("--guidance_scale", type=float, default=5.5)
    p.add_argument("--steps", type=int, default=30)
    # 提示词：二选一
    p.add_argument("--prompt", default="A cat walks on the grass, realistic",
                   help="单个提示词，若与 --batch 一起使用将被复制 batch 次")
    p.add_argument("--prompts", default=None,
                   help="用 '||' 分隔的多个提示词，例如：'cat on grass||a dog running'")
    p.add_argument("--negative_prompt", default=("Bright tones, overexposed, static, blurred details, subtitles, "
                                                 "style, works, paintings, images, static, overall gray, worst quality, "
                                                 "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                                                 "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                                                 "misshapen limbs, fused fingers, still picture, messy background, "
                                                 "three legs, many people in the background, walking backwards"))
    # 批量大小（当没提供 --prompts 时生效）
    p.add_argument("--batch", type=int, default=4, help="一次生成的样本数量（与 prompts 数量一致）")
    # 随机种子（可给每个样本一个；长度需= batch；若不提供则自动生成 base_seed+i）
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--base_seed", type=int, default=1000)
    # 设备/内存
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--offload", action="store_true", help="启用 CPU-Offload 以省显存")
    p.add_argument("--t5_cpu", action="store_true", help="把 T5 文本编码器放到 CPU")
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

def split_tensor_first_dim(x, B):
    """把第 0 维是 batch 的张量切成 B 份；否则保持不变。"""
    import torch
    if not isinstance(x, torch.Tensor):
        return [x] * B
    if x.dim() >= 1 and x.size(0) == B:
        return [x[i:i+1].contiguous() for i in range(B)]  # 保留 batch 维为 1，便于一致性
    else:
        # 若不是按 batch 堆叠的张量（例如常量、标量），复制引用
        return [x] * B

def split_args_by_batch(args_tuple, B):
    """
    将一次 call 的 inputs（tuple）按 batch 切分成 B 份，返回 list[tuple(...)]。
    对 tuple 中每个张量，若第 0 维是 B，则沿第 0 维切分；否则原样复制。
    """
    per_sample = [list() for _ in range(B)]
    for item in args_tuple:
        parts = split_tensor_first_dim(item, B)
        for i in range(B):
            per_sample[i].append(parts[i])
    return [tuple(x) for x in per_sample]

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

    # prompts 解析
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split("||") if p.strip()]
    else:
        prompts = [args.prompt] * int(args.batch)
    B = len(prompts)
    print(f"📦 Batch size (B) = {B}")

    # seeds
    if args.seeds is None:
        seeds = [int(args.base_seed) + i for i in range(B)]
    else:
        if len(args.seeds) != B:
            raise ValueError(f"--seeds 长度({len(args.seeds)})必须与 batch 大小({B})一致")
        seeds = [int(s) for s in args.seeds]

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

    # 保存根目录
    save_root = Path(args.save_root).expanduser().resolve()
    tag = time.strftime("%Y%m%d-%H%M%S")
    session_dir = save_root / f"session-{tag}"
    (session_dir / "batched").mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录：{session_dir}")

    # 捕获容器（批量）
    dit_inputs_args_batched = []   # list[tuple(...)]（各张量 shape 的第 0 维是 B）
    dit_outputs_batched = []       # list[tensor (B,C,T,H,W)]

    # hooks
    def pre_hook(mod, inputs, kwargs=None):
        dit_inputs_args_batched.append(tuple(to_cpu_detached(inputs)))
    def fwd_hook(mod, inputs, output):
        if isinstance(output, (list, tuple)):
            out = output[0]
        else:
            out = output
        if isinstance(out, torch.Tensor):
            dit_outputs_batched.append(out.detach().to("cpu"))
        else:
            dit_outputs_batched.append(torch.tensor(0))

    # 注册 hooks（优先 with_kwargs）
    try:
        pre_h = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    except TypeError:
        def pre_hook_no_kwargs(mod, inputs):
            pre_hook(mod, inputs, kwargs=None)
        pre_h = pipe.transformer.register_forward_pre_hook(pre_hook_no_kwargs)
    fwd_h = pipe.transformer.register_forward_hook(fwd_hook, with_kwargs=False)

    # 为每个样本构造独立 generator 列表
    gen_device = "cuda" if args.device == "cuda" else "cpu"
    generators = [torch.Generator(device=gen_device).manual_seed(s) for s in seeds]

    print("🎬 单次批量生成并捕获 latent …")
    try:
        with torch.no_grad():
            result = pipe(
                prompt=prompts,                       # 传入列表，形成 batch
                negative_prompt=[args.negative_prompt]*B,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=generators,                 # 与 batch 对齐的 generator 列表
            )
    finally:
        try: pre_h.remove()
        except Exception: pass
        try: fwd_h.remove()
        except Exception: pass

    # === 保存批量原始捕获（方便后续自定义拆分/分析） ===
    torch.save(dit_inputs_args_batched, session_dir / "batched" / "dit_inputs_batched.pt")
    torch.save(dit_outputs_batched,    session_dir / "batched" / "dit_outputs_batched.pt")

    # === 拆分并分别为每个样本保存 ===
    approx_calls_per_step = 2 if args.guidance_scale and args.guidance_scale > 1.0 else 1
    frames_list = result.frames  # list 长度 B
    assert len(frames_list) == B, "返回的 frames 数量与 batch 不匹配"

    for i in range(B):
        sample_dir = session_dir / f"sample{i:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 导出视频
        from diffusers.utils import export_to_video
        export_to_video(frames_list[i], str(sample_dir / "output.mp4"), fps=args.fps)

        # 拆分 inputs：每次 call 的 tuple -> 切出第 i 个样本
        per_sample_inputs = []
        for call_args in dit_inputs_args_batched:
            per_sample_inputs.append(split_args_by_batch(call_args, B)[i])
        torch.save(per_sample_inputs, sample_dir / "dit_inputs.pt")

        # 拆分 outputs：每次 call 的张量 -> 取第 i 个 batch 切片
        per_sample_outputs = []
        for out in dit_outputs_batched:
            if isinstance(out, torch.Tensor) and out.dim() >= 1 and out.size(0) == B:
                per_sample_outputs.append(out[i:i+1].contiguous())  # 保留 batch 维=1
            else:
                per_sample_outputs.append(out)  # 罕见情况：非张量或无 batch 维
        torch.save(per_sample_outputs, sample_dir / "dit_outputs.pt")

        # 元信息
        meta = {
            "index": i,
            "seed": seeds[i],
            "prompt": prompts[i],
            "width": args.width, "height": args.height, "num_frames": args.num_frames, "fps": args.fps,
            "guidance_scale": args.guidance_scale, "num_inference_steps": args.steps,
            "dtype": str(dtype), "device": args.device,
            "model_dir": str(local_dir), "save_dir": str(sample_dir),
            "batch_size": B,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 步数映射（启发式）
        call_map = []
        for call_idx in range(len(dit_outputs_batched)):
            step_idx = call_idx // approx_calls_per_step
            is_cfg_negative = (call_idx % approx_calls_per_step == 0 and approx_calls_per_step == 2)
            call_map.append({"call_idx": call_idx, "approx_step": int(step_idx), "is_cfg_negative": int(is_cfg_negative)})
        with open(sample_dir / "step_map.json", "w", encoding="utf-8") as f:
            json.dump({"calls": call_map}, f, ensure_ascii=False, indent=2)

    print("\n✅ 完成。所有结果位于：", session_dir)

if __name__ == "__main__":
    main()

