#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.1 T2V-1.3B (Diffusers) —— 小分辨率/短视频 + 捕获 DiT 输入输出 latent 到本地
保存内容：
  - dit_inputs.pt:  [Tensor, Tensor, ...]  （按调用顺序）
  - dit_outputs.pt: [Tensor, Tensor, ...]  （按调用顺序）
  - step_map.json:  { "calls": [ {"call_idx": i, "step": s, "is_cfg_negative": 0/1}, ... ] }
  - metadata.json:  运行参数、形状、dtype、设备等
  - output.mp4:     生成视频

建议在运行前卸载 xformers/flash-attn（避免你之前碰到的导入冲突）：
  pip uninstall -y xformers flash-attn flash_attn flash_attn_2
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# 避免 xformers 带来的导入问题（可选）
os.environ.setdefault("XFORMERS_DISABLED", "1")

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 T2V-1.3B: capture DiT latents")
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HF repo id")
    p.add_argument("--local_dir", default="./Wan2.1-T2V-1.3B-Diffusers", help="本地权重目录")
    p.add_argument("--save_dir", default="./wan_dit_dump", help="保存输出与latent的目录")
    p.add_argument("--prompt", default="A cat walks on the grass, realistic", help="正向提示词")
    p.add_argument("--negative_prompt", default=("Bright tones, overexposed, static, blurred details, subtitles, "
                                                 "style, works, paintings, images, static, overall gray, worst quality, "
                                                 "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                                                 "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                                                 "misshapen limbs, fused fingers, still picture, messy background, "
                                                 "three legs, many people in the background, walking backwards"),
                   help="反向提示词")
    # 小一些的分辨率与帧数
    p.add_argument("--width", type=int, default=448, help="宽（需为32的倍数）")
    p.add_argument("--height", type=int, default=256, help="高（需为32的倍数）")
    p.add_argument("--num_frames", type=int, default=33, help="帧数（越小越省资源）")
    p.add_argument("--fps", type=int, default=15, help="视频帧率")
    p.add_argument("--guidance_scale", type=float, default=5.0, help="CFG 引导系数")
    p.add_argument("--steps", type=int, default=20, help="去噪步数（越小越快）")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    p.add_argument("--offload", action="store_true", help="启用 CPU-Offload 省显存")
    p.add_argument("--t5_cpu", action="store_true", help="T5 文本编码器放到 CPU")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
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
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ 未检测到 GPU，自动切换到 CPU（会很慢）")
        args.device = "cpu"

    # dtype 选择：GPU优先 bf16 -> fp16，CPU 用 fp32
    if args.device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # 基本校验
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("width/height 必须为 32 的倍数，例如 448x256")

    save_root = Path(args.save_dir).expanduser().resolve()
    ts_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = save_root / f"run-{ts_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 下载或使用本地权重
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
        torch_dtype=dtype,
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

    # 找到 DiT / transformer 模块（WanPipeline 默认应为 pipe.transformer）
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("未在 WanPipeline 中找到 transformer（DiT）模块，无法挂钩捕获 latent。")

    transformer = pipe.transformer

    # 捕获容器
    dit_inputs, dit_outputs = [], []
    call_records = []  # 记录每次调用大致对应的 step 与是否负分支（CFG）
    # 注：严格的“步号/正负分支”在不改 pipeline 内部代码时难以精准拿到（Diffusers内部会把条件/无条件拼接/拆分）。
    # 这里采用启发式：按调用顺序记录，并在 metadata 里保存 num_inference_steps 与 guidance_scale，便于你后处理映射。

    def _to_cpu_detached(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu")
        if isinstance(x, (list, tuple)):
            return type(x)(_to_cpu_detached(xx) for xx in x)
        if isinstance(x, dict):
            return {k: _to_cpu_detached(v) for k, v in x.items()}
        return x

    def pre_hook(mod, inputs):
        # inputs 是一个 tuple；通常第一个是噪声 latent（或其变体）
        dit_inputs.append(_to_cpu_detached(inputs))
        # 记录基本信息（步号推断交给后处理，这里只占位）
        call_records.append({"call_idx": len(dit_inputs)-1})

    def fwd_hook(mod, inputs, output):
        # output 可能是张量或 tuple
        if isinstance(output, (list, tuple)):
            out = output[0]
        else:
            out = output
        dit_outputs.append(_to_cpu_detached(out))

    # 注册 hook
    pre_h = transformer.register_forward_pre_hook(pre_hook, with_kwargs=False)
    fwd_h = transformer.register_forward_hook(fwd_hook, with_kwargs=False)

    # 随机种子
    if args.seed is not None:
        import random, numpy as np
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    print("🎬 开始生成（小分辨率+短视频）并捕获 DiT latent …")
    try:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,   # 控制步数（Diffusers支持）
        )
    finally:
        # 确保移除 hook，避免潜在的内存/重复注册问题
        try: pre_h.remove()
        except Exception: pass
        try: fwd_h.remove()
        except Exception: pass

    frames = result.frames[0]

    # 保存视频
    out_video = run_dir / "output.mp4"
    export_to_video(frames, str(out_video), fps=args.fps)
    print(f"✅ 已保存视频：{out_video}")

    # 保存 latent（统一用 torch.save）
    import torch as _torch
    _torch.save(dit_inputs, run_dir / "dit_inputs.pt")
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
        "notes": "DiT 输入/输出是按 forward 调用顺序保存的；若使用了 CFG，每一步通常会有两次调用（无条件/有条件）。",
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 简单的调用到步数的启发式映射（不保证严格准确，但便于后处理）
    # 常见情况下：每个去噪步 step，会有 1（无CFG）或 2（有CFG）次 DiT 调用
    approx_calls_per_step = 2 if args.guidance_scale and args.guidance_scale > 1.0 else 1
    call_map = []
    for i in range(len(dit_inputs)):
        step_idx = i // approx_calls_per_step
        is_cfg_negative = (i % approx_calls_per_step == 0 and approx_calls_per_step == 2)  # 粗略标注
        call_map.append({"call_idx": i, "approx_step": int(step_idx), "is_cfg_negative": int(is_cfg_negative)})
    with open(run_dir / "step_map.json", "w", encoding="utf-8") as f:
        json.dump({"calls": call_map}, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 DiT latent 与元信息到目录：{run_dir}")
    print("   - dit_inputs.pt / dit_outputs.pt / metadata.json / step_map.json / output.mp4")
    print("ℹ️ 如需严格对齐每次调用与具体 scheduler 步，请考虑修改 Diffusers 源码或在 pipeline 内部 callback 处打点。")

if __name__ == "__main__":
    main()

