import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

# 减少 OOM 风险的辅助设置（可选）
torch.cuda.empty_cache()

model_id = "./Wan2.1-T2V-1.3B"  # 本地路径
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
pipe.to("cuda:1")

# 你的视频生成 prompt
prompt = "A cat fights with a dog on the grass, realistic"
negative_prompt = "overexposed, blurry, static, jpeg artifacts, bad quality"

# ✅ 优化分辨率与帧数，降低显存需求
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=384,       # ↓ 分辨率降低
    width=640,
    num_frames=90,    # ↓ 帧数减半
    guidance_scale=5.0
).frames[0]

export_to_video(output, "output.mp4", fps=15)

