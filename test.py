import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "./Wan2.1-T2V-1.3B"  # 本地路径
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda:1")

prompt = "A cat walks on the grass, realistic"
negative_prompt = "overexposed, blurry, static, jpeg artifacts, bad quality"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]

export_to_video(output, "output.mp4", fps=15)

