import torch
import os
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
import numpy as np

class LatentSaver:
    def __init__(self, save_dir="latents_output"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.input_latents = []
        self.output_latents = []
        self.step_count = 0
    
    def save_latent(self, latent, name):
        """保存单个latent张量"""
        filepath = os.path.join(self.save_dir, f"{name}.pt")
        torch.save(latent.cpu(), filepath)
        print(f"Saved latent to {filepath}")
    
    def save_all_latents(self):
        """保存所有收集的latents"""
        if self.input_latents:
            torch.save(self.input_latents, os.path.join(self.save_dir, "all_input_latents.pt"))
            print(f"Saved {len(self.input_latents)} input latents")
        
        if self.output_latents:
            torch.save(self.output_latents, os.path.join(self.save_dir, "all_output_latents.pt"))
            print(f"Saved {len(self.output_latents)} output latents")

class WanPipelineWithLatentSaving(WanPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_saver = LatentSaver()
    
    def __call__(self, *args, **kwargs):
        # 保存原始的transformer forward方法
        original_forward = self.transformer.forward
        
        def forward_hook(self, hidden_states, *args, **kwargs):
            # 保存输入latent
            self.latent_saver.input_latents.append(hidden_states.detach().cpu())
            self.latent_saver.save_latent(
                hidden_states, 
                f"input_latent_step_{self.latent_saver.step_count}"
            )
            
            # 调用原始forward方法
            output = original_forward(hidden_states, *args, **kwargs)
            
            # 保存输出latent
            if hasattr(output, 'sample'):
                output_latent = output.sample
            else:
                output_latent = output
            
            self.latent_saver.output_latents.append(output_latent.detach().cpu())
            self.latent_saver.save_latent(
                output_latent, 
                f"output_latent_step_{self.latent_saver.step_count}"
            )
            
            self.latent_saver.step_count += 1
            return output
        
        # 临时替换forward方法
        self.transformer.forward = forward_hook.__get__(self.transformer, self.transformer.__class__)
        
        try:
            # 执行原始的__call__
            result = super().__call__(*args, **kwargs)
            
            # 保存所有latents
            self.latent_saver.save_all_latents()
            
            return result
        finally:
            # 恢复原始forward方法
            self.transformer.forward = original_forward

def main():
    # 减少 OMP 风险的辅助设置
    torch.cuda.empty_cache()
    
    model_id = "./Wan2.1-T2V-1.3B"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanPipelineWithLatentSaving.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.to("cuda:1")
    
    # 视频生成 prompt
    prompt = "A cat fights with a dog on the grass, realistic"
    negative_prompt = "overexposed, blurry, static, jpeg artifacts, bad quality"
    
    print("Starting video generation with latent saving...")
    
    # 生成视频并保存latents
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=384,
        width=640,
        num_frames=90,
        guidance_scale=5.0
    ).frames[0]
    
    # 导出视频
    export_to_video(output, "output_with_saved_latents.mp4", fps=15)
    
    print("Video generation completed!")
    print(f"Latents saved in: {pipe.latent_saver.save_dir}")

if __name__ == "__main__":
    main()