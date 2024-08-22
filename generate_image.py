import torch
from diffusers import FluxPipeline, StableDiffusionPipeline
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.DEBUG)

with open('token', 'r') as f:
    token = f.read().strip()
login(token=token)

# This is required if you don't have NVIDIA GPU, since by default torch uses CUDA.
# pretrained_data = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32)
pretrained_data = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

# Optimization for memory usage (ram and swap)
# pretrained_data.enable_sequential_cpu_offload()
# pretrained_data.enable_gradient_checkpointing()

pretrained_data.to('cpu')
# pretrained_data.enable_model_cpu_offload()


# This is if the GPU is nvidia
# pretrained_data = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#
# bellow is to offload some data to CPU.
# pretrained_data.to('cpu')
# pretrained_data.enable_model_cpu_offload()


input_data = input('Enter your request:\n')

image = pretrained_data(
    input_data,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=25,
    generator=torch.manual_seed(0)
).images[0]

image.save('generated_image.png')

