# define dataloaders for training and validation


# import for training
import os
import torch
from PIL import Image

from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)


if __name__ == '__main__':
    print("hello")