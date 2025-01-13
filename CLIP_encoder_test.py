from transformers import (
    CLIPModel, CLIPProcessor, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration, 
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection,
    CLIPImageProcessor, CLIPVisionModel, CLIPVisionModelWithProjection,
)

import os
import PIL
import time
import torch
import argparse

from accelerate import Accelerator

# Seed
generator = torch.manual_seed(0)

# Accelerator Setting
accelerator = Accelerator(
    mixed_precision="fp16",
)

image_path = "example/legacy_images/Hollywood-Sign.jpg"

# Load the image
image = PIL.Image.open(image_path)


# Load CLIP-L 
# CLIPL = CLIPVisionModelWithProjection.from_pretrained("/workspace/sd_models/clip-vit-large-patch14")
# CLIPL_processor = AutoProcessor.from_pretrained("/workspace/sd_models/clip-vit-large-patch14")

# Load OpenCLIP-G
CLIPG = CLIPVisionModelWithProjection.from_pretrained("/workspace/sd_models/CLIP-ViT-bigG-14-laion2B-39B-b160k")
CLIPG_processor = AutoProcessor.from_pretrained("/workspace/sd_models/CLIP-ViT-bigG-14-laion2B-39B-b160k")


# CLIPG, CLIPL, image = accelerator.prepare(CLIPG, CLIPL, image)
# CLIPL, CLIPL_processor, image = accelerator.prepare(CLIPL, CLIPL_processor, image)

# Encode the image
# image_input = CLIPL_processor(images=image, return_tensors="pt")['pixel_values']
image_input = CLIPG_processor(images=image, return_tensors="pt")['pixel_values']

image_features = CLIPG(image_input)

CLIPL, CLIPL_processor, image_input = accelerator.prepare(CLIPL, CLIPL_processor, image_input)