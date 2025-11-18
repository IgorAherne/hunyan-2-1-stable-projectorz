# hy3dpaint\utils\image_super_utils.py

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
import numpy as np
from PIL import Image
from typing import List, Union
import gc

class imageSuperNet:
    def __init__(self, config) -> None:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # FIX: Enable tiling (tile=512) to strictly respect VRAM limits (e.g., 8GB).
        # tile=0 means "process full image", which causes OOM/Shared Memory spill on 3K images.
        upsampler = RealESRGANer(
            scale=4,
            model_path=config.realesrgan_ckpt_path,
            dni_weight=None,
            model=model,
            tile=512,       # Optimization: Process in 512px chunks
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=self.device,
        )
        self.model = upsampler.model
        self.upsampler = upsampler # Keep reference to use built-in enhance method with tiling support
        self.half = True

    @torch.no_grad()
    def __call__(self, image: Union[Image.Image, List[Image.Image]], batch_size=1) -> Union[Image.Image, List[Image.Image]]:
        """
        Run super resolution. 
        Default batch_size=1 to prevent shared memory spill on 8GB cards.
        """
        is_list = isinstance(image, list)
        images = image if is_list else [image]
        
        results = []
        
        # Process images one by one (or in small batches) to keep VRAM low
        for img_pil in images:
            # Convert PIL to numpy (RGB)
            img_np = np.array(img_pil)
            
            # RealESRGANer expects numpy input. 
            # We use the built-in .enhance() method because it handles the TILING logic automatically.
            # The direct model() call in previous optimization skipped tiling logic.
            try:
                # enhance returns (output_bgra, mode)
                output, _ = self.upsampler.enhance(img_np, outscale=4)
                
                # Output is numpy array, convert back to PIL
                results.append(Image.fromarray(output))
            except RuntimeError as e:
                print(f"Error in SuperResolution: {e}")
                # If tiling fails (rare), try to clear cache and retry
                torch.cuda.empty_cache()
                output, _ = self.upsampler.enhance(img_np, outscale=4)
                results.append(Image.fromarray(output))
        
        if is_list:
            return results
        return results[0]

    def free_memory(self):
        """Explicitly clear model from memory"""
        del self.model
        del self.upsampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
