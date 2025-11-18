# hy3dpaint\utils\multiview_utils.py

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

import os
import gc
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler
from diffusers.models.transformers.transformer_2d import BasicTransformerBlock

from hy3dpaint.hunyuanpaintpbr.pipeline import HunyuanPaintPipeline
from hy3dpaint.hunyuanpaintpbr.unet.modules import UNet2p5DConditionModel, Basic2p5DTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        # custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr") # to be deleted
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        model_path = huggingface_hub.snapshot_download(
            repo_id=config.multiview_pretrained_path,
            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        )

        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        
        # Manually load components to ensure local code is used, bypassing diffusers' caching of custom pipelines.
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        
        # Use the custom UNet's from_pretrained method to load the model with the correct architecture.
        unet_path = os.path.join(model_path, "unet")
        unet = UNet2p5DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)

        feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

        # Set the memory format for the models that perform heavy convolution operations.
        vae = vae.to(memory_format=torch.channels_last)
        unet = unet.to(memory_format=torch.channels_last)

        pipeline = HunyuanPaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            use_torch_compile=True, 
        )

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        #pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))

        # Apply VRAM optimizations to both the main UNet and the dual-stream UNet.
        main_unet = pipeline.unet.unet
        dual_unet = pipeline.unet.unet_dual

        # 1. Enable Gradient Checkpointing for both
        main_unet.enable_gradient_checkpointing()
        if dual_unet is not None:
            dual_unet.enable_gradient_checkpointing()

        # 2. Manually enable Forward Chunking for both
        for unet_model in [main_unet, dual_unet]:
            if unet_model is None:
                continue
            for mod in unet_model.modules():
                if isinstance(mod, (BasicTransformerBlock, Basic2p5DTransformerBlock)):
                    mod._chunk_size = 1
                    mod._chunk_dim = 1

        # Enable model CPU offloading to save VRAM
        pipeline.enable_model_cpu_offload()

        self.pipeline = pipeline # Keep on CPU/managed by accelerate

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            # Keep DINO on CPU to save VRAM

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    def free_memory(self):
        """Frees up memory by deleting models and clearing cache."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if hasattr(self, 'dino_v2'):
            del self.dino_v2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, cache=None, num_inference_steps=None):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input, cache=cache,
            num_inference_steps=num_inference_steps
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, cache=None, num_inference_steps=None):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        
        # Use the pipeline's execution device for the generator
        pipeline_device = self.pipeline._execution_device

        kwargs = dict(generator=torch.Generator(device=pipeline_device).manual_seed(0))
        if cache is not None:
            kwargs["cache"] = cache

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            # Check cache first for DINO features
            if cache is not None and "dino_hidden_states" in cache:
                print("Reusing cached DINO hidden states.")
                dino_hidden_states = cache["dino_hidden_states"].to(self.pipeline._execution_device)
                self.dino_v2.to("cpu") # Make sure dino remains on CPU
            else:
                # If not in cache, compute and store them
                import time
                dino_device = self.pipeline._execution_device
                self.dino_v2.to(dino_device)
                
                torch.cuda.synchronize()
                dino_start_time = time.time()
                
                dino_hidden_states = self.dino_v2(input_images[0])
                
                torch.cuda.synchronize()
                print(f"[PROFILING] DINO feature extraction took: {time.time() - dino_start_time:.2f}s")
                
                self.dino_v2.to("cpu") # Offload DINO back to CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if cache is not None:
                    cache["dino_hidden_states"] = dino_hidden_states.cpu() # Store on CPU

            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }
        # Use the passed num_inference_steps only if it's provided (for the warm-up).
        # Otherwise, fall back to the scheduler-specific dictionary for the main run.
        num_inference_steps = num_inference_steps if num_inference_steps is not None else infer_steps_dict[self.pipeline.scheduler.__class__.__name__]

        import time
        torch.cuda.synchronize()
        pipeline_start_time = time.time()
        
        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images

        torch.cuda.synchronize()
        print(f"[PROFILING] Main diffusion pipeline call took: {time.time() - pipeline_start_time:.2f}s")

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image