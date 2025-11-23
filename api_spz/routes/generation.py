# api_spz/routes/generation.py

import logging
import time
import traceback
from typing import Optional, List, Dict
import asyncio
import io
import xatlas
import base64
import trimesh
from trimesh.visual import material
import numpy as np

from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from PIL import Image

from api_spz.core.files_manage import file_manager
from api_spz.core.state_manage import state
from api_spz.core.models_pydantic import (
    GenerationArgForm,
    GenerationResponse,
    TaskStatus,
    StatusResponse,
)
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap

router = APIRouter()
logger = logging.getLogger("hunyuan3d_api")

# State Management in the main process
generation_lock = asyncio.Lock()
current_generation = {
    "status": TaskStatus.FAILED, "progress": 0, "message": "",
}

def is_generation_in_progress() -> bool:
    return generation_lock.locked()

def reset_current_generation():
    current_generation.update({
        "status": TaskStatus.PROCESSING, "progress": 0, "message": "Initializing...",
    })

def update_current_generation(status: Optional[TaskStatus] = None, progress: Optional[int] = None, message: Optional[str] = None):
    if status: current_generation["status"] = status
    if progress is not None: current_generation["progress"] = progress
    if message: current_generation["message"] = message
    logger.info(f"[STATE UPDATE] Status: {current_generation['status']}, Progress: {current_generation['progress']}%, Message: '{current_generation['message']}'")


# Worker Function (Runs in a separate thread)
def _blocking_full_generation_task(pil_images: List[Image.Image], arg: GenerationArgForm):
    """
    This function runs the ENTIRE blocking generation workload in a worker thread.
    """
    try:
        # 1. Prepare input for shape generation
        update_current_generation(progress=5, message="Preparing images...")
        image_for_worker = None
        is_multiview_model = 'mv' in state.pipeline_config.get('model_path', '').lower()

        if is_multiview_model:
            image_for_worker = {}
            view_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
            for i, img in enumerate(pil_images):
                if i < len(view_names):
                    image_for_worker[view_names[i]] = state.rembg(img.convert('RGB'))
        else:
            if len(pil_images) > 1: logger.warning("[WORKER] Using first image for single-view model.")
            image_for_worker = state.rembg(pil_images[0].convert('RGB'))

        # 2. Run shape generation
        update_current_generation(progress=10, message="Generating 3D shape...")
        args_dict = {
            **state.pipeline_config, 'image': image_for_worker, 'steps': arg.num_inference_steps,
            'guidance_scale': arg.guidance_scale, 'seed': arg.seed,
            'octree_resolution': arg.octree_resolution, 'num_chunks': arg.num_chunks * 1000,
        }

        mesh = state.execute_shape_generation(args_dict)

        # 3. Post-process and apply USER simplification (Reducer) - BEFORE texturing
        update_current_generation(progress=50, message="Post-processing mesh...")
        mesh = state.floater_remover(mesh)
        mesh = state.degenerate_face_remover(mesh)
        
        simplify_ratio = arg.mesh_simplify / 100.0 if arg.mesh_simplify > 1.0 else arg.mesh_simplify
        if simplify_ratio < 1.0:
            update_current_generation(progress=55, message="Simplifying mesh...")
            target_faces = int(len(mesh.faces) * simplify_ratio)
            mesh = state.face_reducer(mesh, max_facenum=target_faces)
        
        # 3.5 Conditional UV Unwrapping
        if getattr(arg, 'unwrap_uv', False) and not arg.apply_texture:
            logger.info("[WORKER] Starting UV Unwrapping (Shape Only)...")
            update_current_generation(progress=58, message="Unwrapping UVs...")
            
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            import numpy as np
            import xatlas
            
            vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
            faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

            # 1. Chart Options: REDUCE CRUMBS
            chart_options = xatlas.ChartOptions()
            # maxCost: Higher = allow more stretch (distortion) to keep charts together.
            # Default is 2.0. Raising to 4.0 or higher reduces cuts.
            chart_options.maxCost = 8.0  
            # normalDeviationWeight: Lower = ignore surface bumps/curvature more.
            # Default is 2.0. Lowering to 0.5 helps wrap around bumpy generated geometry.
            chart_options.normalDeviationWeight = 0.5 

            # 2. Pack Options: SPEED
            pack_options = xatlas.PackOptions()
            pack_options.bruteForce = False
            pack_options.resolution = 1024
            # padding: distance between charts. 2 is standard for 1024.
            pack_options.padding = 2 

            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            
            # Apply both options
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            
            vmapping, indices, uvs = atlas[0]
            
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                visual=trimesh.visual.TextureVisuals(uv=uvs),
                process=False 
            )

        # 4. Apply texture if requested. The texture pipeline will perform its own
        #    internal UV unwrapping.
        if arg.apply_texture:
            update_current_generation(progress=60, message="Applying texture...")
            temp_obj_path = file_manager.get_temp_path("temp_for_texture.obj")
            mesh.export(str(temp_obj_path))
            
            conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768, view_chunk_size=arg.num_view_chunks)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            conf.texture_size = arg.texture_size
            texture_pipeline = Hunyuan3DPaintPipeline(conf)
            
            output_textured_obj_path = file_manager.get_temp_path("textured_mesh.obj")
            try:
                texture_pipeline(
                    mesh_path=str(temp_obj_path), 
                    image_path=pil_images[0],
                    output_mesh_path=str(output_textured_obj_path), 
                    save_glb=False,
                    use_remesh=False
                )
                # After texturing, create a GLB with ONLY the albedo texture for StableProjectorz.
                # The other maps (metallic, roughness) remain on disk but are not included in the download.
                mesh = trimesh.load(str(output_textured_obj_path), force="mesh")
                
                # Manually load just the albedo texture
                albedo_path = output_textured_obj_path.with_suffix(".jpg")
                if albedo_path.exists():
                    albedo_image = Image.open(albedo_path)
                    
                    # Create a simple PBR material that only uses the albedo texture
                    albedo_material = material.PBRMaterial(baseColorTexture=albedo_image)
                    
                    # Create a new texture visual for the mesh, assigning our albedo-only material
                    texture_visual = trimesh.visual.TextureVisuals(
                        uv=mesh.visual.uv,
                        material=albedo_material
                    )
                    mesh.visual = texture_visual
                else:
                    logger.warning(f"Albedo texture not found at {albedo_path}, exporting GLB without texture.")

            finally:
                texture_pipeline.free_memory()
            update_current_generation(progress=95, message="Texture applied.")

        # 5. Export final model. NO simplification happens after this point.
        update_current_generation(progress=98, message="Exporting final model...")
        model_path = file_manager.get_temp_path(f"model.{arg.output_format}")
        mesh.export(str(model_path))

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"[WORKER] Error in blocking task: {e}\n{error_trace}")
        update_current_generation(status=TaskStatus.FAILED, message=str(e))
        raise


# API Endpoints
@router.get("/ping")
async def ping():
    return {"status": "running", "message": "API is operational", "busy": is_generation_in_progress()}

@router.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(**current_generation, busy=is_generation_in_progress())

@router.post("/generate", response_model=GenerationResponse)
async def process_ui_generation_request(data: dict):
    """
    This is a SYNCHRONOUS, blocking endpoint from the client's perspective.
    It holds the connection open until generation is complete, as expected by the C# client.
    """
    logger.info(f"[ENDPOINT /generate] Received request.")
    
    try:
        # CORRECTED: Use asyncio.wait_for for a non-blocking lock acquisition attempt.
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        logger.warning("[ENDPOINT /generate] Server is busy. Rejecting request.")
        raise HTTPException(status_code=503, detail="Server is busy with another generation.")

    try:
        start_time = time.time()
        file_manager.clear_current_generation_folder()
        reset_current_generation()

        operation = data.get("generate_what", "make_mesh") # Default to shape-only generation
        should_texture = (operation == "make_meshes_and_tex")

        images_data = data.pop("single_multi_img_input", [])
        if not images_data:
            raise HTTPException(status_code=400, detail="No images provided.")
        
        # Fast pre-processing in the main thread
        pil_images = []
        for b64_str in images_data:
            if "base64," in b64_str: b64_str = b64_str.split("base64,")[1]
            img_bytes = base64.b64decode(b64_str)
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGBA"))
        
        arg = GenerationArgForm(**data)

        # CRITICAL CHANGE
        # We now AWAIT the result of the long-running task.
        # This will block this endpoint handler, but because it runs in a thread,
        # the main server can still answer other requests like /status or /ping.
        await asyncio.to_thread(_blocking_full_generation_task, pil_images, arg)
        
        # If we get here, the task succeeded.
        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete\n"+"-"*100)
        file_manager.cleanup_intermediate_files(keep_model=True)
        duration = time.time() - start_time
        logger.info(f"[ENDPOINT /generate] Generation successful in {duration:.2f} seconds.")

        # Return the FINAL "COMPLETE" response.
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"
        )

    except Exception as e:
        logger.error(f"[ENDPOINT /generate] Generation failed: {e}")
        update_current_generation(status=TaskStatus.FAILED, message=str(e))
        file_manager.clear_current_generation_folder()
        # FastAPI will catch this and return a 500 error, which is correct.
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.get("/info/supported_operations")
async def get_supported_operations():
    return ["make_meshes_and_tex"]

@router.get("/download/model")
async def download_model():
    glb_path = file_manager.get_temp_path("model.glb")
    obj_path = file_manager.get_temp_path("model.obj")
    
    if glb_path.exists():
        model_path = glb_path; media_type = "model/gltf-binary"; filename = "model.glb"
    elif obj_path.exists():
        model_path = obj_path; media_type = "text/plain"; filename = "model.obj"
    else:
        raise HTTPException(status_code=404, detail="Model file not found.")

    return FileResponse(str(model_path), media_type=media_type, filename=filename)

@router.get("/download/spz-ui-layout/generation-3d-panel")
async def get_generation_panel_layout():
    try:
        file_path = "api_spz/routes/layout_generation_3d_panel.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content=content, media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Layout file not found")