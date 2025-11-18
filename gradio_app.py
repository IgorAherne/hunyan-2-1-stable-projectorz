# gradio_app.py

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

# Apply torchvision compatibility fix before other imports

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Set the torch.compile cache directory *before* importing torch, 
# using the modern Inductor environment variables.
import os

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


import random
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid
import numpy as np
import gc
from multiprocessing import Process, Queue
import multiprocessing as mp

from hy3dshape.utils import logger
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


# Set to True to skip shape generation and use a placeholder mesh for testing texturing.
# You must provide a mesh at `assets/debug_mesh.obj` for this to work.
DEBUG_SKIP_SHAPE_GENERATION = False #MODIF

MAX_SEED = 1e7
ENV = "Local" # "Huggingface"
if ENV == 'Huggingface':
    """
    Setup environment for running on Huggingface platform.

    This block performs the following:
    - Changes directory to the differentiable renderer folder and runs a shell 
        script to compile the mesh painter.
    - Installs a custom rasterizer wheel package via pip.

    Note:
        This setup assumes the script is running in the Huggingface environment 
        with the specified directory structure.
    """
    import os, spaces, subprocess, sys, shlex
    print("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    os.system("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    print('install custom')
    subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"),
                   check=True)
else:
    """
    Define a dummy `spaces` module with a GPU decorator class for local environment.

    The GPU decorator is a no-op that simply returns the decorated function unchanged.
    This allows code that uses the `spaces.GPU` decorator to run without modification locally.
    """
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func 

def get_example_img_list():
    """
    Load and return a sorted list of example image file paths.

    Searches recursively for PNG images under the './assets/example_images/' directory.

    Returns:
        list[str]: Sorted list of file paths to example PNG images.
    """
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    """
    Load and return a list of example text prompts.

    Reads lines from the './assets/example_prompts.txt' file, stripping whitespace.

    Returns:
        list[str]: List of example text prompts.
    """
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def gen_save_folder(max_size=200):
    """
    Generate a new save folder inside SAVE_DIR, maintaining a maximum number of folders.

    If the number of existing folders in SAVE_DIR exceeds `max_size`, the oldest folder is removed.

    Args:
        max_size (int, optional): Maximum number of folders to keep in SAVE_DIR. Defaults to 200.

    Returns:
        str: Path to the newly created save folder.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")
    return new_folder


# Removed complex PBR conversion functions - using simple trimesh-based conversion
def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    Export a mesh to a file in the specified folder, optionally including textures.

    Args:
        mesh (trimesh.Trimesh): The mesh object to export.
        save_folder (str): Directory path where the mesh file will be saved.
        textured (bool, optional): Whether to include textures/normals in the export. Defaults to False.
        type (str, optional): File format to export ('glb' or 'obj' supported). Defaults to 'glb'.

    Returns:
        str: The full path to the exported mesh file.
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path




def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
    # 执行转换
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)
            


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" \
height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html file {output_html_path}, \
{os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

def clear_gpu_memory():
    """Collects garbage and clears CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_shape_generation_in_process(queue, args_dict):
    """
    This function runs in a separate process to ensure all memory (RAM and VRAM)
    is released upon completion.
    """
    try:
        # Re-import and initialize everything within the new process
        import torch
        import trimesh
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.pipelines import export_to_trimesh

        print("Worker Process: Loading shape generation pipeline...")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args_dict['model_path'], subfolder=args_dict['subfolder'], use_safetensors=False, 
            device=args_dict['device'], dtype=torch.float16
        )
        if args_dict['enable_flashvdm']:
            shape_pipeline.enable_flashvdm(mc_algo='mc' if args_dict['device'] in ['cpu', 'mps'] else args_dict['mc_algo'])
        
        generator = torch.Generator()
        generator = generator.manual_seed(int(args_dict['seed']))
        
        # We ask for the raw mesh output here to avoid complex object serialization
        outputs = shape_pipeline(
            image=args_dict['image'],
            num_inference_steps=args_dict['steps'],
            guidance_scale=args_dict['guidance_scale'],
            generator=generator,
            octree_resolution=args_dict['octree_resolution'],
            num_chunks=args_dict['num_chunks'],
            output_type='mesh' # Ask for raw mesh, not trimesh
        )
        # Convert to trimesh inside the worker
        mesh = export_to_trimesh(outputs)[0]
        
        stats = { 'number_of_faces': mesh.faces.shape[0], 'number_of_vertices': mesh.vertices.shape[0] }

        print("Worker Process: Freeing shape generation pipeline memory...")
        shape_pipeline.free_memory()
        del shape_pipeline
        
        # Send simple, pickleable data (vertices and faces) instead of the complex Trimesh object
        mesh_data = (mesh.vertices, mesh.faces)
        queue.put(('success', mesh_data, stats))

    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.put(('error', str(e)))
    finally:
        # FIX: Explicitly close the queue to prevent deadlocks on exit
        # This ensures all data is flushed before the process tries to terminate its CUDA context.
        print("Worker Process: Closing queue.")
        queue.close()
        queue.join_thread()


@spaces.GPU(duration=60)
def generation_all(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    time_meta = {}

    # Part 1: Get the initial mesh
    if DEBUG_SKIP_SHAPE_GENERATION:
        print("\n--- DEBUG MODE: Skipping Shape Generation\n")
        if image is None:
            raise gr.Error("An image prompt is required to test texturing directly in debug mode.")

        debug_mesh_path = "assets/debug_mesh.obj"
        if not os.path.exists(debug_mesh_path):
            raise gr.Error(f"Debug mesh not found at '{debug_mesh_path}'. Please provide a mesh to test texturing.")
        
        mesh = trimesh.load(debug_mesh_path, force="mesh")
        save_folder = gen_save_folder()
        stats = {'mode': 'texture_only_debug', 'debug_mesh': debug_mesh_path, 'time': {}}
        # Use the seed from the UI if available
        seed = int(randomize_seed_fn(seed, randomize_seed))
    
    else: # Always use the worker process for shape generation now
        print("Spawning separate process for shape generation.")
        
        # Handle Multi-View image dictionary creation
        image_for_worker = image
        if MV_MODE:
            if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
                raise gr.Error("Please provide at least one view image.")
            image_for_worker = {}
            if mv_image_front: image_for_worker['front'] = mv_image_front
            if mv_image_back: image_for_worker['back'] = mv_image_back
            if mv_image_left: image_for_worker['left'] = mv_image_left
            if mv_image_right: image_for_worker['right'] = mv_image_right
        
        # Handle Text-to-Image generation in the main process
        if image_for_worker is None and caption:
            worker_to_use = None
            if args.enable_t23d:
                from hy3dgen.text2image import HunyuanDiTPipeline
                print("Main Process: Loading T2I pipeline on-demand...")
                worker_to_use = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
            
            start_t2i = time.time()
            if worker_to_use is not None:
                try:
                    image_for_worker = worker_to_use(caption)
                except Exception as e:
                    raise gr.Error(f"Text-to-3D failed: {e}")
                finally:
                    if worker_to_use is not None:
                        del worker_to_use
                        clear_gpu_memory()
            else:
                 raise gr.Error("Text-to-3D is disabled. Please enable it with `--enable_t23d`.")
            time_meta['text2image'] = time.time() - start_t2i
        
        # This image is used for both the worker and the subsequent texture pipeline
        image = image_for_worker 
        seed = int(randomize_seed_fn(seed, randomize_seed))

        args_dict = {
            'model_path': args.model_path, 'subfolder': args.subfolder, 'device': args.device,
            'enable_flashvdm': args.enable_flashvdm, 'mc_algo': args.mc_algo,
            'image': image, 'steps': steps, 'guidance_scale': guidance_scale,
            'seed': seed, 'octree_resolution': octree_resolution, 'num_chunks': num_chunks,
        }
        
        queue = Queue()
        process = Process(target=run_shape_generation_in_process, args=(queue, args_dict))
        
        print("Main Process: Starting shape generation worker...")
        start_shape_gen = time.time()
        process.start()
        
        # FIX: Read from the queue BEFORE joining the process to prevent deadlock
        result = queue.get() # This will block until the worker puts the result in the queue
        process.join() # Now, wait for the (now unblocked) process to terminate

        time_meta['shape generation'] = time.time() - start_shape_gen
        print("Main Process: Shape generation worker finished.")

        if result is None:
             raise gr.Error("Shape generation worker failed to return a result. Check console for errors.")
        
        if result[0] == 'error':
            raise gr.Error(f"Shape generation failed in worker process: {result[1]}")
        
        _, mesh_data, stats_from_worker = result
        mesh_vertices, mesh_faces = mesh_data
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        
        save_folder = gen_save_folder()
        stats = stats_from_worker
        stats['time'] = time_meta

    # Part 2: Common Mesh Post-Processing
    path = export_mesh(mesh, save_folder, textured=False, type='obj')
    print(f"Exported untextured mesh to {path}")

    tmp_time = time.time()
    mesh = face_reduce_worker(mesh)
    path = export_mesh(mesh, save_folder, textured=False, type='obj') # Overwrite with reduced mesh path
    stats['time']['face reduction'] = time.time() - tmp_time

    # Load, run, and unload texture pipeline
    print("Loading texture generation pipeline...")
    conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768, view_chunk_size=args.view_chunk_size)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    texture_pipeline = Hunyuan3DPaintPipeline(conf)
        
    tmp_time = time.time()
    text_path = os.path.join(save_folder, 'textured_mesh.obj')
    path_textured = texture_pipeline(mesh_path=path, image_path=image, output_mesh_path=text_path, save_glb=False)
    stats['time']['texture generation'] = time.time() - tmp_time

    print("Freeing texture generation pipeline memory...")
    texture_pipeline.free_memory()
    del texture_pipeline
    clear_gpu_memory()
  
    # Part 4: Common final steps
    tmp_time = time.time()
    glb_path_textured = os.path.join(save_folder, 'textured_mesh.glb')
    quick_convert_with_obj2gltf(path_textured, glb_path_textured)
    stats['time']['convert textured OBJ to GLB'] = time.time() - tmp_time 
    stats['time']['total'] = time.time() - start_time_0
    
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
    
    return (
        gr.update(value=path),
        gr.update(value=glb_path_textured),
        model_viewer_html_textured,
        stats,
        seed,
    )



@spaces.GPU(duration=60)
def shape_generation(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    
    print("Spawning separate process for shape generation.")
    
    # Handle Multi-View image dictionary creation
    image_for_worker = image
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
            raise gr.Error("Please provide at least one view image.")
        image_for_worker = {}
        if mv_image_front: image_for_worker['front'] = mv_image_front
        if mv_image_back: image_for_worker['back'] = mv_image_back
        if mv_image_left: image_for_worker['left'] = mv_image_left
        if mv_image_right: image_for_worker['right'] = mv_image_right

    # Handle Text-to-Image generation in the main process
    if image_for_worker is None and caption:
        worker_to_use = None
        if args.enable_t23d:
            from hy3dgen.text2image import HunyuanDiTPipeline
            print("Main Process: Loading T2I pipeline on-demand...")
            worker_to_use = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        
        if worker_to_use is not None:
            try:
                image_for_worker = worker_to_use(caption)
            except Exception as e:
                raise gr.Error(f"Text-to-3D failed: {e}")
            finally:
                if worker_to_use is not None:
                    del worker_to_use
                    clear_gpu_memory()
        else:
             raise gr.Error("Text-to-3D is disabled. Please enable it with `--enable_t23d`.")
    
    seed = int(randomize_seed_fn(seed, randomize_seed))

    args_dict = {
        'model_path': args.model_path, 'subfolder': args.subfolder, 'device': args.device,
        'enable_flashvdm': args.enable_flashvdm, 'mc_algo': args.mc_algo,
        'image': image_for_worker, 'steps': steps, 'guidance_scale': guidance_scale,
        'seed': seed, 'octree_resolution': octree_resolution, 'num_chunks': num_chunks,
    }
    
    queue = Queue()
    process = Process(target=run_shape_generation_in_process, args=(queue, args_dict))
    
    print("Main Process: Starting shape generation worker...")
    process.start()
    
    # FIX: Read from the queue BEFORE joining the process to prevent deadlock
    result = queue.get() # This will block until the worker puts the result in the queue
    process.join() # Now, wait for the (now unblocked) process to terminate
    
    print("Main Process: Shape generation worker finished.")

    if result is None:
        raise gr.Error("Shape generation worker failed to return a result. Check console for errors.")

    if result[0] == 'error':
        raise gr.Error(f"Shape generation failed in worker process: {result[1]}")
    
    _, mesh_data, stats_from_worker = result
    mesh_vertices, mesh_faces = mesh_data
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    save_folder = gen_save_folder()
    stats = stats_from_worker
    stats['time'] = {'total': time.time() - start_time_0}
    mesh.metadata['extras'] = stats

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
    )


def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'

    title = 'Hunyuan-3D-2.1'
        
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">

    {title}
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    """
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }

    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.1', analytics_enabled=False, css=custom_css) as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        caption = gr.State(None)
#                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
#                        caption = gr.Textbox(label='Text Prompt',
#                                             placeholder='HunyuanDiT will be used to generate image.',
#                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                        # gr.Label('Please upload at least one front image.')
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(
                            label='Generation Mode',
                            info='Recommendation: Turbo for most cases, \
Fast for very complex cases, Standard seldom use.',
                            choices=['Turbo', 'Fast', 'Standard'], 
                            value='Turbo')
                        decode_mode = gr.Radio(
                            label='Decoding Mode',
                            info='The resolution for exporting mesh from generated vectset',
                            choices=['Low', 'Standard', 'High'],
                            value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(
                                value=True, 
                                label='Remove Background', 
                                min_width=100)
                            randomize_seed = gr.Checkbox(
                                label="Randomize seed", 
                                value=True, 
                                min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, 
                                                          minimum=16, 
                                                          value=256, 
                                                          label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
                    with gr.Tab("Export", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', 
                                                    choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', 
                                                      value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=8000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', 
                                id='tab_img_gallery', 
                                visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        #if HAS_T2I:
        #    tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], 
                           outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, 
                            reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=True)
            else:
                mesh = trimesh.load(file_out)
                mesh = floater_remove_worker(mesh)
                mesh = degenerate_face_remove_worker(mesh)
                if reduce_face:
                    mesh = face_reduce_worker(mesh, target_face_num)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=False)
            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='./save_dir')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--view_chunk_size', type=int, default=3, help="Number of views to process in a single batch for texture generation. Set to 0 to disable chunking.")
    parser.add_argument('--enable_disk_cache', action='store_true', help="Enable persistent, on-disk caching for torch.compile.")
    args = parser.parse_args()
    
    DTYPE = torch.float16

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    # Initialize pipeline variables to None
    tex_pipeline = None
    i23d_worker = None
    t2i_worker = None
    
    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            # Apply torchvision fix before importing basicsr/RealESRGAN
            print("Applying torchvision compatibility fix for texture generation...")
            try:
                from torchvision_fix import apply_fix
                fix_result = apply_fix()
                if not fix_result:
                    print("Warning: Torchvision fix may not have been applied successfully")
            except Exception as fix_error:
                print(f"Warning: Failed to apply torchvision fix: {fix_error}")
            
            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            
            HAS_TEXTUREGEN = True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading texture generator: {e}")
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    HAS_T2I = args.enable_t23d
    if HAS_T2I:
        print("Pre-loading T2I pipeline for standard VRAM mode...")
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')

    from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
        Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    
    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()


    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()
    
    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    # FIX: Set the start method to 'spawn' for CUDA safety
    # This must be done once, at the very beginning of the main execution block.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set.")

    torch.cuda.empty_cache()

    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port)
