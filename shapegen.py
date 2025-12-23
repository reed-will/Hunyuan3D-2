import argparse
import pathlib
import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def process_image_views(file_path,t_names):
        t_exts = {".jpg", ".jpeg", ".png"}
        if file_path.stem.lower() in t_names and file_path.suffix.lower() in t_exts:
            return file_path
        else:
            return None


def gen_from_images(subfolder, pipeline):
    # Initialize dictionary with None
    images = {'front': None, 'left': None, 'back': None, 'right': None}
    
    # 1. Collection Phase: Find the files
    for file in subfolder.iterdir():
        if file.is_file():
            # Check for each view, but only if we haven't found it yet
            if not images['front']: images['front'] = process_image_views(file, {"f", "front"})
            if not images['left']:  images['left']  = process_image_views(file, {"l", "left"})
            if not images['back']:  images['back']  = process_image_views(file, {"b", "back"})
            if not images['right']: images['right'] = process_image_views(file, {"r", "right"})

    view_order = ['front', 'left', 'back', 'right']
    final_views_paths = {}

    for view_name in view_order:
        path = images[view_name]
        if path is not None:
            # Open and Remove Background immediately
            img = Image.open(path).convert("RGBA")
            processed_img = rembg_model(img) # Using the BackgroundRemover
            final_views_images[view_name] = processed_img
        else:
            # We hit a gap! Stop adding any further views in the sequence.
            break

    # 3. Final Check: Do we at least have the front view?
    if 'front' not in final_views_images:
        print(f"Skipping {subfolder.name}: No valid 'front' view found to start the sequence.")
        return

    processed_images = {}
    print(f"Generating mesh for '{subfolder.name}' with {len(final_views_images)} views...")    
    start_time = time.time()
    mesh = pipeline(
        image=processed_images,
        num_inference_steps=5,
        octree_resolution=512,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    print(f"--- Generation took {time.time() - start_time:.2f} seconds ---")
    mesh.export(f"{subfolder.name}_test.glb")
    return 0

def main():
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv-turbo',
        variant='fp16'
    )
    pipeline.enable_flashvdm()
    
    script_root = pathlib.Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(description="Generate meshes from folders")
    
    parser.add_argument("folder", help="The name/path of the folder")
    parser.add_argument("flag", nargs="?", default=None, help="Set to 'batch' for batch processing")

    args = parser.parse_args()
    root_path = pathlib.Path(args.folder)

    if not root_path.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        return

    if args.flag == "batch":
        print(f"--- BATCH MODE: Scanning subfolders in {root_path.name} ---")
        
        # 1. Find all subfolders inside the root folder
        subfolders = [d for d in root_path.iterdir() if d.is_dir()]

        for subfolder in subfolders:
            print(f"\nEntering subfolder: {subfolder.name}")
            gen_from_images(subfolder, pipeline)
    else:
        print(f"--- STANDARD MODE: Scanning files in {root_path.name} ---")
        gen_from_images(root_path, pipeline)

if __name__ == "__main__":
    main()