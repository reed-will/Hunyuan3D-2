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


def gen_from_images(subfolder, pipeline, rembg_model, res, save_images_flag):
    
    debug_dir = subfolder / "processed_views" if save_images_flag else None
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)

    images = {'front': None, 'left': None, 'back': None, 'right': None}
    
    for file in subfolder.iterdir():
        if file.is_file():
            # Check for each view, but only if we haven't found it yet
            if not images['front']: images['front'] = process_image_views(file, {"f", "front"})
            if not images['left']:  images['left']  = process_image_views(file, {"l", "left"})
            if not images['back']:  images['back']  = process_image_views(file, {"b", "back"})
            if not images['right']: images['right'] = process_image_views(file, {"r", "right"})

    view_order = ['front', 'left', 'back', 'right']
    final_views_images = {}
    print(f"found front image: {images['front']}")
    for view_name in view_order:
        path = images[view_name]
        if path is not None:
            img = Image.open(path).convert("RGBA")
            processed_img = rembg_model(img) # Using the BackgroundRemover
            final_views_images[view_name] = processed_img
            if save_images_flag:
                processed_img.save(debug_dir / f"{view_name}_processed.png")
        else:
            # We hit a gap! Stop adding any further views in the sequence.
            break

    if 'front' not in final_views_images:
        print(f"Skipping {subfolder.name}: No valid 'front' view found to start the sequence.")
        return

    print(f"Generating mesh for '{subfolder.name}' with {len(final_views_images)} views...")    
    start_time = time.time()
    mesh = pipeline(
        image=final_views_images,
        num_inference_steps=6,
        octree_resolution=res,
        num_chunks=30000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    print(f"--- Generation took {time.time() - start_time:.2f} seconds ---")
    mesh.export(f"{subfolder.name}_test.glb")
    return 0

def main():
    print("Loading models to GPU...")
    try:
        rembg_model = BackgroundRemover()
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mv',
            subfolder='hunyuan3d-dit-v2-mv-turbo',
            variant='fp16'
        )
        pipeline.to("cuda")
        pipeline.enable_flashvdm()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load models.\n{e}")
        return
    
    parser = argparse.ArgumentParser(description="Generate meshes from folders")
    parser.add_argument("folder", help="The name/path of the folder")
    parser.add_argument("flag", nargs="?", default=None, help="Set to 'batch' for batch processing")
    parser.add_argument("--res", type=int, default=512, help="Octree resolution (default: 512)")
    parser.add_argument("--save-images", action="store_true", help="Save background-removed images")
    
    args = parser.parse_args()

    root_path = pathlib.Path(args.folder).resolve()
    if not root_path.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        return

    # Determine which folders to process
    targets = [root_path] if args.flag != "batch" else [d for d in root_path.iterdir() if d.is_dir()]

    print(f"Starting process for {len(targets)} target(s)...")

    for target in targets:
        print(f"\n{'='*30}")
        print(f"Processing: {target.name}")
        try:
            gen_from_images(target, pipeline, rembg_model, args.res, args.save_images)
        except torch.cuda.OutOfMemoryError:
            print(f"FAILED: Out of GPU Memory on {target.name}. Try reducing octree_resolution.")
            # Clear cache to attempt next folder
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED: An unexpected error occurred on {target.name}")
            print(f"Error details: {e}")
            # Optional: print(traceback.format_exc()) for full debug info
        else:
            print(f"SUCCESS: Finished {target.name}")
        finally:
            # Always clear memory after each folder to prevent "memory creep"
            torch.cuda.empty_cache()

    print(f"\n{'='*30}")
    print("Batch processing complete.")

if __name__ == "__main__":
    main()