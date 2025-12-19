import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


images = {
    'front':  "assets/demo-mv/F.png"
    ,'left':   "assets/demo-mv/L.png"
    ,'back':   "assets/demo-mv/B.png"
#    ,'right':  "assets/demo-mv/R.png"
}

for key in images:
    image = Image.open(images[key]).convert("RGBA")
    if image.mode == 'RGBA':
        rembg = BackgroundRemover()
        image = rembg(image)
    images[key] = image

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv-fast',
    variant='fp16'
)
pipeline.enable_flashvdm()
start_time = time.time()
mesh = pipeline(
    image=images,
    num_inference_steps=5,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
print("--- %s seconds ---" % (time.time() - start_time))
mesh.export(f'demo_mv3.glb')
