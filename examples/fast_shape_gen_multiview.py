import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

images_path = [
    "assets/demo-mv/F.png",
    "assets/demo-mv/L.png",
    "assets/demo-mv/B.png"
]

images = []
for image_path in images_path:
    image = Image.open(image_path)
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    images.append(image)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv-turbo',
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
