import numpy as np 
from diffusers.utils import load_image
from run import DenseposeDetector
from PIL import Image

def execute( image, model="densepose_r50_fpn_dl.torchscript", cmap="parula", resolution=512):
    
    model = DenseposeDetector \
                .from_pretrained(filename=model) \
                .to('cuda')
    np_image =  np.asarray(image, dtype=np.uint8)
    results = model(np_image, output_type="np", detect_resolution=resolution , cmap = cmap)
    return results
if __name__ == "__main__":
    image = load_image('/workspace/training_sdxl_pti/0.png')
    ori_size = image.size
    rs = execute(image = image)
    mask = rs > 1
    rs[mask] = 255
    rs = Image.fromarray(rs).resize(ori_size)
    rs.save('0.png')


