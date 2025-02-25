import torch
import numpy as np
import torch
from PIL import Image
import numpy as np
import glob
import os

def psnr(img1, img2):
    min_width = min(img1.width, img2.width)
    min_height = min(img1.height, img2.height)
    img1 = img1.resize((min_width, min_height))
    img2 = img2.resize((min_width, min_height))

    img1 = torch.tensor(np.array(img1).astype(np.float32))
    img2 = torch.tensor(np.array(img2).astype(np.float32))
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    
    # 避免MSE接近于零，将其替换为一个很小的非零值
    epsilon = 1e-10
    mse = torch.max(mse, torch.tensor(epsilon))
    
    psnr_value = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr_value.mean().item(), mse.mean().item()


psnr_values, mse_values, count = 0, 0, 0
test_path = "/224010165/Project/dx/watermark/NIRW/output-ablation2"
# test_path = '/224010165/Project/dx/watermark/tree-ring-watermark/output-inverse'

all_clean_images = glob.glob(os.path.join(test_path, "orig", "*orig*"))
# all_clean_images = glob.glob(os.path.join(test_path, "orig", "*"))


for image_path in all_clean_images:
    image_id = image_path.split('/')[-1].replace("orig", "w")
    adv_image_path = os.path.join(test_path, "watermarked", image_id)
    # adv_image_path = image_path.replace("orig", "watermarked")
    img1 = Image.open(image_path)
    img2 = Image.open(adv_image_path)
    psnr_value, mse_value = psnr(img1, img2)
    psnr_values += psnr_value
    mse_values += mse_value
    count += 1
    
print('psnr:', psnr_values/count)
print('mse:', mse_values/count)
