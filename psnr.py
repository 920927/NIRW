import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import json
import shutil
import tqdm
from pathlib import Path
from PIL import Image, ImageFilter
import glob
import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from pathlib import Path
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path

import lpips
import glob
# from tqdm import tqdm

from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn import metrics


def get_img_metric(orig_imgs, watermarked_imgs, num_imgs=None):
    orig_imgs = glob.glob(os.path.join(orig_imgs, "*"))
    log_stats = []
    for ii, orig_img in enumerate(tqdm.tqdm(orig_imgs)):
        filename = orig_img.split('/')[-1].replace('orig','w')
        watermarked_img = os.path.join(watermarked_imgs, filename)
        orig_img = Image.open(orig_img)
        watermarked_img =  Image.open(watermarked_img)
        min_width = min(orig_img.width, watermarked_img.width)
        min_height = min(orig_img.height, watermarked_img.height)
        orig_img = orig_img.resize((min_width, min_height))
        watermarked_img = watermarked_img.resize((min_width, min_height))
        orig_img = np.asarray(orig_img)
        watermarked_img = np.asarray(watermarked_img)
        log_stat = {
            'filename': orig_img,
            'ssim': structural_similarity(orig_img, watermarked_img, channel_axis=2),
            'psnr': peak_signal_noise_ratio(orig_img, watermarked_img),
            'linf': np.amax(np.abs(orig_img.astype(int)-watermarked_img.astype(int)))
        }
        log_stats.append(log_stat)
    return log_stats


print(f'>>> Computing img-2-img stats...')
orig_imgs = '/224010165/Project/dx/watermark/NIRW/output-ablation2/orig'
watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-ablation2/watermarked'
img_metrics = get_img_metric(orig_imgs, watermarked_imgs)
img_df = pd.DataFrame(img_metrics)
ssims = img_df['ssim'].tolist()
psnrs = img_df['psnr'].tolist()
linfs = img_df['linf'].tolist()
ssim_mean, ssim_std, ssim_max, ssim_min = np.mean(ssims), np.std(ssims), np.max(ssims), np.min(ssims) 
psnr_mean, psnr_std, psnr_max, psnr_min = np.mean(psnrs), np.std(psnrs), np.max(psnrs), np.min(psnrs)
linf_mean, linf_std, linf_max, linf_min = np.mean(linfs), np.std(linfs), np.max(linfs), np.min(linfs)



base_path = '/224010165/Project/dx/watermark/NIRW/output-ablation2'
# base_path = '/224010165/Project/dx/watermark/tree-ring-watermark/output-inverse'
orig_paths = glob.glob(os.path.join(base_path,'orig','*'))

distance,count = 0,0
for orig_path in orig_paths:
    filename = orig_path.split('/')[-1].replace("orig","w")
    watermarked_path = os.path.join(base_path,'watermarked',filename)
    image1 = Image.open(orig_path)
    image2 = Image.open(watermarked_path) 


    # 调整图像大小为相同尺寸
    min_width = min(image1.width, image2.width)
    min_height = min(image1.height, image2.height)
    image1 = image1.resize((min_width, min_height))
    image2 = image2.resize((min_width, min_height))


    # 加载预训练的LPIPS模型
    lpips_model = lpips.LPIPS(net="alex")

    # 将图像转换为PyTorch的Tensor格式
    image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 使用LPIPS模型计算距离
    distance += lpips_model(image1_tensor, image2_tensor)
    count += 1
    
print("LPIPS distance:", distance.item()/count)
print(f"SSIM: {ssim_mean:.4f}±{ssim_std:.4f} [{ssim_min:.4f}, {ssim_max:.4f}]")
print(f"PSNR: {psnr_mean:.4f}±{psnr_std:.4f} [{psnr_min:.4f}, {psnr_max:.4f}]")
print(f"Linf: {linf_mean:.4f}±{linf_std:.4f} [{linf_min:.4f}, {linf_max:.4f}]")