import os
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
import glob

from pathlib import Path
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset

from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import utils_img

from evaluate.models_hidden import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from evaluate.attenuations import JND

from rivagan import RivaGAN
from utils import ssl_watermarking
import cv2
from imwatermark import WatermarkDecoder
# ------------------------------------------------------------------------------------------------------------------------------------
def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]


class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w


def model_transfer2(adv_img_paths,keys = None, msg_decoder = None,attacks = None):
    
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    adv_img_paths = glob.glob(os.path.join(adv_img_paths,"*"))
    
    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))

        bit_acc,word_acc,count = 0,0,0
        
        for adv_img in adv_img_paths:
            img = Image.open(adv_img)
            img = transform_imnet(img).unsqueeze(0).to("cuda")
            img = attack(img)
            decoded = msg_decoder(img) # b c h w -> b k
            keys = [int(bit) for bit in keys]
            keys = torch.tensor(keys, dtype=torch.float32).to('cuda:0')
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k

            bit_acc += torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_acc += (bit_acc == 1) # b
            count += 1

        print("Accuracy on bit: {}%, Accuracy on word: {}%".format((bit_acc/count).item(),(word_acc/count).item()))

# ------------------------------------------------------------------------------------------------------------------------------------

transform_imnet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


attacks = {
    'none': lambda x: x,
    # 'crop_01': lambda x: utils_img.center_crop(x, 0.1),
    # 'crop_05': lambda x: utils_img.center_crop(x, 0.5),
    # 'rot_25': lambda x: utils_img.rotate(x, 25),
    # 'rot_90': lambda x: utils_img.rotate(x, 90),
    # 'resize_03': lambda x: utils_img.resize(x, 0.3),
    # 'resize_07': lambda x: utils_img.resize(x, 0.7),
    # 'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
    # 'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
    # 'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
    # 'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
    'gaussian_noise': lambda x: utils_img.gaussian_noise(x, 0.1),
    'uniform_noise': lambda x:utils_img.uniform_noise(x, -10, 10),
    # 'salt_pepper_noise': lambda x: utils_img.salt_pepper_noise(x, 0.01),
    'exponential_noise': lambda x:utils_img.exponential_noise(x, 0.5),
    'poisson_noise': lambda x:utils_img.poisson_noise(x),
    # 'filter': lambda x:utils_img.mean_filter(x,3)
    }


model_type = 'NIRW'
key = '111010110101000001010111010011010100010000100111' #hidden
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

if model_type == 'NIRW':
    key = '110011000111101001011011110010110010011010110110' # model key
    watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-coco/dx-signature/watermarked'
    # watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-imagenet/watermarked'
    
    # watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-ablation2/watermarked'
    
    msg_decoder = torch.jit.load("/224010165/Project/dx/watermark/NIRW/model/decoder_model/dec_48b_whit.torchscript.pt").to('cuda:0')
    model_transfer2(watermarked_imgs, keys=key, msg_decoder=msg_decoder,attacks=attacks)

elif model_type == 'stable_signature':
    watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-coco/stable-signature/val/watermarked'
    # watermarked_imgs = '/224010165/Project/dx/watermark/stable_signature/output_imagenet/val/watermarked'
    msg_decoder = torch.jit.load("/224010165/Project/dx/watermark/NIRW/model/decoder_model/dec_48b_whit.torchscript.pt").to('cuda:0')
    model_transfer2(watermarked_imgs, keys=key, msg_decoder=msg_decoder,attacks=attacks)

elif model_type == 'hidden':
    
    watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-coco/hidden/output-watermarked'
    # watermarked_imgs = '/224010165/Project/dx/watermark/stable_signature/hidden/output-watermarked-imagenet'
    
    default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    params = Params(
        encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
        attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
    )
    
    decoder = HiddenDecoder(
        num_blocks=params.decoder_depth, 
        num_bits=params.num_bits, 
        channels=params.decoder_channels
    )
    encoder = HiddenEncoder(
        num_blocks=params.encoder_depth, 
        num_bits=params.num_bits, 
        channels=params.encoder_channels
    )
    attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
    encoder_with_jnd = EncoderWithJND(
        encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
    )
    
    ckpt_path = "model/hidden/hidden_replicate.pth"
    
    state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
    encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
    decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
    
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    
    encoder_with_jnd = encoder_with_jnd.to(device).eval()
    decoder = decoder.to(device).eval()

    msg_ori = torch.Tensor(str2msg(key)).unsqueeze(0).to('cuda')
    msg = 2 * msg_ori.type(torch.float) - 1 # b k
    
    default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    
    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))
    
        count,bit_acc =0,0
        adv_img_paths = glob.glob(os.path.join(watermarked_imgs,"*"))
        for adv_img in adv_img_paths:
            img = Image.open(adv_img).convert('RGB')
            img = img.resize((256, 256), Image.BICUBIC)
            img = default_transform(img).unsqueeze(0).to('cuda')
            ft = decoder(attack(img))
            decoded_msg = ft > 0 # b k -> b k
            accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k
            bit_acc += accs.sum().item() / params.num_bits
            count += 1
    
        print(f"Bit Accuracy: {bit_acc/count}")

elif model_type == 'rivagan':
    watermarked_imgs = '/224010165/Project/dx/watermark/RivaGAN/output-coco'
    # watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/WatermarkAttacker/output-attacked/rivagan'
    model = RivaGAN.load("/224010165/Project/dx/watermark/RivaGAN/experiments/attention-coco/model.pt")
    def get_acc(y_true, y_pred):
        assert y_true.size() == y_pred.size()
        return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()
    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))
    
        count,bit_acc =0,0
        adv_img_paths = glob.glob(os.path.join(watermarked_imgs,"*"))
        for adv_img in adv_img_paths:
            frame = cv2.imread(adv_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transforms.ToTensor()(frame).unsqueeze(0)
            bgr = attack(frame)
            bgr = (bgr.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bgr = torch.FloatTensor(bgr).permute(2, 0, 1).unsqueeze(1).cuda() / 127.5 - 1.0
            bgr = bgr.unsqueeze(0)    
            # print(bgr.shape)
            decoded = model.decode(bgr)
            decoded = torch.from_numpy(decoded).to('cuda:0')
            # print(decoded)
            keys = [int(bit) for bit in key]
            keys = torch.tensor(keys, dtype=torch.float32).to('cuda:0')
            bit_acc += get_acc(keys, decoded)
            count += 1
        print(f"Bit Accuracy: {bit_acc/count}")

elif model_type == 'dwtDct' or model_type == 'dwtDctSvd':
    watermarked_imgs = f'/224010165/Project/dx/watermark/invisible-watermark/output-coco/{model_type}'
    # key = '011110111011111110010101111010010101001001111011'
    # watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-coco/dwt-Dct/watermarked'
    decoder = WatermarkDecoder('bits', 48)
    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))
    
        count,bit_acc =0,0
        adv_img_paths = glob.glob(os.path.join(watermarked_imgs,"*"))
        for adv_img in adv_img_paths:
            bgr = cv2.imread(adv_img, cv2.IMREAD_UNCHANGED)
            bgr = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            bgr = transforms.ToTensor()(bgr).unsqueeze(0)
            bgr = attack(bgr)
            bgr = (bgr.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bgr = cv2.cvtColor(np.array(bgr),cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr,(256, 256))    
            watermark = decoder.decode(bgr, model_type)
            watermark = torch.tensor(list(watermark), dtype=torch.uint8)
            keys = [int(bit) for bit in key]
            keys = torch.tensor(keys, dtype=torch.uint8)
            diff = (~torch.logical_xor(watermark>0, keys>0)) # b k -> b k
            bit_acc += torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            count += 1
            
        print(f"Bit Accuracy: {bit_acc/count}")

elif model_type == 'ssl_watermarking':
    watermarked_imgs = '/224010165/Project/dx/watermark/NIRW/output-coco/ssl/watermarked'
    # watermarked_imgs = '/224010165/Project/dx/watermark/ssl_watermarking/output/imgs'

    model_path = '/224010165/Project/dx/watermark/ssl_watermarking/models/dino_r50_plus.pth'
    model_name = 'resnet50'
    # normlayer_path = '/224010165/Project/dx/watermark/ssl_watermarking/normlayers/out2048_yfcc_resized.pth'
    normlayer_path = '/224010165/Project/dx/watermark/ssl_watermarking/normlayers/out2048_yfcc_orig.pth'
    # key = '111010110101000001010111010011010100010000100111'
    key = '000010010101100101101111110101011000101001001010'
    
    backbone = ssl_watermarking.build_backbone(path=model_path, name=model_name)
    normlayer = ssl_watermarking.load_normalization_layer(path=normlayer_path)
    model = ssl_watermarking.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    carrier_path = '/224010165/Project/dx/watermark/ssl_watermarking/carriers/carrier_48_2048.pth'
    carrier = torch.load(carrier_path)
    carrier = carrier.to(device, non_blocking=True) # direction vectors of the hyperspace
    default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

    msg_ori = torch.Tensor(str2msg(key)).unsqueeze(0).to('cuda')
    msg = 2 * msg_ori.type(torch.float) - 1 # b k
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE_IMAGENET,
    ])
    
    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))
    
        count,bit_acc =0,0
        adv_img_paths = glob.glob(os.path.join(watermarked_imgs,"*"))
        for adv_img in adv_img_paths:
            img = Image.open(adv_img)
            img = default_transform(img).unsqueeze(0).to('cuda', non_blocking=True)

            ft = model(attack(img))
            dot_product = ft @ carrier.T # 1xD @ DxK -> 1xK
            msg = torch.sign(dot_product).squeeze() > 0

            keys = [int(bit) for bit in key]
            keys = torch.tensor(keys, dtype=torch.float32).to('cuda:0')
            diff = (~torch.logical_xor(msg>0, keys>0)) # b k -> b k

            bit_acc += torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            count += 1
    
        print(f"Bit Accuracy: {bit_acc/count}")