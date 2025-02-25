import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from PIL import Image

from art.estimators.classification import PyTorchClassifier
import timm
from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
import warnings
import pytorch_fid.fid_score as fid_score
from Finegrained_model import model as otherModel
from utils import utils_img

warnings.filterwarnings("ignore")


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

def model_transfer1(clean_img_paths, adv_img_paths, label, res, save_path=r"output", args=None,keys=None, msg_decoder=None):
    log = open(os.path.join(save_path, "log_attacks.txt"), mode="w", encoding="utf-8")
    
    bit_acc,word_acc,count = 0,0,0
    
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    attacks = {
        'none': lambda x: x,
        }
    bit_accs,word_accs = [],[]

    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))
        print("\n*********Type of attack {}********".format(name), file=log)

        for adv_img in adv_img_paths:
            img = Image.open(adv_img)
            
            img = transform_imnet(img).unsqueeze(0).to("cuda")
            img = attack(img)
            decoded = msg_decoder(img) # b c h w -> b k

            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_acc += torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_acc += (bit_acc == 1) # b
            count += 1
        print("Accuracy on bit: {}%, Accuracy on word: {}%".format((bit_acc/count).item(),(word_acc/count).item()))
        print("Accuracy on bit: {}%, Accuracy on word: {}%".format((bit_acc/count).item(),(word_acc/count).item()),file=log)

    log.close()


def model_transfer(clean_img_paths, adv_img_paths, label, res, save_path=r"output", args=None,keys=None, msg_decoder=None):
    log = open(os.path.join(save_path, "log_attacks.txt"), mode="w", encoding="utf-8")
    
    bit_acc,word_acc,count = 0,0,0
    
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    attacks = {
        'none': lambda x: x,
        'crop_01': lambda x: utils_img.center_crop(x, 0.1),
        'crop_05': lambda x: utils_img.center_crop(x, 0.5),
        'rot_25': lambda x: utils_img.rotate(x, 25),
        'rot_90': lambda x: utils_img.rotate(x, 90),
        'resize_03': lambda x: utils_img.resize(x, 0.3),
        'resize_07': lambda x: utils_img.resize(x, 0.7),
        'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
        'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
        'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
        'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }

    bit_accs,word_accs = [],[]

    for name, attack in attacks.items():
        print("\n*********Type of attack {}********".format(name))
        print("\n*********Type of attack {}********".format(name), file=log)

        for adv_img in adv_img_paths:
            img = Image.open(adv_img)
            
            img = transform_imnet(img).unsqueeze(0).to("cuda")
            img = attack(img)
            decoded = msg_decoder(img) # b c h w -> b k

            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_acc += torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_acc += (bit_acc == 1) # b
            count += 1
        print("Accuracy on bit: {}%, Accuracy on word: {}%".format((bit_acc/count).item(),(word_acc/count).item()))
        print("Accuracy on bit: {}%, Accuracy on word: {}%".format((bit_acc/count).item(),(word_acc/count).item()),file=log)

    log.close()

