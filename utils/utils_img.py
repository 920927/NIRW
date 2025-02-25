# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyright: reportMissingModuleSource=false

import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image
import io
from PIL import Image, ImageFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)


def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)


def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))



def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)


def gaussian_blur_r(x,blur_r):
    # x = unnormalize_img(x)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    pil_img = to_pil(unnormalize_img(x))
    return to_tensor(x.filter(ImageFilter.GaussianBlur(radius=blur_r)))

def gaussian_noise(x, gaussian_std):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_shape = np.array(pil_img).shape
        g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img_aug[ii] = to_tensor(Image.fromarray(np.clip(np.array(pil_img) + g_noise, 0, 255)))
    return normalize_img(img_aug)
    # return img_aug

def uniform_noise(x, low, high):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_shape = np.array(pil_img).shape
        uniform_noise = np.random.uniform(low, high, img_shape)
        noisy_img = (np.array(pil_img) + uniform_noise).clip(0, 255).astype(np.uint8)
        noisy_pil_img = Image.fromarray(noisy_img)
        img_aug[ii] = to_tensor(noisy_pil_img)
        
    return normalize_img(img_aug)

def mean_filter(x, kernel_size=3):
    """
    Apply mean filter to the PIL image.
    :param image: PIL Image object
    :param kernel_size: Size of the kernel (should be an odd number)
    :return: Filtered PIL Image object
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        image_array = np.array(pil_img)
    
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        filtered_array = np.zeros_like(image_array)
        pad_size = kernel_size // 2
        padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        
        for i in range(pad_size, padded_image.shape[0] - pad_size):
            for j in range(pad_size, padded_image.shape[1] - pad_size):
                # Apply the kernel to the image
                window = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
                filtered_array[i-pad_size, j-pad_size] = np.sum(window * kernel, axis=(0, 1))
    
        filtered_image = Image.fromarray(filtered_array.astype('uint8'))
        img_aug[ii] = to_tensor(filtered_image)
    return normalize_img(img_aug)


def salt_pepper_noise(x, prob):
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    imgaug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pilimg = topil(unnormalize_img(img))
        imgshape = np.array(pilimg).shape
        salt_pepper_noise = np.random.choice([0, 1, 2], size=imgshape, p=[prob/2, (1-prob), prob/2])
        noisy_img = np.where(salt_pepper_noise == 1, 255, np.where(salt_pepper_noise == 2, 0, np.array(pilimg)))
        noisy_pil_img = Image.fromarray(noisy_img)
        imgaug[ii] = totensor(noisy_pil_img)
    return normalize_img(imgaug)


def poisson_noise(x):
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    imgaug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pilimg = topil(unnormalize_img(img))
        noisy_img = np.random.poisson(np.array(pilimg)).astype(np.uint8)
        noisy_pil_img = Image.fromarray(noisy_img)
        imgaug[ii] = totensor(noisy_pil_img)
    return normalize_img(imgaug)


def exponential_noise(x, lambd):
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    imgaug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pilimg = topil(unnormalize_img(img))
        imgshape = np.array(pilimg).shape
        exponential_noise = np.random.exponential(lambd, imgshape)
        noisy_img = (np.array(pilimg) + exponential_noise).clip(0, 255).astype(np.uint8)
        noisy_pil_img = Image.fromarray(noisy_img)
        imgaug[ii] = totensor(noisy_pil_img)
    return normalize_img(imgaug)


