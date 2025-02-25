import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
import random
import numpy as np
import glob
import re
from PIL import Image
import json

from diffusers import StableDiffusionPipeline, DDIMScheduler

from utils.attentionControl import AttentionControlEdit
from utils.imagenet_classes import IMAGENET2012_CLASSES
import diff_latent_attack_copy
from other_attacks import model_transfer


#-------------------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--is_test', default=False, type=bool, help='whether to test the robustness of the generated adversarial examples')

# model
parser.add_argument('--model_name', default='resnet', type=str, help='the surrogate model from which the adversarial examples are crafted')
parser.add_argument('--pretrained_diffusion_path', default='model/stable-diffusion-2-base', type=str, help='path to the pretrained model')
parser.add_argument('--msg_decoder_path', type=str, default='model/decoder_model/dec_48b_whit.torchscript.pt', help='Path to the hidden decoder for the watermarking model')

# directory
parser.add_argument('--save_dir', default='output', type=str, help='where to save the results')
parser.add_argument('--data_path', default='data', type=str, help='the clean images root direvtory')
parser.add_argument('--test_path', default="output", type=str, help='the output images root directory')

# dataset
parser.add_argument('--dataset_name', default='coco', type=str, help='the dataset name for generating watermarking examples')
parser.add_argument('--num', default=1000, type=int, help='the number of images')
parser.add_argument('--res', default=256, type=int, help='input image resized resolution')

# diffusion
parser.add_argument('--diffusion_steps', default=120, type=int, help='total DDIM sampling steps')
parser.add_argument('--start_step', default=119, type=int, help='which DDIM step to start the watermarking')
parser.add_argument('--iterations', default=100, type=int, help='iterations of optimizing the watermarked image')
parser.add_argument('--guidance', default=4.5, type=float, help='guidance scale of the diffusion models')

# mask
parser.add_argument('--is_apply_mask', default=True, type=bool, help='whether to leverage pseudo mask for better imperceptibility')
parser.add_argument('--is_hard_mask', default=False, type=bool, help='which type of mask to leverage')

# loss
parser.add_argument('--decoded_loss_weight', default=100, type=int, help='decoded loss weight factor')
parser.add_argument('--mse_img_loss_weight', default=20, type=int, help='mse loss weight factor')
parser.add_argument('--self_attn_loss_weight', default=80, type=int, help='self attention loss weight factor')

# watermark
parser.add_argument("--num_bits", type=int, default=48, help="Number of bits in the watermark")
parser.add_argument("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
parser.add_argument("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
parser.add_argument("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")
parser.add_argument("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
parser.add_argument("--loss_i", type=str, default="mse", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
parser.add_argument("--topN", default=1,type=int)

#-------------------------------------------------------------------------------------------------------------------------------------------------------

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(42)

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]
    
# def get_coco_label(annotation_path):
#     with open(annotation_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)

#     if isinstance(data, dict):
#         print("JSON is a dictionary with keys:")
#         print(list(data.keys()))
#     annotations, categories = data['annotations'], data['categories']
#     image_id = {item['image_id']: item['category_id'] for item in annotations}
#     label_id = {item['id']: item['name'] for item in categories}

#     return image_id, label_id

def get_coco_label(annotation_path):
    with open(annotation_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, dict):
        print("JSON is a dictionary with keys:")
        print(list(data.keys()))
    annotations, categories = data['annotations'], data['categories']
    image_id = {idx: item['category_id'] for idx, item in enumerate(annotations)}
    label_id = {item['id']: item['name'] for item in categories}

    return image_id, label_id

def get_imagenet_label(annotation_path):
    with open(annotation_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, dict):
        print("JSON is a dictionary with keys:")
        print(list(data.keys()))
    annotations, categories = data['annotations'], data['categories']
    image_id = {item['image_id']: item['category_id'] for item in annotations}
    label_id = {item['id']: item['name'] for item in categories}

    return image_id, label_id
        
def run_diffusion_attack(image, label, diffusion_model, diffusion_steps, guidance=2.5,
                         self_replace_steps=1., save_dir="output", filename=None,res=224,
                         model_name="inception", start_step=15, iterations=30, args=None, key=None,msg_decoder=None,prompt=None):
    controller = AttentionControlEdit(diffusion_steps, self_replace_steps, args.res)
    adv_path, keys, decoded = diff_latent_attack_copy.diffsignature(diffusion_model, label, controller,
                                                                  num_inference_steps=diffusion_steps,
                                                                  guidance_scale=guidance,
                                                                  image=image,
                                                                  save_path=save_dir, 
                                                                  filename=filename,
                                                                  res=res, model_name=model_name,
                                                                  start_step=start_step,
                                                                  iterations=iterations, args=args,key=key,msg_decoder=msg_decoder, prompt=prompt)

    return adv_path, keys, decoded  


if __name__ == "__main__":
    # 命令行配置
    args = parser.parse_args()
    assert args.res % 32 == 0 and args.res >= 96, "Please ensure the input resolution be a multiple of 32 and also >= 96."
    guidance = args.guidance
    diffusion_steps = args.diffusion_steps
    start_step = args.start_step
    iterations = args.iterations
    res = args.res
    model_name = args.model_name

    save_dir = args.save_dir  
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,'orig',),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'watermarked'),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'diff'),exist_ok=True)
    log = open(os.path.join(save_dir, "log.txt"), mode="w", encoding="utf-8")

    # dataset
    if args.dataset_name == "coco":
        data_path = args.data_path
        # all_images = glob.glob(os.path.join(data_path, 'coco2014/train2014', '*'))
        selected_images = glob.glob(os.path.join('/224010165/Project/dx/watermark/NIRW/output-coco/dx-signature/orig', '*'))
        
        annotation_path = os.path.join(data_path, 'coco2014/annotations', 'instances_train2014.json')
        image_to_id, id_to_label = get_coco_label(annotation_path)
        # print('len of all_images:', len(all_images))
        # selected_images = random.sample(all_images, args.num)
        
        all_images, labels, prompts = [], [], []
        # for image_path in selected_images:
        for image_path in selected_images:
            all_images.append(image_path)
            image_id = image_path.split('/')[-1].split('_')[0]
            image_id = int(image_id)
            
            # image_path = image_path.split('_')[-1]
            # print('image_path:', image_path)
            # match = re.search(r'(\d+)\.jpg', image_path)
            # image_id = int(match.group(1))

            label = image_to_id[image_id]
            prompt = id_to_label[label]
            labels.append(label)    
            prompts.append(prompt)
            
    elif args.dataset_name == 'imagenet':
        data_path = args.data_path
        all_images = glob.glob(os.path.join(data_path, 'imagenet/train2012', '*'))
        
        selected_images = random.sample(all_images, args.num)
        all_images, labels, prompts = [], [], []
        for image_path in selected_images:
            all_images.append(image_path)
            label = image_path.split('_')[0].split('/')[-1]
            prompt = IMAGENET2012_CLASSES[label]
            print(label, prompt)
            # print('image_path:', image_path)
            
            labels.append(label)    
            prompts.append(prompt)        

    all_images = np.array(all_images)
    labels = np.array(labels)

    is_test = args.is_test  

    # Loads hidden decoder
    print(f'>>> Building hidden decoder with weights from {args.msg_decoder_path}...')
    msg_decoder = torch.jit.load(args.msg_decoder_path).to('cuda:0')
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to('cuda:0')).shape[-1]

    # Freeze LDM and hidden decoder
    for param in msg_decoder.parameters():
        param.requires_grad = False

    print(f"\n******Watermarked based on Diffusion, Watermarked Dataset: {args.dataset_name}*********")

    pretrained_diffusion_path = args.pretrained_diffusion_path
    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path,revision="fp16").to('cuda:0')
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)


    images, watermarked_images = [], []
    clean_all_acc, adv_all_acc = 0, 0

    # key
    key_dict = {}
    print(f'\n>>> Creating key with {nbit} bits...')
    key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device='cuda:0')
    key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
    print(f'Key: {key_str}')


    adv_paths = []
    for ind, image_path in enumerate(all_images):
        tmp_image = Image.open(image_path).convert('RGB')
        
        if args.dataset_name == 'coco':
            # image_path = image_path.split("_")[-1]
            # match = re.search(r'(\d+)\.jpg', image_path)
            # filename = str(match.group(1))
            filename = image_path.split('/')[-1].split('_')[0]
            
        elif args.dataset_name == 'imagenet':
            filename = image_path.split('_')[0].split('/')[-1]

        resized_image = tmp_image.resize((res,res))
        resized_image.save(os.path.join(save_dir,'orig', filename + "_orig.png"))
        
        adv_path, keys, decoded = run_diffusion_attack(tmp_image, labels[ind:ind + 1],
                                                     ldm_stable,
                                                     diffusion_steps, guidance=guidance,
                                                     res=res, model_name=model_name,
                                                     start_step=start_step,
                                                     iterations=iterations,
                                                     save_dir=save_dir,
                                                     filename=filename,
                                                     args=args,key=key,msg_decoder=msg_decoder,prompt=prompts[ind:ind + 1])

        
        adv_paths.append(adv_path)
        key_str = "".join([ str(int(ii)) for ii in keys.tolist()[0]])
        bool_msg = (decoded>0).squeeze().cpu().numpy().tolist()
        decoded_keys = msg2str(bool_msg)        

        print("keys: {}, decoded_keys: {}".format(key_str, decoded_keys))

        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_acc = (bit_acc == 1) # b

        print("Accuracy on bit: {}%, Accuracy on word: {}%".format(torch.mean(bit_acc).item(),torch.mean(word_acc.type(torch.float)).item()))

        print(adv_path,file=log)
        print("keys: {}, decoded_keys: {}".format(key_str, decoded_keys),file=log)
        print("Accuracy on bit: {}%, Accuracy on word: {}%".format(torch.mean(bit_acc).item(),torch.mean(word_acc.type(torch.float)).item()),file=log)

        name, extension = os.path.splitext(filename)
        parts = name.split('_')
        result = "_".join(parts[:2])
        key_dict[result] = key_str

    # 指定保存文件的路径
    save_file_path = os.path.join(save_dir, "info.txt")

    # 将字典保存到文件中
    with open(save_file_path, 'w') as file:
        json.dump(key_dict, file) 

    # test
    image_paths = [path.replace("w", "orig") for path in adv_paths]
    image_paths = [path.replace("watermarked", "orig") for path in image_paths]
    model_transfer(image_paths, adv_paths, label, res, save_path=save_dir,args=args,keys=keys,msg_decoder=msg_decoder)

