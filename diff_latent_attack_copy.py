import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils.utils import view_images, aggregate_attention
from utils.distances import LpDistance
import other_attacks
from torchvision import transforms
from utils import utils_img
import torch.nn.functional as F
import random

from noisenetwork.Noise import Noise
from utils import spectral_diffusion
from loss.loss_provider import LossProvider


# 图像预处理
def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


# 获得潜在向量
def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888) #随机数生成器
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    # vae编码之后的潜在分布中采样
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


# DDIM 反向采样过程：输入图像、文本提示、用于反向采样的模型、反向采样的步骤数、文本提示的权重、res图像分辨率
@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    batch_size = 1
    max_length = 77

    # 空文本（无条件）输入
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

     # 文本条件输入
    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    # 连接无条件输入和文本输入的嵌入表示
    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    # 设置模型调度器的时间步数为反向采样的步数
    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    # 从第一个到倒数第二个进行遍历，最后一个时间步骤不进行反向采样
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        
        # latents_input作为unet模块的输入，根据当前时间步和编码器的隐藏状态预测噪声
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        
        # 将噪声拆分为无条件部分和文本指导部分
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        # next_timestep 的计算方式是当前时间步 t 加上训练步骤数除以推理步骤数的结果
        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        
        # 如果下一个时间步 next_timestep 在训练范围内（小于或等于训练步骤数），则从调度器的累积 alpha 列表 alphas_cumprod 中获取对应的 alpha_bar_next
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        # 更新潜在表示
        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents



# 注册注意力控制器到给定模型的交叉注意力模块中
def register_attention_control(model, controller):
    # 用于交叉注意力模块的前向传播操作
    def ca_forward(self, place_in_unet):

        def forward(x, context=None):
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)

            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        return forward
    
    # 用于在模型中注册交叉注意力模块的前向传播函数
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    # 在主函数中，通过遍历模型的子模块，对其中的下采样、上采样和中间模块应用 register_recr 函数，并将交叉注意力模块的数量赋值给控制器的属性 num_att_layers。
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


# 重置给定模型中的交叉注意力模块（将交叉注意力模块恢复到其原始的工作状态，去除控制逻辑的影响）
def reset_attention_control(model):
    def ca_forward(self):
        def forward(x, context=None):
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)

            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


# 执行扩散模型中的一个步骤，更新潜在向量
# context上下文信息英语指导生成过程，t代表当前时间步，引导尺度用于平衡文本嵌入和噪声嵌入
# 将当前潜在向量复制一份，然后与上下文信息连接在一起作为输入传入unet模型，unet模型输出噪声的预测，分别用于无条件和文本嵌入
# 融合两部分预测，得到最终的噪声预测
# 利用scheduler模块进行潜在向量的更新，生成下一个时间步的潜在向量，并返回更新后的潜在向量
def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


# 使用@torch.enable_grad()装饰器确保了在 diffattack 函数中的所有操作都会启用 PyTorch 的自动求导机制，这意味着所有对张量的操作都会被记录下来，以便后续进行梯度计算。
@torch.enable_grad()
def diffsignature(model, label, controller, num_inference_steps: int = 20, guidance_scale: float = 2.5, image=None, 
        model_name="inception", save_path=r"C:\Users\PC\Desktop\output", filename=None, res=224, 
        start_step=15, iterations=30, verbose=True, args=None, key=None, msg_decoder=None, prompt=None,
        ):

    #将模型中的 VAE、文本编码器和 UNet 设置为不需要梯度计算
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    # Create losses
    print(f'>>> Creating losses...')
    if args.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif args.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if args.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif args.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to('cuda:0')
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif args.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to('cuda:0')
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif args.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to('cuda:0')
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError

    # data preprocess
    height = width = res
    transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_image = transform(image).unsqueeze(0).to('cuda:0')

    # noise_layers = ['Combined([JpegMask(50),Jpeg(50),Identity()])']
    
    # 定义噪声层
    noise_layers = [
        'Combined([JpegMask(50),Jpeg(50),Identity()])'
    ]
    # 定义噪声模型
    noise_model = Noise(noise_layers)
    # out_image_attack = noise_model([out_image,test_image])

    spd_model = spectral_diffusion.SPD()
    # out_image_attack = spd([out_image,test_image])

    attacks = {
        # 'none': lambda x: x,
        'crop_01': lambda x: utils_img.center_crop(x, 0.1),
        'crop_05': lambda x: utils_img.center_crop(x, 0.5),
        'rot_25': lambda x: utils_img.rotate(x, 25),
        'rot_90': lambda x: utils_img.rotate(x, 90),
        'resize_03': lambda x: utils_img.resize(x, 0.3),
        'resize_07': lambda x: utils_img.resize(x, 0.7),
        'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
        'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
        'noise': lambda x: noise_model([x,test_image]),
        # 'spd': lambda x: spd_model([x,test_image]),
        }

    
    #水印初始化
    keys = key.repeat(test_image.shape[0], 1)
    decoded = msg_decoder(test_image)

    # label = torch.from_numpy(label).long().cuda() #将输入的标签转换为 PyTorch 张量
    # print('label:', label)

    # prompt = id_to_label[label.item()]
    prompt = [prompt[0]] * 2
    print('target_prompt:', prompt)

    true_label = model.tokenizer.encode(prompt)

    #使用 DDIM 方法对输入图像进行反演，得到潜在空间表示
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model, num_inference_steps, 0, res=height)
    inversion_latents = inversion_latents[::-1]

    #初始化提示信息和潜在空间，并根据指定的起始步骤选择潜在空间
    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]


    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    # 第一个优化器：优化无条件嵌入，优化目标是最小化均方误差来接近预期的初始嵌入
    uncond_embeddings.requires_grad_(False)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            # optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
            # loss.backward()
            # optimizer.step()
            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)

        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())


    # 停止优化无条件嵌入
    uncond_embeddings.requires_grad_(False)
    register_attention_control(model, controller)
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    #克隆当前潜在向量
    original_latent = latent.clone()

    # 对潜在向量进行优化
    latent.requires_grad_(True)

    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    #  “Pseudo” Mask for better Imperceptibility, yet sacrifice the transferability. Details please refer to Appendix D.
    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    pbar = tqdm(range(iterations), desc="Iterations")
    for _, _ in enumerate(pbar):
        controller.loss = 0

        #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        controller.reset()
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        before_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False)
        after_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=False)

        before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label) - 1]

        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

        if init_mask is None:
            init_mask = torch.nn.functional.interpolate((before_true_label_attention_map.detach().clone().mean(
                -1) / before_true_label_attention_map.detach().clone().mean(-1).max()).unsqueeze(0).unsqueeze(0),
                                                        init_image.shape[-2:], mode="bilinear").clamp(0, 1)

            if hard_mask:
                init_mask = init_mask.gt(0.5).float()
        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (
                1 - init_mask) * init_image

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)


        attack_key = random.sample(list(attacks.keys()), 1)[0]
        # print(attack_key)
        attack =attacks[attack_key]
        out_image_attack = attack(out_image)

        decoded = msg_decoder(out_image_attack) # b c h w -> b k

        # compute loss
        attack_loss = loss_w(decoded, keys) * args.decoded_loss_weight
        # “Deceive” Strong Diffusion Model. Details please refer to Section 3.3
        # variance_cross_attn_loss = after_true_label_attention_map.var() * args.cross_attn_loss_weight
        # Preserve Content Structure. Details please refer to Section 3.4
        self_attn_loss = controller.loss * args.self_attn_loss_weight

        # mse_img_loss = torch.mean((out_image - test_image)**2) * args.mse_img_loss_weight
        mse_img_loss = loss_i(out_image,test_image) * args.mse_img_loss_weight

        loss = self_attn_loss + attack_loss + mse_img_loss 
        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"mse_img_loss: {mse_img_loss.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item():.5f} "
                f"loss: {loss.item():.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        controller.loss = 0
        controller.reset()

        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
            1 - init_mask) * init_image
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)


    # 将潜在向量转化为图像
    image = latent2image(model.vae, latents.detach())

    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
            1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    image = (perturbed * 255).astype(np.uint8)


    # 0.95new+0.05ori
    # perturbed = perturbed*0.8+real*0.2
    # view_images(np.concatenate([real, perturbed]) * 255, show=False,
    #             save_path=save_path + "_diff_{}_image_{}.png".format(model_name,
    #                                                                  "ATKSuccess" if pred_accuracy == 0 else "Fail"))
    view_images(np.concatenate([real, perturbed]) * 255, show=False,
                save_path=os.path.join(save_path, "diff", "{}_diff_image.png".format(filename)))
    view_images(perturbed * 255, show=False, save_path=os.path.join(save_path ,"watermarked" , "{}_w.png".format(filename)))


    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    reset_attention_control(model)

    adv_path = os.path.join(save_path ,"watermarked" , "{}_w.png".format(filename))
    return adv_path, keys, decoded