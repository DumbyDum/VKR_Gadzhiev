import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from copy import deepcopy
import cv2

import os
import math
import argparse
import random
import logging
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

import base64
import gradio as gr

from scipy.ndimage import zoom

import matplotlib.pyplot as plt
from demo import EditGuard_Hide, EditGuard_Reveal
from model_invert import *
import config as c

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision

def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = img_to_base64("logo.png")

html_content = f"""
<div style='display: flex; align-items: center; justify-content: center; padding: 20px;'>
    <img src='data:image/png;base64,{logo_base64}' alt='Logo' style='height: 50px; margin-right: 20px;'>
    <strong><font size='8'>OmniGuard<font></strong>
</div>
"""

# Examples
examples = [
    ["examples/0000.png"],
    ["examples/0001.png"],
    ["examples/0011.png"],
    ["examples/0012.png"],
    ["examples/0014.png"],
    ["examples/0002.png"],
    ["examples/0003.png"],
    ["examples/0005.png"],
    ["examples/0006.png"],
    ["examples/0007.png"],
    ["examples/0008.png"],
    ["examples/0009.png"],
]

default_example = examples[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe_dtype = torch.float16 if device.type == "cuda" else torch.float32

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=pipe_dtype,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

if device.type == "cpu":
    pipe.enable_attention_slicing()

def load(net, name):
    state_dicts = torch.load(name, map_location=device, weights_only=False)
    network_state_dict = {
        k: v for k, v in state_dicts['net'].items()
        if ('tmp_var' not in k) and ('bm' not in k)
    }

    if not isinstance(net, torch.nn.DataParallel):
        network_state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in network_state_dict.items()
        }

    net.load_state_dict(network_state_dict, strict=False)

def hiding(image_input, model):

    cover = np.array(image_input) / 255.
    cover = torch.from_numpy(cover).permute(2,0,1).unsqueeze(0).float().to(device)

    container = EditGuard_Hide(model, cover)

    return container, container

def rand(num_bits=100):
    random_str = ''.join([str(random.randint(0, 1)) for _ in range(num_bits)])
    return random_str

def ImageEdit(img, prompt, model_index):
    image = img["image"]
    mask = img["mask"]

    if image is None or mask is None:
        raise gr.Error("Сначала загрузите изображение и выделите область редактирования.")

    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise gr.Error("Ожидается цветное RGB-изображение.")

    h, w, _ = image.shape
    h8 = max(8, (h // 8) * 8)
    w8 = max(8, (w // 8) * 8)

    prompt = (prompt or "").strip()
    if not prompt:
        prompt = "remove the selected object and fill the area naturally"

    image_init = Image.fromarray(image, mode="RGB").resize((w8, h8))

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask, mode="L").resize((w8, h8))

    image_inpaint = pipe(
        prompt=prompt,
        image=image_init,
        mask_image=mask_pil,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    image_inpaint = image_inpaint.resize((w, h))
    image_inpaint = np.array(image_inpaint).astype(np.float32) / 255.0

    image_float = image.astype(np.float32) / 255.0
    mask_float = np.array(mask_pil.resize((w, h))).astype(np.float32) / 255.0
    mask_float = np.expand_dims(mask_float, axis=-1)

    received_image = image_float * (1.0 - mask_float) + image_inpaint * mask_float
    received_image_uint8 = (np.clip(received_image, 0, 1) * 255).astype(np.uint8)

    return received_image_uint8, received_image_uint8, received_image_uint8


def image_model_select(ckp_index=0):
    net = Model().to(device)
    init_model(net)

    if device.type == "cuda":
        net = torch.nn.DataParallel(net, device_ids=c.device_ids)

    load(net, c.MODEL_PATH + c.suffix)
    net.eval()

    return net

def wam_model_select(ckp_index=0):
    exp_dir = "/data03/zxy/OmniGuard/wam/checkpoints"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    print("zxy", wam)
    
    return wam

def revealing(image_edited, model_list, model):
    if model is None:
        raise gr.Error("Модель ещё не загружена. Подождите и повторите попытку.")

    if image_edited is None:
        raise gr.Error("Сначала выполните шаг 2 или загрузите изображение для проверки.")

    image_edited = np.array(image_edited)

    if image_edited.dtype != np.uint8:
        if image_edited.max() <= 1.0:
            image_edited = (np.clip(image_edited, 0, 1) * 255).astype(np.uint8)
        else:
            image_edited = np.clip(image_edited, 0, 255).astype(np.uint8)

    steg = image_edited.astype(np.float32) / 255.0
    steg = torch.from_numpy(steg).permute(2, 0, 1).unsqueeze(0).float().to(device)

    mask = EditGuard_Reveal(model, steg)
    mask = mask.permute(0, 2, 3, 1).cpu().squeeze().numpy() * 255
    mask = Image.fromarray(mask.astype(np.uint8))

    return mask


def equalize_string_lengths(bin_str1, bin_str2):
    # 获取两个字符串的长度
    len1 = len(bin_str1)
    len2 = len(bin_str2)
    
    # 比较长度并根据需要填充'0'
    if len1 < len2:
        # 对bin_str1进行填充
        bin_str1 = bin_str1.zfill(len2)
    elif len2 < len1:
        # 对bin_str2进行填充
        bin_str2 = bin_str2.zfill(len1)
    
    return bin_str1, bin_str2

def calculate_ber(string1, string2):

    string1, string2 = equalize_string_lengths(string1, string2)
    # 检查两个字符串长度是否相等
    if len(string1) != len(string2):
        raise ValueError("两个字符串长度必须相等")
    
    # 检查字符串是否只包含0和1
    if not all(c in '01' for c in string1 + string2):
        raise ValueError("字符串只能包含0和1")
    
    # 计算错位数
    errors = sum(1 for a, b in zip(string1, string2) if a != b)
    
    # 计算总位数
    total_bits = len(string1)
    
    # 计算误码率
    ber = errors / total_bits
    
    return ber


# Description
title = "<center><strong><font size='8'>OmniGuard<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title="OmniGuard — защита изображений") as demo:
    gr.HTML(html_content)
    model = gr.State(value = None)
    model_wam = gr.State(value = None)
    save_h = gr.State(value = None)
    save_w = gr.State(value = None)
    sam_global_points = gr.State([])
    sam_global_point_label = gr.State([])
    sam_original_image = gr.State(value=None)
    sam_mask = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem('OmniGuard'):

            DESCRIPTION = """
## Инструкция по работе:
- Загрузите изображение и нажмите кнопку «Встроить водяной знак», чтобы получить защищённую версию изображения.
- Выделите область редактирования и выполните изменение изображения с помощью алгоритма инпейнтинга.
- Нажмите кнопку «Извлечь и проверить», чтобы определить область возможного вмешательства."""
            
            gr.Markdown(DESCRIPTION)
            save_inpainted_image = gr.State(value=None)
            with gr.Column():
                with gr.Row():
                    model_list = gr.Dropdown(
                        label="Выбор модели",
                        choices=["OmniGuard"],
                        value="OmniGuard",
                        type='index'
                    )
                    clear_button = gr.Button("Очистить всё")
                with gr.Box():
                    gr.Markdown("# 1. Встраивание водяного знака")
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(source='upload', label="Исходное изображение", interactive=True, type="numpy", value=default_example[0])
                            hiding_button = gr.Button("Встроить водяной знак")
                        with gr.Column():
                            image_watermark = gr.Image(source="upload", label="Изображение со встроенным водяным знаком", interactive=True, type="numpy")


                with gr.Box():
                    gr.Markdown("# 2. Редактирование изображения")
                    with gr.Row():
                        with gr.Column():
                            image_edit = gr.Image(source='upload', tool="sketch", label="Выделите область редактирования", interactive=True, type="numpy")
                            inpainting_model_list = gr.Dropdown(label="Выбор модели редактирования", choices=["SD Inpainting"], type='index')
                            text_prompt = gr.Textbox(label="Текстовый запрос для редактирования")
                            inpainting_button = gr.Button("Выполнить редактирование")
                        with gr.Column():
                            image_edited = gr.Image(source="upload", label="Результат редактирования", interactive=True, type="numpy")
                

                with gr.Box():
                    gr.Markdown("# 3. Извлечение водяного знака и поиск правок")
                    with gr.Row():
                        with gr.Column():
                            image_edited_1 = gr.Image(
                                source="upload",
                                label="Изображение для проверки (заполняется автоматически после шага 2)",
                                interactive=True,
                                type="numpy"
                            )
                            
                            revealing_button = gr.Button("Извлечь и проверить")
                        with gr.Column():
                            edit_mask = gr.Image(source='upload', label="Предсказанная маска области правок", interactive=True, type="numpy")
                
                gr.Examples(
                            examples=examples,
                            inputs=[image_input],
                        )


                demo.load(
                    image_model_select,
                    inputs=[model_list],
                    outputs=[model]
                )
                model_list.change(
                    image_model_select, inputs = [model_list], outputs=[model]
                    )
                hiding_button.click(
                    hiding, inputs=[image_input, model], outputs=[image_watermark, image_edit]
                    )


                inpainting_button.click(
                    ImageEdit, inputs = [image_edit, text_prompt, inpainting_model_list], outputs=[image_edited, image_edited_1, save_inpainted_image]
                    )

                revealing_button.click(
                    revealing,
                    inputs=[image_edited_1, model_list, model],
                    outputs=[edit_mask]
                )


demo.launch(server_name="0.0.0.0", server_port=10049, share=True, favicon_path='logo.png')

