# Author: Therefore Games
# https://github.com/ThereforeGames/txt2img2img

from math import floor
from pathlib import Path
import modules.scripts as scripts
import gradio as gr

from modules import processing

import torch
import numpy
import cv2
import requests
import os.path

from clipseg.clipseg import CLIPDensePredT
from PIL import ImageDraw, ImageChops, Image, ImageFilter
from torchvision import transforms


def download_file(filename, url):
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        for block in response.iter_content(4096):
            fout.write(block)


def base_path(*parts):
    filename = os.path.join(scripts.basedir(), *parts)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    return filename


def output_path(*parts):
    return base_path("outputs/txt2mask", *parts)


def mask_from_pred(pred, mask_precision):
    arr = torch.sigmoid(pred).cpu()
    arr = (arr.numpy() * 256).astype(numpy.uint8)
    _, bw_image = cv2.threshold(
        arr, mask_precision, 255, cv2.THRESH_BINARY)

    return Image.fromarray(bw_image)


def debug_mask(img, color, alpha=0.25):
    mask = img.convert('L')
    result = img.convert("RGBA")
    draw = ImageDraw.Draw(result)
    draw.rectangle([(0, 0), img.size], fill=color)
    result.putalpha(mask.point(lambda i: floor(i*alpha)))
    return result


def add_masks(masks):
    if len(masks) > 1:
        pending = masks[::-1]
        current = pending.pop()
        while len(pending) > 0:
            next = pending.pop()
            current = ImageChops.add(current, next)

        return current

    if len(masks) == 1:
        return masks[0]

    return None


def mask_from_preds(preds, mask_precision):
    masks = [
        mask_from_pred(pred[0], mask_precision) for pred in preds
    ]
    return (add_masks(masks), masks)


def download_weights(file):
    print("Downloading clipseg model weights...")
    download_file(
        file, "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni-refined.pth")


def model_path():
    path = Path("./models/clipseg/rd64-uni-refined.pth")
    if not os.path.exists(path):
        # Download model weights if we don't have them yet
        path.parent.mkdir(parents=True, exist_ok=True)
        download_weights(path)
    return path


def parse_prompt(mask_prompt, delimiter_string="|"):
    return [part.strip() for part in mask_prompt.split(delimiter_string) if len(part.strip()) > 0]


def load_model():
    # load model
    model = CLIPDensePredT(
        version='ViT-B/16',
        reduce_dim=64,
        complex_trans_conv=True)

    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(
        model_path(), map_location=torch.device('cuda')), strict=False)

    return model


def predict_prompts(image, prompts, negative_prompts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])

    img = transform(image).unsqueeze(0)

    prompts_len = len(prompts)
    neg_prompts_len = len(negative_prompts)

    model = load_model()

    preds = []
    negative_preds = []
    with torch.no_grad():
        if (prompts_len):
            preds = model(img.repeat(prompts_len, 1, 1, 1), prompts)[0]
        if (neg_prompts_len):
            negative_preds = model(img.repeat(
                neg_prompts_len, 1, 1, 1), negative_prompts)[0]

    return (preds, negative_preds)


def blur(image, radius):
    if (int(radius) > 0):
        blur = image.filter(
            ImageFilter.GaussianBlur(radius=radius))
        return blur.point(lambda x: 255 * (x > 0))
    return image


def apply_brush_mask(image, brush_mask, brush_mask_mode):
    if brush_mask:
        brush_mask = brush_mask.resize(image.size).convert("L")
        if brush_mask_mode == 1:
            return (ImageChops.add(image, brush_mask), brush_mask)
        elif brush_mask_mode == 2:
            return (ImageChops.subtract(image, brush_mask), brush_mask)

    return (image, None)


def mask_preview(image, pred_mask, neg_mask, brush_mask, brush_mask_mode):
    green = "#00FF00"
    red = "#FF0000"

    result = image.convert("RGBA")

    if pred_mask:
        result = Image.alpha_composite(result, debug_mask(pred_mask, green))

    if neg_mask:
        result = Image.alpha_composite(
            result, debug_mask(neg_mask, red))

    if brush_mask:
        if brush_mask_mode == 1:
            result = Image.alpha_composite(
                result, debug_mask(brush_mask, green))
        elif brush_mask_mode == 2:
            result = Image.alpha_composite(
                result, debug_mask(brush_mask, red))

    return result


def predict_mask(image,
                 mask_prompt, negative_mask_prompt,
                 mask_precision, negative_mask_precision,
                 debug=False):
    # prediction

    prompts = parse_prompt(mask_prompt)
    negative_prompts = parse_prompt(negative_mask_prompt)
    preds, negative_preds = predict_prompts(image, prompts, negative_prompts)

    # masking
    pred_mask, prompt_masks = mask_from_preds(preds, mask_precision)
    pred_mask = pred_mask and pred_mask.resize(image.size)

    if debug:
        for i, m in enumerate(prompt_masks):
            m.resize(image.size).save(
                output_path(f"prompt-{prompts[i]}.png"))

    neg_mask, negs_masks = mask_from_preds(
        negative_preds, negative_mask_precision)
    neg_mask = neg_mask and neg_mask.resize(image.size)

    if debug:
        for i, m in enumerate(negs_masks):
            m.resize(image.size).save(
                output_path(f"neg_prompt-{negative_prompts[i]}.png"))

    return (pred_mask, neg_mask)


class Script(scripts.Script):
    def title(self):
        return "txt2mask"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        # TODO: show only in inpaint tab
        with gr.Group():
            with gr.Row():
                mask_prompt = gr.Textbox(label="txt2mask Prompt", lines=1)
                negative_mask_prompt = gr.Textbox(
                    label="Negative prompt", lines=1)
            with gr.Row():
                mask_precision = gr.Slider(
                    label="Prompt precision", minimum=0.0, maximum=255.0, step=1.0, value=100.0)
                negative_mask_precision = gr.Slider(
                    label="Negative prompt precision", minimum=0.0, maximum=255.0, step=1.0, value=100.0)
            with gr.Row():
                mask_padding = gr.Slider(
                    label="Mask padding", minimum=0.0, maximum=500.0, step=1.0, value=0.0)
                negative_mask_padding = gr.Slider(
                    label="Negative mask padding", minimum=0.0, maximum=500.0, step=1.0, value=0.0)

            with gr.Row():
                brush_mask_mode = gr.Radio(label="Brush mask mode", choices=[
                    'Discard', 'Add', 'Substract'], value='Discard', type="index")
                debug = gr.Checkbox(label="Debug", value=True)

        return [
            mask_prompt, negative_mask_prompt,
            mask_precision, negative_mask_precision,
            mask_padding, negative_mask_padding,
            brush_mask_mode,
            debug]

    def run(self, params,
            mask_prompt, negative_mask_prompt,
            mask_precision, negative_mask_precision,
            mask_padding, negative_mask_padding,
            brush_mask_mode,
            debug):

        image = params.init_images[0]
        pred_mask, neg_mask = predict_mask(image,
                                           mask_prompt, negative_mask_prompt,
                                           mask_precision, negative_mask_precision,
                                           debug)

        pred_mask = blur(pred_mask.resize(image.size),
                         mask_padding) if pred_mask else None

        neg_mask = blur(neg_mask.resize(image.size),
                        negative_mask_padding) if neg_mask else None

        if pred_mask and neg_mask:
            merged = ImageChops.subtract(pred_mask, neg_mask)
        elif pred_mask:
            merged = pred_mask
        elif neg_mask:
            merged = ImageChops.invert(neg_mask)
        else:
            merged = Image.new("RGBA", image.size)

        mask, brush_mask = apply_brush_mask(
            merged, params.image_mask, brush_mask_mode)

        # Set up processor parameters correctly
        params.mode = 1
        params.mask_mode = 1
        params.image_mask = mask
        params.mask_for_overlay = params.image_mask
        params.latent_mask = None  # fixes inpainting full resolution

        if (debug):
            preview = mask_preview(image,
                                   pred_mask,
                                   neg_mask,
                                   brush_mask, brush_mask_mode)

            if pred_mask:
                pred_mask.save(output_path("prompt.png"))
            if neg_mask:
                neg_mask.save(output_path("neg_prompt.png"))
            if brush_mask:
                brush_mask.save(output_path("brush.png"))
            mask.save(output_path("final.png"))
            merged.save(output_path("merged.png"))
            preview.save(output_path("preview.png"))

        processed = processing.process_images(params)

        if (debug):
            processed.images.append(mask)
            processed.images.append(preview)

        return processed
