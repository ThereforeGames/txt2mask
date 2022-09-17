# Author: Therefore Games
# https://github.com/ThereforeGames/txt2img2img

import modules.scripts as scripts
import gradio as gr

from modules import processing, images, shared, sd_samplers
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state, Options

import torch
import cv2
import base64

from repositories.clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

class Script(scripts.Script):
	def title(self):
		return "txt2mask v0.0.1"

	def show(self, is_img2img):
		return is_img2img

	def ui(self, is_img2img):
		if not is_img2img:
			return None

		mask_prompt = gr.Textbox(label="Mask prompt", lines=1)
		mask_precision = gr.Slider(label="Mask precision", minimum=0.0, maximum=255.0, step=1.0, value=96.0)

		plug = gr.HTML(label="plug",value='<div class="gr-block gr-box relative w-full overflow-hidden border-solid border border-gray-200 gr-panel"><p>If you like my work, please consider showing your support on <strong><a href="https://patreon.com/thereforegames" target="_blank">Patreon</a></strong>. Thank you! &#10084;</p></div>')

		return [mask_prompt,mask_precision, plug]

	def run(self, p, mask_prompt, mask_precision, plug):

		def get_mask():
			# load model
			model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
			model.eval();

			# non-strict, because we only stored decoder weights (not CLIP weights)
			model.load_state_dict(torch.load('./repositories/clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False);			

			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				transforms.Resize((512, 512)),
			])
			img = transform(p.init_images[0]).unsqueeze(0)

			prompts = mask_prompt.split("|")
			prompt_parts = len(prompts)

			# predict
			with torch.no_grad():
				preds = model(img.repeat(prompt_parts,1,1,1), prompts)[0]

			filename = f"mask.png"
			plt.imsave(filename,torch.sigmoid(preds[0][0]))

			# TODO: Figure out how to convert the plot above to numpy instead of re-loading image
			img = cv2.imread(filename)

			gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			(thresh, bw_image) = cv2.threshold(gray_image, mask_precision, 255, cv2.THRESH_BINARY)

			# blur_image = cv2.GaussianBlur(bw_image, (mask_blur, mask_blur), 0)

			# For debugging only:
			cv2.imwrite(filename,bw_image)

			# fix color format
			cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

			return (Image.fromarray(bw_image))
						

		# Set up processor parameters correctly
		p.mode = 1
		p.mask_mode = 1
		p.image_mask = get_mask()

		processed = processing.process_images(p)

		return processed