# txt2mask for Stable Diffusion
Automatically create masks for inpainting with Stable Diffusion using natural language.

## Introduction

txt2mask is an addon for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows you to enter a text string in img2img mode which automatically creates an image mask. It is powered by [clipseg](https://github.com/timojl/clipseg). No more messing around with that tempermental brush tool. 😅

This script is still under active development.

![image](https://user-images.githubusercontent.com/95403634/190878562-d020887c-ccb0-411c-ab37-38e2115552eb.png)

## Installation

Simply clone or download this repo and place the files in the base directory of Automatic's web UI.

## Usage

From the img2img screen, select txt2mask as your active script:

![image](https://user-images.githubusercontent.com/95403634/190878234-43134aff-0843-4caf-a0ea-146d6e1891dc.png)

In the `Mask Prompt` field, enter the text to search for within your image. (In the case of the topmost screenshot, this value would be 'business suit' and the prompt box at the top of your UI would say 'sci-fi battle suit.')

Adjust the `Mask Precision` field to increase or decrease the confidence of that which is masked. Lowering this value too much means it may select more than you intend.

Press Generate. That's it!

## Known Issues

- This uses a different underlying tech for language interpretation, so entering a finetuned object into the Mask Prompt won't work. In general, less is more for masking: instead of trying to mask "a one-armed man doing a backflip off a barn" you will probably have more luck writing "a man."