import sys
from PIL import Image
import os, pickle
import pdb
import cm

CM = cm.ContextManager()

img1 = Image.open('sample_imgs/frames1.jpg').resize((768, 768))
img2 = Image.open('sample_imgs/frames2.jpg').resize((768, 768))

prompt = 'anime, ultra hd, high quality, 8k wallpaper, cinematic, highly detailed'
n_prompt = 'text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'

qc_prompt = 'portrait, centered, hyperrealistic, unreal engine, cinematic'
qc_neg_prompt = 'text, signature, logo, distorted, ugly, bad anatomy, weird face, weird eyes, asymmetrical face, bad anatomy, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='out/frames100.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=100, ddim_steps=200, num_frames=9, guide_scale=10, schedule_type='linear', out_dir='out')
