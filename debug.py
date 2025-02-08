import argparse
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from PIL import Image
import torch.nn.functional as F
from utils import *
from inference import *
import os
from huggingface_hub import hf_hub_download
import re

# Argument parser for source and save directories
# Download the model checkpoint
ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='/root/data/model', resume_download=True)
print(ckpt_path)

# Setup device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = DINOv2_MLP(
    dino_mode='large',
    in_dim=1024,
    out_dim=360+180+180+2,
    evaluate=True,
    mask_dino=False,
    frozen_back=False
)

dino.eval()
print('Model created')
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
print('Weights loaded')
val_preprocess = AutoImageProcessor.from_pretrained("facebook/dinov2-large", cache_dir='/root/data/model')

def sort_filenames(filenames):
    """ 파일 리스트를 숫자 인덱스를 기준으로 정렬하는 함수 """
    def extract_third_number(filename):
        match = re.search(r'prompt_\d+_orientation_\d+_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')  # 정수 변환하여 정렬
    return sorted(filenames, key=extract_third_number)

image_folder = "/root/code/ReNO/results/resampled/sd-turbo/reg_True_lr_3.0_seed_0_noise_optimize_False_noises_0"
output_file = "orientations.txt"

filenames = [f for f in os.listdir(image_folder) if f.endswith(".png") and not f.endswith("orientation.png") and not f.endswith("init.png") and not f.endswith("result.png")]
sorted_filenames = sort_filenames(filenames)
with open(os.path.join(image_folder, output_file), "w") as file:
    for image_path in sorted_filenames:
        image = Image.open(os.path.join(image_folder, image_path)).convert('RGB')

        angles = get_3angle(image, dino, val_preprocess, device)
        azimuth = float(np.radians(angles[0]))
        polar = float(np.radians(angles[1]))
        rotation = float(angles[2])
        confidence = float(angles[3])
        if image_path.startswith("prompt_0_orientation_0"):
            result_line = f"filename: {image_path}, azimuth: {angles[0]}\n"
            file.write(result_line)
            print(result_line.strip())  # 화면에도 출력