import torch
from PIL import Image
from utils import *
import torch.nn.functional as F
import numpy as np

def dino_denormalize_and_save(tensor_image: torch.Tensor, save_path: str):
    """
    DINO 모델을 위한 정규화된 Tensor 이미지를 역변환(denormalize)하고 PIL 이미지로 저장.

    Args:
        tensor_image (torch.Tensor): (C, H, W) 형태의 정규화된 이미지 Tensor
        save_path (str): 저장할 파일 경로 (예: "output.jpg")
    """
    # DINO 정규화에 사용된 mean 및 std
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    # 1. Denormalize: (x' * std) + mean
    tensor_image = (tensor_image * std) + mean
    
    # 2. 다시 0~255 범위로 변환
    tensor_image = tensor_image * 255.0

    # 3. Clamp(0,255)로 안전한 범위 내에서 유지 (Overflow 방지)
    tensor_image = torch.clamp(tensor_image, 0, 255)

    # 4. PyTorch Tensor → PIL 변환 (C, H, W → H, W, C)
    pil_image = Image.fromarray(tensor_image.byte().permute(1, 2, 0).cpu().numpy())

    # 5. 저장
    pil_image.save(save_path)

    print(f"Image saved to {save_path}")

def get_3angle(image, dino, val_preprocess, device):
    
    image_inputs = val_preprocess(images = image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred   = torch.argmax(dino_pred[:, 0:360], dim=-1)
    gaus_pl_pred   = torch.argmax(dino_pred[:, 360:360+180], dim=-1)
    gaus_ro_pred   = torch.argmax(dino_pred[:, 360+180:360+180+180], dim=-1)
    confidence     = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]
    angles = torch.zeros(4)
    angles[0]  = gaus_ax_pred
    angles[1]  = gaus_pl_pred - 90
    angles[2]  = gaus_ro_pred - 90
    angles[3]  = confidence
    return angles


def get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device):
    
    # image = Image.open(image_path).convert('RGB')
    image = get_crop_images(origin_img, num=3) + get_crop_images(rm_bkg_img, num=3)
    image_inputs = val_preprocess(images = image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred   = torch.argmax(dino_pred[:, 0:360], dim=-1).to(torch.float32)
    gaus_pl_pred   = torch.argmax(dino_pred[:, 360:360+180], dim=-1).to(torch.float32)
    gaus_ro_pred   = torch.argmax(dino_pred[:, 360+180:360+180+180], dim=-1).to(torch.float32)
    
    gaus_ax_pred   = remove_outliers_and_average_circular(gaus_ax_pred)
    gaus_pl_pred   = remove_outliers_and_average(gaus_pl_pred)
    gaus_ro_pred   = remove_outliers_and_average(gaus_ro_pred)
    
    confidence = torch.mean(F.softmax(dino_pred[:, -2:], dim=-1), dim=0)[0]
    angles = torch.zeros(4)
    angles[0]  = gaus_ax_pred
    angles[1]  = gaus_pl_pred - 90
    angles[2]  = gaus_ro_pred - 90
    angles[3]  = confidence
    return angles