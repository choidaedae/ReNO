import ImageReward as RM
import torch

from rewards.base_reward import BaseRewardLoss
from huggingface_hub import hf_hub_download
from vision_tower import DINOv2_MLP
import numpy as np
import torchvision.transforms as T
import rembg
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# background preprocessing

def resize_foreground_torch(
    image: torch.Tensor,
    ratio: float,
    nonzero_coords: torch.Tensor
) -> torch.Tensor:
    
    # Assume image shape is (C, H, W)
    assert image.ndim == 3 and image.shape[0] == 3  # Ensure image is (C, H, W)
    if nonzero_coords[0].numel() == 0:
        y2 = image.shape[1] # 기본값 설정 (예시)
        y1 = 0
    else:
        y1, y2 = nonzero_coords[0].min().item(), nonzero_coords[0].max().item()
    if nonzero_coords[1].numel() == 0:
        x2 = image.shape[2] # 기본값 설정 (예시)
        x1 = 0
    else:
        x1, x2 = nonzero_coords[1].min().item(), nonzero_coords[1].max().item()

    # Crop the foreground (keeping all channels)
    fg = image[:, y1:y2, x1:x2]  # Cropping on H and W dimensions

    # Pad to square
    size = max(fg.shape[1], fg.shape[2])  # Compare H and W dimensions
    ph0, pw0 = (size - fg.shape[1]) // 2, (size - fg.shape[2]) // 2
    ph1, pw1 = size - fg.shape[1] - ph0, size - fg.shape[2] - pw0

    # Pad to make the cropped foreground square
    new_image = torch.nn.functional.pad(
        fg,
        (pw0, pw1, ph0, ph1),  # (left, right, top, bottom) for padding
        mode="constant",
        value=0
    )

    # Compute padding according to the ratio
    new_size = int(new_image.shape[1] / ratio)  # Use height (or width, since it's square now)

    # Pad to the new size
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0

    new_image = torch.nn.functional.pad(
        new_image,
        (pw0, pw1, ph0, ph1),
        mode="constant",
        value=0
    )

    return new_image

def differentiable_background_preprocess(decoded_x0: torch.Tensor, seg: bool = False, predefined_bbox=None) -> torch.Tensor:
    # Preprocess background with differentiable manner
    """
    Input:
        decoded_x0: torch.Tensor,  (1, C, H, W)
    """
    decoded_x0 = decoded_x0.squeeze(0)
    C, H, W = decoded_x0.shape
    
    img = T.ToPILImage()(decoded_x0.detach().clone().cpu().to(torch.float32))
    rembg_session = rembg.new_session()
    removed_image = rembg.remove(img, session=rembg_session)

    alpha = torch.from_numpy(np.array(removed_image))[..., 3] > 0
    nonzero_coords = torch.nonzero(alpha, as_tuple=True)

    if len(nonzero_coords[0]) == 0 or len(nonzero_coords[1]) == 0:
        return decoded_x0.unsqueeze(0)

    # Use bounding box and expand it
    y_min, x_min = nonzero_coords[0].min().item(), nonzero_coords[1].min().item()
    y_max, x_max = nonzero_coords[0].max().item(), nonzero_coords[1].max().item()

    # Expand bounding box by a factor of 1/0.85
    scale_factor = 1 / 0.85
    box_height = y_max - y_min + 1
    box_width = x_max - x_min + 1

    if seg:
        mask = torch.from_numpy(np.array(removed_image)[:, :, :-1] != 0).to(decoded_x0.dtype).to(decoded_x0.device).permute(2, 0, 1)

        removed_x0 = mask * decoded_x0
    else:
        # Calculate new expanded coordinates
        new_y_min = max(0, int(y_min - (scale_factor - 1) * box_height / 2))
        new_y_max = min(H, int(y_max + (scale_factor - 1) * box_height / 2))
        new_x_min = max(0, int(x_min - (scale_factor - 1) * box_width / 2))
        new_x_max = min(W, int(x_max + (scale_factor - 1) * box_width / 2))
        # mask = torch.from_numpy(np.array(removed_image)[:, :, :-1] != 0).to(decoded_x0.dtype).to(decoded_x0.device).permute(2, 0, 1)
        # Create expanded bounding box mask
        bounding_box_mask = torch.zeros_like(decoded_x0).to(decoded_x0.dtype).to(decoded_x0.device)
        bounding_box_mask[:, new_y_min:new_y_max+1, new_x_min:new_x_max+1] = 1.0

        # Apply mask
        removed_x0 = bounding_box_mask * decoded_x0

    if predefined_bbox:
        new_x_min, new_x_max, new_y_min, new_y_max = predefined_bbox
        # Create expanded bounding box mask
        bounding_box_mask = torch.zeros_like(decoded_x0).to(decoded_x0.dtype).to(decoded_x0.device)
        bounding_box_mask[:, new_y_min:new_y_max+1, new_x_min:new_x_max+1] = 1.0

        # Apply mask
        removed_x0 = bounding_box_mask * decoded_x0

    # Resize foreground object
    removed_x0 = resize_foreground_torch(removed_x0, ratio=1.0, nonzero_coords=nonzero_coords)
    
    return removed_x0.unsqueeze(0)


class OrientLoss:
    """Image reward loss for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        background_preprocess = False,
        sigmas = (20.0, 2.0, 1.0),
        memsave: bool = False,
    ):
        self.name = "Orient"
        self.log = ""
        self.weighting = weighting
        self.dtype = dtype
        ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir=cache_dir, resume_download=True) # large    
        self.orient_estimator = DINOv2_MLP(
            dino_mode='large',
            in_dim=1024,
            out_dim=360+180+180+2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        )
        self.orient_estimator = self.orient_estimator.to(device=device, dtype=self.dtype)
        self.device = device
        self.background_preprocess = background_preprocess
    
        self.azimuth_sigma, self.polar_sigma, self.rotation_sigma = sigmas

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),  # Bicubic Resize
            transforms.CenterCrop((224, 224)),  # Center Crop to 224x224
            transforms.Lambda(lambda x: (x - self.mean) / self.std)  # Normalize
        ])

        # for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        self.orient_estimator.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.orient_estimator.eval()
        BaseRewardLoss.freeze_parameters(self.orient_estimator.parameters())

    def orientation_to_distribution(self, orientation):
        """
        Generate a Gaussian distribution for the given angle.
        
        Args:
            theta (int): The ground truth angle in degrees.
            sigma (float): Variance for the Gaussian distribution.
            angle_range (int): The maximum angle value (e.g., 180 or 360).
            
        Returns:
            np.ndarray: Discretized probability distribution.
        """

        azimuth, polar, rotation = orientation
        
        polar_sigma = self.polar_sigma
        polar_range = 180
        angles = np.arange(1, polar_range + 1)
        polar_distribution = np.exp(-((angles - polar) ** 2) / (2 * polar_sigma ** 2))
        polar_distribution = torch.tensor(polar_distribution / np.sum(polar_distribution), dtype=torch.float32)
        
        azimuth_sigma = self.azimuth_sigma
        azimuth_range = 360
        angles = np.arange(1, azimuth_range + 1)
        # Compute circular distance
        circular_distance = np.minimum(np.abs(angles - azimuth), azimuth_range - np.abs(angles - azimuth))
        # Compute Gaussian distribution using circular distance
        azimuth_distribution = np.exp(-0.5 * (circular_distance / azimuth_sigma) ** 2)
        # Normalize to ensure sum = 1
        azimuth_distribution /= np.sum(azimuth_distribution)
        # Convert to torch tensor
        azimuth_distribution = torch.tensor(azimuth_distribution, dtype=torch.float32)

        rotation_sigma = self.rotation_sigma
        rotation_range = 180
        angles = np.arange(1, rotation_range + 1)
        rotation_distribution = np.exp(-((angles - rotation) ** 2) / (2 * rotation_sigma ** 2))
        rotation_distribution = torch.tensor(rotation_distribution / np.sum(rotation_distribution), dtype=torch.float32)

        return azimuth_distribution, polar_distribution, rotation_distribution

    def __call__(self, image: torch.Tensor, orientation: np.ndarray) -> torch.Tensor:
        orient_score, fig = self.score_diff(orientation, image)
        return orient_score, fig

    def score_diff(self, target_orientation, image):
        azimuth_distribution, polar_distribution, rotation_distribution = self.orientation_to_distribution(target_orientation)
        gt_distribution = torch.unsqueeze(torch.cat([azimuth_distribution, polar_distribution, rotation_distribution]), dim=0).to(dtype=self.dtype, device=self.device)
        if self.background_preprocess == "seg":
            image = differentiable_background_preprocess(image, seg=True)
        elif self.background_preprocess == "bbox":
            image = differentiable_background_preprocess(image, seg=False)
        elif self.background_preprocess == "predefined_bbox":
            image = differentiable_background_preprocess(image, seg=False, predefined_bbox=self.predefined_bbox)

        cur_distribution = self.orient_estimator.inference((self.transform(image)))[:, :-2]

        self.log += f"azimuth: {torch.argmax(cur_distribution[:, 0:360], dim=1).item()}, polar: {torch.argmax(cur_distribution[:, 360:540], dim=1).item()}, rotation: {torch.argmax(cur_distribution[:, 540:720], dim=1).item()}\n"
        # print(f"azimuth: {torch.argmax(cur_distribution[:, 0:360], dim=1).item()}, polar: {torch.argmax(cur_distribution[:, 360:540], dim=1).item()}, rotation: {torch.argmax(cur_distribution[:, 540:720], dim=1).item()}")
        azimuth_rewards = torch.nn.functional.kl_div(cur_distribution[:, 0:360].log(), gt_distribution[:, 0:360], reduction='batchmean')
        polar_rewards = torch.nn.functional.kl_div(cur_distribution[:, 360:540].log(), gt_distribution[:, 360:540], reduction='batchmean')
        rotation_rewards = torch.nn.functional.kl_div(cur_distribution[:, 540:720].log(), gt_distribution[:, 540:720], reduction='batchmean')
        reward_items = [azimuth_rewards.clone().detach().cpu().item(), polar_rewards.clone().detach().cpu().item(), rotation_rewards.clone().detach().cpu().item()] 

        split_sizes = [360, 180, 180]
        angles = ['azimuth', 'polar', 'rotation']
        split_tensors = np.split(cur_distribution[0].clone().detach().cpu(), np.cumsum(split_sizes)[:-1])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        for i, split_tensor in enumerate(split_tensors):
            axes[i].plot(split_tensor)
            axes[i].set_title(f'{angles[i]}: {np.argmax(split_tensor)}, loss: {reward_items[i]}')

        plt.tight_layout()
        #plt.savefig('orientation_distribution.png')
        rewards = azimuth_rewards + polar_rewards + rotation_rewards
        return rewards, fig
