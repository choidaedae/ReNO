import ImageReward as RM
import torch

from rewards.base_reward import BaseRewardLoss
from huggingface_hub import hf_hub_download
from vision_tower import DINOv2_MLP
import numpy as np

class OrientLoss:
    """Image reward loss for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.name = "Orient"
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
        
        polar_sigma = 2.0
        polar_range = 180
        angles = np.arange(1, polar_range + 1)
        polar_distribution = np.exp(-((angles - polar) ** 2) / (2 * polar_sigma ** 2))
        polar_distribution = torch.tensor(polar_distribution / np.sum(polar_distribution), dtype=torch.float32)

        azimuth_sigma = 20.0
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

        rotation_sigma = 1.0
        rotation_range = 180
        angles = np.arange(1, rotation_range + 1)
        rotation_distribution = np.exp(-((angles - rotation) ** 2) / (2 * rotation_sigma ** 2))
        rotation_distribution = torch.tensor(rotation_distribution / np.sum(rotation_distribution), dtype=torch.float32)

        return azimuth_distribution, polar_distribution, rotation_distribution

    def __call__(self, image: torch.Tensor, orientation: np.ndarray) -> torch.Tensor:
        orient_score = self.score_diff(orientation, image)
        return orient_score

    def score_diff(self, target_orientation, image):
        azimuth_distribution, polar_distribution, rotation_distribution = self.orientation_to_distribution(target_orientation)
        gt_distribution = torch.unsqueeze(torch.cat([azimuth_distribution, polar_distribution, rotation_distribution]), dim=0).to(dtype=self.dtype, device=self.device)
        cur_distribution = self.orient_estimator.inference((image - self.mean) / self.std)[:, :-2]
        rewards = torch.nn.functional.kl_div(cur_distribution.log(), gt_distribution, reduction='batchmean')

        return rewards
