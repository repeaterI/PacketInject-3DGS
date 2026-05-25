import torch


class PositionSampler:
    """投放位置采样器基类。"""

    def sample(self, num_centers, gaussian_model):
        raise NotImplementedError


class UniformRandomSampler(PositionSampler):
    """在场景包围盒内均匀采样包中心。"""

    def __init__(self, bbox_min, bbox_max):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def sample(self, num_centers, gaussian_model):
        device = self.bbox_min.device
        dtype = self.bbox_min.dtype
        if num_centers <= 0:
            return torch.empty((0, 3), device=device, dtype=dtype)
        rand = torch.rand((num_centers, 3), device=device, dtype=dtype)
        return self.bbox_min + rand * (self.bbox_max - self.bbox_min)


class GradientGuidedSampler(PositionSampler):
    """基于梯度累计信息进行采样，若无梯度则退化为均匀采样。"""

    def __init__(self, bbox_min, bbox_max, jitter_scale=0.02):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.jitter_radius = jitter_scale * torch.norm(bbox_max - bbox_min)

    def sample(self, num_centers, gaussian_model):
        device = self.bbox_min.device
        dtype = self.bbox_min.dtype
        if num_centers <= 0:
            return torch.empty((0, 3), device=device, dtype=dtype)
        if gaussian_model is None or gaussian_model.xyz_gradient_accum.numel() == 0:
            return UniformRandomSampler(self.bbox_min, self.bbox_max).sample(num_centers, gaussian_model)

        weights = gaussian_model.xyz_gradient_accum.view(-1).to(device)
        weights = torch.relu(weights)
        if torch.isnan(weights).any() or weights.sum() <= 0:
            return UniformRandomSampler(self.bbox_min, self.bbox_max).sample(num_centers, gaussian_model)

        probs = weights / weights.sum()
        indices = torch.multinomial(probs, num_centers, replacement=True)
        centers = gaussian_model.get_xyz[indices].to(device)

        if self.jitter_radius > 0:
            centers = centers + torch.randn_like(centers) * self.jitter_radius
            centers = torch.max(torch.min(centers, self.bbox_max), self.bbox_min)

        return centers
