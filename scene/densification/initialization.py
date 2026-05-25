import torch
from utils.general_utils import inverse_sigmoid
from .packet import GaussianPacket


class PacketInitializer:
    """包初始化器基类。"""

    def initialize(self, packet_size, centers):
        raise NotImplementedError


class RandomInitializer(PacketInitializer):
    """随机初始化：先采样包中心，再在中心附近生成包内高斯体。"""

    def __init__(self, bbox_min, bbox_max, packet_radius_scale=0.02, sh_degree=3, radius_basis=None):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.radius_basis = radius_basis if radius_basis is not None else torch.norm(bbox_max - bbox_min)
        self.radius = packet_radius_scale * self.radius_basis
        self.sh_degree = sh_degree

    def initialize(self, packet_size, centers):
        if centers is None or packet_size <= 0:
            raise ValueError("packet_size 和 centers 必须有效")
        device = self.bbox_min.device
        dtype = self.bbox_min.dtype
        centers = centers.to(device=device, dtype=dtype)

        num_centers = centers.shape[0]
        total_points = packet_size * num_centers
        centers_expanded = centers.repeat_interleave(packet_size, dim=0)

        # 包内高斯：围绕中心做高斯扰动，并裁剪到场景包围盒内
        offsets = torch.randn((total_points, 3), device=device, dtype=dtype) * self.radius
        xyz = centers_expanded + offsets
        xyz = torch.max(torch.min(xyz, self.bbox_max), self.bbox_min)

        sh_coeffs = (self.sh_degree + 1) ** 2
        features_dc = torch.full((total_points, 1, 3), 0.5, device=device, dtype=dtype)
        features_rest = torch.zeros((total_points, sh_coeffs - 1, 3), device=device, dtype=dtype)
        scaling = torch.log(torch.ones((total_points, 3), device=device, dtype=dtype) * 0.01)
        rotation = torch.zeros((total_points, 4), device=device, dtype=dtype)
        rotation[:, 0] = 1.0
        opacity = inverse_sigmoid(torch.ones((total_points, 1), device=device, dtype=dtype) * 0.1)

        return GaussianPacket(xyz, features_dc, features_rest, scaling, rotation, opacity)


class SfMPerturbInitializer(PacketInitializer):
    """基于 SfM 点云扰动的初始化器（预留）。"""

    def __init__(self, *args, **kwargs):
        self._not_ready = True

    def initialize(self, packet_size, centers):
        raise NotImplementedError("SfMPerturbInitializer 预留，尚未启用")
