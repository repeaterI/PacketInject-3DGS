from dataclasses import dataclass

import torch


@dataclass
class PacketSpatialConfig:
    reference: str
    sample_min: torch.Tensor
    sample_max: torch.Tensor
    radius_basis: float
    bbox_diag: float
    cameras_extent: float
    center_offset: float


class PacketSpatialPreprocessor:
    """根据场景统计量选择包投放参考系，不修改原始数据。"""

    def __init__(self, reference_mode="auto", auto_switch_ratio=3.0, auto_center_ratio=0.5):
        self.reference_mode = reference_mode
        self.auto_switch_ratio = auto_switch_ratio
        self.auto_center_ratio = auto_center_ratio

    def resolve(self, scene):
        bbox_min, bbox_max = scene.get_bounding_box()
        bbox_min = bbox_min.to("cuda")
        bbox_max = bbox_max.to("cuda")
        bbox_diag = torch.norm(bbox_max - bbox_min).item()
        cameras_extent = max(float(scene.cameras_extent), 1e-6)

        bbox_center = (bbox_min + bbox_max) * 0.5
        camera_center = getattr(scene, "camera_center", None)
        center_offset = 0.0
        if camera_center is not None:
            center_offset = torch.norm(bbox_center - camera_center.to("cuda")).item()
        
        
        # 如果你已经知道这个场景 bbox 和相机尺度差得很大，可以直接强制用相机 extent 作为参考系（--packet_reference extent）
        #  --packet_reference bbox： 强制用 bbox 作为参考系
        reference = self.reference_mode
        if reference == "auto":
            bbox_extent_ratio = bbox_diag / cameras_extent
            center_ratio = center_offset / cameras_extent if cameras_extent > 0 else 0.0
            if bbox_extent_ratio > self.auto_switch_ratio or center_ratio > self.auto_center_ratio:
                reference = "extent"
            else:
                reference = "bbox"

        if reference == "extent":
            camera_center = camera_center.to("cuda") if camera_center is not None else bbox_center
            sample_min = camera_center - cameras_extent
            sample_max = camera_center + cameras_extent
            radius_basis = cameras_extent
        else:
            sample_min = bbox_min
            sample_max = bbox_max
            radius_basis = bbox_diag

        return PacketSpatialConfig(
            reference=reference,
            sample_min=sample_min,
            sample_max=sample_max,
            radius_basis=radius_basis,
            bbox_diag=bbox_diag,
            cameras_extent=cameras_extent,
            center_offset=center_offset,
        )