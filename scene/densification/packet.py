import torch


class GaussianPacket:
    """包级数据结构，存放一组高斯体的参数张量。"""

    def __init__(self, xyz, features_dc, features_rest, scaling, rotation, opacity):
        self.xyz = xyz
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.scaling = scaling
        self.rotation = rotation
        self.opacity = opacity

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to(self, device):
        for attr in self.__dict__:
            value = getattr(self, attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
        return self
