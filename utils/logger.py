#
# Copyright (C) 2026
#

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class DensifyLogger:
    """包投放式致密化实验的 TensorBoard 记录器封装。"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir) if _TENSORBOARD_AVAILABLE else None

    def __bool__(self):
        return self.writer is not None

    def add_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def add_images(self, tag, images, global_step=None):
        if self.writer:
            self.writer.add_images(tag, images, global_step=global_step)

    def add_histogram(self, tag, values, step):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_densify(self, step, ema_psnr, phase, packet_size, total_gaussians):
        # 记录包投放过程的关键曲线，便于观察慢启动节奏
        if not self.writer:
            return
        self.writer.add_scalar("densify/ema_psnr", ema_psnr, step)
        self.writer.add_scalar("densify/phase", self._phase_to_int(phase), step)
        self.writer.add_scalar("densify/packet_size", packet_size, step)
        self.writer.add_scalar("densify/total_gaussians", total_gaussians, step)

    def log_eval(self, step, psnr, ssim, lpips):
        # 统一记录评估指标，便于对比 PSNR-高斯数量曲线
        if not self.writer:
            return
        self.writer.add_scalar("eval/psnr", psnr, step)
        self.writer.add_scalar("eval/ssim", ssim, step)
        self.writer.add_scalar("eval/lpips", lpips, step)

    @staticmethod
    def _phase_to_int(phase):
        mapping = {
            "exponential": 0,
            "linear": 1,
            "saturated": 2,
        }
        return mapping.get(phase, -1)
