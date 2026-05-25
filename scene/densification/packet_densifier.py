import os
from utils.system_utils import mkdir_p


class PacketDensifier:
    """包投放调度器：组合采样器、初始化器和慢启动策略。"""

    def __init__(self, scheduler, sampler, initializer, logger=None, output_dir=None, save_every=1):
        self.scheduler = scheduler
        self.sampler = sampler
        self.initializer = initializer
        self.logger = logger
        self.output_dir = output_dir
        self.save_every = save_every
        self.packet_count = 0

    def densify(self, gaussian_model, current_psnr, iteration):
        packet_size, should_deploy = self.scheduler.step(current_psnr)
        if not should_deploy or packet_size <= 0:
            return

        # 每次投放一个包中心，包内包含 packet_size 个高斯体
        centers = self.sampler.sample(1, gaussian_model)
        new_packet = self.initializer.initialize(packet_size, centers)
        gaussian_model.add_gaussians(new_packet)

        if self.logger:
            self.logger.log_densify(
                iteration,
                self.scheduler.ema_psnr,
                self.scheduler.phase,
                packet_size,
                len(gaussian_model),
            )

        self.packet_count += 1
        if self.output_dir and self.save_every > 0 and (self.packet_count % self.save_every == 0):
            self._save_packet_snapshot(gaussian_model, iteration)

    def _save_packet_snapshot(self, gaussian_model, iteration):
        # 保存当前高斯体坐标，便于离线对比空间分布
        snapshot_dir = os.path.join(self.output_dir, "packet_snapshots")
        mkdir_p(snapshot_dir)
        ply_path = os.path.join(snapshot_dir, f"packet_{self.packet_count:04d}_iter_{iteration}.ply")
        gaussian_model.save_ply(ply_path)
