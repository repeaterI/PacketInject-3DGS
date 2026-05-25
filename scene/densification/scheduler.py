class GrowthScheduler:
    """慢启动调度器：指数 -> 线性 -> 饱和。"""

    def __init__(self, base_packet_size, psnr_threshold=25.0, ema_alpha=0.3, sat_window=3, sat_delta=0.05, max_packet_size=8000):
        self.base_packet_size = base_packet_size
        self.psnr_threshold = psnr_threshold
        self.ema_alpha = ema_alpha
        self.sat_window = sat_window
        self.sat_delta = sat_delta
        self.max_packet_size = max_packet_size

        self.phase = "exponential"
        self.ema_psnr = None
        self.deploy_count = 0
        self.sat_counter = 0
        self.last_ema_psnr = None

    def step(self, current_psnr):
        if current_psnr is None:
            return 0, False
        current_psnr = float(current_psnr)

        # 更新 EMA
        if self.ema_psnr is None:
            self.ema_psnr = current_psnr
        else:
            self.ema_psnr = self.ema_alpha * current_psnr + (1.0 - self.ema_alpha) * self.ema_psnr

        # 阶段切换与投放逻辑
        if self.phase == "exponential":
            if self.ema_psnr >= self.psnr_threshold:
                self.phase = "linear"
                self.deploy_count = 0
                self.sat_counter = 0
                self.last_ema_psnr = None
            else:
                packet_size = self.base_packet_size * (2 ** self.deploy_count)
                packet_size = min(packet_size, self.max_packet_size)
                self.deploy_count += 1
                return packet_size, True

        if self.phase == "linear":
            if self.last_ema_psnr is not None:
                delta = self.ema_psnr - self.last_ema_psnr
                if delta < self.sat_delta:
                    self.sat_counter += 1
                else:
                    self.sat_counter = 0
            self.last_ema_psnr = self.ema_psnr

            if self.sat_counter >= self.sat_window:
                self.phase = "saturated"
                return 0, False

            return self.base_packet_size, True

        return 0, False
