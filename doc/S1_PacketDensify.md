# 3DGS 包投放式致密化机制实验开发文档

## 1. 核心目标
在训练阶段以**最少的“包”（Packet）数量**逼近原版 3DGS 的 PSNR 指标，通过从根本上控制高斯体增长避免冗余，为后续高斯体压缩研究提供高信息密度的初始模型。

## 2. 验证目标与阶段规划

| 版本 | 投放策略 | 增长策略 | 核心验证问题 |
|------|----------|----------|--------------|
| v0.1 | 随机投放 | 慢启动（指数→线性→饱和） | 慢启动框架本身是否优于原版 split/clone？ |
| v0.2 | 密度感知随机 | 慢启动 | 局部密度引导能否优于纯随机？ |
| v0.3 | 梯度引导随机 | 慢启动 | 梯度引导与慢启动结合能否进一步压缩高斯体数量？ |
| v0.4 | 根据“包满载率”阈值和梯度引导迁移形变投放（可能变成刺球） | 慢启动 | 包级投放与压缩能否实现更激进的高斯体数量控制？ |

**当前阶段：v0.1** 搭建完整框架并完成随机投放基线，验证慢启动逻辑的可靠性与可观测性。

## 3. 基线对比方案
- **原版 3DGS**：使用标准训练流程（split/clone + prune），记录最终 PSNR、SSIM、LPIPS 及高斯体总数。
- **本方案（v0.1）**：仅使用包投放 + 慢启动，Prune 逻辑暂保留原版，记录 PSNR‑高斯数量曲线。  
  *注：暂不进行“原版+后处理压缩”的对比，聚焦于训练侧内生稀疏性。*

测试场景优先选用 **Horse**（中等规模、清晰几何与纹理），确认机制可行性后再扩展至 Mip-NeRF 360 等复杂场景。

## 4. 机制设计与实现框架

### 4.1 整体文件结构（基于原仓库扩展）
```
gaussian-splatting/
├── arguments/
│   └── __init__.py                # 新增 packet 相关参数
├── scene/
│   ├── gaussian_model.py          # 保留原版，移除 split/clone 调用
│   └── densification/             # 新增模块
│       ├── __init__.py
│       ├── packet.py              # Packet 数据结构
│       ├── packet_densifier.py    # 核心投放调度器
│       ├── initialization.py      # PacketInitializer（随机、扰动）
│       ├── scheduler.py           # GrowthScheduler（慢启动）
│       └── sampler.py             # PositionSampler（随机、梯度引导）
├── utils/
│   └── logger.py                  # 新增统一 TensorBoard 记录器
└── train.py                       # 整合新致密化流程
```

### 4.2 Packet 数据结构与初始化
**包定义**：一个 Packet 包含固定数量（`packet_size`）的高斯体，包内高斯位置相互靠近，包中心随机投放。

- `Packet` 类：存储 `xyz, features_dc, features_rest, scaling, rotation, opacity` 等张量。
- 包内位置生成：先采样包中心，再从 `N(center, radius²I)` 采样 `packet_size` 个点（`radius = scene_diagonal / 50`，由 `packet_radius_scale` 控制）。
- 其他属性（颜色、尺度、旋转等）按原版初始化逻辑填充。

**初始化策略**（通过 `PacketInitializer` 抽象）：
- `RandomInitializer`（当前使用）：完全随机产生包中心及包内高斯。
- `SfMPerturbInitializer`（预留）：基于 SfM 点云添加扰动。

**初始包数**：`init_packets = 1`，即训练开始时仅有 1000 个高斯体（`packet_size=1000`），以极端匮乏的初始条件凸显慢启动行为。

### 4.3 慢启动调度器（GrowthScheduler）
状态机实现 TCP 慢启动思想：

```
指数增长阶段 ──(EMA_PSNR ≥ 25 dB)──► 线性增长阶段
线性增长阶段 ──(连续N次ΔEMA_PSNR < δ)──► 饱和阶段（停止投放）
```

**关键参数**：
- `psnr_threshold = 21.0`：切换阈值
- `ema_alpha = 0.3`：PSNR 指数移动平均系数
- `sat_window = 3`：饱和判定窗口（连续投放次数）
- `sat_delta = 0.3`：PSNR 增量阈值 (dB)

**每步决策**（`step(ema_psnr) -> (packet_size_to_use, should_deploy)`）：
- 指数阶段：`packet_size` 随已投放次数指数增长（例如 1000, 2000, 4000…），EMA_PSNR < 25dB 时投放。
- 线性阶段：`packet_size` 固定为基数（如当前 packet_size），每次触发且未饱和时投放。
- 饱和阶段：`packet_size = 0`，`should_deploy = False`。

### 4.4 投放位置采样器（PositionSampler）
抽象接口 `sample(n, gaussian_model) -> positions`，所有采样器均接收 `gaussian_model` 以便未来使用梯度信号。

- `UniformRandomSampler`：忽略梯度，在场景包围盒内均匀采样包中心。
- `GradientGuidedSampler`（当前版本实现但默认禁用）：基于 `gaussian_model.xyz_gradient_accum` 构建概率分布，`torch.multinomial` 抽取区域，添加随机偏移。

**调用点**：`PacketDensifier.deploy()` 内部获取位置后，调用 `PacketInitializer` 生成高斯体。

### 4.5 PacketDensifier 核心逻辑
`PacketDensifier` 组合三个策略模块：

```python
class PacketDensifier:
    def __init__(self, scheduler, sampler, initializer, logger):
        self.scheduler = scheduler
        self.sampler = sampler
        self.initializer = initializer
        self.logger = logger
        self.packet_count = 0

    def densify(self, gaussian_model, ema_psnr, iteration):
        packet_size, deploy = self.scheduler.step(ema_psnr)
        if not deploy:
            return
        positions = self.sampler.sample(packet_size, gaussian_model)
        new_gaussians = self.initializer.initialize(packet_size, positions)
        gaussian_model.add_gaussians(new_gaussians)
        # 日志记录
        self.logger.log_densify(iteration, ema_psnr, self.scheduler.phase,
                                packet_size, len(gaussian_model))
        self.packet_count += 1
```

原版 `train.py` 中，用 `packet_densifier.densify()` 替换 `gaussian_model.densify_and_prune()` 中的 split/clone 部分，prune 仍保留。

### 4.6 训练流程集成
- 保留原版 `densification_interval` 触发机制。
- 每轮评估后计算 EMA_PSNR（使用测试集 PSNR 或验证集 PSNR，按需选择）。
- 调用 `packet_densifier.densify()`，传递当前 EMA_PSNR 和 iteration。
- 新增高斯体后，扩展优化器参数（与原版扩展方式相同）。

## 5. 数据记录与可视化（TensorBoard 配置）

### 5.1 记录器设计
新建 `utils/logger.py`，封装 `torch.utils.tensorboard.SummaryWriter`，并扩展原版已有的日志记录功能：

```python
class DensifyLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    
    def log_densify(self, step, ema_psnr, phase, packet_size, total_gaussians):
        self.writer.add_scalar('densify/ema_psnr', ema_psnr, step)
        self.writer.add_scalar('densify/phase', phase_to_int(phase), step)
        self.writer.add_scalar('densify/packet_size', packet_size, step)
        self.writer.add_scalar('densify/total_gaussians', total_gaussians, step)
    
    def log_eval(self, step, psnr, ssim, lpips):
        self.writer.add_scalar('eval/psnr', psnr, step)
        self.writer.add_scalar('eval/ssim', ssim, step)
        self.writer.add_scalar('eval/lpips', lpips, step)
```

**原版修改点**：
- 在 `train.py` 中，将原版的 `tb_writer` 替换为 `DensifyLogger` 实例，保留其原有损失记录功能，增加 densify 相关标量。
- 移除原版 `densify_and_prune` 中的 split/clone 相关日志（如点数变化原因等），改用新日志。

### 5.2 关键图表与存储变量
每个实验保存独立的 TensorBoard 日志文件，可通过以下面板监控：

| 图表名称 | 横轴 | 纵轴 | 说明 |
|----------|------|------|------|
| `eval/psnr` vs `densify/total_gaussians` | 高斯总数 | PSNR | 效率曲线，标注阶段切换点 |
| `densify/packet_size` over steps | 迭代次数 | Packet 大小 | 慢启动速度变化 |
| `densify/ema_psnr` | 迭代次数 | EMA-PSNR | 饱和趋势 |
| `gaussian_heatmap` | 自定义图像 | - | 高斯中心投影密度热力图（通过 `utils/graphics_utils` 导出点云渲染） |

**额外存储**：在每次投放时将当前所有高斯体坐标保存为 `.ply` 文件（每隔 N 次 densify 一次），用于离线对比不同版本的空间分布。

## 6. 实验执行计划（Horse 场景）

1. **环境准备**：基于原仓库 `gaussian-splatting` 创建分支 `feature/packet-densification`。
2. **实现 v0.1 框架**：
   - 添加 `arguments` 参数（见附录）。
   - 实现 `densification/` 全部模块，`utils/logger.py`。
   - 修改 `train.py` 集成新致密化流程。
3. **运行 Horse 场景**：
   - 训练原版 3DGS，记录基线（PSNR ~ 28-29 dB，高斯数 ~ 百万级）。
   - 训练 v0.1 版本，参数：`init_packets=1, packet_size=1000, psnr_threshold=21.0`。
4. **观察与调参**：
   - 检查指数→线性切换点是否在 EMA-PSNR ≈ 21.0 dB。
   - 若饱和过早（高斯数远小于原版，PSNR 偏低），减小 `sat_delta` 或增大 `sat_window`。
   - 若饱和过晚（继续投放但 PSNR 几乎不变），增大 `sat_delta`。
   - 记录最终高斯体数量及对应的 PSNR。
5. **迭代至 v0.2/v0.3**：在确认慢启动有效后，启用 `DensityAwareSampler` 或 `GradientGuidedSampler`，对比 PSNR-高斯数量曲线。

## 附录：新增超参数列表

在 `arguments/__init__.py` 中追加以下参数：

```python
# Packet densification
parser.add_argument('--packet_size', type=int, default=1000)
parser.add_argument('--init_packets', type=int, default=1)
parser.add_argument('--packet_radius_scale', type=float, default=0.02)  # 相对场景对角线
parser.add_argument('--psnr_threshold', type=float, default=21.0)
parser.add_argument('--ema_alpha', type=float, default=0.3)
parser.add_argument('--sat_window', type=int, default=3)
parser.add_argument('--sat_delta', type=float, default=0.3)
# 采样器选择
parser.add_argument('--position_sampler', type=str, default='uniform',
                    choices=['uniform', 'gradient_guided'])
```

通过配置文件或命令行传入，确保实验可复现。

## 7. 关键实现细节与代码扩展

### 7.1 Packet 类实现
```python
# scene/densification/packet.py
import torch

class GaussianPacket:
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
            if isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
        return self
```

### 7.2 包初始化器详细设计
`RandomInitializer` 需利用场景边界（从训练相机中估算或由原版代码中的 `scene.bounding_box` 提供）：

```python
class RandomInitializer(PacketInitializer):
    def __init__(self, bbox_min, bbox_max, packet_radius_scale=0.02):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.diagonal = torch.norm(bbox_max - bbox_min)
        self.radius = packet_radius_scale * self.diagonal

    def initialize(self, n: int, positions: torch.Tensor) -> GaussianPacket:
        # positions: (n, 3) 包中心，由 sampler 返回
        # 为每个高斯生成相对于中心的随机偏移
        offsets = torch.randn(n, 3, device=positions.device) * self.radius
        xyz = positions + offsets
        # 裁剪到场景边界内
        xyz = torch.max(torch.min(xyz, self.bbox_max), self.bbox_min)

        # 其他属性沿用原版初始化策略（例如基于点云平均值）
        # 这里简化：颜色使用随机值或固定灰色，尺度统一较小值等
        features_dc = torch.zeros(n, 1, 3)
        features_rest = torch.zeros(n, 15, 3)  # 假设最高阶数
        scaling = torch.ones(n, 3) * 0.01
        rotation = torch.zeros(n, 4); rotation[:, 0] = 1.0
        opacity = torch.ones(n, 1) * 0.1

        return GaussianPacket(xyz, features_dc, features_rest, scaling, rotation, opacity)
```

后期可通过 `SfMPerturbInitializer` 继承，用 SfM 点云作为中心，添加噪声。

### 7.3 慢启动调度器逻辑细节
```python
class GrowthScheduler:
    def __init__(self, psnr_threshold=21.0, ema_alpha=0.3, sat_window=3, sat_delta=0.3):
        self.psnr_threshold = psnr_threshold
        self.ema_alpha = ema_alpha
        self.sat_window = sat_window
        self.sat_delta = sat_delta
        self.phase = "exponential"  # exponential | linear | saturated
        self.ema_psnr = None
        self.deploy_count = 0       # 累计投放次数（用于指数增长）
        self.sat_counter = 0        # 连续低增益计数
        self.last_ema_psnr = None

    def step(self, current_psnr: float) -> tuple:
        # 更新 EMA
        if self.ema_psnr is None:
            self.ema_psnr = current_psnr
        else:
            self.ema_psnr = self.ema_alpha * current_psnr + (1 - self.ema_alpha) * self.ema_psnr

        # 阶段切换逻辑
        if self.phase == "exponential" and self.ema_psnr >= self.psnr_threshold:
            self.phase = "linear"
            # 重置线性阶段相关计数
            self.deploy_count = 0
            self.sat_counter = 0
        elif self.phase == "exponential":
            # 指数阶段：投放大小随累计投放次数指数增长
            packet_size = base_packet_size * (2 ** self.deploy_count)  # base_packet_size=1000
            self.deploy_count += 1
            return packet_size, True
        elif self.phase == "linear":
            # 线性阶段：固定 packet_size，每次触发投放
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
            # 线性增长：可以采用固定增量或恒定大小，这里假设恒定大小（效率最高）
            return base_packet_size, True
        else:  # saturated
            return 0, False
```

**注意**：指数阶段 `packet_size` 可能变得过大。需要设置上限，如不超过 `max_packet_size`（比如 5000），防止显存爆增。可在调度器中加入 `max_packet_size` 参数并限制。

### 7.4 位置采样器梯度引导实现概要
`GradientGuidedSampler` 将在 v0.3 启用，关键实现步骤：
1. 获取 `gaussian_model.xyz_gradient_accum`（需确保梯度已累积并归一化）。
2. 若梯度积累为空（首次投放），退化为均匀随机。
3. 将梯度作为权重，使用 `torch.multinomial` 抽取高斯体索引，以这些高斯体所在位置为中心生成新包中心（可添加随机偏移）。
4. 返回包中心坐标 `(num_centers, 3)`，每个中心对应一个 packet，内部高斯再由 initializer 生成偏移。

### 7.5 与原版代码的具体整合点
在 `train.py` 中，找到原 `densify_and_prune` 调用处（通常在 `training` 循环中，每隔 `densification_interval` 次调用）：

```python
# 原版：
# gaussians.densify_and_prune(max_grad, min_opacity, extent, max_screen_size)

# 修改为：
if iteration % opt.densification_interval == 0 and iteration > 0:
    # 计算当前 EMA_PSNR (使用最近评估的 PSNR)
    current_psnr = last_eval_psnr  # 由测试循环更新
    packet_densifier.densify(gaussians, current_psnr, iteration)
    # 保留原始 prune 逻辑（或独立调用）
    gaussians.prune(min_opacity, extent, max_screen_size)
```

需确保 `gaussians` 模型提供 `add_gaussians(packet)` 方法，将 packet 内的所有张量拼接到现有参数中，并更新优化器状态。原版 `densification.py` 中的 `densify_and_split` 等函数可参考其拼接方式。

### 7.6 TensorBoard 增强建议
- 使用 `add_scalars` 将 `phase` 的离散状态映射为整数值以便可视化（例如 0:exponential, 1:linear, 2:saturated）。
- 在 `eval/psnr` 图表中，通过叠加 `densify/total_gaussians` 作为横坐标（用自定义标量布局，或使用 `add_scalar` 并离线绘图）。TensorBoard 原生不支持双横轴，建议训练后使用 `matplotlib` 绘制 PSNR-高斯数量曲线，或记录 `custom_scalars` 布局。
- 添加高斯体密度热力图：定期保存所有高斯中心点云，利用 `open3d` 或 `plotly` 渲染俯视密度图，作为图像写入 TensorBoard (`add_image`)。

## 8. 潜在问题与应对策略

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 指数阶段 packet 过大导致 OOM | `2^deploy_count * 1000` 无上限 | 设置 `max_packet_size=8000`，或改为线性增长早期就切换 |
| EMA-PSNR 长时间不达标，一直指数增长 | 场景复杂，随机投放难以达到 21.0dB | 调整 `psnr_threshold` 降低至 18.0 dB，或增加初始包数 |
| 线性阶段过早饱和 | 增量阈值过严或窗口过小 | 增大 `sat_window` 至 5，减小 `sat_delta` 至 0.02 |
| 梯度引导采样器无梯度数据 | 首次投放或刚刚剪枝后梯度积累为空 | 检测并回退到均匀采样 |
| 包内高斯分布范围过大导致模糊 | `packet_radius_scale` 太大 | 默认值 0.02 通常合适，可启动时用更小值 |
| 与 prune 的交互导致高斯数反复波动 | prune 删除无用高斯后 PSNR 波动，干扰饱和判定 | 考虑在 prune 后延迟一次 densify 检测，或使用更平滑的 EMA（α=0.1） |

## 9. 后续实验扩展路线
当 v0.1 验证完成后，按以下顺序扩展：
1. **v0.2 密度感知**：实现 `DensityAwareSampler`，计算局部高斯密度（如一定半径内邻居数量），在高密度区域降低采样概率，低密度区域提高。
2. **v0.3 梯度引导**：启用 `GradientGuidedSampler`，结合梯度积累，分析是否用更少的 packet 达到同等 PSNR。
3. **v0.4 包级压缩**：引入包整体属性（如重要性分数），允许删除或合并整个 packet，进一步降低高斯数。
4. **跨场景泛化**：在 Mip-NeRF 360、T&T 等数据集上测试最佳策略，并调整超参数。
