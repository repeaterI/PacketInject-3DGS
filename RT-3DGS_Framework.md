# RT-3DGS（收敛版核心方法重构）
## Gaussian Packet Topology System based on Deformation-driven and Multi-view Epipolar Consensus

---

# 摘要 | Abstract

本文档重构 3D Gaussian Splatting 的核心范式，将其从“高斯点独立优化问题”升级为“高斯包拓扑系统优化问题”。

**核心定位**：Gaussian Field Compression under Multi-view Geometric Consensus

**两个核心机制**：

1. **Spawn（投放机制）**："形变驱动 + 多视角共识" 的 AND 门机制
2. **Drop（丢包机制）**："几何-语义冗余竞争" 的拓扑压缩机制

**最终优化目标**：

$$
\min \mathcal{L}_{render} + \lambda \cdot |P|
$$

$$
s.t. \quad \mathcal{L}_{view-consistency} < \epsilon
$$

---

# 第一部分：基础理论框架

## 1.1 核心范式转变

### 传统 3DGS 的问题

| 传统方法 | 问题本质 | 导致的结果 |
|---------|---------|-----------|
| 高误差区域分裂 | 局部启发式响应 | 高斯数量指数增长 |
| 瞬时梯度驱动 | 瞬时响应式决策 | 拓扑震荡（Topology Thrashing） |
| 独立高斯优化 | 非协调优化 | 多视图不一致 |

### RT-3DGS 的核心思想

> **将 3DGS 重构为受几何应力、路由拓扑与多视图共识共同约束的动态稀疏拓扑系统。**

关键转变：

- **从点到包**：高斯不再独立存在，而是组织为 Gaussian Packet
- **从瞬时到共识**：拓扑增长必须经过多视角验证
- **从删除到竞争**：丢包是基于冗余度的能量竞争，而非简单删除

---

## 1.2 Gaussian Packet（高斯包）形式化定义

### 1.2.1 Packet 结构

一个 Packet 是局部高斯活动域的基本组织单元：

$$
P_k = \{g_i\}_{i=1}^{n_k}
$$

其中 $n_k$ 是第 $k$ 个 Packet 包含的高斯数量。

### 1.2.2 Packet 的组成

每个 Packet 由两部分组成：

$$
P_k = (H_{P_k}, \mathcal{P}_{P_k})
$$

#### Header（包头）$H_{P_k}$

$$
H_{P_k} = [\mu_k, \Sigma_k, F_k, TTL_k, RouteHistory_k]
$$

| 字段 | 符号 | 定义 |
|------|------|------|
| 几何中心 | $\mu_k$ | Packet 内所有高斯位置的加权平均 |
| 协方差包络 | $\Sigma_k$ | Packet 内高斯协方差的聚合 |
| 特征集合 | $F_k$ | DINOv2 特征聚合：$F_k = \frac{1}{n_k}\sum_i \phi(g_i)$ |
| TTL 状态 | $TTL_k \in \mathbb{R}^2$ | $[TTL_a, TTL_s]$：活跃寿命 + 结构寿命 |
| 路由历史 | $RouteHistory_k$ | 该 Packet 经过的视角集合 |

#### Payload（包载荷）$\mathcal{P}_{P_k}$

$$
\mathcal{P}_{P_k} = [x_i, \Sigma_i, SH_i, \alpha_i, \Delta x_i, \Delta h_i]_{i=1}^{n_k}
$$

| 字段 | 符号 | 定义 |
|------|------|------|
| 中心位置 | $x_i$ | 第 $i$ 个高斯的3D位置 |
| 协方差矩阵 | $\Sigma_i$ | 第 $i$ 个高斯的协方差（形状控制） |
| 球谐系数 | $SH_i$ | 颜色/视角相关外观 |
| 不透明度 | $\alpha_i$ | 透射率控制 |
| 几何偏移 | $\Delta x_i$ | 局部几何微调 |
| 高频残差 | $\Delta h_i$ | 高频细节修复 |

---

# 第二部分：机制一 —— Spawn（投放机制）

## 核心思想

> **Spawn = "结构拉伸 + 多视角几何共识" 的 AND 门机制**
>
> - 形变（Deformation）只是 **Proposal Generator（提案生成器）**
> - 多视角共识（Multi-view Consensus）才是真正的 **Spawn Trigger（投放触发器）**

---

## 2.1 长短轴形变 = 局部结构缺陷检测器

### 2.1.1 协方差矩阵特征分解

对 Packet 内所有高斯的协方差进行聚合分析：

$$
\Sigma_k = U \Lambda U^T
$$

其中：
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \lambda_3)$，特征值满足 $\lambda_1 \geq \lambda_2 \geq \lambda_3$
- $U = [\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]$，$\mathbf{v}_i$ 是对应的特征向量

### 2.1.2 形变比定义

定义形变比（Dimensionality Ratio）：

$$
\rho_k = \frac{\lambda_{min}}{\lambda_{max}} = \frac{\lambda_3}{\lambda_1}
$$

| $\rho_k$ 值域 | 物理含义 | 本质解释 |
|--------------|---------|---------|
| $\rho_k \approx 1$ | 近似球形 | 梯度各向同性，无明显缺失方向 |
| $\rho_k \ll 1$ | 极度拉长 | 在某方向“被拉伸” = 该方向梯度缺失 |

### 2.1.3 形变的本质含义

> **$\rho_k \uparrow$ 的本质 = 梯度缺失方向**

当 Packet 在某方向被显著拉长时：

- 长轴方向 $\mathbf{v}_1$：低梯度区域，几何一致性好
- 短轴方向 $\mathbf{v}_3$：**高梯度区域，需要细化**

```
视觉理解：
        长轴 (v₁)
          ↑
          ‖
          ‖  ← 低梯度区域
          ‖
    ──────●──────→ 短轴 (v₃)
          ‖          ↑
          ‖          高梯度区域
          ‖          (需要 Spawn)
          ‖
```

---

## 2.2 关键创新：形变只是 Proposal Generator

### 2.2.1 为什么形变不能直接触发 Spawn？

如果仅依赖形变比 $\rho_k > \tau_r$ 就触发 Spawn，会导致：

| 问题 | 原因 | 结果 |
|------|------|------|
| 伪影放大 | 单视角瞬时形变可能是噪声 | 在错误位置生成高斯 |
| 过拟合局部 | 局部最优解被强化 | 泛化能力下降 |
| 拓扑不稳定 | 形变方向随训练波动 | 拓扑震荡 |

### 2.2.2 Proposal vs Trigger 的区别

| 概念 | 定义 | 在 Spawn 中的角色 |
|------|------|-----------------|
| **Proposal Generator** | 检测“可能需要改进的区域” | 形变比 $\rho_k > \tau_r$ |
| **Spawn Trigger** | 确认“确实需要在该位置生成” | 多视角极线共识验证 |

```
┌─────────────────────────────────────────────────────────┐
│                    Spawn 决策流程                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Proposal Generation（提案生成）                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  形变比检测: ρ_k = λ_min/λ_max                   │   │
│  │  IF ρ_k > τ_r:                                   │   │
│  │      生成候选方向 d_k = U_max（短轴方向）         │   │
│  │  END                                             │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ⚠️ 注意：此时只是"可能需要"，不是"确定要"               │
│                         ↓                               │
│  Step 2: Multi-view Consensus Verification（共识验证）  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  对相邻视角 v_j 执行极线投影验证                 │   │
│  │  统计满足条件的视角数量                          │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Step 3: AND Gate Decision（与门决策）                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  IF (形变条件 AND 共识条件):                     │   │
│  │      EXECUTE Spawn                               │   │
│  │  ELSE:                                          │   │
│  │      DO NOT Spawn                                │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2.3 多视角极线共识验证（核心创新点）

### 2.3.1 问题定义

对于候选方向 $\mathbf{d}_k = \mathbf{v}_3$（短轴方向，即梯度缺失方向），我们需要验证：

> **该方向的结构缺陷是否在多个视角中一致存在？**

### 2.3.2 Step 1：当前视角缺陷检测

首先在当前视角 $v_i$ 检测 Packet $P_k$ 是否存在梯度缺失：

$$
\|\nabla \mathcal{L}_{v_i}(P_k)\| > \tau
$$

其中：
- $\nabla \mathcal{L}_{v_i}(P_k)$：视角 $v_i$ 下 Packet $P_k$ 的渲染损失梯度
- $\tau$：梯度阈值（推荐值：0.01）

**含义**：梯度大 = 该区域重建误差大 = 存在结构缺陷

### 2.3.3 Step 2：极线传播验证（关键）

对每个相邻视角 $v_j$，执行极线投影验证：

#### 极线投影计算

将当前视角 $v_i$ 中检测到的缺陷点 $\mathbf{x}_k$ 投影到相邻视角 $v_j$ 的极线上：

$$
\mathbf{x}_k \xrightarrow{\text{epipolar}(v_j)} \mathbf{l}_{kj}
$$

其中 $\mathbf{l}_{kj}$ 是从 $\mathbf{x}_k$ 到视角 $v_j$ 的极线。

#### 极线上的梯度验证

在相邻视角 $v_j$ 的极线 $\mathbf{l}_{kj}$ 上，检查是否存在一致的梯度缺失：

$$
I(\|\nabla \mathcal{L}_{v_j}(P_k)\| > \tau)
= 
\begin{cases}
1 & \text{if } \|\nabla \mathcal{L}_{v_j}(P_k)\| > \tau \\
0 & \text{otherwise}
\end{cases}
$$

**关键洞察**：
- 如果极线上在多个视角都检测到大梯度
- 说明这不是单视角的局部噪声
- 而是真实的3D结构缺陷

```
极线几何验证示意图：

    视角 v_i                      视角 v_j
       📷                           📷
         \                         /
          \                       /
           \                     /
            \                   /
             \                 /
              \               /
               \             /
                \           /
                 \         /
                  \       /
                   \     /
                    \   /
                     \ /
                      ● ← 3D 缺陷点 (x_k)
                      |
                      | ← 极线 l_kj
                      |
                      |
```

### 2.3.4 Step 3：多视角统计

统计所有相邻视角中满足条件的数量：

$$
\text{ConsensusCount} = \sum_{j \in \mathcal{N}(i)} I(\|\nabla \mathcal{L}_{v_j}(P_k)\| > \tau)
$$

其中 $\mathcal{N}(i)$ 是视角 $v_i$ 的邻居视角集合。

---

## 2.4 Spawn 条件（AND 门核心）

### 2.4.1 形式化定义

**Spawn 触发当且仅当以下两个条件同时满足**：

$$
\underbrace{\rho_k > \tau_r}_{\text{形变条件：结构被拉伸}} 
\wedge 
\underbrace{\sum_{j} I(\|\nabla \mathcal{L}_{v_j}\| > \tau) \geq K}_{\text{共识条件：多视角验证通过}}
$$

| 条件 | 符号 | 含义 | 推荐阈值 |
|------|------|------|---------|
| 形变条件 | $\rho_k > \tau_r$ | Packet 在某方向被显著拉伸 | $\tau_r \in [0.3, 0.5]$ |
| 共识条件 | $\sum_j I(...) \geq K$ | 至少 K 个视角验证了该缺陷 | $K \in [2, 3]$ |

### 2.4.2 物理意义解释

```
┌────────────────────────────────────────────────────────────────┐
│                    Spawn AND Gate 决策表                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  形变条件(ρ>τᵣ)  │  共识条件(≥K)  │  是否 Spawn  │     原因      │
│  ─────────────────────────────────────────────────────────────│
│       ✓           │       ✓        │     YES     │ 真实结构缺陷  │
│       ✓           │       ✗        │     NO      │ 局部噪声/伪影  │
│       ✗           │       ✓        │     NO      │ 误检/阈值问题  │
│       ✗           │       ✗        │     NO      │ 无需改进      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.4.3 Spawn 执行操作

当 AND 门打开时，执行以下操作：

```python
def spawn_new_packet(P_k, d_k, view_j):
    """
    Spawn 新 Packet
    
    Args:
        P_k: 源 Packet（存在结构缺陷）
        d_k: 分裂方向（短轴方向 v_3）
        view_j: 共识验证通过的视角
    
    Returns:
        P_new: 新生成的 Packet
    """
    # 1. 在短轴方向偏移位置创建新 Packet
    offset = d_k * spawn_distance  # spawn_distance 与 λ_3 相关
    P_new_center = P_k.center + offset
    
    # 2. 新 Packet 继承源 Packet 的特征
    P_new.features = P_k.features  # DINOv2 特征继承
    P_new.TTL_a = TTL_init  # 初始化为活跃状态
    P_new.TTL_s = TTL_init
    
    # 3. 记录路由历史
    P_new.RouteHistory = P_k.RouteHistory + [view_j]
    
    # 4. 调整源 Packet（降低密度以维持总数）
    P_k.covariance = P_k.covariance * 0.9  # 收缩
    
    return P_new
```

---

## 2.5 Spawn 机制详细步骤总结

### 步骤清单

| 步骤 | 编号 | 名称 | 核心操作 | 输出 |
|------|------|------|---------|------|
| 1 | 2.5.1 | 协方差计算 | 对 Packet 内高斯计算 $\Sigma_k$ | $\Sigma_k$ |
| 2 | 2.5.2 | 特征分解 | $\Sigma_k = U\Lambda U^T$ | $\{\lambda_i\}, \{\mathbf{v}_i\}$ |
| 3 | 2.5.3 | 形变比计算 | $\rho_k = \lambda_{min}/\lambda_{max}$ | $\rho_k$ |
| 4 | 2.5.4 | 形变条件判断 | $\rho_k > \tau_r$? | True/False |
| 5 | 2.5.5 | 候选方向确定 | $\mathbf{d}_k = \mathbf{v}_3$ | $\mathbf{d}_k$ |
| 6 | 2.5.6 | 极线投影 | $\mathbf{x}_k \rightarrow \mathbf{l}_{kj}$ | 极线集合 |
| 7 | 2.5.7 | 梯度验证 | $I(\|\nabla \mathcal{L}_{v_j}\| > \tau)$ | 二值向量 |
| 8 | 2.5.8 | 共识计数 | $\sum_j I(...)$ | 计数 $C$ |
| 9 | 2.5.9 | 共识条件判断 | $C \geq K$? | True/False |
| 10 | 2.5.10 | AND 门决策 | 形变 AND 共识 | 执行/Skip |
| 11 | 2.5.11 | Spawn 执行 | 创建新 Packet | $P_{new}$ |

---

### 每一步详细说明

#### 步骤 2.5.1：协方差计算

**目标**：计算 Packet 内所有高斯的聚合协方差

**方法**：

$$
\Sigma_k = \frac{1}{n_k} \sum_{i=1}^{n_k} \Sigma_i^{(g)}
$$

其中 $\Sigma_i^{(g)}$ 是第 $i$ 个高斯的协方差矩阵。

**代码实现**：

```python
def compute_packet_covariance(packet):
    """
    计算 Packet 聚合协方差
    
    Args:
        packet: GaussianPacket 对象
    
    Returns:
        Sigma_k: 3x3 协方差矩阵 (numpy array)
    """
    n_gaussians = len(packet.gaussians)
    
    # 初始化
    Sigma_k = np.zeros((3, 3))
    
    # 加权平均
    total_weight = 0.0
    for g in packet.gaussians:
        weight = g.alpha  # 使用不透明度作为权重
        Sigma_k += weight * g.covariance
        total_weight += weight
    
    # 归一化
    if total_weight > 0:
        Sigma_k /= total_weight
    
    return Sigma_k
```

**注意事项**：
- 如果 $n_k = 1$（单个高斯），直接返回该高斯的协方差
- 权重使用不透明度 $\alpha$ 而非均匀权重，因为半透明高斯贡献较小

---

#### 步骤 2.5.2：特征分解

**目标**：将协方差矩阵分解为特征值和特征向量

**方法**：

```python
def eigendecomposition(Sigma_k):
    """
    协方差矩阵特征分解
    
    Args:
        Sigma_k: 3x3 协方差矩阵
    
    Returns:
        eigenvalues: 3个特征值（降序排列）
        eigenvectors: 3x3 特征向量矩阵（按特征值排序）
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_k)
    
    # numpy.linalg.eigh 返回升序排列，需要反转
    eigenvalues = eigenvalues[::-1]  # λ₁ ≥ λ₂ ≥ λ₃
    eigenvectors = eigenvectors[:, ::-1]  # 对应列
    
    return eigenvalues, eigenvectors
```

**输出解释**：

| 特征向量 | 对应特征值 | 几何含义 |
|---------|-----------|---------|
| $\mathbf{v}_1$ | $\lambda_1$（最大） | 高斯长轴方向 = 低梯度方向 |
| $\mathbf{v}_2$ | $\lambda_2$（中等） | 高斯中轴方向 = 中等梯度区域 |
| $\mathbf{v}_3$ | $\lambda_3$（最小） | 高斯短轴方向 = **高梯度方向** |

---

#### 步骤 2.5.3：形变比计算

**目标**：量化 Packet 的各向异性程度

**公式**：

$$
\rho_k = \frac{\lambda_{min}}{\lambda_{max}} = \frac{\lambda_3}{\lambda_1}
$$

**代码实现**：

```python
def compute_deformation_ratio(eigenvalues):
    """
    计算形变比
    
    Args:
        eigenvalues: [λ₁, λ₂, λ₃]，已降序排列
    
    Returns:
        rho: 形变比 ∈ (0, 1]
    """
    lambda_max = eigenvalues[0]  # λ₁
    lambda_min = eigenvalues[2]  # λ₃
    
    # 防止除零
    if lambda_max < 1e-10:
        return 1.0
    
    rho = lambda_min / lambda_max
    
    # 限制在合理范围
    return np.clip(rho, 0.0, 1.0)
```

**形变比解释**：

| $\rho_k$ 范围 | Packet 形状 | 物理含义 |
|--------------|------------|---------|
| $\rho_k \geq 0.7$ | 近似球形 | 各向同性，无需分裂 |
| $0.4 \leq \rho_k < 0.7$ | 轻微拉伸 | 中等各向异性，可能需要观察 |
| $0.2 \leq \rho_k < 0.4$ | 明显拉伸 | 显著各向异性，触发 proposal |
| $\rho_k < 0.2$ | 极度拉长 | 严重各向异性，高优先级 proposal |

---

#### 步骤 2.5.4：形变条件判断

**目标**：判断 Packet 是否满足形变条件（Proposal 生成条件）

**条件**：

$$
\text{DeformationCondition} = (\rho_k > \tau_r)
$$

其中 $\tau_r$ 是形变阈值，推荐值 $\tau_r \in [0.3, 0.5]$。

**代码实现**：

```python
def check_deformation_condition(rho_k, tau_r=0.4):
    """
    检查形变条件
    
    Args:
        rho_k: 形变比
        tau_r: 形变阈值
    
    Returns:
        is_proposal: 是否生成提案
        direction: 分裂方向（如果生成提案）
    """
    if rho_k > tau_r:
        return False, None  # 不满足形变条件
    
    # 满足形变条件，生成 Proposal
    return True, "generate_proposal"
```

**阈值选择依据**：

| 阈值 $\tau_r$ | 灵敏度 | 适用场景 |
|--------------|-------|---------|
| 0.3（严格） | 低 | 高质量要求，减少冗余生成 |
| 0.4（平衡） | 中 | 默认推荐，平衡质量与数量 |
| 0.5（宽松） | 高 | 快速收敛，可能产生更多高斯 |

---

#### 步骤 2.5.5：候选方向确定

**目标**：确定 Spawn 新高斯的方向

**规则**：

$$
\mathbf{d}_k = \mathbf{v}_3
$$

即沿短轴方向（特征值最小对应的特征向量）分裂。

**代码实现**：

```python
def get_spawn_direction(eigenvectors):
    """
    获取 Spawn 方向
    
    Args:
        eigenvectors: 特征向量矩阵，列对应特征值
    
    Returns:
        direction: 归一化的分裂方向向量
    """
    # 第三列是 λ₃ 对应的特征向量（短轴方向）
    short_axis = eigenvectors[:, 2]
    
    # 归一化
    direction = short_axis / (np.linalg.norm(short_axis) + 1e-10)
    
    return direction
```

**分裂距离确定**：

```python
def compute_spawn_distance(eigenvalues):
    """
    计算 Spawn 分裂距离
    
    Args:
        eigenvalues: [λ₁, λ₂, λ₃]
    
    Returns:
        distance: 分裂距离
    """
    # 分裂距离与短轴长度相关
    lambda_3 = eigenvalues[2]  # 最小特征值
    
    # 使用 √λ₃ 作为分裂距离的参考
    distance = np.sqrt(lambda_3) * spawn_scale_factor
    
    return distance
```

**分裂策略可视化**：

```
分裂方向示意图：

    长轴 v₁
       ↑
       ‖
       ‖        ┌─────────────────┐
       ‖        │                 │
       ‖        │      ●──┐       │  ← 源 Packet
       ‖        │      │  │       │    中心在 μₖ
       ‖        │      ↓  │       │
       ‖        │    ┌───┴──┐     │  沿短轴 v₃ 分裂
       ‖        │    │ μₖ  ●──────┼──→ ● P_new (新 Packet)
       ‖        │    └───┬──┘     │    中心在 μₖ + dₖ·dist
       ‖        │        │       │
       ‖        └────────┴───────┘
       ‖          
    ───┼──────────────────────────→ 短轴 v₃
       ‖            (分裂方向)
```

---

#### 步骤 2.5.6：极线投影

**目标**：将当前视角检测到的缺陷点投影到相邻视角的极线上

**极线几何基础**：

给定：
- 视角 $v_i$ 的相机中心 $\mathbf{C}_i$
- 视角 $v_j$ 的相机中心 $\mathbf{C}_j$
- 缺陷点 $\mathbf{x}_k$ 在 $v_i$ 的像平面投影 $\mathbf{p}_k$

极线 $\mathbf{l}_{kj}$ 是从 $\mathbf{C}_j$ 到 $\mathbf{p}_k$ 的极平面与 $v_j$ 像平面的交线。

**代码实现**：

```python
def compute_epipolar_projection(x_k, camera_i, camera_j):
    """
    计算极线投影
    
    Args:
        x_k: 3D 缺陷点位置 (numpy array, shape [3])
        camera_i: 当前视角相机参数 (内参 + 外参)
        camera_j: 目标视角相机参数
    
    Returns:
        epipolar_line: 在 camera_j 像平面上的极线参数
        is_valid: 投影是否有效
    """
    # 1. 构建基础矩阵 F（相机对极几何）
    F = compute_fundamental_matrix(camera_i, camera_j)
    
    # 2. 将 3D 点投影到 camera_i 的像平面
    p_i = project_to_image_plane(x_k, camera_i)
    
    # 3. 计算在 camera_j 上的极线
    # l_j = F * p_i（齐次坐标）
    p_i_homogeneous = np.append(p_i, 1.0)  # [x, y, 1]
    epipolar_line = F @ p_i_homogeneous
    
    # 4. 极线归一化
    line_norm = np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2)
    if line_norm < 1e-10:
        return None, False
    epipolar_line /= line_norm
    
    return epipolar_line, True


def compute_fundamental_matrix(camera_i, camera_j):
    """
    计算基础矩阵 F
    
    Args:
        camera_i, camera_j: 相机参数
    
    Returns:
        F: 3x3 基础矩阵
    """
    # 提取相机参数
    R_i, t_i = camera_i.R, camera_i.t
    R_j, t_j = camera_j.R, camera_j.t
    K_i = camera_i.K
    
    # 相对位姿
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i
    
    # 本质矩阵 E = [t_rel]_× @ R_rel
    skew_t = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    E = skew_t @ R_rel
    
    # 基础矩阵 F = K_j^{-T} @ E @ K_i^{-1}
    F = np.linalg.inv(camera_j.K.T) @ E @ np.linalg.inv(camera_i.K)
    
    return F
```

**极线投影可视化**：

```
极线几何示意图：

           视角 v_i                          视角 v_j
              📷                                 📷
               |                                 |
               |                                 |
               |        3D 场景                  |
               |           *                     |
               |          /                      |
               |         /                       |
               |        /                        |
               |       /                         |
               |      /                          |
               ↓     /                           ↓
            ┌──────●──────┐                  ┌──────●──────┐
            │      x_k     │  ←───────────────│      x_k     │ ← 极线上搜索
            │   (缺陷点)   │      极线 l_kj    │   (缺陷点)   │
            └─────────────┘                  └─────────────┘
              像平面 i                        像平面 j
              
    极线 l_kj 是像平面 j 上的一条直线（极线搜索在该线上进行）
```

---

#### 步骤 2.5.7：梯度验证

**目标**：在极线上验证是否存在一致的梯度缺失

**方法**：

在极线 $\mathbf{l}_{kj}$ 附近的邻域内，计算渲染损失的梯度范数：

$$
g_{kj} = \|\nabla \mathcal{L}_{v_j}(\mathbf{l}_{kj})\| = \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{l}_{kj}}\right\|
$$

然后判断：

$$
I(g_{kj} > \tau) = 
\begin{cases}
1 & \text{if } g_{kj} > \tau \\
0 & \text{otherwise}
\end{cases}
$$

**代码实现**：

```python
def verify_gradient_on_epipolar_line(epipolar_line, camera_j, packet, tau=0.01):
    """
    在极线上验证梯度
    
    Args:
        epipolar_line: 极线参数 [a, b, c]，ax + by + c = 0
        camera_j: 目标视角相机
        packet: 当前验证的 Packet
        tau: 梯度阈值
    
    Returns:
        verified: 是否通过验证
        gradient_norm: 实际梯度范数
    """
    # 1. 在极线附近采样多个点
    line_points = sample_points_on_line(epipolar_line, n_samples=10)
    
    # 2. 对每个采样点计算渲染损失梯度
    gradients = []
    for point_2d in line_points:
        # 反投影到 3D 射线
        ray_3d = backproject_to_3d(point_2d, camera_j)
        
        # 计算该射线方向的渲染损失
        loss = compute_rendering_loss_along_ray(ray_3d, packet)
        gradients.append(loss)
    
    # 3. 计算平均梯度
    gradient_norm = np.mean(np.abs(gradients))
    
    # 4. 与阈值比较
    verified = gradient_norm > tau
    
    return verified, gradient_norm


def sample_points_on_line(line_params, n_samples=10):
    """
    在极线上均匀采样点
    
    Args:
        line_params: [a, b, c]，ax + by + c = 0
        n_samples: 采样点数
    
    Returns:
        points: 采样点列表
    """
    a, b, c = line_params
    
    # 获取图像边界
    h, w = 480, 640  # 假设图像尺寸
    
    points = []
    
    if abs(b) > abs(a):
        # 水平方向变化为主
        x_range = np.linspace(0, w-1, n_samples)
        for x in x_range:
            y = (-a * x - c) / (b + 1e-10)
            if 0 <= y < h:
                points.append([x, y])
    else:
        # 垂直方向变化为主
        y_range = np.linspace(0, h-1, n_samples)
        for y in y_range:
            x = (-b * y - c) / (a + 1e-10)
            if 0 <= x < w:
                points.append([x, y])
    
    return points
```

**搜索邻域设置**：

```python
def verify_gradient_with_neighborhood(epipolar_line, camera_j, packet, 
                                      tau=0.01, search_radius=5.0):
    """
    在极线邻域内验证梯度
    
    Args:
        epipolar_line: 极线参数
        camera_j: 相机参数
        packet: 待验证 Packet
        tau: 梯度阈值
        search_radius: 搜索半径（像素）
    
    Returns:
        verified: 验证结果
        max_gradient: 最大梯度范数
    """
    # 获取极线附近区域内的所有像素
    nearby_pixels = get_nearby_pixels(epipolar_line, search_radius)
    
    # 计算每个像素的梯度
    gradients = []
    for pixel in nearby_pixels:
        ray = backproject_to_3d(pixel, camera_j)
        loss = compute_rendering_loss_along_ray(ray, packet)
        gradients.append(loss)
    
    max_gradient = np.max(np.abs(gradients))
    verified = max_gradient > tau
    
    return verified, max_gradient
```

---

#### 步骤 2.5.8：共识计数

**目标**：统计所有相邻视角中验证通过的数量

**公式**：

$$
C = \sum_{j \in \mathcal{N}(i)} I(\|\nabla \mathcal{L}_{v_j}(P_k)\| > \tau)
$$

**代码实现**：

```python
def count_consensus_views(packet, source_view, neighboring_views, tau=0.01):
    """
    统计共识视角数量
    
    Args:
        packet: 待验证的 Packet
        source_view: 当前视角
        neighboring_views: 相邻视角列表
        tau: 梯度阈值
    
    Returns:
        consensus_count: 共识视角数量
        verified_views: 通过验证的视角列表
    """
    consensus_count = 0
    verified_views = []
    
    # 获取当前视角中 Packet 的位置
    packet_center = packet.center
    
    for view_j in neighboring_views:
        # 跳过自身
        if view_j.id == source_view.id:
            continue
        
        # 计算极线投影
        epipolar_line, is_valid = compute_epipolar_projection(
            packet_center, 
            source_view.camera,
            view_j.camera
        )
        
        if not is_valid:
            continue
        
        # 在极线上验证梯度
        verified, gradient_norm = verify_gradient_on_epipolar_line(
            epipolar_line, 
            view_j.camera, 
            packet, 
            tau
        )
        
        if verified:
            consensus_count += 1
            verified_views.append({
                'view_id': view_j.id,
                'gradient_norm': gradient_norm
            })
    
    return consensus_count, verified_views
```

**邻居视角选择**：

```python
def get_neighboring_views(source_view, all_views, max_neighbors=5):
    """
    获取相邻视角
    
    Args:
        source_view: 源视角
        all_views: 所有视角列表
        max_neighbors: 最大邻居数量
    
    Returns:
        neighbors: 相邻视角列表（按重叠率降序）
    """
    neighbors = []
    
    for view in all_views:
        if view.id == source_view.id:
            continue
        
        # 计算视角重叠率
        overlap = compute_view_overlap(source_view, view)
        
        if overlap > tau_overlap_min:  # 最小重叠阈值
            neighbors.append((view, overlap))
    
    # 按重叠率降序排列
    neighbors.sort(key=lambda x: x[1], reverse=True)
    
    return [v for v, _ in neighbors[:max_neighbors]]
```

---

#### 步骤 2.5.9：共识条件判断

**目标**：判断共识数量是否达到触发 Spawn 的阈值

**条件**：

$$
\text{ConsensusCondition} = (C \geq K)
$$

其中 $K$ 是触发 Spawn 所需的最少共识视角数，推荐值 $K \in [2, 3]$。

**代码实现**：

```python
def check_consensus_condition(consensus_count, K=2):
    """
    检查共识条件
    
    Args:
        consensus_count: 共识视角数量
        K: 所需最小共识数
    
    Returns:
        satisfied: 是否满足共识条件
    """
    return consensus_count >= K
```

**K 值选择的影响**：

| K 值 | 严格程度 | 优点 | 缺点 |
|------|---------|------|------|
| K=1 | 最低 | 收敛快，不遗漏 | 可能产生冗余高斯 |
| K=2 | 中等（推荐） | 平衡质量与数量 | 可能遗漏部分边缘情况 |
| K=3 | 最高 | 质量最高，冗余最少 | 可能遗漏有效结构缺陷 |

---

#### 步骤 2.5.10：AND 门决策

**目标**：综合形变条件和共识条件，做出最终 Spawn 决策

**AND 门逻辑**：

$$
\text{SpawnDecision} = \underbrace{\rho_k > \tau_r}_{\text{形变条件}} \wedge \underbrace{C \geq K}_{\text{共识条件}}
$$

**代码实现**：

```python
def spawn_and_gate(packet, source_view, all_views, config):
    """
    Spawn AND 门决策
    
    Args:
        packet: 待处理的 Packet
        source_view: 当前视角
        all_views: 所有视角
        config: 配置参数
    
    Returns:
        should_spawn: 是否执行 Spawn
        spawn_info: Spawn 详细信息（如果执行）
    """
    # ========== 步骤 1: 形变条件检查 ==========
    Sigma_k = compute_packet_covariance(packet)
    eigenvalues, eigenvectors = eigendecomposition(Sigma_k)
    rho_k = compute_deformation_ratio(eigenvalues)
    
    deformation_satisfied = (rho_k > config.tau_r)
    
    if not deformation_satisfied:
        return False, None
    
    # ========== 步骤 2: 共识条件检查 ==========
    neighboring_views = get_neighboring_views(
        source_view, 
        all_views, 
        max_neighbors=config.max_neighbors
    )
    
    consensus_count, verified_views = count_consensus_views(
        packet,
        source_view,
        neighboring_views,
        tau=config.tau_gradient
    )
    
    consensus_satisfied = (consensus_count >= config.K_consensus)
    
    # ========== 步骤 3: AND 门决策 ==========
    should_spawn = deformation_satisfied and consensus_satisfied
    
    spawn_info = {
        'packet_id': packet.id,
        'deformation_ratio': rho_k,
        'consensus_count': consensus_count,
        'verified_views': verified_views,
        'spawn_direction': eigenvectors[:, 2],  # 短轴方向
        'spawn_distance': np.sqrt(eigenvalues[2]) * config.spawn_scale
    }
    
    return should_spawn, spawn_info
```

**决策真值表**：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Spawn AND 门决策表                           │
├───────────────────────┬───────────────────────┬──────────────────┤
│  形变条件 (ρ > τᵣ)    │  共识条件 (C ≥ K)     │    Spawn 决策    │
├───────────────────────┼───────────────────────┼──────────────────┤
│         ✓            │          ✓            │     执行 Spawn    │
├───────────────────────┼───────────────────────┼──────────────────┤
│         ✓            │          ✗            │     不执行        │
│                       │                       │  (单视角伪影/噪声) │
├───────────────────────┼───────────────────────┼──────────────────┤
│         ✗            │          ✓            │     不执行        │
│                       │                       │  (形变不足)       │
├───────────────────────┼───────────────────────┼──────────────────┤
│         ✗            │          ✗            │     不执行        │
│                       │                       │  (无需改进)       │
└───────────────────────┴───────────────────────┴──────────────────┘
```

---

#### 步骤 2.5.11：Spawn 执行

**目标**：创建新的 Packet

**代码实现**：

```python
def execute_spawn(packet, spawn_direction, spawn_distance, 
                  verified_views, config):
    """
    执行 Spawn 操作
    
    Args:
        packet: 源 Packet
        spawn_direction: 分裂方向（短轴方向）
        spawn_distance: 分裂距离
        verified_views: 通过验证的视角列表
        config: 配置参数
    
    Returns:
        new_packet: 新创建的 Packet
    """
    # 1. 计算新 Packet 位置
    new_center = packet.center + spawn_direction * spawn_distance
    
    # 2. 创建新 Packet
    new_packet = GaussianPacket()
    new_packet.id = generate_unique_id()
    new_packet.center = new_center
    
    # 3. 继承源 Packet 的特征
    new_packet.features = copy.deepcopy(packet.features)
    
    # 4. 初始化 TTL
    new_packet.TTL_a = config.TTL_init
    new_packet.TTL_s = config.TTL_init
    
    # 5. 记录路由历史
    new_packet.RouteHistory = packet.RouteHistory + [v['view_id'] for v in verified_views]
    
    # 6. 创建初始高斯（从源 Packet 复制部分高斯）
    n_new_gaussians = max(1, len(packet.gaussians) // 2)
    for i in range(n_new_gaussians):
        g = Gaussian()
        g.center = new_center + np.random.randn(3) * spawn_distance * 0.1
        g.covariance = packet.covariance * 0.5  # 新高斯协方差缩小
        g.alpha = packet.gaussians[i].alpha
        g.SH = copy.deepcopy(packet.gaussians[i].SH)
        new_packet.gaussians.append(g)
    
    # 7. 调整源 Packet（收缩以维持平衡）
    packet.covariance = packet.covariance * config.source_shrink_factor
    packet.TTL_a *= config.TTL_decay_on_spawn  # Spawn 后源 Packet TTL 略微衰减
    
    return new_packet
```

**Spawn 后的拓扑变化**：

```
Spawn 操作示意图：

    Spawn 前：                     Spawn 后：
    
         P_k                          P_k ← 收缩
          ●                            ●
          │                          ╱
          │                        ╱
          ↓                       ↓
     (分裂方向)              (分裂方向)
          
                                 ● P_new  ← 新 Packet
                                 
    |P_k| = N                 |P_k| = N/2, |P_new| = N/2
```

---

## 2.6 Spawn 机制完整伪代码

```python
def spawn_mechanism(gaussians, views, config):
    """
    Spawn 机制完整流程
    
    Args:
        gaussians: 所有 GaussianPacket 列表
        views: 所有视角列表
        config: 配置参数
    
    Returns:
        new_packets: 新创建的 Packet 列表
    """
    new_packets = []
    
    for packet in gaussians:
        for view_i in views:
            # 检查当前视角是否可见该 Packet
            if not is_visible(packet, view_i):
                continue
            
            # ========== AND 门检查 ==========
            
            # 1. 形变条件
            Sigma_k = compute_packet_covariance(packet)
            eigenvalues, eigenvectors = eigendecomposition(Sigma_k)
            rho_k = compute_deformation_ratio(eigenvalues)
            deformation_ok = (rho_k > config.tau_r)
            
            if not deformation_ok:
                continue
            
            # 2. 获取邻居视角
            neighbors = get_neighboring_views(view_i, views, 
                                              max_neighbors=config.max_neighbors)
            
            # 3. 极线投影验证
            consensus_count = 0
            for view_j in neighbors:
                # 计算极线
                epipolar_line, valid = compute_epipolar_projection(
                    packet.center, view_i.camera, view_j.camera
                )
                if not valid:
                    continue
                
                # 梯度验证
                verified, _ = verify_gradient_on_epipolar_line(
                    epipolar_line, view_j.camera, packet, config.tau_gradient
                )
                
                if verified:
                    consensus_count += 1
            
            # 4. 共识条件
            consensus_ok = (consensus_count >= config.K_consensus)
            
            # 5. AND 门
            if deformation_ok and consensus_ok:
                # 执行 Spawn
                spawn_direction = eigenvectors[:, 2]  # 短轴方向
                spawn_distance = np.sqrt(eigenvalues[2]) * config.spawn_scale
                
                new_packet = execute_spawn(
                    packet, 
                    spawn_direction, 
                    spawn_distance,
                    consensus_count,
                    config
                )
                
                new_packets.append(new_packet)
    
    return new_packets
```

---

# 第三部分：机制二 —— Drop（丢包机制）

## 核心思想

> **Drop ≠ 删除**
>
> **Drop = 拓扑压缩中的"能量竞争机制"（Graph Compression over Gaussian Semantic-field）**

---

## 3.1 Packet 冗余定义

### 3.1.1 冗余的两种维度

冗余度由两个正交维度定义：

| 维度 | 类型 | 符号 | 定义 |
|------|------|------|------|
| 空间维度 | 几何冗余 | $D_{geo}$ | 位置重叠程度 |
| 语义维度 | 语义冗余 | $D_{sem}$ | 特征相似程度 |

### 3.1.2 几何冗余定义

两个 Packet 之间的几何冗余：

$$
D_{geo}(P_i, P_j) = \|\mu_i - \mu_j\|_2
$$

其中 $\mu_i, \mu_j$ 是两个 Packet 的中心位置。

**物理含义**：
- $D_{geo}$ 小 → 两个 Packet 位置接近 → 可能存在冗余
- $D_{geo}$ 大 → 两个 Packet 位置分离 → 各自独立

### 3.1.3 语义冗余定义（DINOv2 特征）

使用预训练的 DINOv2 模型提取语义特征：

$$
D_{sem}(P_i, P_j) = 1 - \cos(F_i, F_j) = 1 - \frac{F_i \cdot F_j}{\|F_i\| \|F_j\|}
$$

其中 $F_i, F_j$ 是两个 Packet 的 DINOv2 特征聚合：

$$
F_k = \frac{1}{n_k} \sum_{i=1}^{n_k} \phi(g_i)
$$

其中 $\phi(\cdot)$ 是 DINOv2 的特征提取函数。

**物理含义**：
- $D_{sem}$ 小 → 两个 Packet 外观相似 → 可能存在冗余
- $D_{sem}$ 大 → 两个 Packet 外观不同 → 各自独立

```
语义冗余可视化：

    P_i 的特征空间                    P_j 的特征空间
         F_i ───────────────────────── F_j
        
    夹角 θ 越小，cos(F_i, F_j) 越大，D_sem 越小
    → 语义冗余度越高
    
    P_i 的外观        P_j 的外观        D_sem
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  红色   │      │  红色   │      │   小    │  ← 高冗余
    │  墙面   │      │  墙面   │      │         │
    └─────────┘      └─────────┘      └─────────┘
    
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  红色   │      │  绿色   │      │   大    │  ← 低冗余
    │  墙面   │      │  树叶   │      │         │
    └─────────┘      └─────────┘      └─────────┘
```

### 3.1.4 综合冗余定义

将几何冗余和语义冗余加权组合：

$$
R(P_i, P_j) = \alpha D_{geo}(P_i, P_j) + \beta D_{sem}(P_i, P_j)
$$

| 参数 | 符号 | 推荐值 | 含义 |
|------|------|--------|------|
| 几何权重 | $\alpha$ | 0.6 | 位置重叠的相对重要性 |
| 语义权重 | $\beta$ | 0.4 | 外观相似的相对重要性 |

**推荐值设置理由**：
- 几何冗余权重稍高，因为空间重叠是冗余的直接证据
- 语义冗余权重适当降低，避免因外观相似但位置不同的 Packet 被误判

**冗余度解释表**：

| $R(P_i, P_j)$ 范围 | 冗余程度 | 处理策略 |
|-------------------|---------|---------|
| $R < 0.2$ | 极高冗余 | 强制 Merge |
| $0.2 \leq R < 0.4$ | 高冗余 | 进入竞争 |
| $0.4 \leq R < 0.6$ | 中等冗余 | 观察 |
| $R \geq 0.6$ | 低冗余 | 保持独立 |

---

## 3.2 丢包规则（能量竞争机制）

### 3.2.1 竞争触发条件

当两个 Packet 的综合冗余度低于阈值时：

$$
R(P_i, P_j) < \tau_r
$$

触发能量竞争机制。

其中 $\tau_r$ 是冗余阈值，推荐值 $\tau_r \in [0.3, 0.4]$。

### 3.2.2 TTL 竞争规则

进入竞争的 Packet 执行 TTL 衰减：

$$
TTL_k \leftarrow TTL_k - \alpha \cdot \text{DropRate} + \beta \cdot \text{ViewSupport}
$$

**参数说明**：

| 参数 | 符号 | 推荐值 | 含义 |
|------|------|--------|------|
| 丢包率系数 | $\alpha$ | 0.01 | 每次竞争失败，TTL 减少多少 |
| 视角支持系数 | $\beta$ | 0.05 | 每个支持视角，TTL 增加多少 |
| DropRate | - | 1.0 | 每次竞争的丢包压力 |

### 3.2.3 ViewSupport（视角支持度）

衡量 Packet 被多少视角验证：

$$
\text{ViewSupport}(P_k) = |\{v_j \in \mathcal{V} : P_k \in \text{Visible}(v_j)\}|
$$

**TTL 更新示例**：

```python
# 假设 Packet P_k 与 P_j 冗余，进入竞争
initial_TTL = 0.5
alpha = 0.01  # 丢包率系数
beta = 0.05   # 视角支持系数
view_support = 3  # 3个视角支持该 Packet

# TTL 更新
TTL_k = initial_TTL - alpha * 1.0 + beta * view_support
TTL_k = 0.5 - 0.01 + 0.15 = 0.64

# 视角支持让 TTL 增加了！
```

### 3.2.4 竞争保留规则

竞争结束后，保留 TTL 最高的 Packet：

$$
P^* = \arg\max_{P \in \{P_i, P_j\}} TTL(P)
$$

**关键洞察**：
- **被越多视角支持的 Packet，TTL 越高，越不容易被删除**
- **长期无视角支持的 Packet，TTL 衰减越快，越容易被删除**

---

## 3.3 丢包机制详细步骤

### 步骤清单

| 步骤 | 编号 | 名称 | 核心操作 | 输出 |
|------|------|------|---------|------|
| 1 | 3.3.1 | 候选对生成 | 遍历所有 Packet 对 | 候选对集合 |
| 2 | 3.3.2 | 几何冗余计算 | $D_{geo} = \|\mu_i - \mu_j\|_2$ | $D_{geo}$ |
| 3 | 3.3.3 | 语义冗余计算 | $D_{sem} = 1 - \cos(F_i, F_j)$ | $D_{sem}$ |
| 4 | 3.3.4 | 综合冗余计算 | $R = \alpha D_{geo} + \beta D_{sem}$ | $R$ |
| 5 | 3.3.5 | 冗余条件判断 | $R < \tau_r$? | True/False |
| 6 | 3.3.6 | ViewSupport 计算 | 统计支持视角数 | $N_{support}$ |
| 7 | 3.3.7 | TTL 衰减执行 | $TTL -= \alpha + \beta \cdot N_{support}$ | $TTL_{new}$ |
| 8 | 3.3.8 | 竞争保留 | 保留 TTL 最高者 | $P^*$ |

---

### 每一步详细说明

#### 步骤 3.3.1：候选对生成

**目标**：找出所有可能冗余的 Packet 对

**方法**：遍历所有 Packet 对，筛选出可能冗余的候选

```python
def generate_candidate_pairs(packets, config):
    """
    生成候选冗余 Packet 对
    
    Args:
        packets: 所有 Packet 列表
        config: 配置参数
    
    Returns:
        candidates: 候选对列表 [(P_i, P_j, initial_score), ...]
    """
    candidates = []
    n = len(packets)
    
    # 空间索引加速（使用 KD-Tree）
    positions = np.array([p.center for p in packets])
    kdtree = KDTree(positions)
    
    for i in range(n):
        # 快速近邻搜索（只搜索空间上接近的 Packet）
        neighbors = kdtree.query_ball_point(positions[i], r=config.search_radius)
        
        for j in neighbors:
            if i >= j:  # 避免重复
                continue
            
            # 初步评分（仅基于几何距离）
            dist = np.linalg.norm(packets[i].center - packets[j].center)
            
            if dist < config.distance_threshold:
                candidates.append((i, j, dist))
    
    # 按距离升序排列（近的优先处理）
    candidates.sort(key=lambda x: x[2])
    
    return candidates
```

**搜索半径设置**：

| 搜索半径 | 效果 | 适用场景 |
|---------|------|---------|
| 0.1m | 严格 | 高密度区域，减少误判 |
| 0.2m（推荐） | 平衡 | 默认场景 |
| 0.5m | 宽松 | 低密度区域，避免遗漏 |

---

#### 步骤 3.3.2：几何冗余计算

**目标**：计算两个 Packet 之间的几何距离

**公式**：

$$
D_{geo}(P_i, P_j) = \|\mu_i - \mu_j\|_2
$$

**代码实现**：

```python
def compute_geometric_redundancy(packet_i, packet_j):
    """
    计算几何冗余度
    
    Args:
        packet_i, packet_j: 两个 Packet
    
    Returns:
        D_geo: 几何冗余度（欧氏距离）
    """
    # 方法1：直接使用中心距离
    center_distance = np.linalg.norm(
        packet_i.center - packet_j.center
    )
    
    # 方法2：考虑 Packet 大小（加权距离）
    # 两个 Packet 的"有效半径"
    radius_i = np.sqrt(np.max(packet_i.covariance)) if hasattr(packet_i, 'covariance') else 0.1
    radius_j = np.sqrt(np.max(packet_j.covariance)) if hasattr(packet_j, 'covariance') else 0.1
    
    # 加权距离（考虑 Packet 大小）
    effective_distance = center_distance / (radius_i + radius_j + 1e-6)
    
    return center_distance, effective_distance
```

---

#### 步骤 3.3.3：语义冗余计算

**目标**：计算两个 Packet 之间的语义相似度

**公式**：

$$
D_{sem}(P_i, P_j) = 1 - \cos(F_i, F_j) = 1 - \frac{F_i \cdot F_j}{\|F_i\| \|F_j\|}
$$

**代码实现**：

```python
def compute_semantic_redundancy(packet_i, packet_j):
    """
    计算语义冗余度（DINOv2 特征）
    
    Args:
        packet_i, packet_j: 两个 Packet
    
    Returns:
        D_sem: 语义冗余度 ∈ [0, 2]
    """
    # 获取 DINOv2 特征
    F_i = packet_i.features  # 假设已提取
    F_j = packet_j.features
    
    # 归一化
    F_i_norm = F_i / (np.linalg.norm(F_i) + 1e-6)
    F_j_norm = F_j / (np.linalg.norm(F_j) + 1e-6)
    
    # 余弦相似度
    cosine_sim = np.dot(F_i_norm, F_j_norm)
    
    # 夹角余弦 → 冗余度
    D_sem = 1.0 - cosine_sim  # ∈ [0, 2]
    
    return D_sem
```

**DINOv2 特征提取流程**：

```python
def extract_dinov2_features(packet, backbone):
    """
    提取 DINOv2 特征
    
    Args:
        packet: GaussianPacket
        backbone: DINOv2 模型
    
    Returns:
        features: 聚合特征向量
    """
    # 1. 将 Packet 内的高斯渲染到虚拟视角
    virtual_view = render_packet_to_virtual_view(packet)
    
    # 2. 使用 DINOv2 提取特征
    with torch.no_grad():
        features = backbone(virtual_view)  # [H, W, D]
    
    # 3. 全局平均池化
    pooled_features = torch.mean(features, dim=[0, 1])
    
    return pooled_features.numpy()
```

---

#### 步骤 3.3.4：综合冗余计算

**目标**：融合几何冗余和语义冗余

**公式**：

$$
R(P_i, P_j) = \alpha D_{geo}^{norm} + \beta D_{sem}
$$

其中 $D_{geo}^{norm}$ 是归一化后的几何距离：

$$
D_{geo}^{norm} = \frac{D_{geo} - D_{geo}^{min}}{D_{geo}^{max} - D_{geo}^{min} + \epsilon}
$$

**代码实现**：

```python
def compute_comprehensive_redundancy(packet_i, packet_j, config):
    """
    计算综合冗余度
    
    Args:
        packet_i, packet_j: 两个 Packet
        config: 配置参数
    
    Returns:
        R: 综合冗余度
        details: 详细分解
    """
    # 1. 几何冗余
    D_geo, D_geo_effective = compute_geometric_redundancy(packet_i, packet_j)
    
    # 2. 语义冗余
    D_sem = compute_semantic_redundancy(packet_i, packet_j)
    
    # 3. 归一化几何距离（使用全局统计）
    D_geo_normalized = D_geo / (config.geo_normalization_scale + 1e-6)
    D_geo_normalized = np.clip(D_geo_normalized, 0, 1)
    
    # 4. 加权组合
    R = config.alpha * D_geo_normalized + config.beta * D_sem
    
    details = {
        'D_geo': D_geo,
        'D_geo_normalized': D_geo_normalized,
        'D_sem': D_sem,
        'alpha': config.alpha,
        'beta': config.beta,
        'R': R
    }
    
    return R, details
```

---

#### 步骤 3.3.5：冗余条件判断

**目标**：判断 Packet 对是否进入竞争流程

**条件**：

$$
\text{RedundantCondition} = (R(P_i, P_j) < \tau_r)
$$

**代码实现**：

```python
def check_redundancy_condition(R, tau_r=0.35):
    """
    检查冗余条件
    
    Args:
        R: 综合冗余度
        tau_r: 冗余阈值
    
    Returns:
        is_redundant: 是否冗余
        redundancy_level: 冗余等级
    """
    if R < 0.2:
        return True, 'high'
    elif R < tau_r:
        return True, 'medium'
    else:
        return False, 'low'
```

---

#### 步骤 3.3.6：ViewSupport 计算

**目标**：统计有多少视角支持（可见）该 Packet

**公式**：

$$
\text{ViewSupport}(P_k) = \sum_{v_j \in \mathcal{V}} \mathbb{1}[P_k \in \text{Visible}(v_j)]
$$

**代码实现**：

```python
def compute_view_support(packet, all_views):
    """
    计算视角支持度
    
    Args:
        packet: 待计算的 Packet
        all_views: 所有视角列表
    
    Returns:
        view_support: 支持视角数量
        supporting_views: 支持视角 ID 列表
    """
    view_support = 0
    supporting_views = []
    
    for view in all_views:
        # 检查 Packet 是否在该视角可见
        is_visible = check_packet_visibility(packet, view)
        
        if is_visible:
            # 检查该视角对 Packet 是否有贡献
            contribution = compute_packet_contribution(packet, view)
            
            if contribution > 0.01:  # 阈值
                view_support += 1
                supporting_views.append(view.id)
    
    return view_support, supporting_views


def check_packet_visibility(packet, view):
    """
    检查 Packet 是否在视角中可见
    
    Args:
        packet: GaussianPacket
        view: View 对象
    
    Returns:
        visible: 是否可见
    """
    # 1. 检查视锥体内
    if not is_in_frustum(packet.center, view.camera):
        return False
    
    # 2. 检查遮挡
    ray = view.camera.get_ray_to(packet.center)
    depth_at_point = query_depth_map(ray, view.depth_map)
    
    if depth_at_point < packet.depth - epsilon:
        return False  # 被遮挡
    
    return True
```

---

#### 步骤 3.3.7：TTL 衰减执行

**目标**：更新竞争失败者的 TTL

**公式**：

$$
TTL_k^{(new)} = TTL_k^{(old)} - \alpha \cdot \text{DropRate} + \beta \cdot \text{ViewSupport}
$$

**代码实现**：

```python
def update_ttl_on_competition(packet, view_support, config):
    """
    在竞争中进行 TTL 更新
    
    Args:
        packet: 待更新的 Packet
        view_support: 视角支持数
        config: 配置参数
    
    Returns:
        old_ttl: 更新前的 TTL
        new_ttl: 更新后的 TTL
        ttl_change: TTL 变化量
    """
    old_ttl = packet.TTL_a  # 活跃 TTL
    
    # TTL 更新
    delta_decay = config.alpha_drop * config.DropRate
    delta_support = config.beta_support * view_support
    
    new_ttl = old_ttl - delta_decay + delta_support
    
    # 限制在 [0, 1] 范围
    new_ttl = np.clip(new_ttl, 0.0, 1.0)
    
    packet.TTL_a = new_ttl
    ttl_change = new_ttl - old_ttl
    
    return old_ttl, new_ttl, ttl_change
```

**TTL 更新示例**：

```
TTL 更新示例表：

Packet    初始TTL   ViewSupport   衰减(α·DR)   增长(β·N)    新TTL
─────────────────────────────────────────────────────────────────
P₁        0.50      3             -0.01        +0.15       0.64    ↑ 增加！
P₂        0.40      1             -0.01        +0.05       0.44    ↑ 略增
P₃        0.30      0             -0.01        +0.00       0.29    ↓ 减少
P₄        0.10      0             -0.01        +0.00       0.09    ↓ 接近删除
```

---

#### 步骤 3.3.8：竞争保留

**目标**：在竞争的 Packet 中选择保留者

**规则**：

$$
P^* = \arg\max_{P \in \{P_i, P_j\}} TTL(P)
$$

**代码实现**：

```python
def resolve_competition(packet_i, packet_j, config):
    """
    解决竞争，保留获胜者
    
    Args:
        packet_i, packet_j: 竞争的 Packet 对
        config: 配置参数
    
    Returns:
        winner: 获胜的 Packet
        loser: 失败的 Packet（待删除）
        merge_info: 合并信息
    """
    # 1. 计算两个 Packet 的综合评分
    score_i = compute_packet_score(packet_i)
    score_j = compute_packet_score(packet_j)
    
    # 2. 选择获胜者
    if score_i >= score_j:
        winner, loser = packet_i, packet_j
    else:
        winner, loser = packet_j, packet_i
    
    # 3. 获胜者 TTL 略微提升（奖励）
    winner.TTL_a = min(1.0, winner.TTL_a + config.winner_bonus)
    
    # 4. 生成 Merge 信息
    merge_info = {
        'winner_id': winner.id,
        'loser_id': loser.id,
        'winner_score': max(score_i, score_j),
        'loser_score': min(score_i, score_j),
        'merged_features': (winner.features + loser.features) / 2
    }
    
    return winner, loser, merge_info


def compute_packet_score(packet):
    """
    计算 Packet 综合评分
    
    Args:
        packet: Packet 对象
    
    Returns:
        score: 综合评分
    """
    # TTL 评分
    ttl_score = packet.TTL_a
    
    # 结构 TTL 评分（保护稀有结构）
    ttl_structural_score = packet.TTL_s * 0.5
    
    # 路由历史深度评分
    route_depth_score = min(len(packet.RouteHistory) / 10.0, 1.0)
    
    # 协方差行列式（体积越小，精度越高）
    det_sigma = np.linalg.det(packet.covariance) if hasattr(packet, 'covariance') else 1.0
    quality_score = 1.0 / (det_sigma + 1e-6)
    quality_score = np.clip(quality_score, 0, 10) / 10.0
    
    # 综合评分
    score = (0.4 * ttl_score + 
             0.2 * ttl_structural_score + 
             0.2 * route_depth_score + 
             0.2 * quality_score)
    
    return score
```

---

## 3.4 Drop 机制完整伪代码

```python
def drop_mechanism(packets, views, config):
    """
    Drop 机制完整流程
    
    Args:
        packets: 所有 GaussianPacket 列表
        views: 所有视角列表
        config: 配置参数
    
    Returns:
        surviving_packets: 存活 Packet 列表
        deleted_packets: 删除的 Packet 列表
    """
    surviving_packets = list(packets)  # 复制
    deleted_packets = []
    
    # 1. 生成候选冗余对
    candidates = generate_candidate_pairs(packets, config)
    
    for i, j, dist in candidates:
        P_i = surviving_packets[i]
        P_j = surviving_packets[j]
        
        # 2. 计算综合冗余度
        R, details = compute_comprehensive_redundancy(P_i, P_j, config)
        
        # 3. 检查冗余条件
        is_redundant, level = check_redundancy_condition(R, config.tau_r)
        
        if not is_redundant:
            continue
        
        # 4. 计算 ViewSupport
        support_i, _ = compute_view_support(P_i, views)
        support_j, _ = compute_view_support(P_j, views)
        
        # 5. TTL 竞争更新
        old_i, new_i, _ = update_ttl_on_competition(P_i, support_i, config)
        old_j, new_j, _ = update_ttl_on_competition(P_j, support_j, config)
        
        # 6. 解决竞争
        winner, loser, merge_info = resolve_competition(P_i, P_j, config)
        
        # 7. 处理失败者
        if loser.TTL_a < config.ttl_death_threshold:
            # 从存活列表移除
            surviving_packets.remove(loser)
            deleted_packets.append(loser)
            
            # 获胜者继承失败者的部分信息
            winner.features = merge_info['merged_features']
            winner.RouteHistory = list(set(winner.RouteHistory + loser.RouteHistory))
    
    # 8. 清理 TTL 过低的 Packet
    surviving_packets = [p for p in surviving_packets if p.TTL_a > config.ttl_death_threshold]
    deleted_packets += [p for p in packets if p not in surviving_packets]
    
    return surviving_packets, deleted_packets
```

---

# 第四部分：机制三 —— 视角路由系统（Routing View System）

## 核心思想

> **视角 = 路由节点，Packet = 路由对象**
>
> **整个系统变成：$G_{views} = (V, E)$，其中 Edge 表示 Packet 转移关系**

---

## 4.1 视角图的构建

### 4.1.1 图结构定义

$$
G_{views} = (V, E)
$$

其中：
- $V = \{v_1, v_2, ..., v_n\}$：视角节点集合
- $E = \{(v_i, v_j) : \text{Packet 从 } v_i \text{ 转移至 } v_j\}$：Packet 转移边

### 4.1.2 视角节点的功能

每个视角 $v_i$ 作为路由节点，具有以下功能：

| 功能 | 描述 |
|------|------|
| **接收 Packets** | 从相邻视角接收转发来的 Packets |
| **生成 Packets** | 在本地 Spawn 新的 Packets |
| **转发 Packets** | 将本地 Packets 转发给相邻视角 |
| **丢弃 Packets** | 对低 TTL Packets 执行 Drop |

### 4.1.3 边权重的定义

边 $(v_i, v_j)$ 的权重 $w_{ij}$ 定义为：

$$
w_{ij} = \text{overlap}(v_i, v_j) \times \text{information\_gain}(v_j)
$$

其中：
- $\text{overlap}(v_i, v_j)$：视角重叠率
- $\text{information\_gain}(v_j)$：视角 $v_j$ 的信息增益

---

## 4.2 TTL 的本质：拓扑活跃度

### 4.2.1 TTL 不是"时间"，而是"活跃度"

传统 TTL（Time To Live）通常表示"剩余存活时间"。

在 RT-3DGS 中，TTL 的物理含义是：

> **TTL = 该 Packet 被多视角网络验证的强度**

### 4.2.2 TTL 更新机制

$$
TTL_k \leftarrow TTL_k - \alpha \cdot \text{DropRate} + \beta \cdot \text{ViewSupport}
$$

**解释**：
- **$- \alpha \cdot \text{DropRate}$**：被判定为冗余 → TTL 降低
- **$+ \beta \cdot \text{ViewSupport}$**：被越多视角支持 → TTL 上升

### 4.2.3 TTL 与拓扑的关系

```
TTL 与拓扑活跃度的关系：

高 TTL (TTL > 0.8)
    │
    │    ┌─────────────────────────────────────┐
    │    │  多个视角一致验证                    │
    │    │  Packet 位于拓扑核心                │
    │    │  贡献度高                          │
    │    └─────────────────────────────────────┘
    │
    │  ┌─────────────────────────────────────┐
    │  │  少数视角验证                        │
    │  │  Packet 位于拓扑边缘                │
    │  │  贡献度中等                          │
    │  └─────────────────────────────────────┘
    │
低 TTL (TTL < 0.2)
    │    ┌─────────────────────────────────────┐
    │    │  无视角验证（长期遮挡/冗余）        │
    │    │  Packet 处于拓扑边缘                │
    │    │  贡献度低，可能被删除              │
    │    └─────────────────────────────────────┘
    │
    └────────────────────────────────────────────→ TTL 值
```

---

# 第五部分：机制四 —— Recovery（恢复机制）

## 核心思想

> **Recovery = "低密度区域的自发再生机制"**

---

## 5.1 Recovery 的触发条件

Recovery 触发需要**同时**满足以下条件：

| 条件 | 描述 |
|------|------|
| **低密度检测** | 局部空间没有 overlapping packet |
| **残差累积** | 连续 $K$ 个视角检测到 residual error |

### 5.1.1 低密度检测

定义空间密度：

$$
\rho_{spatial}(\mathbf{x}) = \sum_{P_k} \mathbb{1}[\|\mathbf{x} - \mu_k\| < r]
$$

其中 $r$ 是检测半径（推荐值：0.3m）。

当 $\rho_{spatial}(\mathbf{x}) = 0$ 时，位置 $\mathbf{x}$ 处于低密度区域。

### 5.1.2 残差累积检测

定义视角 $v_j$ 在位置 $\mathbf{x}$ 的残差：

$$
r_j(\mathbf{x}) = \|\nabla \mathcal{L}_{v_j}(\mathbf{x})\|
$$

当连续 $K$ 个视角的残差都超过阈值时：

$$
\sum_{j=1}^{K} \mathbb{1}[r_j(\mathbf{x}) > \tau_{residual}] = K
$$

触发 Recovery。

---

## 5.2 Recovery 执行

### 5.2.1 TTL 恢复

当 Recovery 条件满足时：

$$
TTL_k \rightarrow TTL_k + \Delta_{recover}
$$

其中 $\Delta_{recover}$ 是恢复增量（推荐值：0.2）。

### 5.2.2 Packet 重激活

```python
def execute_recovery(low_density_point, views, config):
    """
    在低密度区域执行 Recovery
    
    Args:
        low_density_point: 低密度区域的位置
        views: 所有视角
        config: 配置参数
    
    Returns:
        recovered_packet: 恢复的 Packet
    """
    # 1. 检查是否已有 Packet（可能被误判为低密度）
    existing_packet = find_nearest_packet(low_density_point, all_packets)
    
    if existing_packet is not None:
        # 已有 Packet，直接提升 TTL
        existing_packet.TTL_a = min(1.0, existing_packet.TTL_a + config.delta_recover)
        return existing_packet
    
    # 2. 创建新 Packet
    new_packet = GaussianPacket()
    new_packet.center = low_density_point
    new_packet.TTL_a = config.delta_recover * 0.5  # 初始 TTL 较低
    new_packet.TTL_s = config.delta_recover  # 结构 TTL 较高（保护新生成）
    new_packet.RouteHistory = []
    
    return new_packet
```

---

# 第六部分：总体优化目标

## 6.1 最终优化目标

RT-3DGS 的最终优化目标是：

$$
\min \mathcal{L}_{render} + \lambda \cdot |P|
$$

约束条件：

$$
s.t. \quad \mathcal{L}_{view-consistency} < \epsilon
$$

其中：

| 符号 | 定义 | 含义 |
|------|------|------|
| $\mathcal{L}_{render}$ | 渲染损失 | 像素级重建误差 |
| $\lambda \cdot |P|$ | 稀疏性惩罚 | 鼓励使用最少的 Packet |
| $\mathcal{L}_{view-consistency}$ | 视图一致性损失 | 多视图几何一致性约束 |
| $\epsilon$ | 一致性阈值 | 允许的最大不一致度 |

---

## 6.2 各损失项详细定义

### 6.2.1 渲染损失

$$
\mathcal{L}_{render} = \underbrace{\frac{1}{N} \sum_p |I_{gt}(p) - I_{render}(p)|^2}_{\text{光度损失}} + \gamma \underbrace{D_{SSIM}(I_{gt}, I_{render})}_{\text{结构损失}}
$$

### 6.2.2 稀疏性惩罚

$$
\lambda \cdot |P| = \lambda \sum_{k=1}^{K} \mathbb{1}[TTL_a^{(k)} > \tau_{active}]
$$

即 $\lambda$ 乘以活跃 Packet 的数量。

### 6.2.3 视图一致性损失

$$
\mathcal{L}_{view-consistency} = \frac{1}{M} \sum_{m=1}^{M} \sum_{(i,j) \in \mathcal{P}_m} \|x_i^m - x_j^m\|^2
$$

其中 $\mathcal{P}_m$ 是视角 $m$ 观测到的 Packet 对集合。

---

## 6.3 收敛判定标准

| 指标 | 判定条件 | 说明 |
|------|---------|------|
| 损失平稳 | $\|L_{total}(t) - L_{total}(t-k)\| < \tau_{loss}$ | 连续 $k$ 轮损失变化小于阈值 |
| 拓扑稳定 | $\|Route(t) - Route(t-k)\|_F < \tau_{route}$ | 路由矩阵 Frobenius 范数变化小于阈值 |
| 数量稳态 | $\|G(t) - G(t-k)\| < \tau_{gaussians}$ | 高斯数量变化小于阈值 |

---

# 第七部分：完整训练流程

```python
def train_rt3dgs(scene_data, config):
    """
    RT-3DGS 完整训练流程
    
    Args:
        scene_data: 包含多视角图像和相机参数
        config: 超参数配置
    
    Returns:
        gaussian_packets: 训练后的高斯包集合
    """
    # ========== 初始化阶段 ==========
    
    # Step 1: 场景语义扫描
    blocks = semantic_scanning(scene_data.images)
    
    # Step 2: 计算每个 Block 的频率特征
    for block in blocks:
        block.frequency = compute_frequency(block)
    
    # Step 3: 非均匀高斯播种
    packets = []
    for block in blocks:
        n_gaussians = config.N_base + config.alpha * block.frequency
        packet = create_gaussian_packet(block, n_gaussians)
        packets.append(packet)
    
    # ========== 视角图构建 ==========
    
    # Step 4: 构建初始路由拓扑
    view_graph = build_view_graph(scene_data.views, config)
    
    # ========== 迭代训练 ==========
    
    for iteration in range(config.max_iterations):
        for view in scene_data.views:
            # 1. 前向传播
            rendered = render_packets(packets, view)
            
            # 2. 计算损失
            loss = compute_total_loss(rendered, view, packets)
            
            # 3. 反向传播
            loss.backward()
            
            # 4. 更新高斯参数
            update_gaussians(packets)
            
            # 5. Spawn 机制
            new_packets = spawn_mechanism(packets, scene_data.views, config)
            packets.extend(new_packets)
            
            # 6. Drop 机制
            packets, deleted = drop_mechanism(packets, scene_data.views, config)
            
            # 7. Recovery 机制
            low_density_points = detect_low_density_regions(packets, config)
            for point in low_density_points:
                recovered = execute_recovery(point, scene_data.views, config)
                if recovered not in packets:
                    packets.append(recovered)
            
            # 8. TTL 更新
            update_all_ttl(packets, scene_data.views, config)
            
            # 9. 周期性路由更新
            if iteration % config.route_update_freq == 0:
                view_graph = optimize_view_graph(view_graph, packets, config)
        
        # 10. 收敛检查
        if check_convergence(packets, config):
            break
    
    return packets
```

---

# 第八部分：超参数推荐值

## 核心超参数

| 参数 | 符号 | 推荐值 | 说明 |
|------|------|--------|------|
| **形变阈值** | $\tau_r$ | 0.3-0.5 | 触发 proposal 的形变比阈值 |
| **共识数量** | $K$ | 2-3 | 触发 Spawn 的最少验证视角数 |
| **梯度阈值** | $\tau$ | 0.01 | 梯度残差阈值 |
| **冗余阈值** | $\tau_r$ | 0.3-0.4 | 触发竞争的冗余度阈值 |
| **几何权重** | $\alpha$ | 0.6 | 冗余度计算中几何权重 |
| **语义权重** | $\beta$ | 0.4 | 冗余度计算中语义权重 |
| **丢包率系数** | $\alpha$ | 0.01 | TTL 衰减系数 |
| **视角支持系数** | $\beta$ | 0.05 | TTL 上升系数 |
| **恢复增量** | $\Delta_{recover}$ | 0.2 | Recovery 时 TTL 增量 |
| **稀疏权重** | $\lambda$ | 0.001 | 损失函数中的 Packet 数量权重 |
| **一致性阈值** | $\epsilon$ | 0.1 | 视图一致性约束阈值 |

## Spawn 专用参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| max_neighbors | 5 | 最大邻居视角数量 |
| spawn_scale | 1.0 | 分裂距离缩放因子 |
| source_shrink_factor | 0.9 | Spawn 后源 Packet 收缩比例 |
| TTL_decay_on_spawn | 0.95 | Spawn 后源 Packet TTL 衰减 |

## Drop 专用参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| search_radius | 0.2m | 冗余检测搜索半径 |
| ttl_death_threshold | 0.1 | 删除 TTL 阈值 |
| winner_bonus | 0.05 | 竞争获胜者 TTL 奖励 |

## TTL 专用参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| TTL_init | 0.5 | 初始 TTL 值 |
| TTL_a_decay | 0.98 | 活跃 TTL 每帧衰减率 |
| TTL_s_decay | 0.995 | 结构 TTL 每帧衰减率 |

---

# 附录 A：一句话论文定义

> We reformulate 3D Gaussian Splatting as a packetized topology compression problem, where Gaussian packets are dynamically spawned based on anisotropic deformation and multi-view epipolar consensus, and pruned via semantic-geometric redundancy under a network-inspired TTL propagation system, aiming to achieve minimal packet representation while preserving rendering fidelity.

---

# 附录 B：核心公式速查表

## Spawn 机制

| 公式 | 含义 |
|------|------|
| $\rho_k = \lambda_{min}/\lambda_{max}$ | 形变比 |
| $\rho_k > \tau_r$ | 形变条件（Proposal 生成） |
| $\sum_j I(\|\nabla \mathcal{L}_{v_j}\| > \tau) \geq K$ | 共识条件（Trigger 确认） |
| $\text{Spawn} = (\rho > \tau_r) \wedge (C \geq K)$ | AND 门决策 |

## Drop 机制

| 公式 | 含义 |
|------|------|
| $D_{geo} = \|\mu_i - \mu_j\|_2$ | 几何冗余 |
| $D_{sem} = 1 - \cos(F_i, F_j)$ | 语义冗余 |
| $R = \alpha D_{geo} + \beta D_{sem}$ | 综合冗余 |
| $TTL \leftarrow TTL - \alpha + \beta \cdot N_{support}$ | TTL 更新 |

## 优化目标

| 公式 | 含义 |
|------|------|
| $\min \mathcal{L}_{render} + \lambda \cdot |P|$ | 最终优化目标 |
| $s.t. \mathcal{L}_{view-consistency} < \epsilon$ | 约束条件 |

---

# 附录 C：检测点清单

## Spawn 机制检测点

- [ ] 协方差矩阵特征分解返回 $\mathbf{v}_1$（长轴）, $\mathbf{v}_3$（短轴）
- [ ] 形变比 $\rho \in (0, 1]$
- [ ] 极线投影计算正确
- [ ] 形变只是 Proposal，不直接触发 Spawn
- [ ] 单视角验证不触发 Spawn
- [ ] 形变 AND 共识同时满足才触发 Spawn

## Drop 机制检测点

- [ ] 几何冗余计算正确
- [ ] 语义冗余计算正确（DINOv2）
- [ ] TTL 衰减符合公式
- [ ] ViewSupport 高的 Packet TTL 上升
- [ ] 竞争保留 TTL 最高者

## Recovery 机制检测点

- [ ] 低密度区域检测正确
- [ ] 连续 $K$ 帧残差检测正确
- [ ] TTL 恢复值合理

## 优化目标检测点

- [ ] 损失函数包含渲染项和稀疏项
- [ ] 收敛条件判定正确
- [ ] 约束条件满足

---

# 附录 D：模块与文件对应关系

| 模块 | 文件 | 核心功能 |
|------|------|---------|
| PacketFoundation | `packet_definition.py` | Packet 数据结构定义 |
| PacketFoundation | `covariance_analysis.py` | 协方差矩阵特征分解 |
| PacketFoundation | `deformation_detection.py` | 形变比计算 |
| PacketFoundation | `feature_aggregation.py` | DINOv2 特征聚合 |
| SpawnMechanism | `anisotropy_proposal.py` | 基于形变的 Proposal 生成 |
| SpawnMechanism | `epipolar_projection.py` | 极线投影计算 |
| SpawnMechanism | `consensus_verification.py` | 多视角共识验证 |
| SpawnMechanism | `spawn_decision.py` | AND 门决策逻辑 |
| DropMechanism | `redundancy_definition.py` | 几何+语义冗余度计算 |
| DropMechanism | `competition_mechanism.py` | TTL 竞争与衰减 |
| DropMechanism | `topology_compression.py` | 拓扑压缩执行 |
| RoutingView | `view_graph_construction.py` | 视角图构建 |
| RoutingView | `ttl_propagation.py` | TTL 传播规则 |
| RoutingView | `packet_transfer.py` | Packet 转移逻辑 |
| RecoveryMechanism | `recovery_trigger.py` | 恢复触发条件检测 |
| RecoveryMechanism | `reactivation.py` | TTL 恢复与重激活 |
| OptimizationTarget | `loss_function.py` | 渲染损失+稀疏约束 |
| OptimizationTarget | `convergence_analysis.py` | 收敛判定标准 |

---

**文档版本**：v2.0（收敛版核心方法重构）

**核心变更**：
1. 明确区分 Proposal Generator（形变）与 Spawn Trigger（共识）
2. 详细定义多视角极线共识验证流程
3. 将 Drop 重新定位为"拓扑压缩中的能量竞争机制"
4. 完善 TTL 的"拓扑活跃度"物理含义
5. 增加每一步的详细代码实现与检测点
