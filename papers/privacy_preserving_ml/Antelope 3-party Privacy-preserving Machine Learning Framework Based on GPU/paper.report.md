# Paper Report: Antelope — 3-party Privacy-preserving Machine Learning Framework Based on GPU

**Authors**: Yu Huan (余欢), Hua Qiang-Sheng (华强胜), Lu Bi-Ran (卢必然), Shi Xuan-Hua (石宣化), Jin Hai (金海)  
**Affiliation**: Huazhong University of Science and Technology (HUST), Wuhan, China  
**Journal**: Journal of Software (软件学报), 2025  
**URL**: http://www.jos.org.cn/1000-9825/7445.htm  
**Analysis Date**: 2026-05-11

---

## 1. 论文定位与核心贡献摘要

Antelope 是一个基于 GPU 的**三方隐私保护机器学习（PPML）框架**，专注于降低安全多方计算（MPC）在神经网络训练/推理中的性能开销。论文的核心贡献有三点：

1. **64 位整数矩阵乘法/卷积的自定义 CUDA 算子**：规避了 cuBLAS 不支持 64 位整数的瓶颈，基于分块 SGEMM 思想实现，减少内存访问延迟。
2. **O(1) 通信轮数的 MSB 获取协议**：基于 0-1 编码方法，将最高符号位（MSB）提取的通信轮数从 O(log l) 降低至 O(1)（不需要离线预处理）。
3. **MixMul 混合乘法协议**：特殊布尔秘密分享与算术秘密分享数据混合相乘的协议，简化了 ReLU 安全比较的通信开销。

---

## 2. 形式化问题定义

### 2.1 安全模型

**定义 1（单腐坏半诚实安全性）**：设 $f: (\mathbb{Z}/2^l\mathbb{Z})^3 \to (\mathbb{Z}/2^q\mathbb{Z})^3$ 是随机可计算函数，$\pi$ 是 3PC 计算协议。称 $\pi$ 满足**单腐坏半诚实安全性**，若存在 PPT 算法 $\mathcal{S}$，使得对每个 $j \in \{0,1,2\}$ 和输入 $x \in (\mathbb{Z}/2^l\mathbb{Z})^3$：

$$\{output_\pi(x),\ view_\pi^j(x)\} \equiv \{f(x),\ \mathcal{S}(j, x_j, f_j(x))\}$$

其中 $\equiv$ 表示两种概率分布多项式时间内（除可忽略概率）不可区分。

**含义**：三方中任意一方观察不到其他方的私有输入。

### 2.2 数据编码

设 $l$ 为位宽，$f$ 为小数精度，实数 $x \in \mathbb{R}$ 的定点数编码为 $\lfloor 2^f x \rceil \in \mathbb{Z}/2^l\mathbb{Z}$，取值范围为 $[-2^{l-1}, 2^{l-1}-1]$。

**复制秘密分享（Replicated Secret Sharing）**：三方持有份额 $(x_0, x_1, x_2)$，满足 $x_0 + x_1 + x_2 = \lfloor 2^f x \rceil$，每方持有两份（$P_j$ 持有 $(x_j, x_{j+1 \bmod 3})$）。

### 2.3 目标

- **输入**：三方各自持有私有数据（如数据集分片、模型参数分片）
- **输出**：安全训练或推理结果，任意单方不泄露他方隐私
- **约束**：
  - 半诚实安全模型（honest-but-curious）
  - 诚实多数（3 方中最多 1 方被腐化）
  - GPU 友好：数据全程留存 GPU 显存，避免 CPU-GPU 频繁迁移

---

## 3. 技术贡献详解

### 3.1 线性层：64 位整数 CUDA 矩阵乘法

**问题**：cuBLAS 不支持 64 位整数乘法；CryptGPU 的折叠浮点法（64 位整数 → 4 个浮点数）带来额外计算开销和内存膨胀；Piranha 手写 CUDA 核函数但未充分利用 GPU 访存特性。

**Antelope 方案**（算法 1 MatMul）：

- 矩阵分块，每个 GPU 线程分块负责结果矩阵的一个子块
- 使用 **GPU 共享内存（Shared Memory）** 缓存分块：先从全局内存拷贝到共享内存，再从共享内存加载到寄存器，显著降低内存访问延迟
- 受 SGEMM 启发，最大化计算/访存比

**卷积层（Conv2D）**：利用 im2col 算法将卷积转化为矩阵乘法 [Figure 2]。

```
截断延迟（Truncation Deferral）：
[[AB; f]] = Truncate(Σ [[A_{j,q} B_{q,k}; 2f]]) 
—— 所有乘加完成后仅做一次截断，减少截断协议调用次数
```

### 3.2 非线性层：O(1) 轮 MSB 获取协议

**引理 1（关键数学性质）**：

设 PRF 生成均匀分布，对于 $x \in \mathbb{Z}/2^l\mathbb{Z}$ 满足 $|x| < 2^{l_x}$，其中 $l_x < l$，复制秘密分享 $[[x]] = (x_0, x_1, x_2)$ 满足：

$$\Pr[(x_0 + x_1)\ \text{与}\ x_2\ \text{正负性相反}] > 1 - (2^{l_x+1-l} + \text{negl}(l))$$

**直觉**：令 $y_0 = x_0 + x_1$，$y_1 = x_2$，则以极高概率 $y_0$ 和 $y_1$ 具有相反符号。

**协议 1（MSB 获取，O(1) 轮）**：

基于 0-1 编码（Encoder）和有限域 $\mathbb{F}_p$（$p = 2^{61}-1$）上的随机线性变换，两方 $P_0, P_1$ 在本地完成，仅 $P_2$ 接收消息：

1. $P_j$（$j=0,1$）本地调用 $A_j[0:l-1] \leftarrow \text{Encoder}_j(|y_j|)$
2. 使用共享随机数 $r, s \leftarrow \mathbb{F}_p$ 计算 $B_j[k] = (rA_j[k] + s) \bmod p$
3. 若 $\exists k: B_0[k] = B_1[k]$，则对应比特位上绝对值相等，据此推断正负关系
4. 输出 $[[[x < 0]]]_{B'} = (b_0, b_1, b_2)$（布尔秘密分享形式）

**定理 1（安全性）**：上述 MSB 协议满足单腐坏半诚实安全性 [Section 3.3.3]。

### 3.3 MixMul 混合乘法协议

**目标**：实现 $[[b \cdot x; f]]$，其中 $b = \text{MSB}(x)$ 以特殊布尔秘密分享 $[[b]]_{B'}$ 存储，$x$ 以算术秘密分享存储。

核心思路：将特殊布尔秘密分享转化为算术域，再执行乘法，通信量极低。

**ReLU 协议**：

$$\text{ReLU}(x) = (1 - \text{MSB}(x)) \cdot x$$

组合 MSB 协议和 MixMul 协议即可实现，且整个 ReLU 协议满足单腐坏半诚实安全性（**定理 2**）[Section 3.3.3]。

### 3.4 批量归一化（BatchNorm）层

归一化公式：
$$x^{(j)} \mapsto \gamma \cdot \frac{x^{(j)} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

安全计算均值 $\mu$（可通过秘密共享加法完成），方差 $\sigma^2$（需乘法协议），以及开方倒数（通过比较协议和迭代算法实现）。引入批量归一化后，Antelope 可以完成 VGG-16 等深层网络的**随机初始化训练**（而非仅 fine-tuning）。

---

## 4. 实验结果分析

### 4.1 实验配置 [Section 4]

- **硬件**：3 台运行 Ubuntu 的机器，每台配备 NVIDIA GPU（具体型号论文未明确标注）
- **网络**：10 Gbps LAN
- **基线对比**：FALCON（CPU，PoPETs 2020）、CryptGPU（GPU，S&P 2021）、Piranha（GPU，USENIX 2022）

### 4.2 训练性能（Table, Section 4）

| 对比系统 | 训练加速比（Antelope vs X） |
|---------|--------------------------|
| FALCON (CPU) | **29–101×** |
| CryptGPU (GPU) | **2.5–3×** |
| Piranha (GPU) | **1.2–1.6×** |

> **来源**: Section 4（摘要及实验对比表）

### 4.3 推理性能

| 对比系统 | 推理加速比（Antelope vs X） |
|---------|--------------------------|
| FALCON (CPU) | **1.6–35×** |
| CryptGPU (GPU) | **11×** |
| Piranha (GPU) | **2.8×** |

具体模型推理时间对比（[Table 6]）：

| 架构 | FALCON | CryptGPU | Piranha | **Antelope** |
|------|--------|----------|---------|-------------|
| LeNet/MNIST | 0.04s | 0.31s | 0.04s | **0.024s** |
| AlexNet/CIFAR-10 | 0.11s | 0.78s | 0.14s | **0.05s** |
| AlexNet/TI | 0.34s | 0.80s | 0.19s | **0.06s** |
| VGG-16/CIFAR-10 | 1.51s | 1.89s | 0.48s | **0.13s** |
| VGG-16/TI | 8.92s | 2.02s | 0.76s | **0.25s** |

### 4.4 安全 ReLU 协议性能对比（[Figure 5]）

在运行时间和通信量两个维度上对比 ReLU 协议：

```
图 5：安全 ReLU 协议运行时间（a）和通信量（b）比较
X 轴：输入尺寸（10^0 ~ 10^5）
Y 轴：运行时间/通信量（对数坐标）
结论：Antelope 在小输入（≤ 10^3）时相对其他方案优势最大
```

> **注意**：[Figure 5] 表明在**大输入尺寸**下 Antelope 与 CryptGPU/Piranha 的差距收窄，对小数据量计算优势最为显著。

### 4.5 模型精度验证（[Table 7]）

| 模型 | 基准（随机猜测） | 明文框架 | **Antelope** |
|------|---------------|---------|-------------|
| LeNet/MNIST | 10% | 96.2% | **96.1%** |
| AlexNet/CIFAR-10 | 10% | 34.3% | **34.2%** |
| VGG-16/CIFAR-10 | 10% | 55.8% | **55.6%** |

精度损失极小（< 0.2%），验证了安全计算协议的正确性。

---

## 5. 反包装审计（Anti-Packaging Audit）

### 5.1 主要声明分解与验证

| 声明 | 子声明 | 证据锚 | 验证状态 |
|------|--------|--------|---------|
| 训练比 FALCON 快 29–101× | 取决于网络结构 | 摘要 + Section 4 实验表 | ✅ 有锚，但具体表格页码论文未清晰标注 |
| 训练比 CryptGPU 快 2.5–3× | GPU 系统间对比 | 摘要 + Section 4 | ✅ 有锚 |
| MSB 协议 O(1) 轮通信 | 不需离线预处理 | Section 3.3.1，协议 1 | ✅ 有完整协议描述 |
| 支持深层网络完整训练（VGG-16 随机初始化） | 依赖 BatchNorm | Section 3.4 + Table 7 | ✅ 验证，精度与明文接近 |
| 安全比较在小输入时优势大 | 小输入尺寸 | Figure 5 | ✅ 图中可见 |

### 5.2 公平性存疑

- **硬件未完全披露**：论文未明确说明实验所用 GPU 型号，无法独立复现性能数据。**标记为公平性未知（Fairness Unknown）**。
- **FALCON 对比不公**：FALCON 是 CPU 实现，与 GPU 系统对比 29–101× 加速比更多体现 GPU vs CPU 的硬件差异，而非协议本身优势。
- **Piranha 多协议 vs 单协议**：Piranha 是通用平台（支持 SecureML/FALCON/FantasticFour），论文对比具体协议未明确说明使用哪种配置。
- **网络模型简单**：实验中 AlexNet/CIFAR-10 精度仅 34.3%（基准 10%），远低于 state-of-the-art（~93%）。作者在 [Section 4.4] 解释为"未经充分调优的训练"（仅 1 epoch），这是一个重要但易被忽视的限制。

### 5.3 隐藏假设

1. **半诚实安全模型**：论文仅保证半诚实（curious but honest），不防御恶意攻击方（Malicious Adversaries）。恶意模型扩展被明确留作未来工作 [Section 2.1]。
2. **诚实多数假设**：至少两方是诚实的（标准 3PC 假设）。
3. **LAN 网络**：实验在 10 Gbps 局域网下，广域网（WAN）下性能会大幅下降（通信轮数虽少，但通信量本身未显著优化）。
4. **同构硬件**：三方使用相同配置的机器，异构环境下性能未评估。
5. **平均值池化代替最大值池化**：为降低非线性层复杂度，牺牲了最大值池化的特性 [Section 3 协议层]。

---

## 6. 局限性与失效条件

1. **半诚实安全性不足**：面对主动攻击（如 Malicious MPC）时不提供安全保障。论文承认这一点并把恶意安全扩展留作未来工作 [Section 4/结论]。
2. **小输入优势区间**：[Figure 5] 清晰显示，当输入尺寸增大到 $10^3$ 以上时，与 Piranha 和 CryptGPU 的差距明显收窄。对超大 batch 或高维度输入，协议优势减弱。
3. **WAN 场景不适用**：MSB 协议虽然降低了通信轮数，但总通信量没有根本性突破，高延迟网络（如跨机构联邦学习）下性能会显著下降。
4. **仅支持特定网络结构**：当前实现支持线性层、ReLU、平均池化、BatchNorm，不支持 attention、Softmax（仅初步设计，未在训练中完整评估）、残差连接等现代架构组件。
5. **精度验证不充分**：实验仅训练 1 个 epoch，精度远低于 SOTA，难以反映实际部署场景中的训练效果。

---

## 7. 与已有工作的对比关系

### 7.1 超越 / 改进的工作

| 论文 | 关系 | 证据 |
|------|------|------|
| **CryptGPU** (Tan et al., S&P 2021) | 训练提速 2.5–3×，推理提速 11×；Antelope 避免了 64 位整数折叠浮点的额外开销 | Section 4, Table 6 |
| **Piranha** (Watson et al., USENIX 2022) | 推理提速 2.8×；Antelope 更充分利用 GPU 并行性和共享内存 | Section 4, Table 6 |
| **FALCON** (Wagh et al., PoPETs 2020) | 提速 29–101× (训练)；但差异主要来自 GPU vs CPU | Section 4 |
| **ABY3** (Mohassel & Rindal, CCS 2018) | Antelope 的 MSB 通信轮数 O(1) vs ABY3 的 O(log l) | Section 3.3.1 |
| **SecureNN** (Wagh et al., PoPETs 2019) | Antelope 简化了算术-二进制共享的转换开销 | Section 1 |

### 7.2 被 Antelope 批评 / 指出局限的工作

- **CryptGPU [Ref 14]**：64 位整数折叠为浮点数的方法带来额外开销和内存占用。
- **Piranha [Ref 20]**：GPU 核函数未充分利用访存特性（共享内存）。
- **文献[21],[23]**（Orca 等）：提出了精度更好的非线性函数算法，但仅有 CPU 实现。

### 7.3 与 Force (Dai et al., NordSec 2023) 的关系

Force 是同期的**四方（4PC）**PPML GPU 框架；Antelope 专注**三方（3PC）**。两者目标相近但方案不同，Force 未在 Antelope 参考文献中出现（时间接近，可能未交叉引用）。

---

## 8. 跨论文影响分析

| 关联论文 | 影响方向 |
|---------|---------|
| **Piranha** | 被 Antelope 超越：GPU 共享内存利用更充分 |
| **CryptGPU** | 被 Antelope 超越：64 位整数计算路径更高效 |
| **Orca** (Jawalkar et al., S&P 2024) [Ref 22] | 后续工作；基于 FSS 技术，不同方向 |
| **SPIN** (Jiang et al., arXiv 2024) [Ref 23] | 同时期并行工作，亦使用 GPU 加速 |

---

## 9. 文献推荐阅读清单

### 9.1 近两年同类问题论文（2023–2025）

1. **Orca: FSS-based Secure Training and Inference with GPUs**  
   Jawalkar N, Gupta K, et al.  
   IEEE Symposium on Security and Privacy (S&P), 2024.  
   DOI: 10.1109/SP54263.2024.00063  
   *基于 FSS（Function Secret Sharing）的 GPU 安全推理框架，与 Antelope 同期竞争*

2. **SPIN: An Efficient Secure Computation Framework with GPU Acceleration**  
   Jiang WX, Song XJ, et al.  
   arXiv:2402.02320, 2024.  
   *GPU 加速安全计算框架，同期并行工作*

3. **Force: Highly Efficient Four-Party Privacy-Preserving Machine Learning on GPU**  
   Tianxiang Dai, Li Duan, et al.  
   NordSec 2023. DOI: 10.1007/978-3-031-47748-5_18  
   *四方版本的 GPU PPML，与 Antelope 互补*

4. **Secure Quantized Training for Deep Learning**  
   Keller M, Sun K.  
   ICML 2022.  
   *量化 + 安全训练结合，解决精度问题的相关方向*

### 9.2 Antelope 使用的基础/基线论文

5. **FALCON: Honest-Majority Maliciously Secure Framework for Private Deep Learning**  
   Wagh S, Tople S, et al.  
   PoPETs 2021, Vol. 1: 188–208.  
   *Antelope 的主要 CPU 基线*

6. **CryptGPU: Fast Privacy-Preserving Machine Learning on the GPU**  
   Tan S, Knott B, Tian Y, Wu DJ.  
   IEEE S&P 2021. DOI: 10.1109/SP40001.2021.00098  
   *Antelope 改进的主要 GPU 基线*

7. **Piranha: A GPU Platform for Secure Computation**  
   Watson JL, Wagh S, Popa RA.  
   USENIX Security 2022: 827–844.  
   *Antelope 改进的另一 GPU 基线*

8. **ABY3: A Mixed Protocol Framework for Machine Learning**  
   Mohassel P, Rindal P.  
   ACM CCS 2018: 35–52.  
   *Antelope 秘密分享方案（复制秘密分享）的来源*

### 9.3 Antelope 所依赖 / 被其批评的工作

9. **SecureNN: 3-Party Secure Computation for Neural Network Training**  
   Wagh S, Gupta D, Chandran N.  
   PoPETs 2019(3): 26–49.  
   *3PC PPML 领域关键先驱工作*

10. **SecureML: A System for Scalable Privacy-Preserving Machine Learning**  
    Mohassel P, Zhang Y.  
    IEEE S&P 2017. DOI: 10.1109/SP.2017.12  
    *2PC PPML 奠基工作*

11. **CrypTen: Secure Multi-Party Computation Meets Machine Learning**  
    Knott B, Venkataraman S, et al.  
    NeurIPS 2021: 4961–4973.  
    *Antelope 基于 PyTorch 接口设计参考*

---

## 10. 文献搜索日志

| 搜索平台 | 查询词 | 结果摘要 |
|---------|--------|---------|
| arXiv | "Antelope" "privacy-preserving" "GPU" | 0 结果（该论文发表于软件学报，非预印本） |
| IACR ePrint | title:antelope | 0 结果 |
| Crossref | Antelope 3-party GPU machine learning | 返回最相关：CryptGPU, Privacy-Preserving Multi-Party ML Based on TEE and GPU |
| OpenReview | Antelope privacy preserving machine learning GPU | 返回 CryptGPU, Force, GTree（无直接命中） |
| DBLP | 3-party GPU machine learning | 服务暂不可用 |

**筛选说明**：本论文为**中文期刊论文**（软件学报），未上传至 arXiv/IACR，主要发布渠道为 http://www.jos.org.cn/1000-9825/7445.htm。

---

## 11. 整体评价

### 真实贡献（实质性的）

- O(1) 轮 MSB 协议是本文最核心的密码学贡献，理论证明完整（引理 1, 2，定理 1, 2）
- 64 位整数 CUDA MatMul 的实现策略实用且有工程价值
- BatchNorm 的安全实现使深层网络完整随机初始化训练成为可能

### 值得怀疑的声明

- 与 FALCON 的对比（29–101×）主要体现 GPU 优势，与协议创新关系不大
- 精度实验仅 1 epoch，不能代表实际训练质量
- GPU 硬件型号未披露，无法评估公平性

### 对领域认知的修正

- **先前认知**：CryptGPU 和 Piranha 是 GPU PPML 的最优实现  
- **修正后**：在 3PC 半诚实设置下，针对 MSB 提取的专用优化可进一步提升 2.5–3× 训练性能，主要来源于 O(1) 轮通信协议 + 共享内存利用率

---

## 附录：完整参考文献列表

1. Esteva A, et al. Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 2017, 542(7639): 115–118. DOI: 10.1038/nature21056
2. Angelini E, Di Tollo G, Roli A. A neural network approach for credit risk evaluation. *Quarterly Review of Economics and Finance*, 2008, 48(4): 733–755. DOI: 10.1016/j.qref.2007.04.001
3. Dowlin N, et al. CryptoNets: Applying neural networks to encrypted data. *ICML 2016*: 201–210.
4. Hesamifard E, Takabi H, Ghasemi M. CryptoDL: Deep neural networks over encrypted data. arXiv:1711.05189, 2017.
5. Demmler D, Schneider T, Zohner M. ABY-A framework for efficient mixed-protocol secure two-party computation. *NDSS 2015*. DOI: 10.14722/ndss.2015.23113
6. Juvekar C, Vaikuntanathan V, Chandrakasan A. GAZELLE: A low latency framework for secure neural network inference. *USENIX Security 2018*: 1651–1669.
7. Mishra P, et al. Delphi: A cryptographic inference service for neural networks. *USENIX Security 2020*: 2505–2522.
8. Rathee D, et al. CrypTFlow2: Practical 2-party secure inference. *ACM CCS 2020*: 325–342.
9. Mohassel P, Rindal P. ABY3: A mixed protocol framework for machine learning. *ACM CCS 2018*: 35–52.
10. Mohassel P, Zhang Y. SecureML: A system for scalable privacy-preserving machine learning. *IEEE S&P 2017*: 19–38.
11. Liu J, et al. Oblivious neural network predictions via minionn transformations. *ACM CCS 2017*: 619–631.
12. Wagh S, Gupta D, Chandran N. SecureNN: 3-party secure computation for neural network training. *PoPETs 2019*(3): 26–49.
13. Wagh S, Tople S, et al. Falcon: Honest-majority maliciously secure framework for private deep learning. *PoPETs 2021*, Vol. 1: 188–208.
14. Tan S, Knott B, Tian Y, Wu DJ. CryptGPU: Fast privacy-preserving machine learning on the GPU. *IEEE S&P 2021*: 1021–1038. DOI: 10.1109/SP40001.2021.00098
15. Chaudhari H, et al. ASTRA: High throughput 3PC over rings with application to secure prediction. *CCSW 2019*: 81–92.
16. Patra A, Suresh A. BLAZE: Blazing fast privacy-preserving machine learning. *NDSS 2020*. DOI: 10.14722/ndss.2020.24202
17. Beaver D. Efficient multiparty protocols using circuit randomization. *CRYPTO 1991*: 420–432.
18. Knott B, Venkataraman S, et al. CrypTen: Secure multi-party computation meets machine learning. *NeurIPS 2021*: 4961–4973.
19. Kumar N, et al. CrypTFlow: Secure tensorflow inference. *IEEE S&P 2020*: 336–353.
20. Watson JL, Wagh S, Popa RA. Piranha: A GPU platform for secure computation. *USENIX Security 2022*: 827–844.
21. Keller M, Sun K. Secure quantized training for deep learning. *ICML 2022*: 10912–10938.
22. Jawalkar N, et al. Orca: FSS-based secure training and inference with GPUs. *IEEE S&P 2024*: 597–616. DOI: 10.1109/SP54263.2024.00063
23. Jiang WX, et al. SPIN: An efficient secure computation framework with GPU acceleration. arXiv:2402.02320, 2024.
24. Araki T, et al. High-throughput semi-honest secure three-party computation with an honest majority. *ACM CCS 2016*: 805–817. DOI: 10.1145/2976749.2978331
25. Chellapilla K, Puri S, Simard P. High performance convolutional neural networks for document processing. *ICDAR 2006*.
26. Canetti R. Universally composable security. *IEEE FOCS 2001*: 136–145.（隐含引用，协议组合定理来源）
