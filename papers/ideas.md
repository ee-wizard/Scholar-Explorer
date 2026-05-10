# Ideas

## 待提炼队列

（当前无）

## 已提炼

### 4. 结构感知自适应剪枝预算
- 问题定义类别：视觉文档多向量检索中的文档级自适应压缩
- 技术路线标签：传统索引、多向量检索、自适应压缩
- 原论文创新点锚点：DocPruner 的 global-token attention 重要性估计与文档级统计阈值
- 改进假设：如果把页面结构区域先验引入 patch 重要性估计，可进一步减少 attention 代理对查询无关噪声的误判
- 改进方案：联合 layout parsing 与 attention score，对标题区、表格区、图像区、正文区设置分层阈值或区域预算，再执行自适应 pruning
- 对比基线：DocPruner 原方法，Prune-then-Merge，固定比例 pruning
- 可验证指标：nDCG@5（效果），平均每页向量数与离线索引时延（效率）

### 5. Layout-Aware Sub-Image 分层编码
- 问题定义类别：视觉文档高压缩下的多向量语义保留
- 技术路线标签：多向量检索、布局感知压缩、结构嵌入
- 原论文创新点锚点：Beyond the Grid（ColParse）将页面按 layout parser 切割成 sub-image，每块独立嵌入后融合
- 改进假设：若对不同语义块（标题/表格/图像/正文）使用异构压缩策略，可进一步提升高压缩率下的 nDCG 稳定性
- 改进方案：在 sub-image 编码后，根据块类型标签为不同块设置不同 pooling ratio，标题块保留高分辨率嵌入，空白/重复纹理块直接 pooling 到单向量
- 对比基线：Beyond the Grid（ColParse），DocPruner，Prune-then-Merge
- 可验证指标：nDCG@5（效果），每页向量数与存储 footprint（效率）

### 6. Query-Aware Key-Token 路由的 Hybrid Retrieval
- 问题定义类别：VDR 两阶段 hybrid 检索中 query-side 冗余计算控制
- 技术路线标签：混合检索、query routing、安全搜索算法
- 原论文创新点锚点：HEAVEN 用 POS tagging 过滤 query 中的非名词 token 来降低 multi-vector scoring 的 FLOPs
- 改进假设：如果用 cross-modal saliency 或 attention-based token importance 替代 POS 规则过滤，可减少误删对精细匹配关键的 query token
- 改进方案：训练轻量 query token importance predictor，只保留对当前文档类型最具区分力的 K 个 token 再做 late interaction
- 对比基线：HEAVEN 原方法（POS filtering），full query late interaction
- 可验证指标：Recall@1（精度），per-query FLOPs（效率）

### 7. 高维多向量的 FDE 近似加速
- 问题定义类别：高维多向量 late interaction 的近似检索加速
- 技术路线标签：传统索引、安全搜索算法、ANN 索引技术
- 原论文创新点锚点：MUVERA 把多向量嵌入映射到固定维度编码（FDE），使其与单向量 ANN 引擎兼容
- 改进假设：若把 FDE 与视觉 patch 的空间分布结构结合（而非无结构 hashing），可在视觉文档场景取得更优精度-速度平衡
- 改进方案：把 MUVERA 的随机特征映射替换为 patch-grid 感知的结构化 feature map，与 ColPali 的 2D patch layout 对齐
- 对比基线：MUVERA 原方法，Visual RAG Toolkit multi-stage retrieval
- 可验证指标：MaxSim 近似误差（精度），检索延迟与 ANN 查询吞吐（效率）

### 8. Training-Free Pooling 中的 Query-Conditioned 激活
- 问题定义类别：无训练多向量压缩中精度损失来源定位
- 技术路线标签：多向量检索、训练无关压缩、query-aware 激活
- 原论文创新点锚点：Visual RAG Toolkit 用静态空间 pooling（tile mean、row mean、Gaussian smoothing）压缩 patch embedding
- 改进假设：若在推理时根据 query 激活不同 pooled tile/row，而非统一做全页 pooling，可在不引入额外训练的前提下减少信息丢失
- 改进方案：在 stage-1 prefetch 阶段，计算 query token 与各 pooled tile 的 similarity，只对高相关 tile 展开原始 patch 做精排，而不是对所有 tile 都跑全精排
- 对比基线：Visual RAG Toolkit 2-stage pooling，HEAVEN VS-pages
- 可验证指标：NDCG@10（效果），per-query FLOPs 与 QPS（效率）

### 1. 结构感知多向量压缩预算
- 问题定义类别：视觉文档多向量检索的存储成本控制
- 技术路线标签：传统索引、多向量检索、索引压缩
- 原论文创新点锚点：ColPali 的 image patch 多向量表示与 late interaction
- 改进假设：如果按页面区域结构为 patch 分配不同压缩预算，可在更低存储成本下保留核心检索信号
- 改进方案：先用布局解析或区域聚类划分页内块，再对不同区域使用差异化 pruning / pooling / merging
- 对比基线：ColPali 原始多向量索引，DocPruner，Prune-then-Merge
- 可验证指标：nDCG@5（效果），每页向量数与存储 footprint（效率）

### 2. 结构感知 Prune-then-Merge 压缩
- 问题定义类别：高压缩率下多向量检索性能稳定性
- 技术路线标签：传统索引、向量检索压缩、鲁棒检索
- 原论文创新点锚点：先剪枝后合并的两阶段协同
- 改进假设：若将文档布局结构（正文/表格/图像）注入剪枝阈值，可进一步降低语义信息误删
- 改进方案：在 Stage-1 增加结构感知重要性重加权，再执行分层合并
- 对比基线：PRUNE-THEN-MERGE 原方法
- 可验证指标：nDCG@5（效果），压缩率与编码时延（效率）

### 3. 动态预算压缩策略
- 问题定义类别：不同文档复杂度下统一压缩率不稳定
- 技术路线标签：多向量检索、自适应压缩、系统优化
- 原论文创新点锚点：自适应剪枝阈值 + 分层聚类合并
- 改进假设：按文档复杂度动态分配压缩预算，可在相同存储预算下提升总体检索质量
- 改进方案：引入文档复杂度估计器，输出文档级 compression budget 后执行两阶段压缩
- 对比基线：固定 k/m 的 PRUNE-THEN-MERGE
- 可验证指标：平均 nDCG@5（效果），单位存储下检索收益（效率）

### 9. FDE 结构化映射迁移至视觉 Patch 场景
- 问题定义类别：视觉文档多向量 late interaction 的 ANN 加速
- 技术路线标签：传统索引、安全搜索算法、近似最近邻
- 原论文创新点锚点：MUVERA 的固定维度编码（FDE）将多向量检索规约为单向量 MIPS，保留 10× 更高 recall，延迟降低 90%
- 改进假设：若将 MUVERA 的随机高斯特征映射替换为利用 ColPali patch 2D grid 空间信息的结构化特征映射，可在相同维度预算下取得更优精度-速度权衡
- 改进方案：设计 patch-grid 感知的 FDE，使得空间相邻 patch 的特征投影向量之间存在平滑约束，再将其接入标准单向量 ANN（HNSW/IVF）；与 MUVERA 随机 FDE 在 ViDoRe V3 10 数据集上对比
- 对比基线：MUVERA 随机 FDE，ColPali 原始多向量 MaxSim，Visual RAG Toolkit 多阶段检索
- 可验证指标：MaxSim 近似误差（精度），检索延迟 P50/P99（效率）

### 10. 残差压缩思想迁移至视觉 Patch 嵌入量化
- 问题定义类别：多向量视觉检索的存储压缩
- 技术路线标签：多向量检索、量化压缩、存储索引
- 原论文创新点锚点：ColBERTv2 的残差压缩：以聚类 centroid 为基准只保存差值，将存储降低 6-10×
- 改进假设：若将 ColBERTv2 的 centroid-residual 量化思路迁移至视觉 patch 嵌入，以图像区域质心作为 centroid，保存 patch 到质心的残差向量，可在相同存储预算下比 ColPali binary/int8 量化保留更多语义细节
- 改进方案：对每页 patch 嵌入先做 K-Means 聚类（region-level），以聚类中心量化残差，并配合 HPC 的二值编码进一步压缩；在 ViDoRe V2/V3 上对比 HPC 量化基线
- 对比基线：ColBERTv2 残差压缩（文本），HPC（Hierarchical Patch Compression），DocPruner
- 可验证指标：nDCG@10（效果），存储 MB/页，压缩率（效率）

### 11. 视觉重排器（Visual Reranker）设计
- 问题定义类别：视觉文档检索流水线中的重排组件设计
- 技术路线标签：重排器、视觉 Cross-encoder、多模态 RAG
- 原论文创新点锚点：ViDoRe V3 量化发现：文本重排器（zerank-2）给视觉流水线带来 +13.2 NDCG@10，而视觉重排器（jina-reranker-m0）仅 +0.2，在 4 个数据集上下降
- 改进假设：若设计基于 VLM 的视觉 cross-encoder 式重排器，用（query, page image）对直接打分而非嵌入相似度，可显著缩小视觉重排与文本重排的差距
- 改进方案：以 ViDoRe V3 公开集（8 数据集）为训练数据，构建（query, 正例页, 负例页）三元组，用 VLM（ColQwen/jina-v4）fine-tune cross-encoder；引入 pairwise 或 listwise 排序损失；用私有测试集盲评
- 对比基线：zerank-2（文本重排器），jina-reranker-m0（现有视觉重排器），无重排基线
- 可验证指标：NDCG@10（效果），重排延迟 ms（效率）

### 12. 多语言视觉检索器训练（非拉丁文字增强）
- 问题定义类别：多语言视觉文档检索能力提升
- 技术路线标签：多语言 VLM、跨语言对齐、视觉检索
- 原论文创新点锚点：MIRACL-VISION 量化发现：视觉检索器平均比文本检索器落后 59.7%，Telugu 完全失效（NDCG < 0.10），非拉丁字母语言差距最大
- 改进假设：通过构建多语言（query, Wikipedia 截图）训练对并引入跨语言对比损失，可将低资源语言的视觉检索性能从 < 0.10 提升至 > 0.40
- 改进方案：基于 MIRACL-VISION 生成训练集（18 语言 × top-100 硬负例对），用跨语言 InfoNCE 对 ColQwen 系列微调；同时引入知识蒸馏（文本检索器→视觉检索器）策略
- 对比基线：gme-Qwen2-VL-2B（现有最优视觉检索器），bge-m3（文本检索器上界），vdr-2b-multi-v1
- 可验证指标：MIRACL-VISION 18 语言平均 NDCG@10 与 Telugu/Arabic/Thai 子集 NDCG@10

### 13. 自适应 Patch 分辨率切割策略（DSE 效率优化）
- 问题定义类别：截图嵌入式视觉文档检索的编码效率
- 技术路线标签：视觉嵌入、patch 压缩、效率优化
- 原论文创新点锚点：DSE 的 $C_x \times C_y$ 子图切割策略：4×4 精度最高（Top-1 NQ=46.2%）但编码速度仅 4.3 doc/sec；1×1 速度最快（12.2 doc/sec）但精度降 11 点
- 改进假设：若根据截图内容自适应选择切割粒度（文字密集区域高分辨率，图像/空白区域低分辨率），可在不损失精度的前提下将编码速度提升至 > 3×
- 改进方案：先用轻量内容分类器（前景密度、文字面积占比）对截图分类，再动态选择 1×1/2×2/4×4 切割方案；目标是在文字密集页面保持 4×4 精度，在图像为主页面使用 1×1/2×2；在 Wiki-SS 和 SlideVQA-open 上评测
- 对比基线：DSE 固定 4×4，DSE 固定 2×2，CLIP（单路 1×1）
- 可验证指标：NQ Top-1 精度（效果），doc/sec 编码速度（效率），检索延迟（系统）

### 14. 自适应 Merging Factor（按页面信息密度动态调整压缩比）
- 问题定义类别：视觉文档多向量检索的内容感知压缩预算分配
- 技术路线标签：多向量检索、自适应压缩、训练无关
- 原论文创新点锚点：Light-ColPali（arXiv:2506.04997）使用语义聚类合并相邻 patch，但使用全局固定的 merging factor
- 改进假设：若按页面信息密度（文字面积占比、色彩方差、图像内容比例）动态分配合并比例，文字密集页保留更多 patch，空白/图像页大幅合并，可在相同平均预算下提升高信息密度页面的检索质量
- 改进方案：训练轻量页面复杂度分类器（基于缩略图特征），输出每页 merging factor $k \in \{4, 8, 16, 32\}$，再用 Light-ColPali 的语义聚类合并；与固定 k 基线和 ColChunk 固定 K=40 对比
- 对比基线：Light-ColPali 固定 merging factor，ColChunk K=40
- 可验证指标：nDCG@5（效果），每页平均向量数与总存储（效率）

### 15. ColChunk + LoRA 微调联合方案（训练增强的 pseudo-multi-vector）
- 问题定义类别：视觉文档检索高压缩比下的检索精度保持
- 技术路线标签：多向量检索、知识蒸馏、微调
- 原论文创新点锚点：ColChunk（arXiv:2604.10167）完全 training-free，通过 HAC 聚类将单向量 patch 嵌入压缩为 K=40 pseudo-multi-vector；Light-ColPali 微调路径显示 fine-tuned 版本可显著提升极高压缩比精度
- 改进假设：若以 ColChunk 作为初始化（training-free pseudo-multi-vector），再用 LoRA 微调单向量骨干，可在不增加推理存储的同时进一步改善 K=10-20 超高压缩比下的精度
- 改进方案：ColChunk(K=20) training-free baseline → 冻结 ColChunk 分组，LoRA 微调骨干 → end-to-end 联合微调；在 ViDoRe V1/V2/V3 对比 training-free ColChunk 和 Light-ColPali 微调版本
- 对比基线：ColChunk K=20（training-free），Light-ColPali fine-tuned，原生单向量基线
- 可验证指标：nDCG@5（效果），每页向量数（效率），LoRA 微调成本（GPU-hours）

### 16. AGREE 局部余弦损失 + 多语言训练数据（解决 AGREE 多语言局限）
- 问题定义类别：视觉文档检索的多语言精度提升
- 技术路线标签：视觉检索、跨语言对齐、局部注意力监督
- 原论文创新点锚点：AGREE（arXiv:2511.13415）用 MLLM 注意力图监督 patch 级局部对齐损失；但当前仅用英语训练数据，非英语文档的注意力监督效果未知
- 改进假设：若引入多语言 MLLM 生成的注意力监督信号（Arabic/Chinese/French VQA 数据），AGREE 的局部对齐损失可推广到多语言视觉文档，降低非英语文档检索的 recall 差距
- 改进方案：用 InternVL/Qwen-VL 多语言 MLLM 生成多语言文档的注意力图，构建多语言（query, doc, attention map）三元组，在 AGREE 框架下多语言联合训练；在 MIRACL-VISION 多语言数据集上评测
- 对比基线：AGREE 英语训练基线，ColPali 多语言微调（vdr-2b-multi-v1）
- 可验证指标：MIRACL-VISION 18 语言平均 NDCG@10，非英语子集（Arabic/Chinese）NDCG@10

### 17. NanoVDR + 多向量文档编码（Light-ColPali 教师）
- 问题定义类别：视觉文档检索非对称编码的全效率优化
- 技术路线标签：知识蒸馏、非对称检索、多向量+单向量混合
- 原论文创新点锚点：NanoVDR（arXiv:2603.12824）冻结 2B 单向量 VLM 作教师（文档端），蒸馏 69M 文本学生（查询端），但教师本身是重量级单向量模型
- 改进假设：若以 Light-ColPali 等高效多向量模型替换 NanoVDR 的 VLM 教师，文档端可获多向量精度，查询端继续用轻量文本学生，两者结合可能兼得多向量精度 + 查询侧超低延迟
- 改进方案：以 Light-ColPali（或 Hierarchical Patch Compression ColPali）作为冻结文档编码教师，生成 K=40 pseudo-multi-vector 文档嵌入；用 NanoVDR 的 cosine alignment 蒸馏轻量文本 query 编码器，使其查询嵌入与教师 MaxSim 结果对齐（近似 MaxSim 分数预测）
- 对比基线：NanoVDR（单向量教师），Light-ColPali（多向量文档），标准 MaxSim ColPali
- 可验证指标：ViDoRe V3 nDCG@5（效果），查询侧 CPU 延迟 ms（效率），总文档存储 GB/百万页（效率）

### 18. Top-k Pooling 替换 MaxSim 端到端重训（防御 Spike Hijacking）
- 问题定义类别：晚交互检索系统的鲁棒性提升与安全加固
- 技术路线标签：安全搜索算法、多向量检索、anti-adversarial
- 原论文创新点锚点：Spike Hijacking（arXiv:2604.05253）证明 MaxSim Gini 系数=0.983（高度集中），K=100 spike patches 使 MaxSim recall 降至 27.8%；Top-k pooling Gini~0.45，保留更好鲁棒性
- 改进假设：若以 Top-k pooling（k=5）从头重训 ColPali，而非 inference 时切换，模型可在学习阶段内化分散梯度信号，在正常精度和鲁棒性之间取得更优 Pareto 平衡
- 改进方案：在 ColPali 训练脚本中将 MaxSim 替换为 Top-k（k=3/5/10），在 ViDoRe V1/V2/V3 上评测正常检索精度；同时测试 spike hijacking 攻击场景（K=5/20/100）下的 recall 保留率
- 对比基线：MaxSim ColPali（vanilla），Softmax pooling（推理时切换，本文基线），inference-time Top-k 切换
- 可验证指标：ViDoRe nDCG@5（正常精度），spike 注入后 Recall@5 保留率（鲁棒性），Gini 梯度集中系数（分析指标）

### 19. AGREE + ColChunk 联合：按 Chunk 做注意力对齐
- 问题定义类别：视觉文档检索局部语义对齐的粒度优化
- 技术路线标签：视觉检索、注意力监督、chunk 级对齐
- 原论文创新点锚点：AGREE（arXiv:2511.13415）用 MLLM 注意力图对单 patch 做局部余弦损失；ColChunk（arXiv:2604.10167）将 patch 聚为 K=40 个语义连贯 chunk
- 改进假设：若以 ColChunk 的 HAC 分组结果为基础，在 chunk 级别（而非单 patch 级别）做 AGREE 的注意力对齐损失，可减少单 patch 注意力噪声，提升局部对齐的语义粒度
- 改进方案：ColChunk 将 patch 聚为 K 组 → 对每组内 patch 的注意力分数做 max-pool/mean-pool 作为组级注意力 → AGREE 的局部余弦损失在组级 chunk 嵌入上计算；在 ViDoRe 数据集上对比单 patch 级别 AGREE 和 chunk 级别 AGREE
- 对比基线：AGREE 原始单 patch 对齐，ColChunk training-free，两者简单组合（分别推理）
- 可验证指标：ViDoRe V2/V3 nDCG@5（效果），注意力对齐信噪比（分析指标）

### 20. 外存版集合级图索引
- 问题定义类别：多向量检索在十亿级语料下的单机可扩展索引构建与查询
- 技术路线标签：传统索引、安全搜索算法、图索引、多向量检索、外存检索
- 原论文创新点锚点：DiskANN 的 Vamana 低直径图、overlapping clusters 合并构图、PQ in-DRAM + full-precision on-SSD 的协同设计
- 改进假设：若把 DiskANN 的外存化图索引框架迁移到 GEM 一类集合级图索引，可在不牺牲过多 MaxSim 质量的前提下，把多向量索引从“只能驻留内存”推进到“单机 SSD 可承载”
- 改进方案：以 GEM 的 set-level graph 为骨架，增加 DiskANN 风格的 fixed-offset SSD layout、beam-style frontier expansion、以及 cluster membership 驱动的 overlapping graph merge；在文本 ColBERT 和视觉 ColPali 上分别评测
- 对比基线：GEM 原始内存驻留版本，ESPN（SSD offload embedding table），HEAVEN 或 MUVERA 两阶段候选生成方案
- 可验证指标：MRR@10 或 nDCG@10（效果），P50/P99 查询延迟与 SSD 读取次数（效率）

### 9. 自适应 Token Pruning 基于查询复杂度（从 ColBERT 启发）
- 问题定义类别：多向量 late interaction 中的查询时间自适应优化
- 技术路线标签：传统索引、安全搜索算法、多向量检索
- 原论文创新点锚点：ColBERT 的 MaxSim 相似度计算对所有查询 token 等权处理
- 改进假设：简单查询（如单一实体名）不需要全量 token-level MaxSim；可根据查询特征（长度、信息熵、命名实体密度）动态选择 K 个关键 token 参与计算
- 改进方案：训练轻量 query complexity classifier，将查询分为 simple/moderate/complex 三级，对应稀疏化系数 {10%, 50%, 100%}；仅保留高权重 token 做 MaxSim
- 对比基线：ColBERT 原方法（全量 token），XTR（已有类似思路），Col-Bandit（bandit pruning）
- 可验证指标：平均每查询 FLOPs（效率），端到端延迟 P50/P99，nDCG@10（精度）
- 预期提升：查询延迟 30-50% 降低，精度损失 <1%

### 10. 两阶段混合检索：单向量候选 + ColBERT 精排
- 问题定义类别：超大规模语料库下的多向量检索可扩展性
- 技术路线标签：传统索引、多向量检索、系统架构
- 原论文创新点锚点：ColBERT token-level embedding 导致索引膨胀（相对单向量 100x），难以部署到亿级文档
- 改进假设：在两阶段框架中，用廉价单向量生成百万级候选，再用 ColBERT 精排千级集合，可平衡精度和延迟
- 改进方案：Stage-1 使用 DPR 快速检索得 1M 候选；Stage-2 加载 ColBERT 压缩向量精排 1K；系统设计为分布式 retriever + compact ranker
- 对比基线：ColBERT 单阶段全量检索，纯单向量 DPR，HEAVEN 的 VS-pages 策略
- 可验证指标：端到端延迟、内存占用（GB）、Recall@1000、nDCG@10
- 预期提升：内存减 90%+，总延迟 <200ms，nDCG 损失 <1 点

### 21. Utility-Aware Skill Retrieval（从 SRA 启发）
- 问题定义类别：Agent 技能检索中“相关但无用”候选导致的端到端收益塌缩
- 技术路线标签：agentic_ai、skill_retrieval、受控技能加载
- 原论文创新点锚点：SRA（arXiv:2604.24594）指出检索指标提升不能直接转化为端到端收益，关键瓶颈在 relevance-aware 与 need-aware 的技能加载决策
- 改进假设：若将检索优化目标从语义相关性改为任务效用（utility），并联合学习“是否需要加载技能”的判别器，可显著降低无效加载与误加载
- 改进方案：训练两阶段系统：阶段一做 utility-aware rerank（以任务成功率增益作为弱监督），阶段二做 need-aware gating（输入为任务置信度、候选技能证据强度、历史调用结果）
- 对比基线：BM25、BGE、BM25+Rerank（相关性导向）；LLM Selection（无显式 need-aware gating）
- 可验证指标：端到端任务准确率（效果），无效加载率与误加载率（稳定性），平均 token 开销与加载次数（效率）

### 22. Skills 质量估计与自适应注入（从 SkillsBench 启发）
- 问题定义类别：Agent 技能增强中的“技能存在但无效/负效”识别
- 技术路线标签：agentic_ai、skill_benchmark、skill_quality_control
- 原论文创新点锚点：SkillsBench（arXiv:2602.12670）显示 curated skills 平均 +16.2pp，但 16/84 任务出现负增益；self-generated skills 平均无收益
- 改进假设：若在注入前进行 skill quality estimation（可执行性、步骤完整度、与任务匹配度、冲突风险），并根据置信度自适应决定“注入/不注入/精简注入”，可减少负增益任务比例
- 改进方案：构建轻量技能评分器（静态文档特征 + 历史轨迹特征），在推理时先排序再门控；对 2-3 个高质量技能优先注入，抑制 4+ 技能堆叠造成的上下文噪声
- 对比基线：无门控的全量注入、仅相关性检索注入、self-generated 技能注入
- 可验证指标：任务通过率（效果）、负增益任务占比（稳定性）、平均上下文 token 开销（效率）

### 23. 混合知识注入调度器（从 Parametric RAG 启发）
- 问题定义类别：RAG 中 in-context 注入与参数注入的协同决策
- 技术路线标签：agentic_ai、parametric_rag、retrieval_augmented_generation
- 原论文创新点锚点：Parametric RAG（SIGIR 2025）提出文档参数化与 Retrieve-Update-Generate；并证明与 in-context 组合（Combine Both）效果更强
- 改进假设：若按问题类型、检索证据密度、时延预算动态选择“上下文注入/参数注入/混合注入”，可在同等预算下得到更高任务成功率
- 改进方案：训练轻量路由器预测注入策略；对每次请求输出 policy（context-only / param-only / both）并最小化多目标损失（效果+时延+成本）
- 对比基线：Standard RAG、Parametric RAG、固定 Combine Both
- 可验证指标：端到端 F1 或 EM（效果），单请求延迟（效率），token 与显存成本（资源）

### 24. 动态检索触发的多信号判别（从 DRAGIN 启发）
- 问题定义类别：动态 RAG 中何时检索的稳定触发
- 技术路线标签：agentic_ai、dynamic_rag、retrieval_augmented_generation
- 原论文创新点锚点：DRAGIN（ACL 2024）提出 RIND（不确定性+语义重要性+后续影响）与 QFS（基于 self-attention 的全局查询构造）
- 改进假设：若将触发决策从单一置信度阈值扩展为多信号融合，并加入时延成本约束，可显著减少过检索与漏检索
- 改进方案：构建轻量触发器，输入 token-level uncertainty、attention entropy、influence score 与预算状态，输出检索开关与查询粒度
- 对比基线：固定步长触发、低置信触发（FLARE）、DRAGIN 原策略
- 可验证指标：任务 EM/F1（效果），每样本检索次数（效率），平均延迟与 token 成本（资源）

### 25. 可执行性导向的多字段工具检索（从 MFTR 启发）
- 问题定义类别：工具检索中“语义相关但不可调用”误检索
- 技术路线标签：agentic_ai、tool_retrieval、multi_field_retrieval
- 原论文创新点锚点：MFTR（arXiv:2602.05366）将工具文档结构化为多字段并做自适应加权，显著优于 Full-Doc 检索
- 改进假设：若将字段级相关性与可执行约束（参数可满足性、输出兼容性）联合建模为统一效用分数，可进一步降低误调用率
- 改进方案：在多字段检索后加入执行可行性判别器，对候选工具进行二次校正排序（utility rerank）
- 对比基线：BM25 Full-Doc、EasyTool、OnlineRAG、MFTR 原始版本
- 可验证指标：NDCG@10 / Recall@10（检索效果），工具调用成功率（执行效果），平均候选规模与延迟（效率）
