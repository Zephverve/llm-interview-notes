# 架构演进与 MoE

> 本章导读：从 BERT 到 GPT，从稠密模型到 MoE，从纯文本到多模态，架构演进的主线是“在给定算力与推理预算下，提升有效容量”。本章先厘清三大架构范式的取舍，再讲清 MoE、MLA、多模态与长上下文的结构设计，最后给出一张主流模型对照表。
>
> 阅读建议：面试中最容易被追问的是“MoE 为什么省算力不省显存”“MLA 压缩了什么”“多模态的视觉 token 如何进入语言模型”。带着这三个问题阅读第 4、5、6 节。

## 核心要点

- Encoder-Only（BERT）、Encoder-Decoder（T5）、Decoder-Only（GPT）没有绝对优劣，取决于任务形态与训练目标；生成式大模型选择 Decoder-Only 是工程与数据的双重选择。
- MoE 用“稀疏激活”把总参数量与单 token 计算量解耦：总参数决定容量与显存，激活参数决定算力。
- MoE 的难点不在前向公式，而在负载均衡、专家并行通信、训练稳定性与推理时的显存驻留。
- MLA 通过低秩压缩 KV 潜向量，把长上下文推理的 KV Cache 显著降低，是 DeepSeek 系列的关键设计。
- 多模态模型的主流结构是“视觉编码器 + 投影层 + 语言模型”，视觉 token 与文本 token 在同一序列中做注意力。
- 长上下文不是单纯“把位置编码拉长”，而是位置外推、注意力实现、训练数据、KV Cache 管理的系统工程。
- 推理模型（reasoning model）把 test-time compute 作为新的扩展维度，后训练从“模仿”转向“可验证奖励的强化学习”。
- 架构选择要回到约束：训练算力、推理成本、延迟、上下文长度、多模态需求、生态与合规。

## 1. 三种架构范式

### 1.1 Encoder-Only：BERT 系

代表：BERT、RoBERTa、DeBERTa、ELECTRA。

训练目标：

- **MLM（Masked Language Modeling）**：随机遮住 15% 的 token，让模型根据双向上下文预测。
- **NSP（Next Sentence Prediction）**：判断两句话是否连续（后续工作发现作用有限）。

特点：

- 双向注意力，每个位置都能看到左右上下文；
- 适合理解类任务：分类、抽取、检索、句向量；
- 不擅长开放式生成，因为训练目标不是自回归。

面试点：BERT 的 [CLS] 向量为什么不能直接当高质量句向量？因为 MLM 目标没有显式优化句子级相似度，[CLS] 未经过对比学习，直接做余弦相似度效果有限；Sentence-BERT 等用对比学习修正。

### 1.2 Encoder-Decoder：T5 系

代表：T5、BART、mT5、原始 Transformer。

结构：

- Encoder 双向编码输入；
- Decoder 自回归生成，并通过交叉注意力读取 Encoder 表示。

训练目标：

- T5 使用 span corruption：把连续片段替换为哨兵 token，让 Decoder 还原被遮住的片段；
- BART 使用去噪自编码：打乱、删除、填充，再重建原文。

特点：

- 输入理解与输出生成分离，适合翻译、摘要、结构化转换；
- 参数量在 Encoder/Decoder 间分配，同等总参数下生成容量可能不如 Decoder-Only；
- 推理时需要维护 Encoder 表示与 Decoder KV Cache，工程更复杂。

### 1.3 Decoder-Only：GPT 系

代表：GPT-3/4、Llama、Qwen、Mistral、DeepSeek。

结构：因果自注意力 + FFN，无交叉注意力。

训练目标：next-token prediction。

为什么成为主流：

1. **数据效率**：任何文本都能构造训练样本，不需要成对标注；预训练语料规模可以无限扩展。
2. **任务统一**：所有任务都可写成“前缀 → 续写”，天然支持 few-shot、指令、工具调用。
3. **工程简单**：只有一种层，KV Cache 管理统一，易于分布式扩展与推理优化。
4. **规模效应**：在足够规模下，in-context learning 弥补了缺少双向编码的不足；需要双向信息时，可以把材料放在前缀里。

局限：

- 单向注意力对某些理解任务不如双向编码；
- 训练效率低于 MLM（每个 token 只预测一次，而 MLM 可以遮多个）；
- 长上下文下 KV Cache 成本高。

> 面试表达：架构没有“唯一正确”，但生成式通用助手选择 Decoder-Only，是数据、任务统一性、工程扩展性共同作用的结果。

## 2. 主流模型家族与设计取舍

以下参数以公开资料为准，面试时说明“量级/版本”，不要把它当作精确规格。

| 模型 | 结构 | 参数量 | 层数 | 隐藏维度 | 注意力 | FFN | 特点 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3 8B | Decoder-Only | 8B | 32 | 4096 | GQA（32 Q / 8 KV） | SwiGLU 14336 | 15T token 训练，词表 128K |
| Llama 3 70B | Decoder-Only | 70B | 80 | 8192 | GQA（64 Q / 8 KV） | SwiGLU 28672 | 同上 |
| Qwen2.5 72B | Decoder-Only | 72B | 80 | 8192 | GQA（64 Q / 8 KV） | SwiGLU | 中文与代码能力强 |
| Mistral 7B | Decoder-Only | 7B | 32 | 4096 | GQA（32 Q / 8 KV） | SwiGLU | 滑动窗口注意力 |
| DeepSeek-V3 | Decoder-Only + MoE | 671B 总 / 37B 激活 | 61 | 7168 | MLA | MoE（256 路由 + 1 共享，top-8） | FP8 训练、MLA、无辅助损失负载均衡 |
| Gemma 2 27B | Decoder-Only | 27B | 46 | 4608 | GQA | GeGLU | 滑动窗口与全局注意力交替 |

从表中可以看出几条设计主线：

- **GQA 成为默认**：在质量损失可控的前提下大幅降低 KV Cache。
- **SwiGLU/GeGLU 成为默认**：门控 FFN 在同等参数量下更优。
- **词表变大**：128K 词表提升多语言与代码压缩率，但 embedding 参数增加。
- **训练 token 远超 Chinchilla 最优**：小模型过训练以降低推理成本。
- **MoE 与 MLA 用于“大容量、低激活、低 KV”**：DeepSeek 系列是典型。

## 3. 稠密模型的参数与计算

对标准 Decoder-Only 稠密模型：

$$
N\approx L\cdot(12d^2)+Vd
$$

单 token 前向 FLOPs 约 $2N$（训练时反向再加约 $4N$）。推理 decode 阶段每生成一个 token 需要读取全部参数一次，因此是 **memory-bound**：

$$
\text{decode 时间下限}\approx\frac{\text{参数量}\times\text{字节数}}{\text{显存带宽}}
$$

例如 7B 模型 FP16 约 14GB，H100 显存带宽约 3.35TB/s，理论下限约 4ms/token，实际因 KV Cache、kernel 开销更高。

这解释了两个趋势：

1. 量化（FP16 → INT8/INT4）能直接提升 decode 速度；
2. MoE 用更少的激活参数降低每 token 的计算与访存，但总参数仍要驻留显存。

## 4. MoE：稀疏专家混合

### 4.1 基本结构

MoE 把 FFN 替换为多个“专家”FFN，并用门控网络为每个 token 选择 top-$k$ 个专家：

$$
y=\sum_{i\in\mathrm{TopK}(g(x))}g_i(x)\cdot E_i(x)
$$

其中：

- $x\in\mathbb{R}^d$ 是 token 表示；
- $E_i$ 是第 $i$ 个专家（通常是 FFN）；
- $g(x)=\mathrm{softmax}(xW_g)$ 是门控分数；
- TopK 只保留分数最高的 $k$ 个专家，其余不计算。

常见设置：$k=1$ 或 $2$（早期），现代大模型常用 top-8（DeepSeek-V3）、top-2（Mixtral）。

### 4.2 总参数 vs 激活参数

假设有 $E$ 个专家，每个专家参数量为 $N_e$，稠密部分（注意力、embedding）为 $N_d$：

$$
N_{\text{total}}=N_d+E\cdot N_e,\qquad
N_{\text{active}}=N_d+k\cdot N_e
$$

DeepSeek-V3 的例子（公开资料）：总参数 671B，每 token 激活 37B，约 5.5% 的激活率。这意味着：

- 训练与推理的**显存**要容纳全部专家（671B）；
- 单 token 的**计算量**接近 37B 稠密模型；
- 通信与路由成为新的瓶颈。

> 面试常见错误：说“MoE 参数量小所以显存小”。MoE 省的是计算，不是显存；所有专家都要加载，除非做专家卸载（offload），但会引入延迟。

### 4.3 负载均衡

如果门控总是选少数专家，会出现“赢家通吃”：热门专家过载、冷门专家得不到训练。常用手段：

**辅助损失（auxiliary loss）**

$$
\mathcal{L}_{\text{aux}}=\alpha\cdot E\sum_{i=1}^{E}f_i\cdot P_i
$$

其中 $f_i$ 是路由到专家 $i$ 的 token 比例，$P_i$ 是门控给专家 $i$ 的平均概率。最小化它可以让路由分布更均匀。

**容量因子（capacity factor）**

每个专家最多处理 $C=\mathrm{capacity\_factor}\times\frac{\text{token 数}}{E}$ 个 token，超出部分被丢弃（token dropping）或走残差。容量因子越大越不丢 token，但显存与计算浪费越多。

**无辅助损失的负载均衡（aux-loss-free）**

DeepSeek-V3 的做法：给每个专家一个偏置 $b_i$，只用于路由选择，不参与梯度；根据负载动态调整 $b_i$。这样避免辅助损失与语言建模目标的冲突。

**共享专家**

设置一个始终激活的共享专家，捕捉通用知识；路由专家专注差异化知识，减少冗余。

### 4.4 专家并行与通信

MoE 训练通常使用专家并行（Expert Parallelism, EP）：不同专家放在不同 GPU 上，token 通过 all-to-all 发送到对应专家，计算后再 all-to-all 返回。

通信特点：

- all-to-all 的通信量与 batch size、序列长度、隐藏维度、top-k 成正比；
- 负载不均会导致某些 GPU 等待；
- 需要与数据并行、张量并行组合，形成复杂的 3D/4D 并行。

推理时的挑战：

- 所有专家常驻显存，单卡放不下时需要多卡 EP；
- 路由是动态的，batch 内不同 token 去不同专家，难以做静态图优化；
- 需要 grouped GEMM、专家缓存、token 重排等 kernel 优化。

### 4.5 MoE 的优缺点

优点：

- 在相同计算预算下，总容量更大，质量通常优于同激活参数的稠密模型；
- 推理时只激活部分专家，单位 token 计算低；
- 天然适合多任务/多语言，专家可能分化。

缺点：

- 显存占用高，部署成本高；
- 路由不稳定、负载不均、训练更难；
- 通信开销大，分布式实现复杂；
- 微调时容易过拟合少数专家，需要谨慎设置学习率与路由；
- 推理框架支持不如稠密模型成熟。

### 4.6 代码：一个简化的 MoE 层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.top_k = top_k

    def forward(self, x):
        # x: [batch, seq, d_model]
        b, s, d = x.shape
        logits = self.gate(x)                       # [b, s, E]
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_idx = probs.topk(self.top_k, dim=-1)  # [b, s, k]

        # 简化实现：逐 token 循环，生产环境应使用 scatter/gather + grouped GEMM
        out = torch.zeros_like(x)
        flat_x = x.view(-1, d)
        flat_out = out.view(-1, d)
        flat_p = topk_p.view(-1, self.top_k)
        flat_i = topk_idx.view(-1, self.top_k)

        for token in range(flat_x.size(0)):
            for j in range(self.top_k):
                expert_id = flat_i[token, j].item()
                weight = flat_p[token, j]
                flat_out[token] += weight * self.experts[expert_id](flat_x[token])

        return out
```

> 这段代码只用于理解路由逻辑，效率很低。真实实现会按专家分组 token，做批量 GEMM，并处理容量因子与负载均衡。

## 5. MLA：多头潜在注意力

### 5.1 动机

标准 MHA 的 KV Cache 随头数线性增长。GQA 通过共享 KV 头降低缓存，但压缩率有限。MLA（Multi-head Latent Attention）换了一个思路：不直接缓存完整的 K/V，而是缓存一个低秩潜向量，推理时再投影出 K/V。

### 5.2 结构

对输入 $h_t$：

$$
c_t^{KV}=W^{DKV}h_t,\qquad
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
$$

其中 $c_t^{KV}\in\mathbb{R}^{d_c}$ 是压缩后的潜向量，$d_c\ll n_h d_h$。推理时只需缓存 $c_t^{KV}$（以及解耦的 RoPE 部分 $k_t^R$），不必缓存每个头的完整 K/V。

DeepSeek-V2 的公开数据：每 token 每层的 KV Cache 从 MHA 的 $2n_hd_h$ 降到 $d_c+d_R\approx576$ 维，压缩率约 93%（对比 MHA），同时通过吸收投影矩阵保持表达能力。

### 5.3 优缺点

优点：

- KV Cache 大幅降低，长上下文推理更省显存与带宽；
- 通过低秩压缩，参数量与计算量可控；
- 与 RoPE 解耦，位置编码仍可正常使用。

缺点：

- 实现复杂，训练时需要多组投影；
- 推理时要把潜向量投影回 K/V，增加计算（可吸收到矩阵乘中优化）；
- 对框架支持要求高。

## 6. 多模态架构

### 6.1 主流三段式

$$
\text{图像}\rightarrow\text{视觉编码器}\rightarrow\text{投影层}\rightarrow\text{语言模型}
$$

- **视觉编码器**：ViT、SigLIP、CLIP ViT，把图像切成 patch，输出 patch 特征。
- **投影层**：MLP、Q-Former、Perceiver Resampler，把视觉特征映射到语言模型维度，并控制视觉 token 数量。
- **语言模型**：把视觉 token 与文本 token 拼接成同一序列，做因果注意力。

代表：

- CLIP：图文对比学习，双塔结构，不是生成模型，但提供强大的视觉-文本对齐表示。
- LLaVA：ViT + 线性/MLP 投影 + Llama，两阶段训练（对齐投影层 → 指令微调）。
- Qwen-VL：ViT + 可学习 query 的 resampler + Qwen，支持 grounding、OCR。
- GPT-4V/Gemini：闭源，细节未公开。

### 6.2 关键设计问题

1. **视觉 token 数量**：高分辨率图像 patch 多，token 多，会挤占上下文与增加计算。解决：patch merging、resampler、动态分辨率、图像分块。
2. **融合方式**：早期融合（像素进 Transformer）vs 晚期融合（各自编码后交叉注意力）vs 中期融合（投影后拼接）。主流是“编码器 + 投影 + 拼接”。
3. **训练阶段**：先冻结语言模型训练投影层（对齐），再联合微调（多模态指令），最后可能做偏好对齐。
4. **位置编码**：图像 patch 需要二维位置编码或行列位置编码，与文本的 RoPE 区分。
5. **评测**：VQA、OCR、图表理解、grounding、视频时序理解；要区分“看得见”和“推理对”。

### 6.3 视频与音频

- 视频：按帧采样 + 时序位置编码 + 时间压缩；长视频需要 token 剪枝/层次化记忆。
- 音频：mel 频谱 → 音频编码器 → 投影 → 语言模型；或语音 tokenizer 离散化后当 token。
- 全模态：统一 token 空间，文本/图像/音频/视频都映射为 token，做同一个自回归目标。

## 7. 长上下文架构

长上下文能力不是单一模块，而是多层配合：

| 层面 | 手段 | 代价 |
| --- | --- | --- |
| 位置编码 | PI、NTK、YaRN、LongRoPE | 外推可能损失局部分辨率 |
| 注意力计算 | FlashAttention、Ring Attention、序列并行 | 实现复杂，通信开销 |
| 注意力模式 | 滑动窗口、稀疏、全局+局部交替 | 可能丢失长程依赖 |
| KV 管理 | GQA/MLA、KV 量化、PagedAttention、prefix cache | 精度/实现复杂度 |
| 训练数据 | 长文本继续预训练、长依赖任务 | 数据稀缺、成本高 |
| 评测 | Needle-in-a-Haystack、RULER、LongBench | 指标与真实能力有差距 |

一个常见误区：把“支持 128K 上下文”等同于“能在 128K 上推理”。前者是窗口大小，后者需要在长距离检索、多跳、指代、摘要等任务上验证。

## 8. 推理模型与 test-time compute

传统扩展靠“更大模型 + 更多数据”；推理模型引入第三个维度：**推理时计算**（test-time compute）。

- 训练：用可验证奖励（答案对错、单元测试通过）做 RL，鼓励模型生成长链推理。
- 推理：允许模型“思考”更多 token，或用多次采样 + 验证器选择。
- 结构：仍然是 Decoder-Only，变化主要在后训练目标与解码策略，而不是全新架构。
- 代价：思考 token 增加延迟与成本；需要控制“过度思考”。

面试点：推理模型的“思考”不是内部隐藏状态无限循环，而是把中间步骤写成 token，利用自回归计算换取更深的有效计算深度。

## 9. 架构选择的决策框架

面对“选什么架构”的问题，可以按以下顺序回答：

1. **任务形态**：理解/抽取 → Encoder 系；生成/对话/Agent → Decoder-Only；翻译/结构化转换 → Encoder-Decoder 可考虑。
2. **算力与成本**：训练预算、推理 QPS、延迟、显存；MoE 适合大容量低激活，但部署复杂。
3. **上下文长度**：长上下文优先 GQA/MLA + FlashAttention + 位置外推 + 长文本数据。
4. **多模态**：是否需要视觉/音频/视频；选择成熟的开源基座或闭源 API。
5. **生态与合规**：框架支持、license、数据出境、私有化部署。
6. **迭代速度**：从零预训练 vs 继续预训练 vs 微调 vs RAG，成本差异巨大。

## 10. 面试高频问题

### Q1：BERT、T5、GPT 三种架构的区别？为什么现在主流是 Decoder-Only？
- BERT 是 Encoder-Only + MLM，双向理解强，不擅生成。
- T5 是 Encoder-Decoder + span corruption，适合输入输出分离的转换任务。
- GPT 是 Decoder-Only + 因果 LM，任务统一、数据效率高、工程简单。
- 生成式助手选择 Decoder-Only，是数据规模、任务统一性、推理工程共同作用的结果。

### Q2：MoE 为什么能“用更少算力获得更大容量”？
- 总参数 $N_{\text{total}}$ 决定容量，激活参数 $N_{\text{active}}$ 决定单 token 计算。
- 门控只选 top-k 专家，计算量约为稠密模型的 $N_{\text{active}}/N_{\text{total}}$。
- 但显存要容纳全部专家，通信与负载均衡是代价。

### Q3：MoE 的负载均衡怎么做？
- 辅助损失：鼓励路由分布均匀。
- 容量因子：限制每个专家处理的 token 数，超出丢弃或走残差。
- 无辅助损失：用路由偏置动态调整，避免与 LM 目标冲突。
- 共享专家：承担通用知识，减少路由专家冗余。

### Q4：MoE 训练/推理的瓶颈分别是什么？
- 训练：all-to-all 通信、负载不均、路由不稳定、专家并行与 TP/DP 的组合。
- 推理：全部专家驻留显存、动态路由难以静态优化、grouped GEMM 与 token 重排开销。
- 共同点：显存容量与带宽，而不是单纯 FLOPs。

### Q5：MLA 和 GQA 的区别？
- GQA 让多个 Query 头共享一组 K/V 头，直接减少 KV 头数。
- MLA 把 K/V 压缩成低秩潜向量，推理时缓存潜向量，再投影回 K/V。
- MLA 压缩率更高，但实现复杂；GQA 简单成熟，框架支持好。

### Q6：多模态模型的基本结构是什么？
- 视觉编码器（ViT/SigLIP）+ 投影层（MLP/Q-Former）+ 语言模型。
- 图像 patch 编码后映射为视觉 token，与文本 token 拼接，做因果注意力。
- 训练分阶段：对齐投影层 → 多模态指令微调 → 偏好对齐。

### Q7：长上下文模型需要哪些技术配合？
- 位置编码外推（PI/NTK/YaRN）。
- 高效注意力（FlashAttention/Ring Attention/序列并行）。
- KV Cache 压缩（GQA/MLA/量化/分页）。
- 长文本训练数据与长依赖任务。
- 科学评测（检索、多跳、摘要、指代）。

### Q8：推理模型和普通模型在架构上有什么不同？
- 架构通常相同，区别在后训练目标（RLVR/可验证奖励）与解码策略（长 CoT、多次采样）。
- 推理时计算成为新的扩展维度，但受延迟与成本约束。
- 需要控制过度思考，并用验证器/早停提升效率。

### Q9：如果让你设计一个 100B 级模型，你会选稠密还是 MoE？
- 先看推理预算：MoE 激活参数低，单 token 算力省，但显存与部署复杂。
- 看训练稳定性与团队工程能力：MoE 的路由、通信、负载均衡门槛高。
- 看上下文与 KV：MLA/GQA 影响长上下文成本。
- 给出取舍：追求极致质量且部署资源充足 → MoE；追求简单可靠、中等规模 → 稠密 + GQA。

### Q10：架构演进的主线是什么？
- 在给定算力/推理预算下提升有效容量与训练效率。
- 从“结构创新”转向“规模 + 数据 + 后训练 + 系统工程”的综合竞争。
- 稀疏激活（MoE）、低秩压缩（MLA）、长上下文、多模态、推理时计算是当前几条主线。

> 本文档为面试复习资料，公式与数字建议结合原始论文复核。
