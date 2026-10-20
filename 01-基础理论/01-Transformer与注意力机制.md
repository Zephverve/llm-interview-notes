# Transformer 与注意力机制

> 本章导读：Transformer 是大模型面试的“必考地基”。面试官通常不会只问“注意力公式是什么”，而是沿着“为什么这样设计 → 复杂度多少 → 参数量怎么算 → 为什么能扩展”一路追问。本章从动机、公式、代码到变体，建立一套能自洽回答的框架。
>
> 阅读建议：先掌握第 3、4、5 节的公式与参数量推导，再对着第 7 节代码手写一遍，最后用第 10、11 节自测。

## 核心要点

- Transformer 的核心优势是**任意两个位置之间的信息交互路径长度为 1**，且序列维度可完全并行；代价是注意力对序列长度呈平方复杂度。
- 缩放点积注意力中除以 $\sqrt{d_k}$ 是为了控制点积方差，避免 softmax 进入饱和区导致梯度消失。
- 标准层参数量约为 $12d^2$：注意力 $4d^2$，FFN $8d^2$（$d_{ff}=4d$ 时）。
- 多头注意力不是简单重复，而是让不同头在不同子空间学习不同的关系模式；头数与 $d_{head}$ 需要联合设计。
- Pre-LN 是现代大模型的默认选择，训练更稳定；Post-LN 表达力略强但对 warmup 更敏感。
- KV Cache 是自回归推理的核心优化，其显存随层数、KV 头数、上下文长度线性增长，是长上下文部署的主要成本。
- FlashAttention 是 IO 感知的精确注意力实现，不是近似算法；它通过分块和在线 softmax 避免物化 $n\times n$ 矩阵。
- “注意力权重可解释”是一个常见误区；注意力权重只是信息路由的一个侧面，不能直接等同于模型决策依据。

## 1. 为什么是 Transformer

在 Transformer 之前，序列建模主要依赖 RNN/LSTM 和 CNN。

| 结构 | 序列并行 | 最长依赖路径 | 单层计算复杂度 | 归纳偏置 |
| --- | --- | --- | --- | --- |
| RNN / LSTM | 否，必须逐步递推 | $O(n)$ | $O(n \cdot d^2)$ | 强时序、局部性 |
| CNN（因果卷积） | 是 | $O(\log_k n)$（需堆叠） | $O(k \cdot n \cdot d^2)$ | 局部性、平移等变 |
| Transformer | 是 | $O(1)$ | $O(n^2 d + n d^2)$ | 弱，依赖数据学习 |

三个关键点：

1. **并行性**：训练时整条序列一次性输入，GPU 利用率高，才能扩展到万亿 token 级别的数据。
2. **长依赖**：任意两个 token 通过一次注意力即可交互，梯度传播路径短，缓解了 RNN 的长程遗忘。
3. **可扩展性**：结构规整（重复堆叠同一层），参数量、层数、宽度可以按比例放大，配合残差连接训练深层网络。

代价也很明确：注意力复杂度随序列长度平方增长；Transformer 本身没有位置概念，必须显式注入位置信息；对数据量和算力的需求远高于之前的模型。

> 面试表达：不要只说“Transformer 比 RNN 好”，要落到“并行度、依赖路径、归纳偏置、计算复杂度”四个维度做取舍。

## 2. 整体结构：从 Encoder-Decoder 到 Decoder-Only

原始 Transformer（Vaswani et al., 2017）是 Encoder-Decoder 结构，用于机器翻译：

- Encoder：多层自注意力 + FFN，双向可见，输出上下文表示。
- Decoder：掩码自注意力 + 对 Encoder 的交叉注意力 + FFN，自回归生成。
- 交叉注意力中，Query 来自 Decoder，Key/Value 来自 Encoder。

今天的生成式大模型绝大多数是 **Decoder-Only**：

- 只有掩码自注意力，没有交叉注意力；
- 训练目标是下一个 token 预测；
- 所有任务（问答、翻译、代码、工具调用）都被统一成“续写”形式。

为什么 Decoder-Only 成为主流？常见回答：

1. 训练目标统一，数据无需成对的“输入-输出”标注，任何文本都能用于预训练。
2. 推理时 KV Cache 结构简单，工程实现成熟。
3. 规模扩展后，few-shot / in-context learning 能力足以覆盖大量任务，不需要为每个任务设计特定结构。
4. 与 Encoder-Decoder 相比，在相同参数量下生成任务更直接；Encoder 的双向表示优势可以通过前缀（prefix）方式近似。

一个有用的心智模型是**残差流（residual stream）**：把每一层的输入输出看成一个共享的向量通道，注意力和 FFN 是挂在通道上的“读取-计算-写回”模块。层与层之间通过残差连接传递信息，归一化则控制通道的数值尺度。

## 3. 缩放点积注意力

### 3.1 直觉：可微的软检索

把每个 token 的表示投影成三个向量：

- Query（查询）：我想找什么信息；
- Key（键）：我能提供什么信息的“索引”；
- Value（值）：我实际携带的内容。

用 Query 和所有 Key 做相似度匹配，softmax 归一化成权重，再对 Value 加权求和。整个过程可微，因此可以端到端训练。

### 3.2 公式与维度

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

- $Q\in\mathbb{R}^{n\times d_k}$：$n$ 个查询向量；
- $K\in\mathbb{R}^{m\times d_k}$：$m$ 个键向量（自注意力中 $m=n$）；
- $V\in\mathbb{R}^{m\times d_v}$：$m$ 个值向量；
- $QK^\top\in\mathbb{R}^{n\times m}$：相似度矩阵；
- softmax 按行归一化，得到每个查询对所有键的注意力分布；
- 输出为 $\mathbb{R}^{n\times d_v}$。

### 3.3 为什么除以 $\sqrt{d_k}$

假设 $q$ 和 $k$ 的每个分量独立、均值为 0、方差为 1，则点积

$$
q\cdot k=\sum_{i=1}^{d_k} q_i k_i
$$

的均值为 0，方差为 $d_k$，标准差为 $\sqrt{d_k}$。当 $d_k$ 较大时（例如 128），点积的数值范围很宽，softmax 容易把最大的那个分数推到接近 1，其余接近 0，进入饱和区。此时 softmax 的雅可比矩阵接近 0，梯度消失，训练难以进行。

除以 $\sqrt{d_k}$ 后，点积方差被拉回约 1，softmax 的输入分布保持在较平缓的区域，梯度更稳定。

> 追问：为什么不是除以 $d_k$？因为标准差是 $\sqrt{d_k}$，除以 $d_k$ 会把方差压到 $1/d_k$，softmax 过于平滑，损失区分度。

### 3.4 掩码（Mask）

掩码通过在 softmax 前加上一个很大的负数实现：

$$
\mathrm{softmax}(S+M),\qquad M_{ij}=\begin{cases}0 & \text{允许}\\-\infty & \text{禁止}\end{cases}
$$

两类常见掩码：

- **Padding Mask**：忽略 batch 中补齐的无效 token。
- **Causal Mask**：下三角为 0、上三角为 $-\infty$，保证位置 $t$ 只能看到 $\le t$ 的 token。

工程细节：

- 用 `-inf` 时，如果某一行全被 mask，softmax 会出现 `NaN`；需要保证每行至少有一个可见位置。
- 混合精度下建议用 `-1e9` 或 `torch.finfo(dtype).min`，避免 `-inf` 参与运算产生 `NaN`。
- 实现上通常把 mask 加在 logits 上，而不是对概率做乘法，后者会破坏归一化。

### 3.5 复杂度

设序列长度为 $n$，模型维度为 $d$，头数为 $h$，$d_{head}=d/h$。

| 操作 | 计算量 | 显存（若不融合） |
| --- | --- | --- |
| $QK^\top$ | $O(n^2 d)$ | $O(n^2 h)$ |
| softmax | $O(n^2 h)$ | $O(n^2 h)$ |
| 加权求和 | $O(n^2 d)$ | $O(n d)$ |
| 四个投影 | $O(n d^2)$ | $O(n d)$ |

当 $n \gg d$ 时，$O(n^2 d)$ 主导；当 $n$ 较小、$d$ 很大时，投影的 $O(n d^2)$ 不可忽略。显存方面，朴素实现需要存储每层的 $n\times n$ 注意力矩阵，长上下文下这是主要瓶颈之一；FlashAttention 通过分块避免物化，把显存降到 $O(n)$。

## 4. 多头注意力

### 4.1 公式

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O
$$

$$
\mathrm{head}_i=\mathrm{Attention}(QW_{Q_i},KW_{K_i},VW_{V_i})
$$

其中 $W_{Q_i},W_{K_i},W_{V_i}\in\mathbb{R}^{d\times d_{head}}$，$W_O\in\mathbb{R}^{d\times d}$。

### 4.2 参数量

四个投影矩阵均为 $d\times d$：

$$
\underbrace{3d^2}_{\text{QKV}}+\underbrace{d^2}_{W_O}=4d^2
$$

### 4.3 为什么要多头

- 单个注意力头只能学习一种“相似度度量”；多个头可以在不同子空间关注不同关系（语法、指代、位置、实体等）。
- 类似 ensemble：多个头的输出拼接后再线性变换，增加表示能力。
- 头数 $h$ 与每个头的维度 $d_{head}$ 需要平衡：总计算量近似不变，但 $d_{head}$ 太小会限制单头表达能力，太大则头数不足。

研究观察：许多头在训练后可以被剪枝而不显著损失性能，说明存在冗余；但“多头完全等价于单头”并不成立。

### 4.4 代码：多头自注意力

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 一次线性变换拆成 Q/K/V，工程上比三个独立矩阵更高效
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        b, n, _ = x.shape

        qkv = self.qkv(x)                       # [b, n, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)          # 各 [b, n, d]
        # 拆头: [b, n, h, d_head] -> [b, h, n, d_head]
        q = q.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

        # 缩放点积注意力；scale 用 1/sqrt(d_head) 控制方差
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [b, h, n, n]

        if causal:
            # 上三角为 True 的位置需要被屏蔽
            mask = torch.triu(
                torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v                          # [b, h, n, d_head]
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.out(out)
```

> 这段代码用于理解原理。生产环境应使用 `F.scaled_dot_product_attention` 或 FlashAttention，它们会做算子融合与显存优化。

## 5. 前馈网络（FFN）

### 5.1 标准形式

$$
\mathrm{FFN}(x)=W_2\,\sigma(W_1 x+b_1)+b_2
$$

通常 $d_{ff}=4d$，两个矩阵的参数量为：

$$
d\cdot 4d + 4d\cdot d = 8d^2
$$

### 5.2 SwiGLU

现代大模型常用门控线性单元：

$$
\mathrm{FFN}(x)=\big(\mathrm{Swish}(xW_1)\odot xW_3\big)W_2
$$

其中 $\mathrm{Swish}(z)=z\cdot\sigma(z)$。因为多了一个 $W_3$，为了保持参数量与 $4d$ 版本接近，通常取 $d_{ff}\approx \frac{8}{3}d$（再对齐到 128 的倍数）。Llama 系列即采用此结构。

### 5.3 FFN 的作用

一种流行观点把 FFN 视为“键值记忆”：第一层把输入映射到高维空间做模式匹配，激活函数筛选，第二层把匹配到的信息写回。经验上，模型的事实知识大量存储在 FFN 参数中，这也是模型编辑（model editing）和知识蒸馏关注 FFN 的原因。

## 6. 残差连接与归一化

### 6.1 残差

$$
x_{l+1}=x_l+\mathrm{Sublayer}(\mathrm{Norm}(x_l))
$$

残差连接让梯度可以跨层直接回传，是训练深层网络的前提。没有残差，几十层的 Transformer 几乎无法稳定训练。

### 6.2 Post-LN 与 Pre-LN

| 结构 | 公式 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Post-LN | $\mathrm{LN}(x+\mathrm{Sublayer}(x))$ | 表达力强、原始论文结构 | 对初始化与 warmup 敏感，深层易不稳定 |
| Pre-LN | $x+\mathrm{Sublayer}(\mathrm{LN}(x))$ | 训练稳定，可用较大学习率 | 表达力略弱，需在最后加一层 LN |

现代大模型基本采用 Pre-LN（或 RMSNorm + Pre-Norm）。原因很简单：规模变大后，训练稳定性优先。

### 6.3 LayerNorm 与 RMSNorm

LayerNorm：

$$
\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm 去掉了均值中心化和偏置：

$$
\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2+\epsilon}}
$$

RMSNorm 的计算更简单、少一次归约，且在实践中效果与 LayerNorm 相当，因而被 Llama、Qwen、Gemma 等广泛采用。

> 注意：LayerNorm/RMSNorm 是在特征维度上归一化，不是 batch 维度；因此与 batch size 无关，推理时行为一致。

## 7. 完整 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: 先归一化再进子层，最后残差相加
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

一个 $L$ 层的 decoder-only 模型，参数量粗略为：

$$
N \approx L\cdot(12d^2) + V\cdot d
$$

其中 $V$ 为词表大小，$Vd$ 是 token embedding；若输出层与 embedding 共享权重（weight tying），则不额外增加参数量。以 $d=4096$、$L=32$、$V=128256$ 为例：

- 每层约 $12\times 4096^2\approx 2.01\times 10^8$；
- 32 层约 $6.44\times 10^9$；
- embedding 约 $128256\times 4096\approx 5.25\times 10^8$；
- 合计约 $6.97\times 10^9$，即约 7B 量级（实际 Llama 3 8B 因词表与 FFN 维度更大，约 8B）。

## 8. 训练与推理的差异

训练时使用 **teacher forcing**：整条序列一次性输入，每个位置并行预测下一个 token，损失为所有位置的交叉熵平均。因此训练可以充分利用 GPU 并行。

推理时必须自回归：每次生成一个 token，再把它拼回输入。朴素实现每步都要重新计算整个前缀，复杂度为 $O(n^3)$（每步 $O(n^2)$，共 $n$ 步）。KV Cache 通过缓存历史 token 的 Key/Value，把每步计算降到 $O(n d)$，是推理加速的基础。

这也带来一个训练-推理不一致问题：训练时模型看到的是真实前缀（teacher forcing），推理时看到的是自己生成的前缀，一旦早期出错会累积，称为 **exposure bias**。大模型通过海量数据和 RL 后训练在一定程度上缓解，但并未完全消除。

## 9. 注意力变体地图

| 变体 | 核心思想 | KV Cache | 代表模型 | 取舍 |
| --- | --- | --- | --- | --- |
| MHA | 每个头独立的 K/V | $2Lhd_h$ | GPT-3、BERT | 表达力强，KV 显存大 |
| MQA | 所有头共享一组 K/V | $2Ld_h$ | PaLM、Falcon | KV 显存降为 $1/h$，质量略降 |
| GQA | 每组 $g$ 个头共享 K/V | $2Lg d_h$ | Llama 2/3、Qwen2.5 | 质量与显存的折中 |
| MLA | 低秩压缩 K/V 到潜向量 | 压缩后的 latent | DeepSeek-V2/V3 | 压缩率高，实现复杂 |
| Sliding Window | 只关注最近 $w$ 个 token | $2Lwd_h$ | Mistral、Gemma | 长上下文近似，需堆叠感受野 |
| Sparse Attention | 只计算部分位置对 | 视模式而定 | Longformer、BigBird | 降低复杂度，但工程与效果权衡 |
| Linear Attention | 用核函数近似 softmax | 固定大小状态 | RWKV、Mamba（非 Transformer） | 线性复杂度，长程记忆有损 |

需要强调：**FlashAttention 不属于数学上的新注意力**，它计算的是与标准注意力完全相同的精确结果，只是通过分块（tiling）、在线 softmax（online softmax）和重计算减少 HBM 读写。把 FlashAttention 说成“稀疏注意力”是常见错误。

## 10. 常见误区与追问

**误区 1：注意力权重就是解释。**
注意力权重只反映“当前层、当前头”的信息路由，后续层、FFN、残差都会改变信息；而且多头之间存在冗余。可以用注意力做粗粒度分析，但不能直接作为模型决策的因果解释。

**误区 2：Transformer 的瓶颈只有 $O(n^2)$。**
在中等长度下，FFN 的参数量与内存带宽往往才是瓶颈；自回归 decode 阶段是典型的 memory-bound，算力利用率很低。优化要分 prefill 和 decode 两个阶段看。

**误区 3：多头越多越好。**
头数增加会降低每个头的维度，且带来更多 KV Cache（MHA 下）和通信开销。GQA 的出现说明“减少 KV 头”在工程上更划算。

**误区 4：参数量公式 $12Ld^2$ 适用于所有模型。**
它假设 $d_{ff}=4d$、无 bias、MHA。使用 SwiGLU、GQA、MoE、MLA 时都要修正。面试时先说明假设，再给数量级。

**追问：为什么 decoder-only 模型训练时能看到未来 token 吗？**
不能。因果掩码保证位置 $t$ 的注意力只覆盖 $\le t$ 的位置，下一个 token 的预测只依赖前缀。

**追问：残差连接为什么能缓解梯度消失？**
梯度沿残差路径回传时包含恒等项，连乘的雅可比不会全部小于 1；同时残差流为信息提供了一条“高速公路”，深层的梯度可以直接传到浅层。

## 11. 面试高频问题

### Q1：手推缩放点积注意力，并解释每个矩阵的维度。
- 输入 $X\in\mathbb{R}^{n\times d}$，投影得 $Q=XW_Q$、$K=XW_K$、$V=XW_V$，均为 $n\times d_k$（$d_v$ 可不同）。
- 计算 $S=QK^\top/\sqrt{d_k}$，$S\in\mathbb{R}^{n\times n}$。
- 对 $S$ 按行 softmax 得 $A$，输出 $O=AV\in\mathbb{R}^{n\times d_v}$。
- 多头时把 $d$ 拆成 $h$ 个 $d/h$，各自计算后拼接，再乘 $W_O$。

### Q2：为什么除以 $\sqrt{d_k}$，不除会怎样？
- 点积方差随 $d_k$ 线性增长，标准差为 $\sqrt{d_k}$。
- 不缩放时 softmax 输入过大，分布尖锐，梯度接近 0，训练不稳定。
- 缩放后方差约为 1，保持 softmax 在敏感区间。

### Q3：MHA、MQA、GQA 的区别？为什么 Llama 用 GQA？
- MHA：每个头独立 K/V，KV Cache 最大。
- MQA：所有头共享一组 K/V，KV Cache 最小，但质量下降明显。
- GQA：分组共享，组数 $g$ 可调，在质量与显存之间折中。
- Llama 2/3 使用 GQA 主要是为了在长上下文、大 batch 推理时降低 KV Cache 显存和带宽压力。

### Q4：Pre-LN 和 Post-LN 的区别？为什么现在都用 Pre-LN？
- Post-LN 在残差后归一化，深层梯度容易不稳定，需要 warmup。
- Pre-LN 在子层前归一化，残差路径干净，梯度更稳，适合大规模训练。
- Pre-LN 需要在最后加一层 LN，且同等深度下表达力略弱。

### Q5：Transformer 的参数量怎么估算？
- 单层注意力 $4d^2$，FFN $8d^2$（$d_{ff}=4d$），共 $12d^2$。
- $L$ 层为 $12Ld^2$，加词表 embedding $Vd$（若 weight tying 则只算一次）。
- SwiGLU 的 FFN 为 $3\times d\times d_{ff}$，取 $d_{ff}=8d/3$ 时仍约为 $8d^2$。
- 给出假设，再给数量级，不要背死数字。

### Q6：注意力复杂度是多少？长上下文怎么优化？
- 时间 $O(n^2d)$，朴素显存 $O(n^2)$。
- 优化方向：FlashAttention（IO 优化、精确计算）、GQA/MLA（压缩 KV）、滑动窗口/稀疏注意力（减少计算对）、线性注意力/状态空间模型（改变结构）、序列并行与 Ring Attention（分布式切分）。
- 面试时要区分“训练侧”和“推理侧”：训练侧关注算力与显存，推理侧 decode 关注带宽。

### Q7：为什么 Transformer 需要位置编码？
- 自注意力对输入顺序是置换不变的：打乱 token 顺序，输出只是对应打乱，模型无法感知顺序。
- 因此必须注入位置信息，常见有绝对位置编码、相对位置偏置、RoPE、ALiBi。
- 细节见《位置编码、归一化与激活函数》。

### Q8：FlashAttention 和标准注意力结果一样吗？
- 一样，是精确计算，不是近似。
- 它通过分块计算和在线 softmax，避免把 $n\times n$ 矩阵写入 HBM，减少显存读写。
- 速度提升来自 IO 优化，显存从 $O(n^2)$ 降到 $O(n)$。

### Q9：注意力头越多越好吗？
- 不是。头数增加会降低单头维度，且 MHA 下 KV Cache 与头数成正比。
- 实践中通过 GQA 减少 KV 头，同时保持 Query 头数；头数、维度、层数需要联合调参。
- 训练后很多头可以被剪枝，说明存在冗余，但不代表多头无意义。

### Q10：如果让你从零实现一个 Transformer，你会注意哪些工程细节？
- 掩码的正确性（causal + padding），避免全 mask 行产生 NaN。
- 权重初始化（残差分支缩放）、Pre-LN、warmup、梯度裁剪。
- 混合精度与 loss scaling，attention 用融合算子。
- 位置编码的数值稳定性与长度外推。
- 训练吞吐：packing、激活重计算、分布式并行。
- 评测与可复现性：固定 seed、记录数据版本、监控 loss spike。

> 本文档为面试复习资料，公式与数字建议结合原始论文复核。
