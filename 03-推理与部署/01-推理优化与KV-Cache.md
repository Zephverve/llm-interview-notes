# 推理优化与 KV Cache：从 Roofline 到 PD 分离

> 本章导读
> 大模型推理分为 prefill 与 decode 两个阶段，前者像大矩阵乘法，后者像逐 token 的矩阵向量乘法，瓶颈截然不同。
> 理解 KV Cache 的显存公式与访存行为，是解释“长上下文为什么贵、并发为什么上不去”的基础。
> 本文串联 Roofline、MQA/GQA/MLA、PagedAttention、FlashAttention、连续批处理、PD 分离与投机解码。
> 数字均为量级/估算，快速变化内容以公开资料为准，落地前应以目标硬件实测为准。

## 核心要点

- prefill 一次处理整个 prompt，计算密集；decode 每步只生成一个 token，通常受显存带宽限制。
- KV Cache 大小与层数、KV 头数、头维度、上下文长度、批大小成正比，长上下文下可超过模型权重。
- MQA 把 KV 头压到 1 个，GQA 折中为若干组，MLA 用低秩潜变量压缩 KV，压缩率与实现复杂度依次上升。
- PagedAttention 用固定大小物理块和块表管理 KV，消除外部碎片，并支持前缀共享与写时复制。
- FlashAttention 通过分块与 online softmax 避免物化 $N \times N$ 注意力矩阵，数学上仍是精确注意力。
- 连续批处理按迭代调度请求，chunked prefill 把长 prompt 切片，PD 分离让 prefill 与 decode 使用独立资源池。
- 投机解码（speculative decoding）用草稿模型提出候选，再由目标模型一次验证，加速比由接受率、草稿长度和相对成本决定。
- 采样参数的作用都是重塑概率分布，temperature、top-k、top-p、min-p 存在先后与叠加关系。

## 1. 自回归推理的两个阶段

### 1.1 prefill 与 decode 的计算特征

给定长度为 $S$ 的 prompt，prefill 一次性计算所有位置的隐藏状态。线性投影为 $Y = XW$，其中 $X \in \mathbb{R}^{S \times d_{model}}$，$W \in \mathbb{R}^{d_{model} \times d_{ff}}$。prefill 计算量近似为：

$$
\mathrm{FLOPs}_{prefill} \approx 2PS + 4L S^2 d_{model}
$$

$P$ 是参数量，$L$ 是层数。第一项是权重矩阵乘法，第二项是注意力中 $QK^T$ 与 $AV$ 的贡献；4K 上下文时注意力只占几个百分点，128K 时不可忽略。

decode 每步只处理 1 个新 token，计算量约 $2P$ FLOPs，却要读取约 $2P$ 字节的 FP16 权重，算术强度接近 1 FLOP/Byte。现代 GPU 的拐点远高于此，因此 decode 是 memory-bound（显存带宽受限）。

| 维度 | prefill | decode |
| --- | --- | --- |
| 一次处理的 token 数 | $S$ 个，可并行 | 每序列 1 个 |
| 主要算子 | GEMM | GEMV 或窄 GEMM |
| 瓶颈 | 算力、张量核心利用率 | 显存带宽、KV 读取 |
| 典型延迟指标 | 首 token 延迟（TTFT） | 每 token 耗时（TPOT）/ token 间隔（ITL） |
| 批处理收益 | 提高算力利用率 | 分摊权重读取，但 KV 流量随批增大 |
| 优化手段 | FlashAttention、融合算子、chunked prefill | KV 压缩、量化、投机解码、连续批处理 |

### 1.2 Roofline 分析

Roofline 用算力上限和带宽上限共同约束性能：$T \ge \max\left(\mathrm{FLOPs}/F_{peak},\ \mathrm{Bytes}/W\right)$。$T$ 是最短执行时间，$F_{peak}$ 是峰值算力（FLOP/s），$W$ 是显存带宽（Byte/s）。定义算术强度与拐点：

$$
I = \frac{\mathrm{FLOPs}}{\mathrm{Bytes}},\qquad I^{*} = \frac{F_{peak}}{W}
$$

当 $I < I^{*}$ 时为 memory-bound，反之为 compute-bound。以 H100 SXM 为例，FP16 稠密算力约 989 TFLOPS，HBM3 带宽约 3.35 TB/s，拐点约为 $989/3.35 \approx 295$ FLOP/Byte（量级估算，实际受缓存、kernel 效率和频率影响）。

decode 单步读取权重 $2P$ 字节、计算 $2P$ FLOPs，$I \approx 1$。批大小为 $B$、上下文为 $S$ 时：

$$
I_{decode} \approx \frac{2PB}{2P + B S m_{kv}}
$$

$m_{kv}$ 是每 token 的 KV 字节数。分子随批增大，分母中的 KV 项也随批增大；当 $B S m_{kv} \gg 2P$ 时，算术强度趋于 $2P/(S m_{kv})$，继续加批也无法脱离 memory-bound。decode 单步时间的下界为：

$$
T_{step} \ge \max\left(\frac{2P}{F_{peak}},\ \frac{2P + B S m_{kv}}{W}\right)
$$

单序列、短上下文时 $T_{step} \approx 2P/W$。7B 模型 FP16 权重约 14 GB，理论下界如下（量级估算，实际效率按 50% 到 70% 折算）：

| GPU | 显存带宽 | 单步下界 | 单序列理论上限 | 实测常见范围 |
| --- | --- | --- | --- | --- |
| H100 80GB | 3.35 TB/s | 约 4.2 ms | 约 240 tok/s | 100 到 160 tok/s |
| A100 80GB | 2.0 TB/s | 约 7.0 ms | 约 143 tok/s | 60 到 100 tok/s |
| L40S 48GB | 864 GB/s | 约 16.2 ms | 约 62 tok/s | 30 到 45 tok/s |
| RTX 4090 24GB | 1008 GB/s | 约 13.9 ms | 约 72 tok/s | 40 到 60 tok/s |

单用户对话的 decode 速度主要由“权重字节数除以带宽”决定，换更快的卡或更小的模型，比换更多卡更直接。张量并行（tensor parallelism, TP）能分摊权重，但会引入通信开销，单序列延迟未必线性改善。

## 2. KV Cache

### 2.1 为什么需要与显存公式

若每步重算整个前缀，第 $t$ 步注意力计算量为 $O(t^2 d)$，累计到长度 $S$ 为 $O(S^3 d)$，MLP 还要重复 $O(S^2 P)$。缓存历史 K/V 后，每步只为新 token 计算一份 K/V 并读取历史缓存，累计注意力降为 $O(S^2 d)$。代价是显存：

$$
M_{KV} = 2 \times L \times n_{kv} \times d_{head} \times S \times B \times b_{kv}
$$

$L$ 是层数；$n_{kv}$ 是 Key/Value 头数（MHA 时等于查询头数，GQA/MQA 更小）；$d_{head}$ 是头维度；$S$ 是上下文长度（含已生成 token）；$B$ 是并发序列数；$b_{kv}$ 是每元素字节数，FP16 为 2，FP8/INT8 为 1；前面的 2 表示 K 和 V 各一份。MLA 则把 $2 n_{kv} d_{head}$ 换成 $d_c + d_R$，即压缩潜变量维度加解耦 RoPE 维度。

### 2.2 典型估算

下表按 FP16、单序列计算，配置取常见公开结构（7B：32 层、32 个查询头；70B：80 层、64 个查询头、GQA 8 个 KV 头），均为量级估算。

| 模型与配置 | 每 token KV | 4K 上下文 | 32K 上下文 | 32K 下 32 并发 |
| --- | --- | --- | --- | --- |
| 7B，MHA，32 KV 头 | 0.5 MiB | 2 GiB | 16 GiB | 512 GiB |
| 7B，GQA，8 KV 头 | 0.125 MiB | 0.5 GiB | 4 GiB | 128 GiB |
| 70B，GQA，8 KV 头 | 0.3125 MiB | 1.25 GiB | 10 GiB | 320 GiB |
| 70B，MHA，64 KV 头 | 2.5 MiB | 10 GiB | 80 GiB | 2560 GiB |

解读：7B 模型 32K 的 KV（16 GiB）已超过 FP16 权重（14 GiB），长上下文服务首先撞上 KV 墙；70B 不做 GQA 时 32K 单序列 80 GiB，单卡 H100 放不下；最后一列刻意给出不可行的组合，说明长上下文乘并发必须靠多卡、KV 量化或更强压缩。实际框架还要为每个序列预留长度、保留激活、CUDA Graph、通信缓冲和碎片，通常再留 10% 到 20%。

### 2.3 权重受限还是 KV 受限

FP16 下权重字节为 $2P$，KV 单步读取量为 $B S m_{kv}$，两者相等时 $B S m_{kv} = 2P$。7B MHA 的 $2P=14$ GB、$m_{kv}=0.5$ MiB，得到 $BS \approx 2.7\times10^{4}$ 个 token；70B GQA 约为 $4.3\times10^{5}$ 个 token。总在途 token 超过阈值后，KV 读取量超过权重读取量，容量规划的重点就要从权重转向 KV。

## 3. MQA、GQA 与 MLA

标准 MHA 中第 $i$ 个头的 K/V 为 $k_t^{(i)}=W_K^{(i)}h_t$、$v_t^{(i)}=W_V^{(i)}h_t$，$i=1,\dots,n_h$，每个头都缓存一份，显存随 $n_h$ 线性增长。

MQA（multi-query attention）让所有查询头共享一组 K/V，即 $n_{kv}=1$，KV 缩小为 $1/n_h$。代价是表达能力下降、训练稳定性变差，PaLM、Falcon 等使用过。GQA（grouped-query attention）把查询头分成 $G$ 组，每组共享一组 K/V，$n_{kv}=G$，第 $i$ 个头属于组 $g(i)=\lfloor i/(n_h/G)\rfloor$。$G=n_h$ 退化为 MHA，$G=1$ 退化为 MQA。Llama-2 70B、Llama-3 8B、Mistral 7B 都采用 GQA，常见配置是 32 个查询头、8 个 KV 头，KV 缩小 4 倍。GQA 论文的做法是用 MHA 权重对同组 K/V 做平均池化初始化，再继续预训练约 5% 的 token。

MLA（multi-head latent attention）由 DeepSeek-V2 提出，先把隐藏状态压缩成低维潜变量 $c_t^{KV}=W^{DKV}h_t$，再上投影出 $k_t^{C}=W^{UK}c_t^{KV}$、$v_t^{C}=W^{UV}c_t^{KV}$。RoPE 位置信息不能直接作用在压缩潜变量上，否则无法在推理时把上投影吸收进其他矩阵，因此额外保留解耦 RoPE Key：

$$
k_t^{R} = \mathrm{RoPE}(W^{KR} h_t)
$$

每个 token 每层只需缓存 $[c_t^{KV}; k_t^{R}]$，元素数为 $d_c + d_R$。推理时把 $W^{UK}$ 吸收进 $W_Q$、把 $W^{UV}$ 吸收进 $W_O$，避免显式展开每个头的 K/V，既省显存又不显著增加计算。

| 方案 | 每层每 token 元素数（$n_h=32$，$d_{head}=128$） | 相对 MHA | 特点 |
| --- | --- | --- | --- |
| MHA | 8192 | 1.0 | 质量好，KV 最大 |
| GQA，8 组 | 2048 | 0.25 | 质量与成本平衡，工业界主流 |
| MQA | 256 | 0.031 | KV 最小，质量与训练稳定性风险最高 |
| MLA，$d_c=512$，$d_R=64$ | 576 | 0.070 | 压缩率高，需吸收权重与解耦 RoPE |

DeepSeek-V2 论文报告，相比其 67B MHA 模型 KV Cache 减少约 93.3%。按每层 576 个元素粗算，相对 128 头 MHA 的压缩比可达 50 倍以上；具体随层数、头数和精度口径变化，引用时应以论文表格为准。MLA 的代价是需要定制吸收后的注意力 kernel，训练时也要保证低秩分解不损失太多表达能力。

## 4. PagedAttention、前缀缓存与 KV 量化

### 4.1 显存碎片与块表

朴素实现按“最大可能长度”为每个请求预留连续显存。请求长度差异大，预留过长产生内部碎片（internal fragmentation），结束后释放又产生外部碎片（external fragmentation）。vLLM 论文指出，已有系统在真实负载下可能有 60% 到 80% 的 KV 显存被浪费。

PagedAttention 借鉴操作系统虚拟内存分页：KV 切成固定大小的块（block，默认 16 个 token），每个序列维护块表（block table），把逻辑块号映射到可不连续的物理块号：

```text
序列 A 逻辑块:  [0]  [1]  [2]
块表:           [7]  [1]  [9]
物理块池: 0:B  1:A  2:B  3:C  4:D  5:E  6:B  7:A  8:D  9:A
```

分配器在生成新 token 时按需分配块。除最后一个块外每块填满，每个序列内部浪费不超过 $block\_size-1$ 个 token；块大小固定，外部碎片为零。vLLM 报告整体浪费可降到 4% 以下。块表还支持前缀共享（相同前缀指向同一组物理块）和写时复制（并行采样或 beam search 分叉时先共享、写入时再复制）。

工程上要权衡块大小：块太小，块表与元数据开销大、kernel 访存不连续；块太大，最后一个块浪费多。16 是常见默认值，长上下文也会尝试 32 或 64。

### 4.2 前缀缓存

对每个完整块计算链式哈希 $h_i = \mathrm{Hash}(h_{i-1},\ \text{tokens}_i,\ \text{model\_id},\ \text{adapter\_id})$，只有完整块才进入缓存，因为部分块内容还会变化。新请求从第一个块开始逐块查哈希，命中后直接复用物理块。SGLang 的 RadixAttention 用基数树（radix tree）组织前缀，支持更细粒度共享与 LRU 淘汰。

前缀缓存对多轮对话、固定系统提示、few-shot 模板和批量评测非常有效。注意：哈希必须包含模型版本、LoRA 适配器和分词器配置；位置编码与块内容绑定，不能跨位置复用；多租户共享可能造成侧信道泄露，生产系统通常按租户隔离；缓存本身也占显存，要设容量上限和淘汰策略。

### 4.3 KV Cache 量化

把 $b_{kv}$ 从 2 降到 1，显存和带宽都减半。Key 的离群值通常集中在某些通道，Value 的离群值更多随 token 变化，因此常见做法是对 Key 做 per-channel 量化、对 Value 做 per-token 量化，或使用分组量化。KIVI 等工作进一步把 Key/Value 压到 2 到 4 bit，并对近期 token 保留高精度。

经验规律（量级判断，需实测）：FP8/INT8 KV 通常只有很小质量损失，适合大多数在线服务；4 bit KV 在长上下文、检索、数学任务上更容易退化，需要逐层或分组缩放；2 bit 往往需要残差或混合精度；KV 量化与权重量化叠加时误差累积，必须端到端评估。

## 5. FlashAttention

### 5.1 标准注意力的问题与 online softmax

标准注意力 $S=QK^{T}/\sqrt{d}$、$P=\mathrm{softmax}(S)$、$O=PV$ 会把 $N \times N$ 的 $S$ 和 $P$ 写入 HBM，$N=32K$ 时单个头就是十亿级元素。FlashAttention 用 online softmax 把 softmax 变成可分块累加：维护当前最大值 $m$、指数和 $l$ 与未归一化输出 $O$，处理新块时

$$
m' = \max(m,\ \max_j s_j),\qquad l' = e^{m-m'} l + \textstyle\sum_j e^{s_j-m'}
$$

$$
O' = \frac{e^{m-m'} l\, O + \sum_j e^{s_j-m'} v_j}{l'}
$$

$s_j$ 是当前块内第 $j$ 个位置的分数，$v_j$ 是对应 Value。最大值更新时用 $e^{m-m'}$ 重新缩放旧累加结果，因此不需要保存完整 $S$。最终结果与一次性 softmax 数学等价，差异只来自浮点舍入。

```python
import torch

def flash_attention_forward(q, k, v, block_size=128):
    # q: [seq_len, head_dim]，简化为单个头、无因果掩码
    n, d = q.shape
    out = torch.zeros_like(q)
    for i in range(0, n, block_size):
        q_block = q[i:i + block_size]
        m = torch.full((q_block.shape[0], 1), -float("inf"), device=q.device)
        l = torch.zeros((q_block.shape[0], 1), device=q.device)
        acc = torch.zeros_like(q_block)
        for j in range(0, n, block_size):
            k_block, v_block = k[j:j + block_size], v[j:j + block_size]
            s = q_block @ k_block.T / (d ** 0.5)
            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            p = torch.exp(s - m_new)          # 用新最大值稳定指数
            alpha = torch.exp(m - m_new)      # 旧结果的缩放因子
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p @ v_block
            m = m_new
        out[i:i + block_size] = acc / l
    return out
```

真实 kernel 还会做因果掩码、跳过多余边界块、使用张量核心 MMA 指令，并把 softmax 除法与最大值更新融合进寄存器。

### 5.2 IO 复杂度与版本演进

以 HBM 访问元素数为度量，标准注意力为 $\Theta(Nd+N^2)$，FlashAttention 为 $\Theta(N^2 d^2 / M)$，$M$ 是 SRAM 可容纳的元素数。$d=128$、SRAM 约 100 KB（FP16 约 5 万元素）时 $d^2/M \approx 0.33$，$N^2$ 项系数显著变小；序列越长、头维越大收益越明显，短序列可能因调度开销收益有限。

| 版本 | 关键改进 | 效果（量级/估算） |
| --- | --- | --- |
| FlashAttention 1 | 分块、online softmax、反向重计算 | 显存 $O(N^2)$ 降到 $O(N)$，长序列提速 2 到 4 倍 |
| FlashAttention 2 | 改进工作划分与 warp 调度，减少非矩阵乘 FLOPs | 比 FA1 再快约 1.5 到 2 倍，成为主流基线 |
| FlashAttention 3 | 面向 Hopper 的 WGMMA、异步流水、FP8 | 在 H100 上比 FA2 再快约 1.5 到 2 倍，随形状变化 |

这些倍数来自论文和官方测试，属于量级估算，实际取决于序列长度、头维度、精度和 GPU 型号。

FlashAttention 没有跳过低权重、没有低秩分解或稀疏化，计算的仍是完整 softmax 注意力，改变的是计算顺序和内存层级：分数留在 SRAM/寄存器中分块累加，反向用输出和 softmax 统计量重计算，用计算换显存。它不会引入近似误差，只可能因运算顺序不同产生微小数值差异。这一点是高频辨析题：FlashAttention 是 IO-aware 的精确算法，稀疏注意力、Linformer、Performer 才是近似或结构改造。对 decode，Query 只有 1 个 token 而 K/V 很长，FlashDecoding 把 KV 沿序列维切块，多个 CTA 并行计算局部结果再 reduction，解决并行度不足。

## 6. 连续批处理（continuous batching）、chunked prefill 与 PD 分离

### 6.1 连续批处理与 chunked prefill

静态批处理（static batching）把一批请求凑齐后一起执行，直到最长的请求结束，短请求结束后槽位空转。连续批处理（continuous batching，也称 iteration-level scheduling）在每个 decode 迭代后重新调度：完成的序列立即退出，等待队列中的新请求立即加入，批大小随时间动态变化，GPU 尽量保持忙碌。

```text
静态批处理:
时间 ->  [A A A A A A A A]
         [B B B B        ]   B 结束后槽位空转
         [C C C C C C    ]

连续批处理:
时间 ->  [A A A A A A A A]
         [B B B B][D D D D]
         [C C C C C C][E E E]
```

连续批处理显著提升吞吐，但带来调度开销、KV 块分配压力和更复杂的抢占逻辑；显存不足时可以交换（swap）到 CPU 或重计算（recompute）被抢占序列的 KV。

长 prompt 的 prefill 会占用大量算力，一次执行可能数百毫秒，导致同批 decode 请求明显卡顿（ITL 抖动）。chunked prefill 把 prompt 切成固定 token 预算的多个块，每个迭代只处理一块，并与 decode token 混合调度。设每迭代 token 预算为 $C$，当前有 $n_d$ 个 decode token，本次可分配给 prefill 的预算约为 $C-n_d$。效果：降低 decode 长尾延迟、让 ITL 更稳定；提高 GPU 利用率；可能增加 TTFT，因为长 prompt 被拆成多次执行；需要处理掩码、位置编码和 KV 写入边界。Sarathi-Serve 等工作进一步用 stall-free batching 控制 prefill 块大小，保证 decode 不被长 prefill 阻塞。

### 6.2 PD 分离

prefill 吃算力，适合大 TP、高算力卡；decode 吃带宽和显存容量，适合大显存、高带宽卡。放在同一实例会互相干扰：长 prefill 抢占算力使 TPOT 抖动，decode 占 KV 显存使 prefill 无法充分利用。PD 分离（prefill-decode disaggregation）把两个阶段部署到独立资源池，中间通过高速网络传输 KV，代表性工作有 DistServe、Mooncake 以及 DeepSeek 的生产系统。

KV 传输时间下界为 $T_{transfer} = M_{KV}/BW_{link}$。70B GQA、4K 上下文单请求 KV 约 1.25 GiB：走 100 Gb/s 普通网络（约 12.5 GB/s）约 0.1 秒，走 NVLink（约 900 GB/s）约 1.4 毫秒。这个对比说明 PD 分离强烈依赖高速互联。它适合长 prompt、严格 ITL、prefill 与 decode 比例差异大、集群规模较大的场景；不适合小规模部署、网络带宽不足或请求很短的场景，还会引入调度复杂度、KV 生命周期管理和故障恢复问题。

## 7. 投机解码（speculative decoding）

### 7.1 原理与加速比

草稿模型先自回归生成 $K$ 个候选 token，目标模型一次前向并行验证这 $K$ 个位置，逐个接受候选，遇到第一个拒绝就停止，并从目标分布重新采样一个 token。对草稿采样的 token $x$，接受概率为：

$$
\alpha(x) = \min\left(1,\ \frac{p(x)}{q(x)}\right),\qquad \alpha = \mathbb{E}_{x \sim q}\left[\min\left(1,\ \frac{p(x)}{q(x)}\right)\right]
$$

$q(x)$ 是草稿分布，$p(x)$ 是目标分布。拒绝时从修正分布 $p'(x) \propto \max(0,\ p(x)-q(x))$ 重采样。这套接受加拒绝机制保证最终输出严格服从目标分布，因此投机解码无损，不改变生成质量。若每个草稿 token 接受率近似独立且为 $\alpha$，一次投机步骤期望产出 token 数为

$$
\mathbb{E}[L] = \sum_{i=0}^{K}\alpha^{i} = \frac{1-\alpha^{K+1}}{1-\alpha}
$$

其中包含全部接受后目标模型额外给出的 bonus token。设草稿单 token 成本与目标单 token 成本之比为 $c$，加速比近似为

$$
S \approx \frac{1+\alpha+\alpha^{2}+\cdots+\alpha^{K}}{1+cK}
$$

分母中的 1 是目标模型的一次并行验证，$cK$ 是 $K$ 个草稿 token 的相对成本。

| 接受率 $\alpha$ | 草稿长度 $K$ | 成本比 $c$ | 近似加速比 |
| --- | --- | --- | --- |
| 0.8 | 4 | 0.05 | 2.8 |
| 0.8 | 4 | 0.10 | 2.4 |
| 0.9 | 4 | 0.10 | 2.9 |
| 0.6 | 4 | 0.10 | 1.65 |
| 0.8 | 8 | 0.05 | 3.1 |

这些是理想化估算，未考虑批处理、验证掩码、KV 写入和采样开销。实际系统中批越大，目标模型越接近 compute-bound，验证 $K+1$ 个位置的成本越接近线性增长，加速比明显下降；因此投机解码最适合低并发、单序列延迟敏感的场景。

### 7.2 草稿模型、Medusa 与 EAGLE

- 草稿模型：用同 tokenizer 的小模型（如 1B 对 70B）做 draft。实现简单，但额外占显存，分布差异大时接受率低。
- Medusa：在目标模型顶部加多个解码头，每个头预测不同未来位置的 token，形成候选树，用树状注意力一次验证，避免独立草稿模型，但需要数据训练解码头。
- EAGLE：在特征层做自回归草稿，利用目标模型隐藏状态预测下一个特征，再由轻量头解码成 token。论文报告 2.7 到 3.5 倍加速（量级，随任务变化），EAGLE-2 进一步提高。
- 树状投机：一次构造多分支候选树，用树掩码并行验证，提高每个验证步骤的期望接受 token 数，但验证计算量增加。

## 8. 采样参数

temperature 在 softmax 前缩放 logits：$p_i = \exp(z_i/T) / \sum_j \exp(z_j/T)$，$z_i$ 是 logit，$T$ 是温度。$T<1$ 让分布更尖锐，$T>1$ 更平坦，$T \to 0$ 等价贪心；温度不改变排序，只改变概率差距。

- top-k：只保留概率最高的 $k$ 个 token，其余置零后重新归一化，$k=1$ 等价贪心；它控制候选数量，但不随分布形状自适应。
- top-p（nucleus sampling）：按概率从高到低累加，保留累计概率达到 $p$ 的最小集合。分布尖锐时集合小，平坦时集合大，比 top-k 更自适应。
- min-p：保留满足 $p_i \ge p_{min} \cdot \max_j p_j$ 的 token，相对最大概率设阈值，高温下自动放宽、低温下收紧，常见取值 0.02 到 0.1。
- repetition penalty：对已出现 token 降低 logit，常见形式为 $z_i' = z_i/\rho$（$z_i>0$）或 $z_i \cdot \rho$（$z_i \le 0$），$\rho>1$。此外还有按出现次数扣减的 frequency penalty 和出现过就扣固定值的 presence penalty，惩罚过强会破坏语法与事实一致性。

不同框架实现顺序略有差异，常见流程是：惩罚、除以 temperature、top-k、top-p、min-p、重新归一化并采样。temperature 与 top-p 叠加时，高温先拉平分布，top-p 再截断长尾，实际多样性可能低于只看 temperature 的预期。调参应固定其他变量并用固定种子对比。

| 场景 | temperature | top-p | top-k | min-p | 重复惩罚 |
| --- | --- | --- | --- | --- | --- |
| 事实问答、抽取 | 0 到 0.3 | 0.9 | 关闭或 20 | 0 | 1.0 到 1.05 |
| 通用对话 | 0.7 | 0.9 到 0.95 | 关闭 | 0 到 0.02 | 1.05 到 1.1 |
| 创意写作 | 0.9 到 1.1 | 0.95 | 关闭 | 0 到 0.05 | 1.05 到 1.15 |
| 代码生成 | 0.2 到 0.4 | 0.95 | 关闭 | 0 | 1.0 到 1.05 |

## 面试高频问题

### 1. 为什么 prefill 是 compute-bound，decode 是 memory-bound？

参考回答要点：

- prefill 一次处理 $S$ 个 token，线性层是 $S \times d$ 大矩阵乘，算术强度约 $S$ FLOP/Byte，长 prompt 下高于 GPU 拐点。
- decode 每步只处理 1 个 token，算术强度约 1 FLOP/Byte，远低于 H100 约 295 FLOP/Byte 的拐点。
- decode 时间下界主要由权重字节数除以带宽决定，例如 7B FP16 在 H100 上约 4 ms/token。
- 长上下文、大 batch 时 KV 读取量超过权重读取量，decode 从权重受限转为 KV 受限。
- 优化手段因此不同：prefill 关注算力利用率与注意力 IO，decode 关注带宽、KV 压缩和批处理。

### 2. 推导 KV Cache 显存公式，并估算 70B 模型 32K 上下文的占用。

参考回答要点：

- 每层每 token 有 K 和 V 各 $n_{kv} d_{head}$ 个元素，乘 2 后再乘层数、序列长度、批大小和字节数。
- 公式为 $2 L n_{kv} d_{head} S B b_{kv}$；MLA 把 $2 n_{kv} d_{head}$ 换成 $d_c + d_R$。
- 70B GQA（80 层、8 个 KV 头、头维 128、FP16）每 token 约 0.3125 MiB。
- 32K 单序列约 10 GiB；若 32 并发约 320 GiB，单张 80GB 卡不可能容纳。
- 实际还要加激活、碎片和框架开销，容量规划留 10% 到 20% 余量。

### 3. MQA、GQA、MLA 各自的取舍是什么？

参考回答要点：

- MQA 只保留一组 K/V，KV 缩小为 $1/n_h$，但表达能力与训练稳定性变差。
- GQA 分成 $G$ 组，KV 缩小为 $n_h/G$，质量接近 MHA，是当前主流折中。
- MLA 用低秩潜变量压缩 KV，并保留解耦 RoPE Key，压缩率最高，但实现复杂。
- GQA 通常从 MHA 平均池化初始化再继续预训练；MLA 需要吸收上投影并定制 kernel。
- 选择要看上下文长度、并发、质量要求和团队工程能力，不能只看压缩比。

### 4. PagedAttention 解决了什么问题？块大小如何取舍？

参考回答要点：

- 朴素按最大长度预留连续显存，产生大量内部和外部碎片，论文报告浪费可达 60% 到 80%。
- 固定大小物理块加块表，实现非连续存储、按需分配，外部碎片为零，内部浪费不超过一块。
- 块表支持前缀共享和写时复制，方便并行采样、beam search 和多轮对话。
- 块太小则元数据与 kernel 开销大、访存不连续；块太大则最后一个块浪费多，16 是常见默认值。
- 还要处理抢占、换入换出、块表并发更新和前缀缓存淘汰。

### 5. FlashAttention 为什么快？它和近似注意力有什么区别？

参考回答要点：

- 标准注意力把 $N \times N$ 的分数矩阵和概率矩阵写入 HBM，带宽成为瓶颈。
- FlashAttention 分块搬入 SRAM，用 online softmax 增量更新最大值、归一化因子和输出，不物化大矩阵。
- HBM 访问从 $\Theta(Nd+N^2)$ 降到 $\Theta(N^2 d^2/M)$，反向用重计算换显存。
- 数学上仍是完整 softmax 注意力，没有稀疏、低秩或跳水，只是运算顺序不同。
- FA2 优化工作划分，FA3 针对 Hopper 的异步与 FP8 进一步提速。

### 6. 连续批处理、chunked prefill、PD 分离分别解决什么问题？

参考回答要点：

- 连续批处理按迭代调度，完成的请求立即退出、新请求立即加入，减少静态批处理的槽位空转。
- chunked prefill 把长 prompt 切片，与 decode 混合执行，降低 ITL 抖动，代价是可能增加 TTFT。
- PD 分离把计算密集的 prefill 和带宽密集的 decode 放到独立资源池，避免相互干扰。
- PD 分离依赖高速 KV 传输，70B 4K 上下文约 1.25 GiB，普通网络约 0.1 秒，NVLink 约毫秒级。
- 三者都增加调度复杂度，需要根据负载长度、SLO 和集群规模决定是否启用。

### 7. 投机解码的加速比由什么决定？为什么高并发下收益下降？

参考回答要点：

- 加速比取决于接受率 $\alpha$、草稿长度 $K$ 和草稿与目标模型的成本比 $c$。
- 理想公式为 $S \approx (1+\alpha+\dots+\alpha^K)/(1+cK)$，接受率越高、草稿越便宜，收益越大。
- 高并发时目标模型接近 compute-bound，一次验证 $K+1$ 个位置的成本接近线性增长，分母变大。
- 草稿模型占用显存和算力，接受率低时可能负收益；草稿与目标分布差异大时接受率低。
- Medusa、EAGLE 通过树状候选和特征级草稿提高接受率，但实现和训练成本更高。

### 8. 前缀缓存为什么只能复用完整块？有哪些坑？

参考回答要点：

- 部分块的内容还会继续追加，KV 尚未稳定，复用会读到不完整或错误的状态。
- 完整块哈希需要包含 token 序列、模型版本、LoRA 适配器和分词器配置。
- 位置编码与块在序列中的位置绑定，不能把不同位置的相同文本直接复用。
- 多租户共享可能造成侧信道泄露，需要按租户隔离或限制共享范围。
- 缓存占显存，要设置容量、LRU 淘汰，并监控命中率与收益。

### 9. 采样参数如何影响生成质量？temperature 和 top-p 如何配合？

参考回答要点：

- temperature 缩放 logits，改变分布尖锐程度，不改变排序，$T \to 0$ 为贪心。
- top-k 固定候选数，top-p 按累计概率自适应截断，min-p 相对最大概率设阈值。
- 高温加 top-p 会先拉平再截断，实际多样性取决于两者组合，不能只看单一参数。
- repetition penalty 过强会破坏语法和事实，代码与数学任务通常用较小值。
- 调参要固定其他变量、固定种子，并用任务指标和人工评估共同判断。

> 本文档为面试复习资料，公式与数字建议结合原始论文复核。
