# SFT 与参数高效微调（PEFT）

**本章导读**

- 监督微调（Supervised Fine-Tuning，SFT）把预训练模型从“续写互联网文本”变成“按指令对话”，核心是数据质量、对话模板与损失掩码，而不是堆参数量。
- 全参微调效果上限高，但 7B 模型仅优化器状态就约 84GB；参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）用少量可训练参数换取可接受的显存与效果。
- 低秩适配（Low-Rank Adaptation，LoRA）用低秩分解近似权重更新，可合并回原权重、无推理开销，是当前最主流的 PEFT；量化低秩适配（Quantized LoRA，QLoRA）再把冻结基座量化到 4-bit，把 65B 模型塞进单张 48GB 卡。
- 微调的副作用是灾难性遗忘与对齐税（alignment tax），需要用数据回放、低秩约束、模型合并和完整评测来控制。

**核心要点**

- SFT 主要教“格式、风格、工具调用与拒答边界”，知识主要来自预训练；LIMA 的 1,000 条高质量样本说明质量比数量更关键。
- 只对助手（assistant）token 计算损失（loss），system/user/tool 片段设为 `-100`；EOS/EOT 必须被监督。
- 全参微调约 16 字节/参数；7B 约 112GB 状态，70B 约 1.12TB，必须分片或 PEFT。
- LoRA 的更新为 $W'=W+\frac{\alpha}{r}BA$，$B$ 零初始化保证训练起点等价于原模型，参数量为 $r(d+k)$。
- 秩 $r$ 控制容量，$\alpha/r$ 控制更新尺度；常见 $r=8$–$64$、$\alpha=2r$，目标模块从注意力（attention）扩展到 MLP 收益更明显。
- QLoRA 用 4-bit NormalFloat（NF4）量化、双重量化与 paged optimizer，把 7B 微调压到约 10GB、65B 压到约 48GB（估算）。
- 灾难性遗忘用 5%–20% 通用数据回放、Kullback-Leibler（KL）正则、LoRA 约束和早期停止缓解；模型合并可把多个专家能力叠加，但会互相干扰。

## 1. SFT 的目标与数据

### 1.1 SFT 在做什么

预训练优化的是无条件的下一个 token 分布 $p_\theta(x_t\mid x_{<t})$；SFT 优化的是给定对话历史 $x$ 时回答 $y$ 的条件分布：

$$\mathcal{L}_{\text{SFT}}(\theta)=-\sum_{(x,y)\in\mathcal{D}}\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,y_{<t})$$

其中 $\mathcal{D}$ 是指令数据，$x$ 是 system+user+历史轮次，$y$ 是 assistant 回答。这个目标本质是行为克隆（behavior cloning）：让模型模仿标注者的回答风格、格式与决策边界。

SFT 能教什么、不能教什么：

- **能教**：对话格式、语气、长度偏好、工具调用 JSON、拒答模板、逐步推理的书写方式、领域术语。
- **不太能教**：预训练中完全没有的事实知识、全新的复杂技能；这些需要继续预训练或更强的 RL/蒸馏。
- **容易过拟合**：模板 artifact、特定标注者的口头禅、数据集里的错误答案。数据质量比数据量更早成为瓶颈。

### 1.2 数据格式

| 格式 | 典型结构 | 适用 | 风险 |
| --- | --- | --- | --- |
| Alpaca | `instruction`、`input`、`output` | 单轮任务、简单 SFT | 多轮/system 支持弱 |
| ShareGPT | `conversations: [{from, value}]` | 多轮对话、社区数据 | 角色映射不统一、噪声多 |
| ChatML | `<\|im_start\|>role ... <\|im_end\|>` | 通用对话、工具调用 | 特殊 token 必须注册 |
| Llama 3 | `<\|start_header_id\|>role<\|end_header_id\|>` | 官方模型对齐 | 模板细节必须完全一致 |
| 纯 JSONL | 每行一条 messages 数组 | 自定义管线 | 需要自己处理截断与 mask |

训练与推理必须使用同一套对话模板（chat template）。常见事故是训练时用 ChatML、推理时用 Llama 3 header，模型把 header 当成普通文本，输出格式崩坏。保存模型时同时保存 tokenizer 与 `chat_template`，评测时复用同一个渲染函数。

### 1.3 质量优先：LIMA 的结论

LIMA（Less Is More for Alignment）用 1,000 条人工精选样本微调 65B LLaMA，在作者的人工评测中表现可与 text-davinci-003 竞争。论文提出的“表层对齐假说”（Superficial Alignment Hypothesis）认为：预训练已经学到了绝大部分知识与能力，SFT 主要是在教“如何调用这些能力”的格式与风格。

这个结论要辩证看：

- 1,000 条足以学风格，但不足以覆盖大量任务类型、工具 schema、领域术语；实际工业 SFT 常在 1 万–10 万条量级（估算）。
- LIMA 的评测规模有限、题目偏开放对话；在代码、数学、函数调用上，数据覆盖度仍然重要。
- “质量”包括正确性、多样性、难度、格式一致性、长度分布；一条错误答案的负面影响可能抵得上十条普通样本。
- 数据去重与去污染同样适用于 SFT：重复样本会让模型复读，测试集泄漏会让评测虚高。

### 1.4 数据量与 epoch

| 目标 | 样本量级（估算） | epoch | 说明 |
| --- | --- | --- | --- |
| 风格/格式对齐 | 1k–10k | 2–3 | 高质量、多样，避免模板单一 |
| 通用对话 | 10k–100k | 1–3 | 覆盖任务类型与拒答边界 |
| 领域专家 | 5k–50k | 2–4 | 领域术语多，需混入通用数据防遗忘 |
| 工具调用 | 5k–30k | 2–3 | schema 要全，负例（不该调用）也要有 |
| 长上下文 | 1k–10k | 1–2 | 样本长，packing 与显存压力大 |

epoch 不是越多越好。第一轮通常带来最大提升，第二轮收益减半，第三轮后开始记忆训练集、通用能力下降。用验证集 loss、任务指标和通用 benchmark 三条曲线共同决定早停；如果任务指标还在涨但通用能力掉，优先加回放数据而不是继续训练。

### 1.5 Loss mask 与多轮拼接

```python
# 只对 assistant 片段计 loss 的最简实现
def encode_chat(tokenizer, messages):
    input_ids, labels = [], []
    for m in messages:
        prefix = f"<|im_start|>{m['role']}\n"
        body = m["content"] + "<|im_end|>\n"
        p_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        b_ids = tokenizer(body, add_special_tokens=False).input_ids
        input_ids += p_ids + b_ids
        if m["role"] == "assistant":
            labels += [-100] * len(p_ids) + b_ids   # 回答与 EOT 参与 loss
        else:
            labels += [-100] * (len(p_ids) + len(b_ids))
    return input_ids, labels
```

要点：

- `-100` 是 `CrossEntropyLoss` 的 `ignore_index`；Hugging Face 的 causal LM 内部会做 shift，不要再手动错位。
- assistant 的 EOS/EOT 必须参与 loss，否则模型学不会停；多轮对话中每一轮 assistant 都应监督。
- 工具调用中 assistant 生成的调用 JSON 参与 loss，role=tool 的返回结果不参与。
- 超长样本按 token 预算截断，优先保留 system 与最近轮次；截断后仍要保证至少一个 assistant 片段有监督。

## 2. 全参微调与 PEFT 对比

### 2.1 显存与效果

| 维度 | 全参微调 | LoRA（bf16 基座） | QLoRA（4-bit 基座） |
| --- | --- | --- | --- |
| 可训练参数 | 100% | 0.1%–1% | 0.1%–1% |
| 7B 权重显存 | 14GB（bf16） | 14GB（冻结） | 约 3.5GB（NF4） |
| 7B 优化器状态 | 约 84GB | 约 0.1–0.2GB | 约 0.1–0.2GB |
| 7B 总显存（含激活，估算） | 112GB+ | 20–40GB | 8–12GB |
| 70B 可行性 | 多机 3D 并行 | 多卡，140GB 权重+激活 | 单张 80GB 可尝试 |
| 训练速度 | 基准 | 快 20%–40%（估算） | 慢 20%–50%（估算，反量化开销） |
| 效果上限 | 最高 | 接近全参（风格/领域） | 略低于 LoRA |
| 推理开销 | 无 | 合并后无，未合并有 | 需反量化，略慢 |
| 适用场景 | 大分布迁移、继续预训练 | 领域/风格/工具适配 | 显存极度受限 |

全参微调的优势是容量不受低秩限制，适合“新知识、新语言、新模态”这类大分布迁移；LoRA 更适合“已有能力的方向性调整”。但“LoRA 一定不如全参”不成立：在数据量小、任务窄的场景，低秩约束反而能防过拟合。选择时先看显存，再看任务分布偏移程度，最后用同一验证集比较。

### 2.2 为什么 PEFT 省显存

PEFT 冻结基座，只训练少量参数：

- 不需要为基座存梯度与 Adam 状态，省掉 $14P$ 字节（bf16 梯度 2P + fp32 master/m/v 12P）。
- 反向仍然要经过基座计算输入梯度（供低秩分支使用），所以激活显存和大部分前向计算仍在。
- 优化器状态只对应 adapter，通常几十到几百 MB；这才是 7B 从 112GB 降到 20GB 量级的主因。
- QLoRA 进一步把冻结权重压到 4-bit，权重从 14GB 降到 3.5GB，但反量化会让计算变慢。

## 3. LoRA

### 3.1 低秩假设与公式

LoRA（Low-Rank Adaptation）假设微调带来的权重更新 $\Delta W$ 具有较低的内在秩（intrinsic rank），因此可以分解为两个小矩阵的乘积。对原权重 $W_0\in\mathbb{R}^{d\times k}$：

$$W'=W_0+\Delta W=W_0+\frac{\alpha}{r}BA$$

其中 $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$，$r\ll\min(d,k)$，$\alpha$ 是缩放超参。前向为：

$$h=W_0x+\frac{\alpha}{r}BAx$$

$A$ 用随机高斯/Kaiming 初始化，$B$ 初始化为零，因此训练开始时 $\Delta W=0$，模型行为与原模型完全一致；若两者都随机初始化，初期会引入噪声，甚至破坏已对齐的行为。

参数量为 $r(d+k)$，而全秩更新是 $dk$。以 $d=k=4096$、$r=8$ 为例：LoRA 每个矩阵 65,536 参数，是全秩的 0.39%；Llama 7B 在 q/k/v/o 四个模块、32 层上加 LoRA，约 840 万参数（0.12%）；若再加上 MLP 的三个矩阵，约 1,470 万（0.21%）（量级估算）。

### 3.2 秩、alpha 与目标模块

- **秩 $r$**：4–16 适合风格/格式，32–64 适合领域知识，128 以上参数量快速上升、收益递减。低秩不是越小越好，任务分布偏移越大，需要的秩越高。
- **alpha**：实际更新尺度是 $\alpha/r$。固定 $\alpha$ 而增大 $r$ 会让单步更新变小，所以常见做法是 $\alpha=2r$（如 $r=8,\alpha=16$）或固定 $\alpha=16$–$32$。rsLoRA 提出用 $\alpha/\sqrt{r}$ 缩放，让大秩时梯度尺度更稳定。
- **目标模块**：只加在 $q,v$ 最省参数；加到 $q,k,v,o$ 提升明显；再加 MLP 的 `gate/up/down` 通常效果最好，但参数量约翻倍。embedding 与 `lm_head` 一般不训，除非要加新 token 或新语言。
- **dropout**：LoRA 分支加 0.05–0.1 的 dropout 可缓解小数据过拟合。
- **学习率**：LoRA 常用 $1\times10^{-4}$–$3\times10^{-4}$，比全参的 $1\times10^{-5}$–$2\times10^{-5}$ 高一个量级；因为只更新低秩分支，且缩放后的梯度较小。

### 3.3 合并与多 LoRA 服务

训练完后可以把 LoRA 合并回基座：

$$W_{\text{merged}}=W_0+\frac{\alpha}{r}BA$$

合并后推理没有任何额外层，延迟与基座一致。注意：

- 合并应在 fp32（或 fp64）下做，再 cast 回推理精度；bf16 直接累加会引入误差。
- 量化基座（NF4/INT4）不能直接合并到 4-bit 权重；要么反量化到 bf16 再合并，要么保留 adapter 在推理时动态加。
- 合并不可逆时先保存原始权重；需要切换多个 adapter 时不要合并，改用多 LoRA 服务。

多 LoRA 服务（multi-LoRA serving）在一个基座上同时加载多个 adapter，按请求路由。S-LoRA、Punica、vLLM 的 multi-LoRA 用统一内存池存放不同秩的 adapter，用 grouped GEMM 把同一 batch 内不同 adapter 的请求分组计算。收益是“一份基座服务多个任务”，代价是调度复杂、adapter 切换有开销、batch 内秩不一致会降低 GPU 利用率。

### 3.4 LoRA 层实现

```python
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if getattr(self, "merged", False):
            return self.base(x)
        out = self.base(x)
        update = self.drop(x) @ self.A.T @ self.B.T
        return out + self.scaling * update

    @torch.no_grad()
    def merge(self):
        self.base.weight += self.scaling * (self.B @ self.A)
        self.merged = True
```

工程细节：`base` 的 bias 不参与 LoRA；若 `base` 是量化层，`base(x)` 内部会反量化，合并要单独处理；合并后把 `forward` 切到只走 `base`，否则会重复加更新。

## 4. QLoRA

### 4.1 NF4 量化

QLoRA 把冻结基座量化到 4-bit NormalFloat（NF4）。NF4 的 16 个量化值是标准正态分布的分位点，因而对近似正态的权重是“信息论上较优”的 4-bit 编码；每个 block（通常 64 个权重）共享一个 absmax 缩放因子：

$$w\approx s\cdot q,\qquad s=\frac{\max(|w_{\text{block}}|)}{|q|_{\max}}$$

其中 $q$ 是 16 个 NF4 码字之一，$s$ 是 block 的缩放因子。反量化在矩阵乘前进行，计算仍用 bf16。

### 4.2 双重量化

第一层量化要为每个 64 权重存一个 32-bit absmax，折合 $32/64=0.5$ bit/参数；双重量化（double quantization）把这些 absmax 再按 256 个一组量化到 8-bit，并存的缩放因子更少：

$$\text{bits/param}\approx4+\frac{8}{64}+\frac{32}{64\times256}\approx4.127$$

相比 4.5 bit/参数，节省约 0.37 bit/参数；7B 模型约省 0.3GB（估算）。数字不大，但对“刚好差一点”的单卡场景很关键。

### 4.3 Paged optimizer

QLoRA 使用 NVIDIA 统一内存的 paged optimizer：优化器状态分页存放，显存不足时自动换出到 CPU，出现长序列或梯度尖峰时避免 OOM。它解决的是“峰值”而不是“平均值”，吞吐会因换页下降；不能把 70B 硬塞进 24GB。

### 4.4 显存估算

| 模型 | NF4 权重 | 量化常数 | LoRA 参数/优化器 | 激活（seq 2k，估算） | 总计（估算） |
| --- | --- | --- | --- | --- | --- |
| 7B | 3.5GB | 0.1–0.2GB | 0.05–0.2GB | 2–5GB | 8–12GB |
| 13B | 6.5GB | 0.2–0.3GB | 0.1–0.3GB | 3–6GB | 12–16GB |
| 70B | 35GB | 1–2GB | 0.5–1GB | 5–10GB | 45–60GB |

QLoRA 论文报告 65B 模型可在单张 48GB 卡上微调，33B 可在 24GB 上微调（以公开资料为准）。实际显存随序列长度、batch、是否 gradient checkpointing 大幅变化；表中小于 80GB 的配置并不保证一定能跑，需留余量。

### 4.5 QLoRA 的取舍

- 优点：显存门槛最低，适合单卡/少卡实验；4-bit 基座 + bf16 adapter 的效果接近 16-bit LoRA。
- 缺点：反量化与 paged optimizer 让训练慢 20%–50%（估算）；4-bit 量化损失对代码、数学等精细任务更敏感；合并麻烦。
- 常见错误：把 LoRA 直接合并进 4-bit 权重导致精度崩坏；忘记 `bnb_4bit_compute_dtype=torch.bfloat16`，用 fp32 计算又慢又占显存；在推理时仍保留训练用的 dropout。

## 5. 其他 PEFT 方法

| 方法 | 注入位置 | 可训练参数 | 推理开销 | 特点 |
| --- | --- | --- | --- | --- |
| Adapter | 每层 attention/FFN 后加瓶颈 MLP | 1%–5% | 有，串行增加层 | 早期方案，稳定但推理变慢 |
| Prefix-Tuning | 每层 KV 前加可学习 prefix | 0.1%–1% | 占用上下文，可预计算 | 训练不稳定，生成任务效果一般 |
| P-Tuning v2 | 每层加 deep prompt | 0.1%–3% | 占用上下文 | NLU 上接近全参，生成任务仍弱 |
| IA3 | 对 K/V/FFN 激活做逐元素缩放 | 约 0.01% | 可合并，无开销 | 极省参数，容量有限 |
| LoRA | 权重旁路低秩分解 | 0.1%–1% | 可合并，无开销 | 通用、生态最好 |
| DoRA | 把权重分解为幅度与方向，方向用 LoRA | 略高于 LoRA | 可合并 | 低秩下效果更好，训练更慢 |

**适配器（Adapter）**：在 Transformer 子层后插入“降维-非线性-升维”的瓶颈模块，只训这部分。优点是模块化、可插拔；缺点是推理时多了串行层，延迟增加，且多任务需要切换 adapter。

**Prefix-Tuning / P-Tuning v2**：在每层 attention 的 K/V 前拼接可学习的虚拟 token。它们不改权重，但会占用上下文长度；prefix 的梯度要穿过整个网络，训练不稳定，对生成任务的效果不如 LoRA 稳定。P-Tuning v2 通过深层 prefix、移除重参数化 MLP，在 NLU 上更强。

**IA3**：学习三个向量，分别缩放 key、value 和 FFN 中间激活：$h\leftarrow l\odot h$。参数量极小（约 0.01%），可合并进权重，适合极端省参数场景；容量有限，复杂任务容易欠拟合。

**权重分解低秩适配（Weight-Decomposed Low-Rank Adaptation，DoRA）**：把权重分解为幅度 $m$ 与方向 $V$：

$$W'=m\frac{W_0+BA}{\|W_0+BA\|_c}$$

其中 $\|\cdot\|_c$ 是列范数，$m$ 是可学习的幅度向量。DoRA 在低秩（$r=4$–$8$）时通常优于 LoRA，接近全参；代价是训练时多一次范数计算，显存与时间略高，推理可合并。

## 6. 灾难性遗忘、数据混合与模型合并

### 6.1 遗忘从哪来

微调数据分布窄、学习率高、训练久，模型会把参数拉向新任务，旧能力下降。表现包括：通用问答变差、代码能力退化、安全拒答边界被破坏、语言混杂。缓解手段：

- **数据回放**：混入 5%–20% 通用 SFT 或预训练数据；领域越窄，回放比例越高。
- **KL 正则**：在损失中加入 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$，约束输出分布不要离基座太远。
- **低秩约束**：LoRA 只更新低秩子空间，天然限制漂移；但不能完全避免。
- **低学习率、短训练、早停**：用通用 benchmark 做早停信号，而不是只看任务指标。
- **模型合并**：把领域模型与通用模型按权重插值，找回部分通用能力。

数据混合（data mixing）是遗忘控制的前置手段，不只指领域比例，还包括任务类型、难度、回答长度、语言、通用/领域比例。经验做法是目标任务占 70%–90%、通用数据占 10%–30%，对稀缺任务用温度采样 $p_i^{1/T}$ 上采样；混合后在验证集上看各能力曲线的帕累托前沿，而不是单一平均分。混合比例本身是超参，需要小模型代理实验或消融确定。

### 6.2 模型合并

**线性插值 / SLERP**：两个同架构模型 $\theta_1,\theta_2$，线性合并 $\theta=(1-\lambda)\theta_1+\lambda\theta_2$；SLERP 在球面上插值，保持方向范数，通常比线性更稳。只适合同源模型，跨架构不行。

**任务算术（Task Arithmetic）**：定义任务向量 $\tau_i=\theta_i-\theta_0$，合并：

$$\theta_{\text{merged}}=\theta_0+\lambda\sum_i\tau_i$$

其中 $\theta_0$ 是基座，$\lambda$ 是缩放系数。它把“微调方向”当成可加减的向量，可叠加多个专家；但任务向量之间会干扰，$\lambda$ 过大时能力互相覆盖。

**TIES-Merging**：先修剪（trim）每个任务向量中绝对值最小的 $p$%（如 80%），再对每个参数做符号选举（elect sign），只保留与多数符号一致的更新，最后对一致部分求平均。它解决“符号冲突导致互相抵消”的问题，适合合并多个 LoRA/全参专家。

**DARE**：随机丢弃 90% 的 delta 参数，并把保留的参数按 $1/(1-p)$ 放大，再合并；利用冗余性减少冲突，常与 TIES 组合。

合并的评估不能只看单任务：要在通用能力、领域任务、安全评测上同时回归；合并后的模型可能“什么都会一点，但都不精”，需要按业务取舍。

### 6.3 超参经验

| 超参 | 全参微调 | LoRA | QLoRA |
| --- | --- | --- | --- |
| 学习率 | $1\times10^{-5}$–$2\times10^{-5}$ | $1\times10^{-4}$–$3\times10^{-4}$ | $1\times10^{-4}$–$2\times10^{-4}$ |
| batch（序列数） | 32–128 | 32–128 | 16–64 |
| epoch | 1–3 | 2–4 | 2–4 |
| warmup | 3%–5% | 3%–5% | 3%–5% |
| weight decay | 0.01–0.1 | 0.01–0.1 | 0.01–0.1 |
| 梯度裁剪 | 1.0 | 1.0 | 1.0 |
| 序列长度 | 2k–8k | 2k–8k | 1k–4k |
| 特殊设置 | ZeRO/TP | dropout 0.05 | NF4 + paged optimizer |

这些是起点而不是定律。小数据（<5k）应降低学习率、增加 dropout、减少 epoch；大数据可以适当提高学习率并延长训练。每次只改一个维度，记录任务指标、通用指标和显存峰值。

## 7. 面试高频问题

### 1. SFT 的目标是什么？它和预训练、RLHF 的分工是什么？

- SFT 是条件行为克隆，最大化给定对话历史下 assistant 回答的似然。
- 预训练学知识与语言能力，SFT 学格式、风格、工具调用与拒答边界，RLHF/DPO 进一步优化偏好与安全性。
- SFT 数据质量决定上限，RL 阶段需要较好的 SFT 基座做探索。
- SFT 不能可靠注入预训练中没有的事实；新知识要靠继续预训练、RAG 或合成数据。

### 2. LIMA 的结论是什么？1,000 条数据真的够吗？

- LIMA 用 1,000 条精选样本微调 65B 模型，证明少量高质量数据可以显著改善对话风格。
- 表层对齐假说：知识在预训练中已获得，SFT 主要教“如何表达”。
- 1,000 条不足以覆盖工具 schema、领域术语、长尾任务；工业界常用 1 万–10 万条。
- 结论的边界：评测规模有限、任务偏开放对话；不能外推为“数据越少越好”。

### 3. 全参微调和 LoRA 的显存、效果、适用场景如何比较？

- 全参约 16 字节/参数，7B 约 112GB；LoRA 冻结基座，只存 adapter 的优化器状态，7B 约 20–40GB。
- 全参容量最高，适合大分布迁移；LoRA 在风格/领域/工具适配接近全参，小数据时更抗过拟合。
- LoRA 可合并、无推理开销；全参推理不变但训练成本高。
- 选择顺序：显存 → 分布偏移 → 数据量 → 用同一验证集做 A/B。

### 4. LoRA 的低秩假设是什么？为什么 B 要初始化为零？

- 微调的权重更新 $\Delta W$ 被认为位于低秩子空间，用 $BA$ 近似，参数量从 $dk$ 降到 $r(d+k)$。
- $B=0$ 时 $\Delta W=0$，训练起点与原模型完全一致，避免随机噪声破坏已对齐行为。
- $A$ 随机初始化提供梯度方向；若 $A$ 也为零，梯度会对称地卡住。
- 缩放 $\alpha/r$ 控制更新幅度，调 $\alpha$ 等价于调学习率尺度。

### 5. LoRA 的秩和 alpha 怎么选？目标模块加在哪里？

- $r=8$–$16$ 适合风格/格式，$r=32$–$64$ 适合领域知识；再大收益递减、接近全参成本。
- 常见 $\alpha=2r$；固定 $\alpha$ 增大 $r$ 会缩小有效步长，需同步调学习率。
- 只加 q/v 最省；q/k/v/o 明显更好；加 MLP 的 gate/up/down 通常最佳但参数翻倍。
- 小数据加 dropout、降秩；大数据可提高秩并训练更久；用消融而不是默认值。

### 6. QLoRA 的 NF4、双重量化和 paged optimizer 分别解决什么？

- NF4 用正态分位点做 4-bit 量化，适配近似正态的权重，比均匀 4-bit 精度更高。
- 双重量化把 block 的 absmax 再量化到 8-bit，把开销从 0.5 bit/参数降到约 0.127 bit/参数。
- paged optimizer 在显存峰值时把优化器状态换页到 CPU，避免 OOM，但会降低吞吐。
- 三者组合把 65B 微调压到约 48GB；代价是反量化变慢、合并麻烦、精细任务精度略降。

### 7. Adapter、Prefix-Tuning、IA3、DoRA 和 LoRA 有什么区别？

- Adapter 加瓶颈 MLP，参数 1%–5%，推理有串行开销；Prefix/P-Tuning 加虚拟 KV，占上下文且训练不稳定。
- IA3 用逐元素缩放向量，参数约 0.01%，可合并，但容量有限。
- DoRA 把权重拆成幅度与方向，低秩时效果更好，训练开销略高。
- LoRA 是通用性与生态最好的折中；选择取决于参数预算、推理延迟和任务难度。

### 8. 微调后灾难性遗忘怎么办？

- 数据回放 5%–20% 通用数据，按能力维度组织，而不只是随机混。
- KL 正则约束输出分布、LoRA 限制更新子空间、低学习率与早停。
- 用通用 benchmark、安全评测、领域任务三条曲线监控，不能只看训练 loss。
- 模型合并可找回部分通用能力，但会引入任务间干扰，需要回归评测。

### 9. 模型合并的 Task Arithmetic、TIES、DARE 有什么区别？

- Task Arithmetic 把 $\theta_i-\theta_0$ 当任务向量相加，简单但向量间会冲突。
- TIES 先修剪小幅更新、再按符号选举、只合并同号部分，减少互相抵消。
- DARE 随机丢弃大部分 delta 并重新缩放，利用冗余降低冲突，常与 TIES 组合。
- 合并只适合同源模型；合并后要在通用、领域、安全三方面回归，避免“平均化”掉关键能力。

### 10. SFT 数据如何构造和评估？

- 覆盖任务类型、难度、长度、拒答与工具调用；正例与负例都要有。
- 统一 chat template，只监督 assistant，EOS 必须纳入 loss；多轮全轮监督。
- 去重、去污染、检查答案正确性；用少量高质量样本优于大量噪声。
- 评估看任务指标、通用能力、安全性和格式遵循；用人工抽检与 LLM-as-judge 互补。

> 本文档为面试复习资料，公式与数字建议结合原始论文复核。
