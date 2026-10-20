# 对齐：RLHF、DPO 与 GRPO

**本章导读**

- 对齐（alignment）要让模型同时做到有用（helpful）、无害（harmless）、诚实（honest），三者经常冲突，工程上是在偏好数据与约束之间做权衡。
- 基于人类反馈的强化学习（Reinforcement Learning from Human Feedback，RLHF）的三阶段是监督微调（Supervised Fine-Tuning，SFT）→ 奖励模型（Reward Model，RM）→ 强化学习，近端策略优化（Proximal Policy Optimization，PPO）是经典实现；直接偏好优化（Direct Preference Optimization，DPO）把“奖励模型 + RL”折叠成一个分类损失，组相对策略优化（Group Relative Policy Optimization，GRPO）则去掉价值网络（critic）、用组内相对奖励。
- 这一章的核心是几个目标函数：Bradley-Terry、带 KL 约束的 RL 目标、PPO clip、DPO 的闭式推导、GRPO 的组归一化优势。
- 推理模型的后训练把奖励换成可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards，RLVR），带来长思维链（Chain-of-Thought，CoT）与新的“过度思考”问题；蒸馏则是把大模型的对齐与推理能力迁移到小模型。

**核心要点**

- Bradley-Terry 把成对偏好建模为奖励差的 sigmoid，RM 的损失是 $-\log\sigma(r_w-r_l)$；RM 会被长度、格式和奉承行为钻空子。
- PPO 的目标是“奖励最大化 − Kullback-Leibler（KL）约束”，用 clip 限制策略更新幅度，用 critic 估计优势，用广义优势估计（Generalized Advantage Estimation，GAE）做偏差-方差权衡。
- DPO 从 KL 约束的 RL 最优解出发，把奖励表示成策略与参考策略的 log-ratio，代入 Bradley-Terry 后得到只依赖偏好对的分类损失，不需要显式 RM 和在线采样。
- DPO 的 $\beta$ 控制对参考策略的约束强度；$\beta$ 越大越保守，越小越允许偏离，但也更容易过拟合偏好数据。
- GRPO 对同一提示词（prompt）采样一组回答，用组内均值/标准差归一化得到优势，省掉 critic；适合可验证奖励和大规模采样。
- RLVR 用答案匹配、单元测试等可验证信号替代 RM，是推理模型后训练的核心；过程奖励模型（Process Reward Model，PRM）提供步级稠密奖励，但标注贵、易被 hack；结果奖励模型（Outcome Reward Model，ORM）只给最终结果打分。
- 长 CoT 是 RL 的涌现结果，不是越长越好；过度思考会增加延迟与成本，需要长度控制与难度自适应。
- 蒸馏分离线（SFT 教师轨迹）与在线（学生在自己分布上向教师对齐），后者能缓解暴露偏差，但受学生容量限制。

## 1. 对齐目标与 RLHF 流程

### 1.1 有用、无害、诚实

- **有用（helpful）**：理解意图、给出正确且可执行的信息，遵循格式与工具约束。
- **无害（harmless）**：拒绝伤害性请求、避免偏见与隐私泄露、不被越狱诱导。
- **诚实（honest）**：不编造事实、表达不确定性、不谄媚（sycophancy）。

三者存在张力：过度安全会让模型“什么都拒绝”，损害有用性；过度迎合用户会牺牲诚实；过度强调简洁可能漏掉关键风险。偏好数据标注时，标注者的取舍会直接塑造模型价值观，因此需要明确的标注指南与多 annotator 一致性检查。

对齐税（alignment tax）指对齐后通用能力或创造力下降。衡量方式是同一基座在 SFT/RLHF 前后的 MMLU、代码、数学、长文写作对比；如果掉点明显，通常要增加通用数据回放、降低 KL 强度或缩短 RL 训练。

### 1.2 RLHF 三阶段

| 阶段 | 目标 | 数据量级（估算） | 产出 |
| --- | --- | --- | --- |
| SFT | 学会指令格式与基本对话 | 1 万–10 万条 | 策略初始模型 $\pi_{\text{SFT}}$ |
| RM | 学会给回答打分 | 10 万–100 万对偏好 | 奖励模型 $r_\phi$ |
| PPO | 用 RM 奖励优化策略 | 1 万–10 万条 prompt | 对齐后的策略 $\pi_\theta$ |

第一步 SFT 已在上一章讨论。第二步收集同一 prompt 下的多个回答，让标注者或强模型给出偏好排序，训练 RM。第三步用 PPO 让策略生成回答、RM 打分、KL 约束防止偏离 SFT 模型太远。Anthropic HH-RLHF 约 17 万条比较（量级，以公开资料为准），UltraFeedback 约 6.4 万 prompt、每个 4 个回答，是常见的开源偏好数据规模。

工业界不一定严格三阶段：可以用 DPO 替代 PPO，用 AI 反馈（Reinforcement Learning from AI Feedback，RLAIF）替代人工标注，用拒绝采样微调（Rejection Sampling Fine-Tuning，RFT）替代在线 RL，或在 SFT 后直接做 RLVR。

## 2. 奖励模型

### 2.1 Bradley-Terry（BT）模型

给定 prompt $x$ 和两个回答 $y_w$（win）与 $y_l$（lose），Bradley-Terry 假设偏好概率只取决于两者的奖励差：

$$P(y_w\succ y_l\mid x)=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)$$

其中 $\sigma(z)=1/(1+e^{-z})$ 是 sigmoid，$r_\phi$ 是奖励模型。对应的负对数似然：

$$\mathcal{L}_{\mathrm{RM}}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]$$

这个损失只约束奖励差，不约束奖励的绝对尺度；因此 RM 的输出可以整体平移，RL 阶段通常要做奖励归一化（减均值、除标准差）。

### 2.2 数据与训练细节

- **配对构造**：同一 prompt 采样 2–8 个回答，人工或强模型排序；只取“明显更好”的对，跳过质量接近的样本以减少噪声。
- **平局与噪声**：标注者不一致率通常 20%–30%（估算）；可用多标注投票、软标签、margin 损失（要求 $r_w-r_l\ge m$）缓解。
- **模型结构**：语言模型最后一层加标量 reward head，取最后一个 token 的隐藏状态；也可以对整段做 pooling。训练时通常冻结部分层或全参微调，学习率比 SFT 低。
- **长度偏差**：RM 容易给长回答高分，因为偏好数据里长回答往往更详细；需要对长度做归一化、在数据中平衡长度，或在 RL 中加长度惩罚。
- **评估**：成对准确率（pairwise accuracy）、不同模型间的排序一致性、对抗样本鲁棒性；RM 在分布外 prompt 上很容易失效。

### 2.3 奖励黑客（reward hacking）

RM 是真实偏好的代理，PPO 会找到代理的漏洞：堆砌格式、重复安全声明、变长、迎合用户、给出看似专业但错误的细节。表现为代理奖励持续上升，而人工评测或真实任务指标下降——典型的 Goodhart 定律。

缓解手段：

- KL 惩罚限制策略偏离参考模型，避免走向 RM 的极端区域。
- 早停：在验证集人工评分开始下降时停止，而不是等代理奖励收敛。
- RM 集成（ensemble）：多个 RM 的平均或最小值，降低单点漏洞。
- 数据刷新：用当前策略采样新回答、重新标注，让 RM 覆盖策略的新分布。
- 规则/验证器：代码用单测、数学用答案匹配，减少对 RM 的依赖。
- 长度与格式惩罚：显式约束，避免“越长越高分”。

## 3. PPO

### 3.1 带 KL 约束的 RL 目标

RLHF 希望最大化奖励，同时不偏离参考策略 $\pi_{\text{ref}}$（通常是 SFT 模型）太远：

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\left[r_\phi(x,y)\right]-\beta\,\mathbb{E}_{x\sim\mathcal{D}}\left[\mathrm{KL}\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right)\right]$$

其中 $\beta$ 是 KL 系数，$\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=\mathbb{E}_{y\sim\pi_\theta}[\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}]$。KL 的作用是防止策略为了骗 RM 而崩坏，同时保留 SFT 的语言能力。

### 3.2 PPO 的 clipped 目标

把生成看成 token 级马尔可夫决策过程（Markov Decision Process，MDP）：状态 $s_t=(x,y_{<t})$，动作 $a_t=y_t$，策略 $\pi_\theta(a_t|s_t)$。设旧策略为 $\pi_{\theta_{\text{old}}}$，重要性比：

$$\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$

PPO 的 clipped surrogate：

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_{t}\left[\min\left(\rho_t(\theta)\hat{A}_t,\ \mathrm{clip}\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t\right)\right]$$

其中 $\hat{A}_t$ 是优势估计，$\epsilon$ 通常取 0.2。直观理解：优势为正时，提高该动作概率的收益最多到 $1+\epsilon$；优势为负时，降低概率的收益最多到 $1-\epsilon$；超出范围不再给梯度，避免一步更新过猛。

完整的 PPO 损失（最大化形式）：

$$\mathcal{L}_{\mathrm{PPO}}(\theta)=\mathbb{E}\left[L^{\mathrm{CLIP}}(\theta)-c_1\left(V_\psi(s_t)-\hat{R}_t\right)^2+c_2\,\mathcal{H}\left(\pi_\theta(\cdot\mid s_t)\right)\right]$$

其中 $V_\psi$ 是 critic（价值网络），$\hat{R}_t$ 是回报目标，$c_1$ 是价值损失系数，$c_2$ 是熵奖励系数，$\mathcal{H}$ 是策略熵。实际训练时最小化 $-\mathcal{L}_{\mathrm{PPO}}$，即对 clip 项取负、价值项取正、熵项取负。

### 3.3 奖励塑形与 GAE

RLHF 通常只在序列末尾得到 RM 分数。为了让每个 token 都有奖励信号，可以把 KL 惩罚拆到 token 级：

$$r_t=r_\phi(x,y)\cdot\mathbb{1}[t=T]-\beta\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}$$

其中 $T$ 是序列末尾，$\mathbb{1}[t=T]$ 表示只有最后一个 token 拿 RM 奖励；KL 项对每个 token 计算，累加后近似序列级 KL。注意符号：策略概率高于参考策略时，KL 惩罚为负，抑制过度偏离。

优势估计用广义优势估计（Generalized Advantage Estimation，GAE）：

$$\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t)$$

$$\hat{A}_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}$$

其中 $\gamma$ 是折扣因子（RLHF 常设 1.0），$\lambda$ 控制偏差-方差权衡（常 0.95）。$\lambda=0$ 只用一步时序差分（Temporal Difference，TD）误差，方差低偏差大；$\lambda=1$ 用完整回报，偏差低方差大。

### 3.4 PPO 为什么不稳定

- **奖励尺度**：RM 输出未归一化，不同 prompt 的奖励方差差异大；需要 running mean/std 归一化。
- **critic 欠拟合**：LLM 的价值函数很难学，尤其是长序列、稀疏终端奖励；critic 误差会直接变成策略梯度噪声。
- **KL 坍塌**：$\beta$ 太小，策略迅速偏离 SFT，语言能力崩坏；$\beta$ 太大，几乎不学习。常用自适应 KL：KL 高于目标值时增大 $\beta$，低于时减小。
- **离策略漂移**：同一批采样轨迹（rollout）上做多个 epoch，$\pi_\theta$ 与 $\pi_{\theta_{\text{old}}}$ 差距变大，clip 只能缓解不能消除。
- **长序列信用分配**：终端奖励要归因到几千个 token，GAE 的 $\lambda$ 与 critic 质量决定信号是否可用。
- **实现细节**：token 级 logprob 与序列级 logprob 混用、padding 位置未屏蔽、EOS 处理不一致、fp16 溢出，都会造成“训练看起来在跑但指标不涨”。
- **熵坍塌**：策略过早确定，探索消失；需要熵奖励或提高采样温度。

PPO 的显存也高：策略、参考策略、RM、critic 四份模型。7B 模型仅 bf16 权重就 56GB，加上优化器与激活，通常需要多卡或 LoRA/offload。

## 4. DPO

### 4.1 从 KL 约束目标推导

从同一个 KL 约束目标出发（这里的 $r$ 是未知的真实奖励，DPO 不显式建模它）：

$$\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot\mid x)}\left[r(x,y)\right]-\beta\,\mathrm{KL}\left(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right)$$

对每个 $x$ 用变分法求最优解，可得：

$$\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

$$Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

其中 $Z(x)$ 是配分函数（partition function），只与 $x$ 有关。把上式取对数并整理，奖励可以写成：

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$$

关键点：$Z(x)$ 与 $y$ 无关，在比较两个回答的奖励差时会抵消。把 $r$ 代入 Bradley-Terry：

$$P(y_w\succ y_l\mid x)=\sigma\left(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)$$

用参数化策略 $\pi_\theta$ 替换 $\pi^*$，得到 DPO 损失：

$$\mathcal{L}_{\mathrm{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

定义隐式奖励 $\hat{r}_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$，则损失就是 $-\log\sigma(\hat{r}_\theta(x,y_w)-\hat{r}_\theta(x,y_l))$，形式与 RM 训练完全一致，只是奖励由策略自身给出。实际实现用整段回答的 token logprob 之和近似 $\log\pi_\theta(y\mid x)$；是否按长度归一化是 DPO 变体的一个设计点。

### 4.2 梯度与 $\beta$ 的作用

对损失求梯度：

$$\nabla_\theta\mathcal{L}_{\mathrm{DPO}}=-\beta\,\mathbb{E}\left[\sigma\left(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w)\right)\left(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\right)\right]$$

直观理解：当模型已经更偏好 $y_w$ 时，$\sigma(\hat{r}_l-\hat{r}_w)$ 很小，梯度衰减；当模型把 $y_l$ 排得更高时，权重接近 1，梯度把 $y_w$ 的概率推高、$y_l$ 的概率压低。它不需要显式 RM，也不需要在线采样。

$\beta$ 的作用：

- $\beta$ 越大，最优策略越接近 $\pi_{\text{ref}}$（因为 $\exp(r/\beta)$ 更平），约束越强，训练越保守。
- $\beta$ 越小，策略可以偏离参考模型更远，表达偏好更强，但容易过拟合偏好数据、损失语言能力。
- 常见取值 0.01–0.5，实践多用 0.1；它同时影响隐式奖励的尺度与梯度大小，改变 $\beta$ 后要重新调学习率。

### 4.3 DPO 的优缺点

优点：

- 不需要训练 RM，不需要 PPO 的在线 rollout 与 critic，工程简单、显存低、稳定。
- 直接优化偏好对，超参少；在开源社区成为 SFT 后的默认对齐手段。
- 隐式奖励可解释：$\beta\log(\pi_\theta/\pi_{\text{ref}})$ 就是模型相对参考策略的偏好增量。

缺点：

- **离线**：只使用固定偏好数据，没有探索；策略一旦偏离数据分布，无法像 PPO 那样用新样本纠正。
- **分布偏移**：DPO 假设偏好对来自参考策略附近，实际数据可能来自更强/更弱的模型，隐式奖励的假设不成立。
- **似然位移（likelihood displacement）**：损失只约束相对差，可能同时降低 $y_w$ 和 $y_l$ 的绝对概率，导致生成质量下降。
- **长度与风格偏差**：DPO 容易学到“更长、更客气”的表面特征；需要长度归一化或 R-DPO 之类变体。
- **参考模型开销**：训练时要同时跑策略与参考模型，显存约 2 份权重；可用 LoRA 或预计算参考 logprob 降低。

### 4.4 常见变体

| 方法 | 一句话区别 |
| --- | --- |
| IPO | 把偏好间隔回归到目标值，用平方损失替代 log-sigmoid，缓解 DPO 对确定性偏好的过拟合。 |
| KTO | 不需要成对数据，只要单条回答的“好/坏”二元标签，用前景理论的非对称损失。 |
| ORPO | 把 SFT 的 NLL 与 odds-ratio 偏好损失合并成一步训练，不需要参考模型。 |
| SimPO | 用长度归一化的平均 logprob 作为隐式奖励，并加目标 margin $\gamma$，不需要参考模型。 |
| CPO | 对比偏好优化，和 SFT 联合训练，强调训练效率与稳定性。 |
| R-DPO | 在 DPO 中加入长度正则，抑制“越长越偏好”的偏差。 |

这些变体大多在“要不要参考模型、要不要成对数据、如何归一化长度、损失形状”四个维度上做取舍；没有 universally 最优，取决于数据形态与任务。

## 5. GRPO

### 5.1 去掉 critic

GRPO（Group Relative Policy Optimization）由 DeepSeekMath 提出，核心是用“同一 prompt 的一组回答”的相对表现作为基线，替代 PPO 的 critic：

1. 对每个 prompt $x$，从旧策略采样 $G$ 个回答 $\{y_1,\dots,y_G\}$。
2. 用奖励函数（RM 或验证器）给每个回答打分 $r_i$。
3. 组内归一化得到优势：

$$\hat{A}_i=\frac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

4. 用 PPO 式 clip 目标更新策略，KL 项直接加在 loss 上。

这样不需要价值网络，省掉一份模型与 critic 的训练；组内比较天然适应“同一题的不同解法”，尤其适合数学、代码这类有验证器的任务。

### 5.2 损失函数

$$\mathcal{L}_{\mathrm{GRPO}}(\theta)=-\mathbb{E}_{x,\{y_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\left[\min\left(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}\left(\rho_{i,t},1-\epsilon,1+\epsilon\right)\hat{A}_i\right)-\beta\,D_{\mathrm{KL}}\right]\right)\right]$$

其中：

$$\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}$$

$|y_i|$ 是回答长度，用于长度归一化；$\epsilon$ 是 clip 范围；$\beta$ 是 KL 系数。KL 常用无偏估计（k3）：

$$D_{\mathrm{KL}}\approx\frac{\pi_{\text{ref}}(y_{i,t}\mid s_{i,t})}{\pi_\theta(y_{i,t}\mid s_{i,t})}-\log\frac{\pi_{\text{ref}}(y_{i,t}\mid s_{i,t})}{\pi_\theta(y_{i,t}\mid s_{i,t})}-1$$

该式恒非负，方差比 $\log$ 比值估计更低。也有实现用 k1：$\log\frac{\pi_\theta}{\pi_{\text{ref}}}$。

### 5.3 GRPO 的注意点

- **组内全对/全错**：若 $G$ 个回答奖励相同，标准差为 0，优势全为 0，没有梯度；需要难度适中的 prompt 与足够的采样。
- **标准差归一化偏差**：除以组内 std 会让“简单题的小差异”被放大；Dr. GRPO 等工作建议去掉 std、只用减均值，或做难度分层。
- **采样成本**：每个 prompt 要生成 $G$ 条回答（常 8–64），长 CoT 下 rollout 成本很高；需要高效推理引擎与异步调度。
- **奖励设计**：规则奖励（答案匹配、单测）比 RM 更难被 hack，但覆盖窄；格式奖励要与正确性奖励解耦，避免“只学会格式”。
- **长度爆炸**：RL 会诱导更长的 CoT，需长度惩罚或最大长度截断；截断会破坏 advantage 的完整性。

### 5.4 RLOO 与拒绝采样微调

**RLOO（REINFORCE Leave-One-Out）**：同样对每个 prompt 采样 $G$ 个回答，但基线用“除自己以外的平均奖励”：

$$\hat{A}_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j$$

它不需要 critic，也不做 std 归一化，估计是无偏的；实现比 GRPO 更简单，但方差略高，常与 PPO 式 clip 结合。

**拒绝采样微调（Rejection Sampling Fine-Tuning，RFT）**：对每个 prompt 采样 $K$ 个回答，用验证器或 RM 筛出正确/高分的，拿这些“好样本”做 SFT。它没有负样本，不直接优化偏好，但实现简单、稳定，适合冷启动；缺点是只模仿成功样本，学不到“为什么错”，且需要大量采样。

RFT 可以迭代：当前模型采样 → 筛选 → SFT → 再采样。STaR 是早期代表；DeepSeek-R1 的后训练也用了拒绝采样生成高质量 CoT 数据。

### 5.5 PPO、DPO、GRPO 对比

| 维度 | PPO | DPO | GRPO |
| --- | --- | --- | --- |
| 奖励来源 | 显式 RM | 偏好对（隐式奖励） | RM 或验证器 |
| critic | 需要 | 不需要 | 不需要 |
| 在线采样 | 需要 | 不需要 | 需要（每组 $G$ 条） |
| 参考模型 | 需要 | 需要 | 需要 |
| 显存 | 最高（4 个模型） | 中（2 个模型） | 中高（策略+参考+采样） |
| 稳定性 | 低，调参多 | 高 | 中，组方差与长度敏感 |
| 适用 | 有可靠 RM、需在线探索 | 有高质量偏好对、算力有限 | 可验证奖励、推理任务 |

## 6. 推理模型的后训练

### 6.1 RLVR

可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards，RLVR）用程序化验证器替代 RM：

- 数学：最终答案与标准答案精确匹配（需处理等价形式、单位、分数）。
- 代码：单元测试通过率、编译错误、运行时限制。
- 逻辑/指令：格式检查、约束满足、工具调用 schema 校验。
- 奖励通常是 0/1，可加格式奖励（如把答案放进 `\boxed{}`）与长度惩罚。

RLVR 的优势是奖励客观、难以被语言模型“讨好”，适合大规模采样；局限是只覆盖有验证器的领域，奖励稀疏，模型可能学会“猜答案”“写测试通过但错误的代码”等捷径。DeepSeek-R1-Zero 用 GRPO + 规则奖励，在没有 SFT 冷启动的情况下观察到长 CoT、自我检查与“aha moment”式行为；但输出可读性差、语言混杂，R1 随后加入冷启动 SFT、拒绝采样与多阶段 RL。

### 6.2 PRM 与 ORM

| 维度 | ORM（Outcome Reward Model） | PRM（Process Reward Model） |
| --- | --- | --- |
| 打分对象 | 最终答案 | 每个推理步骤 |
| 信号密度 | 稀疏 | 稠密 |
| 标注成本 | 低 | 高（人工或 MCTS 自动标注） |
| 信用分配 | 难，长 CoT 尤甚 | 容易定位错误步骤 |
| 风险 | 结果对但过程错也被奖励 | 步级标签噪声、步骤边界定义 |
| 推理用法 | best-of-N、RL 奖励 | 逐步搜索、剪枝、验证器 |

PRM 的典型用途是测试时搜索：生成多个候选步骤，用 PRM 打分，保留高分路径（beam search / tree search）。它能把额外算力转化为准确率，但收益取决于 PRM 的质量与搜索策略；PRM 也会被策略钻空子（写出“看起来正确”的步骤）。

### 6.3 长 CoT 与过度思考

RL 训练会显著增加回答长度：模型发现“多写步骤”更容易得到正确结果，长度与准确率在训练早期正相关。但长度不是能力本身：

- **过度思考（overthinking）**：简单题也生成数千 token，延迟与成本上升，甚至因自我怀疑而答错。
- **长度偏置**：奖励模型偏好长回答，RL 进一步放大；需要长度惩罚、长度归一化或长度分桶。
- **自适应思考**：按难度决定是否展开 CoT，简单题直接答、难题多步推理；可用难度分类器、思考预算 token、早停。
- **测试时扩展**：best-of-N、多数投票、PRM 引导搜索；收益随采样数递增但会饱和，且成本线性增长。

### 6.4 蒸馏与对齐的关系

对齐后的强模型可以作为教师，把“格式、拒答、推理链”蒸馏给学生：

- **离线蒸馏**：教师生成回答，学生做 SFT。简单、稳定，但学生只见到教师的输出分布，存在暴露偏差（exposure bias）。
- **在线蒸馏（on-policy distillation）**：学生自己生成，教师对学生的 token 分布打分，最小化两者 KL。学生在自己会遇到的分布上学习，缓解 exposure bias；代价是需要同时跑教师与学生。
- **序列级 vs token 级**：序列级只用教师最终答案做 SFT；token 级用教师 logits 做 KL，信号更密，但要求词表一致、教师推理成本高。
- **能力边界**：学生容量不足时，长 CoT 蒸馏可能只学到“更长”而不是“更会推理”；需要课程学习、长度控制与验证器筛选。

token 级在线蒸馏的目标可写为：

$$\mathcal{L}_{\text{OPD}}(\theta)=\mathbb{E}_{y\sim\pi_\theta}\left[\sum_{t=1}^{|y|}\mathrm{KL}\left(\pi_T(\cdot\mid s_t)\,\|\,\pi_\theta(\cdot\mid s_t)\right)\right]$$

其中 $\pi_T$ 是教师分布，$s_t=(x,y_{<t})$ 是学生自己生成的前缀。KL 的方向会影响行为：教师到学生的前向 KL 倾向覆盖教师的全部模式，学生到教师的反向 KL 更聚焦高概率模式，但可能忽略教师的部分分布。

蒸馏不是对齐的替代品：它迁移的是教师的行为分布，不能保证学生在新分布上仍然安全；安全对齐通常还要单独的偏好数据与红队评测。

## 7. 关键损失的 PyTorch 实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    # 序列 logprob 为 token logprob 之和；也可按长度归一化
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()

def grpo_advantage(rewards, eps=1e-4):
    # rewards: 同一 prompt 的 G 个回答得分
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)

def ppo_surrogate(new_logp, old_logp, advantage, clip=0.2):
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
    return torch.min(unclipped, clipped)
```

`dpo_loss` 中的 logprob 是整段回答的 token logprob 之和；若做长度归一化，等价于 SimPO 一类变体。`grpo_advantage` 的 `std` 归一化在组内奖励相同时会退化为 0 优势，工程上要加 `eps` 并过滤全对/全错的组。`ppo_surrogate` 只计算 clip 部分，实际损失还要加价值损失与熵项。

## 8. 面试高频问题

### 1. RLHF 的三个阶段分别解决什么问题？

- SFT 教格式与基本对话，给 RL 一个可用的初始策略。
- RM 从成对偏好中学习人类打分，把“好坏”变成可优化的标量。
- PPO 用 RM 奖励优化策略，KL 约束防止偏离 SFT 太远。
- 三阶段的数据量、学习率和目标都不同；工业界常用 DPO/RFT/RLVR 替代其中某些阶段。

### 2. Bradley-Terry 模型的假设和损失是什么？

- 假设偏好只由奖励差决定，$P(y_w\succ y_l)=\sigma(r_w-r_l)$。
- 损失 $-\log\sigma(r_w-r_l)$，只约束奖励差，不约束绝对尺度。
- 假设偏好可传递、无平局、标注噪声可控；现实中常有循环偏好与不一致。
- RM 需要奖励归一化、长度控制、集成与定期重训，否则会被策略 hack。

### 3. PPO 的 clip 目标如何工作？为什么要 KL 惩罚？

- 重要性比 $\rho_t=\pi_\theta/\pi_{old}$ 衡量策略变化；clip 把 $\rho_t$ 限制在 $[1-\epsilon,1+\epsilon]$。
- 优势为正时最多奖励到 $1+\epsilon$，为负时最多惩罚到 $1-\epsilon$，防止一步更新过大。
- KL 惩罚把策略拉在参考模型附近，防止为了骗 RM 而崩坏语言能力。
- clip 不是硬约束，多个 epoch 后仍会漂移；需要监控 KL、clip fraction 与熵。

### 4. GAE 的 $\lambda$ 和 $\gamma$ 分别控制什么？

- $\gamma$ 是折扣因子，RLHF 常设 1.0，因为终端奖励在序列末尾。
- $\lambda$ 控制偏差-方差：0 用一步 TD 误差（低方差高偏差），1 用完整回报（低偏差高方差）。
- GAE 把多步 TD 误差按 $(\gamma\lambda)^l$ 加权，常取 0.95。
- critic 的质量决定优势估计是否可用；critic 欠拟合时 PPO 会退化成高方差策略梯度。

### 5. 请推导 DPO 的损失。

- 从 $\max_\pi \mathbb{E}[r]-\beta KL(\pi\|\pi_{ref})$ 出发，最优解 $\pi^*\propto\pi_{ref}\exp(r/\beta)$。
- 取对数得 $r=\beta\log(\pi^*/\pi_{ref})+\beta\log Z(x)$，$Z$ 与 $y$ 无关。
- 代入 BT，$Z$ 在奖励差中抵消，得到 $\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$。
- 取负对数就是 DPO loss；它隐式定义奖励，不需要显式 RM。

### 6. DPO 的 $\beta$ 有什么作用？DPO 有什么缺点？

- $\beta$ 控制对参考策略的约束：越大越保守，越小越允许偏离。
- 常见 0.1；改变 $\beta$ 会改变隐式奖励尺度与梯度，需要重调学习率。
- 缺点是离线、无探索、分布偏移、似然位移、长度/风格偏置。
- 训练时要同时跑参考模型，显存约两份权重；可用 LoRA 或预计算参考 logprob。

### 7. GRPO 和 PPO 的核心区别是什么？

- GRPO 去掉 critic，用同一 prompt 的一组回答的组内均值/标准差作为基线。
- 优势 $\hat A_i=(r_i-\text{mean})/\text{std}$，不需要价值网络，省一份模型。
- 仍需在线采样、参考模型与 PPO 式 clip；适合可验证奖励与推理任务。
- 组内全对/全错时梯度为 0；std 归一化会放大简单题的小差异，需注意。

### 8. RLOO 和拒绝采样微调分别是什么？

- RLOO 用“留一法”基线：$r_i$ 减去除自己外的平均奖励，无 critic、无 std 归一化。
- RFT 采样多个回答，用验证器筛出正确/高分的做 SFT，不利用负样本。
- RLOO 是在线策略梯度，RFT 是离线/迭代式监督学习；两者常结合使用。
- RFT 简单稳定但只模仿成功样本；RLOO 方差更高但能利用失败信号。

### 9. RLVR 是什么？它和 RLHF 的区别在哪？

- RLVR 用程序化验证器（答案匹配、单测、约束检查）给奖励，不训练 RM。
- 奖励客观、抗奉承，适合数学/代码；但只覆盖可验证领域，奖励稀疏。
- RLHF 用人类偏好训练 RM，覆盖开放式任务，但 RM 会被 hack。
- 推理模型常用 RLVR + GRPO；通用对话仍需要偏好数据与安全对齐。

### 10. PRM 和 ORM 的区别？长 CoT 为什么会出现过度思考？

- ORM 只给最终结果打分，信号稀疏、信用分配难；PRM 给每个步骤打分，信号稠密但标注贵。
- PRM 可用于逐步搜索/剪枝，ORM 多用于 best-of-N 或 RL 终端奖励。
- RL 会奖励“多写步骤”，导致简单题也生成长 CoT，增加延迟与成本。
- 用长度惩罚、难度自适应、思考预算与早停控制；长不等于强，需看准确率-成本曲线。

> 本文档为面试复习资料，公式与数字建议结合原始论文复核。
