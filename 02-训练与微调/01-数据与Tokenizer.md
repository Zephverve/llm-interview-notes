# 预训练数据与分词器（Tokenizer）

**本章导读**

- 预训练模型的能力上限很大程度由数据决定：来源覆盖度、清洗质量、去重强度、配比与课程安排，往往比模型结构改动带来更大收益。
- 分词器（Tokenizer）是文本与参数之间的唯一接口，词表大小、切分粒度、数字与中文处理会直接影响训练效率、推理成本和多语言公平性。
- 样本打包（packing）、损失掩码（loss mask）与多轮拼接决定了模型实际看到什么监督信号，也是 SFT 阶段最容易出错、最容易被面试追问的工程细节。
- 数据污染（benchmark contamination）会让评测虚高，必须把“去污染”当成数据流水线的固定环节，而不是事后补救。

**核心要点**

- 预训练数据以网页为主，代码、书籍/论文、合成数据负责补足推理、长文与稀缺能力；配比需要用代理实验验证。
- 清洗遵循“抽取 → 语言识别 → 启发式过滤 → 分类器打分 → 安全与版权过滤 → 去重”的顺序，越靠前的步骤越便宜。
- MinHash + LSH 用 Jaccard 相似度做近重复检测，阈值由 band 数 $b$ 与行数 $r$ 控制，近似为 $t\approx(1/b)^{1/r}$。
- DoReMi 用小型代理模型学习领域权重，核心是最小化各领域相对参考模型的超额损失，而不是直接优化平均损失。
- 字节对编码（Byte Pair Encoding，BPE）、WordPiece、Unigram 的差别在“合并准则”和“是否有概率模型”；SentencePiece 解决的是语言无关的预处理与可逆解码。
- 词表越大，序列越短但嵌入（embedding）/输出层参数与 softmax 成本越高；中文、代码、多语言场景通常需要 100k 量级的词表。
- 监督微调（Supervised Fine-Tuning，SFT）只对助手（assistant）片段计算 loss；packing 必须配合 document mask 或可接受的跨文档注意力，否则会引入脏上下文。
- 去污染应同时覆盖预训练与 SFT 数据，常用 13-gram 重叠、embedding 近邻和 canary 字符串检测。

## 1. 预训练数据来源

### 1.1 主要来源与角色

| 来源 | 典型语料 | 规模量级（估算） | 主要作用 | 主要风险 |
| --- | --- | --- | --- | --- |
| 网页 | Common Crawl、FineWeb、RefinedWeb | 原始 PB 级；清洗后 3T–15T token | 覆盖世界知识与语言多样性 | 噪声、模板、重复、低质 |
| 代码 | GitHub 宽松许可证仓库、StackExchange | 0.5T–2T token | 结构化推理、工具使用、语法严谨性 | 许可证、漏洞、测试泄漏 |
| 书籍/论文 | Gutenberg、arXiv、PubMed Central | 100B–500B token | 长文连贯性、专业知识 | 版权、过时、公式解析 |
| 百科/问答 | Wikipedia、Wikidata、StackExchange | 50B–200B token | 事实密度高、风格规范 | 覆盖偏斜、更新滞后 |
| 合成数据 | 教科书式改写、指令回译、拒绝采样 | 10B–1T token | 补足推理链与稀缺格式 | 模型自噬、事实幻觉放大 |
| 多语言 | mC4、CC-100、各语言本地语料 | 按语言差异极大 | 非英语能力与公平性 | 语料质量参差、tokenizer  fertility 高 |

Common Crawl 自 2008 年以来累计抓取数百亿页面，每月新增约 30 亿页面（量级估算）。原始 HTML 不能直接训练：正文抽取、语言识别和去噪后，通常只保留原始文本 token 的 10%–25%（估算，随过滤强度变化）。FineWeb 从约 96 个 Common Crawl 快照得到 15T token；RefinedWeb 抽取 5T token 后保留约 600B；Dolma 约 3T；SlimPajama 约 627B；C4 约 156B；The Pile 约 800GB、300B token 量级。上述数字以公开资料为准，不同论文的计数口径（是否含重复、是否用其 tokenizer）不同。

Llama 3 的公开描述是约 15T token、30 种语言、非英语约 5%，配比大致为通用知识 50%、数学与推理 25%、代码 17%、多语言 8%（约数，以论文为准）。这组配比说明：网页仍是主体，但代码和数学被显著上采样，因为这两类数据在自然网页中占比低、对能力提升的边际收益高。

### 1.2 合成数据的边界

合成数据的价值在于“把稀缺结构显式写出来”，例如把网页改写成教科书章节、为代码生成解释、对数学题做多解扩写。风险是模型自噬（model collapse）：若教师模型的事实错误被反复采样，错误会集中放大；若合成数据风格单一，学生模型会丢失分布多样性。经验做法是：

- 合成数据占比控制在 5%–30%（估算），并用真实数据做锚点。
- 用可验证信号过滤（代码单测、数学答案校验、引用回溯），而不是只依赖教师模型打分。
- 保留一定比例的“未改写”原始语料，避免分布收窄。
- 教师模型与被训练模型规模差距越大，蒸馏收益越明显，但事实性仍需外部检索或校验。

## 2. 清洗与质量过滤

### 2.1 流水线

```python
# 数据清洗流水线（伪代码，展示顺序与取舍）
def clean_documents(raw_pages, target_langs=("zh", "en"), quality_threshold=0.6):
    for page in raw_pages:
        text = extract_main_content(page.html)          # trafilatura/resiliparse，去掉导航与广告
        if len(text) < 200:                             # 过短页面信息密度低
            continue
        lang, prob = fasttext_langid(text)              # lid.176，阈值通常 0.65 左右
        if lang not in target_langs or prob < 0.65:
            continue
        if not passes_gopher_rules(text):               # 词数、平均词长、符号比、重复行
            continue
        if classifier_score(text) < quality_threshold:  # 教育/质量分类器
            continue
        text = remove_pii(text)                         # 邮箱、电话、身份证号、密钥
        text = copyright_filter(text)                   # 版权声明、盗版站点、许可白名单
        yield normalize_unicode(text)
```

顺序的原则是“便宜的先做”：URL 与语言过滤能在毫秒级砍掉大量数据；分类器推理贵，放在启发式之后；去重通常放在质量过滤之后，避免在垃圾数据上浪费哈希与索引成本。但若重复数据极多，也可以先做一轮 URL/段落级精确去重。

### 2.2 启发式规则

Gopher 规则是最常被引用的一组硬阈值（不同团队会调整）：

- 词数在 50–100,000 之间；平均词长 3–10 个字符。
- `#` 与省略号的符号/词比例低于 0.1。
- 以 bullet 开头的行占比低于 90%；以省略号结尾的行占比低于 30%。
- 重复行占比、重复段落占比均低于 30%；最高频 2-gram 占比低于 20%。

C4 的规则更偏向“像正常文章”：保留以终止标点结尾的行，删除含 “javascript”、“lorem ipsum”、“terms of use”、脏词或明显代码模板的行，再按 3 句窗口拼接。规则便宜、可解释，但会系统性删掉对话、列表、公式、代码等格式；因此代码和数学语料通常使用独立的清洗器，而不是套用网页规则。

### 2.3 分类器质量过滤

两类做法最常见：

1. **正负样本分类器**：以 Wikipedia、书籍、高评分网页为正例，以随机网页为负例训练 fastText（hashed n-gram，维度 $2^{18}$，2–5 个 epoch）。输出分数高于阈值才保留。优点是速度快、可在线打分；缺点是把“像百科”当成“高质量”，会压制对话、口语和代码。
2. **LLM 标注 + 小模型蒸馏**：用强模型给几十万条样本打 0–5 分（教育价值、信息密度、可读性），再蒸馏成 BERT/fastText 分类器。FineWeb-Edu 用约 46 万条 Llama-3-70B 标注训练分类器，保留分数 $\ge 3$ 的网页，得到约 1.3T token（数字为公开量级）。这种方式更贴近“对训练有用”，但标注成本高、会继承教师偏好。

阈值不是越低越好。经验上，质量过滤保留率从 30% 降到 10% 带来的收益，可能小于多样性损失；小模型阶段尤其如此，因为小模型容量有限，重复的高质数据更容易被记住。需要在下游任务和验证集 loss 上同时评估。

### 2.4 安全、版权与个人信息（PII）

- **敏感内容**：成人、仇恨、暴力、个人信息。常用 DoS（Dolma）与 NSFW 分类器，按概率过滤或降采样；注意“讨论暴力”与“鼓励暴力”的区别，不能只靠关键词。
- **PII**：正则 + 命名实体识别删除邮箱、电话、身份证、地址、API key；代码语料还要扫描 `.env`、私钥、内部域名。
- **版权**：优先使用许可明确的代码（MIT/Apache/BSD），对书籍与新闻尊重 robots.txt 与许可；记录数据来源与许可证，便于追溯和下架。版权是法律问题，不是模型能力问题，工程上应保留文档级元数据。
- **毒性残留**：即使过滤后仍可能有偏斜，需要在对齐阶段用安全数据与红队评测补足。

## 3. 去重

### 3.1 精确去重与子串去重

文档级精确去重：对规范化文本（统一空白、去 HTML、转小写可选）计算 SHA-256，保留首次出现的副本。它能去掉镜像站点、重复抓取和模板页，但对“改了几个词”的近似重复无效。

子串去重更强：把所有文档拼成一个大序列，用后缀数组（suffix array）找出长度超过阈值（如 50 个 token）的重复子串，只保留一处。Google 的“Deduplicating Training Data Makes Language Models Better”工作表明，训练语料中存在大量长重复片段，去掉后能显著降低逐字记忆（memorization），且不损失下游效果。代价是后缀数组构建与扫描成本高，通常离线跑一次。

### 3.2 MinHash + LSH

近重复检测常用局部敏感哈希（Locality-Sensitive Hashing，LSH）把“可能相似”的文档快速分桶。核心是 Jaccard 相似度：

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}$$

其中 $A,B$ 是两篇文档的 shingle 集合（常用 5-gram 词或字符）。直接两两比较是 $O(N^2)$，所以用 MinHash 把集合压成签名：对集合做 $m$ 个随机哈希，取每个哈希的最小值，得到 $m$ 维签名。两个集合的签名在某一维相等的概率恰好等于 $J(A,B)$，因此签名相同的维度比例是 Jaccard 的无偏估计。

LSH 再把 $m$ 维签名切成 $b$ 个 band、每个 band 有 $r$ 行（$m=br$）。只要某个 band 完全相同，两篇文档就成为候选对。候选概率为：

$$P_{\text{candidate}}(J)=1-\left(1-J^{r}\right)^{b}$$

令 $P\approx 1/2$ 可解出近似阈值：

$$t\approx\left(\frac{1}{b}\right)^{1/r}$$

例如 $b=20,r=5$ 时 $t\approx(1/20)^{1/5}\approx 0.55$；$b=8,r=8$ 时 $t\approx(1/8)^{1/8}\approx 0.77$；$b=4,r=6$ 时 $t\approx0.79$。$r$ 越大，曲线越陡（误报少、漏报多）；$b$ 越大，召回越高但候选对更多。工程上常用 $m=128$、$b=16$–$32$，再对候选对做精确 Jaccard 校验。

```python
# MinHash + LSH 的核心逻辑（示意）
def minhash_signature(tokens, num_perm=128):
    shingles = {" ".join(tokens[i:i + 5]) for i in range(len(tokens) - 4)}
    return [min(hash64(s, seed) for s in shingles) for seed in range(num_perm)]

def lsh_bands(sig, num_bands=20):
    rows = len(sig) // num_bands
    for b in range(num_bands):
        yield (b, tuple(sig[b * rows:(b + 1) * rows]))

buckets = {}
for doc_id, tokens in corpus:
    for key in lsh_bands(minhash_signature(tokens)):
        buckets.setdefault(key, []).append(doc_id)   # 同桶即候选对，再做精确校验
```

### 3.3 SimHash

SimHash 把文档映射成 64 位指纹：对每个 token 做 64 位哈希，按 token 权重在每一位上投票（哈希位为 1 加权重，为 0 减权重），最后取符号位。两篇文档的 SimHash 汉明距离（Hamming distance）越小越相似；经验上距离 $\le 3$ 可视为近重复。它比 MinHash 更省空间（每篇 8 字节），适合海量网页的粗筛，但对长文档和不同长度的文档不如 Jaccard 稳定。候选检索可用“鸽巢原理”：把 64 位切成 4 段 16 位，若汉明距离 $\le 3$，至少有一段完全相同，因此只需在同段同值的桶里比较。

### 3.4 语义去重与去重的影响

语义去重先用 embedding 聚类（如 Sentence-BERT、SimHash 的连续版本），再在簇内保留代表样本。它能去掉“换了说法”的重复，但也可能误删不同观点；通常只对高相似度簇（余弦相似度 > 0.95）操作。

去重的收益：减少逐字记忆与隐私泄露、提升每个 token 的有效信息量、降低训练集规模与训练成本。风险：过度去重会删掉合法的常见表达（法律条款、代码模板、数学题题干），导致模型对这些模式欠拟合；对预训练而言，完全去重可能损失自然语言的重复统计特性。实践上按文档级 + 段落级组合，保留少量重复，并用验证集消融确定阈值。

## 4. 数据配比与课程学习

### 4.1 配比为什么重要

不同领域对梯度的贡献不同。代码和数学在网页中占比低，但对推理与工具使用收益高；多语言数据在英语主导的语料中容易被淹没。配比的目标不是“让平均 loss 最低”，而是“在目标能力上取得最好的边际收益”。常见做法：

- **人工先验**：通用网页 50%–70%、代码 10%–20%、数学 5%–15%、多语言 5%–10%（量级估算，随模型定位变化）。
- **小模型代理实验**：在 100M–1B 参数模型上扫多组配比，选下游平均分最高的，再外推到大模型。代理模型与目标模型的排序不一定一致，需要至少两个规模验证。
- **DoReMi**：用参考模型与代理模型的领域超额损失自动学习权重。
- **RegMix**：训练大量小模型拟合“配比 → 验证 loss”的回归，再用优化器搜索配比，适合领域多、组合爆炸的场景。

### 4.2 DoReMi 的思想

DoReMi 先训练一个参考模型（通常用均匀或原始配比），再训练代理模型，并在训练过程中用 group DRO 调整领域权重：

$$\min_{\theta}\max_{w\in\Delta}\sum_{i=1}^{k}w_i\left(L_i(\theta)-L_i^{\mathrm{ref}}\right)$$

其中 $k$ 是领域数，$L_i(\theta)$ 是当前模型在领域 $i$ 上的损失，$L_i^{\mathrm{ref}}$ 是参考模型在同一领域的损失，$w_i$ 是领域权重且 $\sum_i w_i=1$。内层最大化会把权重推向“相对参考模型进步最小”的领域，迫使模型补齐短板；外层最小化则降低这些领域的损失。

实际优化用指数梯度（exponentiated gradient）：

$$w_i^{(t+1)}\propto w_i^{(t)}\exp\left(\eta\left(L_i(\theta_t)-L_i^{\mathrm{ref}}\right)\right)$$

其中 $\eta$ 是步长。若某领域当前损失高于参考模型（超额损失为正），其权重被上调。训练结束后，用学到的 $w$ 在目标大模型上采样数据。DoReMi 的假设是“代理模型学到的领域难度排序可迁移”，但领域粒度过粗、参考模型选择不当、代理规模过小都会让权重失真；因此它通常作为先验，而不是最终答案。

### 4.3 课程学习与退火

课程学习（curriculum learning）按“从易到难”安排数据：先短文本、通用网页，再长文、代码、数学；或在训练后期提高高质量数据的采样概率。更常见的是**学习率退火阶段的数据切换**：最后 10%–20% 的 token 把学习率余弦衰减到接近 0，同时切换到高质量、指令化、长上下文数据。Llama 3.1 在预训练后期用长上下文数据做继续训练，把上下文从 8k 扩到 128k，这类“中期训练（midtraining）”已成为标准流程。

课程学习的风险是“先学简单样本会固化浅层模式”，且难度定义依赖模型本身。可操作的信号包括：序列长度、loss 分桶、语法树深度、代码复杂度、题目所需的推理步数。没有可靠难度信号时，随机打乱 + 后期质量上采样通常更稳。

### 4.4 数据回放防止遗忘

继续预训练或领域微调时，模型会遗忘通用能力（catastrophic forgetting）。数据回放（data replay / rehearsal）是在新数据中混入 1%–20% 的原始预训练数据或通用 SFT 数据，保持旧分布的梯度。回放比例越高，遗忘越少，但新任务收敛越慢；经验上 5%–10% 是常用起点。

回放也可以按“能力”组织：代码微调时混入通用对话，数学微调时混入自然语言推理，安全微调时混入正常请求以避免过度拒答。与 KL 正则、LoRA 低秩约束相比，回放更简单、无需额外参考模型，但需要保存一部分旧数据并重新采样。

## 5. 分词器

### 5.1 BPE

字节对编码（Byte Pair Encoding，BPE）从字符或字节的词表开始，反复统计相邻 token 对的频率，把最高频的一对合并为新 token，直到达到目标词表大小。训练结束后，合并规则按学习顺序保存；编码时按顺序应用合并，或等价地用 rank 做贪心最长匹配。

字节级 BPE（byte-level BPE）以 256 个字节为初始词表，因此任何 Unicode 文本都能被表示，不会出现 `<unk>`；GPT-2 先用正则把文本切成词/数字/标点片段，再在片段内做 BPE，避免跨词合并。BPE 的目标是“用最少的 token 表示语料”，没有概率模型，分词结果是确定性的。

```python
# BPE 训练的核心循环（简化版，展示合并准则）
def train_bpe(corpus, vocab_size):
    vocab = {bytes([i]) for i in range(256)}
    while len(vocab) < vocab_size:
        pairs = Counter()
        for word, freq in corpus.items():          # word 是字节序列
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        best = max(pairs, key=pairs.get)           # 最高频相邻对
        vocab.add(best[0] + best[1])
        corpus = merge_pair(corpus, best)          # 所有出现位置合并
    return vocab
```

### 5.2 WordPiece

WordPiece 同样从字符开始合并，但选择准则不是频率，而是“合并后训练语料似然提升最大”：

$$\text{score}(a,b)=\frac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}$$

它近似于点互信息（PMI），倾向于把经常一起出现、但各自也常见的片段合并。WordPiece 用 `##` 表示非词首子词（如 `playing` → `play` + `##ing`），编码时做贪心最长匹配。BERT 使用 30k 词表，多语言 BERT 使用约 110k–120k。它没有生成式概率解释，训练与解码都比 Unigram 简单。

### 5.3 Unigram

Unigram 语言模型假设文本由独立子词生成，每个子词有概率 $p(t)$，一句话的概率是 $\prod_t p(t)$。训练分三步：

1. 用较大种子词表（所有高频子串或 BPE 词表）初始化。
2. EM：E 步用 Viterbi 找当前概率下的最优切分并统计期望频次；M 步用频次估计 $p(t)$。
3. 剪枝：计算删除某个 token 后语料似然的损失，删掉影响最小的一批，直到目标词表大小；重复 EM。

编码时用 Viterbi 最大化 $\sum_t \log p(t)$，因此同一文本可能有多种切分，Unigram 是概率模型，支持采样与多切分。它倾向于保留高频整词，对中文、日文等无空格语言较友好。

### 5.4 SentencePiece

SentencePiece 是“预处理 + 分词”的完整工具，把原始文本（含空格）直接作为输入，用 `▁` 表示空格，因此解码是可逆的；支持 BPE 与 Unigram 两种算法、字节回退（byte fallback）和 NFKC 规范化。它不依赖语言特定的分词器，适合多语言与中文。Llama 系列、T5、Gemma 使用 SentencePiece；Qwen 使用 tiktoken 风格的字节级 BPE；GPT-4o 的 o200k_base 也是字节级 BPE。

### 5.5 四种方案对比

| 维度 | BPE | WordPiece | Unigram | SentencePiece |
| --- | --- | --- | --- | --- |
| 合并准则 | 最高频相邻对 | 似然增益/PMI | EM + 似然剪枝 | 封装 BPE 或 Unigram |
| 是否有概率模型 | 否 | 否 | 是 | 取决于算法 |
| 是否语言无关 | 字节级可做到 | 需要预分词 | 需要种子词表 | 是，空格转 `▁` |
| OOV | 字节级无未登录词（OOV） | 可能 `<unk>` | 可用 byte fallback | 支持 byte fallback |
| 典型使用者 | GPT 系列、Qwen | BERT 系列 | T5、ALBERT | Llama、Gemma、T5 |
| 解码可逆性 | 需额外处理空格 | 需保留 `##` | 依赖实现 | 好，空格可还原 |

### 5.6 词表大小的权衡

词表大小 $V$ 与隐藏维度 $d$ 决定 embedding 参数量：若输入输出不共享，约 $2Vd$；若权重绑定（weight tying），约 $Vd$。以 $V=128{,}256$、$d=4096$ 为例，单个矩阵约 $5.25\times10^8$ 参数（0.53B），绑定后占 7B 模型的约 7.5%；$V=32{,}000$ 时约 1.31 亿参数，占约 1.9%。词表扩大 4 倍，embedding 参数也扩大 4 倍，而 Transformer 主体不变。

推理时输出层每个 token 要做 $V$ 维 softmax，计算量约 $2Vd$ FLOPs/token；$V=128k,d=4096$ 约 1.05 GFLOP/token，而 7B 模型前向约 14 GFLOP/token，输出层占约 7%。更关键的是显存带宽：bf16 下输出矩阵约 1GB，生成每个 token 都要读一遍，词表越大解码越受带宽限制（估算）。

压缩率用“每 token 字符数”衡量。英文在 32k 词表下约 4 字符/token；中文在英语主导的 32k 词表下可能只有 1.3–1.6 字符/token，因为一个汉字常被拆成 2–3 个字节 token；128k 多语言词表可到 2–3 字符/token（量级估算，随语料与规范化变化）。词表太小 → 序列变长、注意力成本上升、中文/代码被切碎；词表太大 → 稀有 token 训练不足、输出层成本高、多语言之间互相挤占。

常见规模：Llama 2 为 32k，Llama 3 为 128,256，Qwen2 为 151,646，Gemma 为 256k，GPT-4o 的 o200k 约 200k。选择时要看目标语言分布：多语言模型通常需要 100k 以上，并以温度采样平衡各语言的词表分配。

### 5.7 数字、空格与中文

- **数字切分**：GPT-2 的正则把连续数字按 1–3 位分组，有些新词表则把每个数字单独切分。逐位切分让模型更容易做位数对齐和算术，但序列变长；三位分组压缩率更高，却可能让模型把 “123” 当成一个整体 token，泛化到 “124” 时更差。没有定论，需用算术评测消融。
- **前导空格**：`"hello"` 与 `" hello"` 在字节级 BPE 中通常是不同 token。聊天模板、代码缩进、换行都依赖空格 token；构造训练样本时不要随意 `strip()`，否则分布偏移。
- **中文**：中文没有空格，SentencePiece 的 `▁` 只能标在片段开头；字节级 BPE 会把一个汉字拆成 3 个 UTF-8 字节，若词表没有学好中文，压缩率会显著变差。中文还涉及全角/半角、繁简、NFKC 归一化，建议在分词前统一，避免同一语义多种写法。
- **代码**：缩进、换行、运算符周围空格都有意义；代码语料应单独统计词表，避免 Python 缩进被合并成奇怪 token。

### 5.8 特殊 token

常见特殊 token 包括：`<bos>`、`<eos>`、`<pad>`、`<unk>`、角色标记（如 `<|im_start|>`、`<|im_end|>`）、工具调用标记、图像占位符、FIM（fill-in-the-middle）的 `<|fim_prefix|>`/`<|fim_middle|>`/`<|fim_suffix|>`，以及为未来能力预留的 reserved token。

要点：

- 特殊 token 必须在分词器中被注册为“原子”，不能被 BPE 拆开；构造数据时用字符串替换或 `tokenizer.add_special_tokens`。
- SFT 中 EOS/EOT 必须出现在 assistant 回答末尾且参与 loss，否则模型学不会停止。
- padding token 最好独立；若复用 EOS，要确保 label 中的 pad 被设为 `-100`，否则模型会把 padding 当成监督目标。
- 特殊 token 的 embedding 初始化为现有 token 的均值，比随机初始化更稳。
- 词表预留会占用 softmax 维度但不参与训练；reserved token 太多会浪费参数，太少则后续扩展麻烦。

### 5.9 词表扩展与 embedding resize

扩展词表的典型流程：

```python
# 词表扩展：新增领域 token 或特殊 token
new_tokens = ["<|tool_call|>", "<|/tool_call|>", "量子纠缠", "Transformer"]
num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
model.resize_token_embeddings(len(tokenizer))       # 输入 embedding 与输出 head 一起扩展

with torch.no_grad():
    old_mean = model.get_input_embeddings().weight[:-num_added].mean(dim=0)
    model.get_input_embeddings().weight[-num_added:] = old_mean
    if not model.config.tie_word_embeddings:
        model.get_output_embeddings().weight[-num_added:] = old_mean
```

注意事项：

- 新增 token 的 embedding 若随机初始化，初期 loss 会尖峰；用旧词表均值或同标准差的正态初始化，并给新 token 更高学习率（如 5–10 倍）或先冻结旧 embedding 训练若干步。
- 若模型绑定输入输出权重，`resize_token_embeddings` 会重新绑定；若解绑，要同时扩展 `lm_head`，并检查 `config.vocab_size` 与 `len(tokenizer)` 一致。
- 新 token 需要足够出现次数（经验上至少数千到数万次，估算）才能训好；只加特殊 token 而不加训练数据，新 token 的表示会一直接近初始化。
- 扩展后要重新保存 tokenizer（`tokenizer.save_pretrained`），并确认 chat template、`pad_token_id`、`eos_token_id` 都已更新。
- 词表扩展会改变输出层维度，旧 checkpoint 的 logits 无法直接对齐；若做继续预训练，建议先在小学习率下“预热”新 embedding，再全量训练。
- 评测要覆盖旧能力，防止扩展引入的分布偏移导致通用能力下降。

## 6. 训练样本构造

### 6.1 Packing

变长样本直接 pad 到最大长度会浪费大量算力：平均长度 600、最大长度 4096 时，padding 占比可能超过 80%。Packing 把多条样本首尾拼接，填满固定长度（如 4096/8192），把利用率提高到 90% 以上。

代价是跨文档注意力（cross-document attention）：后一条样本的 token 可以注意到前一条样本，造成信息泄漏或上下文污染。对预训练，长文档本身跨段落，影响相对小；对 SFT，若前一条是用户提问、后一条是答案，模型可能把无关内容当成上下文，导致幻觉或“偷看”答案。常见处理：

1. **朴素拼接 + 重置 position_ids**：只解决 RoPE 位置连续性，不阻止注意力跨文档，仍会泄漏。
2. **块对角注意力掩码（attention mask）**：构造 $L\times L$ 的 0/1 mask，文档内可见、文档间不可见；显存 $O(L^2)$，长序列代价高。
3. **FlashAttention varlen**：用 `cu_seqlens` 描述每条样本的起止位置，内核按变长序列计算，等价于块对角 mask，但不需要物化 $L^2$ 矩阵，是主流方案。
4. **按文档边界切包**：尽量让一个 pack 只含同文档或同对话的样本，减少跨文档；配合 best-fit packing 降低碎片。

```python
# 使用 FlashAttention varlen 的 packing 元数据（示意）
# 假设一个 batch 内拼接了 3 条长度分别为 2、3、4 的样本
lengths = torch.tensor([2, 3, 4], dtype=torch.int32, device="cuda")
cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                        lengths.cumsum(0).to(torch.int32)])
max_len = int(lengths.max())
position_ids = torch.cat([torch.arange(n, device="cuda") for n in lengths.tolist()])
# flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
#                        max_seqlen_q=max_len, max_seqlen_k=max_len, causal=True)
# position_ids 每条样本从 0 开始，配合 cu_seqlens 实现文档内可见
```

### 6.2 SFT loss mask

自回归训练的标准目标是：

$$\mathcal{L}(\theta)=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},x)$$

SFT 只希望模型学习“给定对话历史，生成 assistant 的回答”，因此把 system/user/tool 部分的 label 设为 `-100`，只在 assistant token 上计算损失。`-100` 是 PyTorch `CrossEntropyLoss` 的默认 `ignore_index`。

```python
# 构造一条多轮 SFT 样本，只对 assistant 片段计 loss
def build_example(tokenizer, messages):
    input_ids, labels = [], []
    for msg in messages:
        header = f"<|im_start|>{msg['role']}\n"
        content = msg["content"] + "<|im_end|>\n"
        header_ids = tokenizer(header, add_special_tokens=False).input_ids
        content_ids = tokenizer(content, add_special_tokens=False).input_ids
        input_ids += header_ids + content_ids
        if msg["role"] == "assistant":
            labels += [-100] * len(header_ids) + content_ids   # 只监督回答
        else:
            labels += [-100] * (len(header_ids) + len(content_ids))
    return {"input_ids": input_ids, "labels": labels}

# 训练时右移：logits[:, :-1] 预测 labels[:, 1:]
# loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
#                        ignore_index=-100)
```

注意：计算 loss 时 logits 与 labels 要错一位；若先做 padding，`labels` 的 pad 位置也必须是 `-100`；若使用 `labels` 直接传给 Hugging Face 模型，模型内部会做 shift，不要重复移位。

### 6.3 多轮对话拼接

多轮对话有两种监督方式：

- **全轮监督**：每一轮 assistant 回答都算 loss。优点是训练信号多、模型学会在多轮中持续回答；缺点是早期轮次的错误会累积，且长对话中后段梯度占比高。
- **仅最后一轮监督**：只对最终回答算 loss。适合“给定完整历史，生成最终答案”的任务，但模型可能学不会在中途停止或处理追问。

实践上以全轮监督为主，并对过长对话做截断（保留 system + 最近若干轮），或按 token 预算采样。工具调用场景中，assistant 的工具调用 JSON 要算 loss，工具返回结果（role=tool）不算 loss；否则模型会把工具输出当成自己要生成的内容。

### 6.4 常见数据格式

| 格式 | 结构 | 优点 | 注意点 |
| --- | --- | --- | --- |
| Alpaca | `instruction` / `input` / `output` | 简单，单轮任务友好 | 多轮与 system 支持弱 |
| ShareGPT | `conversations: [{from, value}]` | 天然多轮，社区数据多 | 角色命名不统一，需映射 |
| ChatML | `<\|im_start\|>role ... <\|im_end\|>` | 角色清晰、易扩展工具 | 特殊 token 必须注册 |
| Llama 3 header | `<\|start_header_id\|>role<\|end_header_id\|>` | 与官方模型一致 | 模板细节影响很大 |

模板不一致是 SFT 效果差的常见原因：训练用一套模板、推理用另一套，模型会把“格式”当成分布外输入。保存模型时应同时保存 chat template，并在评测中复用同一渲染函数。

## 7. Benchmark 污染

数据污染指评测集（或其近邻）出现在训练数据中，导致分数虚高。常见检测：

- **n-gram 重叠**：取评测样本的 13-gram（或 50 字符）与训练集比对，命中即视为污染。13-gram 对自然语言较稳，短答案题（如选择题选项）容易误报。
- **embedding 近邻**：用句向量检索训练集，余弦相似度高于阈值（如 0.95）的样本人工复核；能发现改写后的题目。
- **perplexity/生成检测**：若模型对某测试样本的 loss 异常低，可能见过；但低 loss 也可能来自题目简单，需结合基线。
- **canary 字符串**：在训练数据中插入唯一标识，评测时检查模型是否能复述；BIG-bench 采用类似机制。
- **时间切分**：用训练截止后发布的 benchmark（如 LiveCodeBench、MathArena）评估，天然抗污染。

污染的影响在闭卷事实题、数学题、代码题上最大：模型可能记住答案而不是学会推理。GSM1k 等“同分布新题”实验显示，部分模型在 GSM8K 上的高分有数个百分点来自污染（具体数字以公开复现为准）。缓解手段：训练前对评测集做精确/近似去重；SFT 数据同样要查重；报告分数时披露重叠率；用私有或动态生成的评测集。

注意，完全避免污染不现实：网页可能引用题目，代码仓库可能包含测试用例。工程目标是“可测量、可披露、可复现”，而不是宣称零污染。

## 8. 面试高频问题

### 1. 预训练数据的清洗流程是什么？如何平衡质量与多样性？

- 顺序：正文抽取 → 语言识别 → 启发式规则 → 分类器质量打分 → 安全/版权/PII → 去重；便宜步骤前置。
- 质量过滤用 Gopher/C4 规则 + fastText/BERT 分类器，阈值通过验证集消融确定，不追求极致保留率。
- 多样性用领域配比、语言采样温度、去重强度共同控制；高质量但单一的数据会让模型风格收窄。
- 监控指标：各领域 token 占比、重复率、下游 loss、benchmark 与人工评测，避免只看平均 loss。

### 2. MinHash + LSH 的原理是什么？如何设置阈值？

- 把文档表示为 shingle 集合，用 $m$ 个哈希的最小值构成签名，签名相同的比例估计 Jaccard。
- 将签名切成 $b$ 个 band、每个 $r$ 行，任一 band 相同即候选，候选概率 $1-(1-J^r)^b$。
- 近似阈值 $t\approx(1/b)^{1/r}$；$r$ 越大曲线越陡，$b$ 越大召回越高。
- 候选对还要精确校验 Jaccard；阈值按业务定，网页近重复常用 0.7–0.8，代码可更低。

### 3. 数据配比如何确定？DoReMi 的假设和局限是什么？

- 先验配比 + 小模型代理实验 + 下游评测；代理模型至少两个规模验证排序稳定性。
- DoReMi 用参考模型的领域超额损失做 group DRO，指数梯度更新权重，偏向“相对进步最小”的领域。
- 假设代理模型与目标模型的领域难度排序可迁移；领域粒度过粗或参考模型偏差会让权重失真。
- 配比不是静态的：退火阶段提高高质量数据，训练后期提高长文/代码/数学比例。

### 4. BPE、WordPiece、Unigram 有什么区别？

- BPE 按最高频相邻对合并，确定性分词，字节级 BPE 无 OOV；GPT/Qwen 常用。
- WordPiece 按似然增益（PMI 近似）合并，`##` 表示词中片段，BERT 常用。
- Unigram 用 EM 学习子词概率并剪枝，Viterbi 解码，是概率模型，支持多切分。
- SentencePiece 是工具层封装，解决空格、Unicode、可逆解码，可跑 BPE 或 Unigram。

### 5. 词表大小如何权衡？中文和数字该怎么处理？

- 词表大 → 压缩率高、序列短，但 embedding/输出层参数与 softmax 成本上升，稀有 token 欠训练。
- 词表小 → 参数省，但中文/代码被拆成字节，序列变长，注意力成本上升。
- 中文用 SentencePiece/多语言 BPE，检查字符/token 压缩率；数字可逐位切分以利于算术，但序列变长，需评测消融。
- 多语言用温度采样分配词表，避免高资源语言挤占低资源语言。

### 6. 扩展词表并 resize embedding 时有哪些坑？

- 新增 token 要用旧 embedding 均值或同分布初始化，否则初期 loss 尖峰。
- 输入 embedding 与输出 head 要同步扩展；绑定权重时注意重新绑定，解绑时别漏 `lm_head`。
- 新 token 需要足够训练数据；只加特殊 token 不训练，表示会停留在初始化附近。
- 保存 tokenizer 与 chat template，检查 `vocab_size`、`pad/eos_token_id` 一致性；扩展后回归旧能力。

### 7. Packing 的跨文档注意力怎么处理？

- 朴素拼接会让孩子文档看到前文，SFT 中可能泄漏答案或引入无关上下文。
- 方案：块对角 mask（显存 $O(L^2)$）、FlashAttention varlen + `cu_seqlens`（主流）、按文档边界装包。
- 仅重置 position_ids 只能修正 RoPE 位置，不能阻止注意力跨文档，别把它当成隔离。
- 预训练可容忍少量跨文档，SFT/长上下文训练建议严格隔离，并用消融验证收益。

### 8. SFT 的 loss mask 和多轮拼接怎么做？

- 只对 assistant token 计 loss，system/user/tool 的 label 设为 `-100`；EOS/EOT 要纳入监督。
- 多轮对话通常全轮监督；长对话按 token 预算截断，保留 system 与最近轮次。
- 注意 logits 与 labels 错位一位；padding 位置也必须忽略；不要重复 shift。
- 工具调用中 assistant 的调用 JSON 计 loss，工具返回结果不计 loss。

### 9. 如何检测和缓解 benchmark 污染？

- 13-gram/50 字符重叠 + embedding 近邻 + 低 perplexity 异常 + canary 字符串。
- 用训练截止后的新 benchmark、私有集、动态生成题目做抗污染评测。
- 训练前对评测集做精确/近似去重，SFT 数据同样处理；报告中披露重叠率。
- 污染无法完全消除，重点是量化影响、避免用被污染的分数做结论。

### 10. 数据回放和课程学习分别解决什么问题？

- 回放把旧数据混入新训练，缓解灾难性遗忘，比例 5%–10% 起步，按能力维度组织。
- 课程学习按难度/长度安排数据，后期退火切换高质量数据，提升收敛与最终效果。
- 两者都需要验证集监控：回放不足看通用能力掉点，回放过量看新任务收敛变慢。
- 课程难度定义要可复现，否则容易变成“拍脑袋”的超参。

> 本文档为面试复习资料，公式与数字建议结合原始论文复核。
