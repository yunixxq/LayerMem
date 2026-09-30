# LayerMem 首版实现 · LoCoMo 验证结果

> 运行日期：2026-09-30 · 代码版本：`src/layermem` 首版
> 数据：LoCoMo `conv0` 单对话（19 sessions / 419 messages / 199 QA，评测阶段实际判定 152 题）

---

## 一、运行配置

| 组件 | 配置 |
|---|---|
| 记忆系统 | LayerMem（本文档验证的实现） |
| Embedding | `all-MiniLM-L6-v2`（384 维，本地） |
| LLM | **Qwen3-1.7B**（本地，MLX 推理，约 25 tok/s） |
| 服务方式 | `mlx_lm.server` 暴露 OpenAI 兼容接口，提取 / 答题 / 评判三处复用同一实例 |
| 向量库 | Qdrant（本地嵌入式模式），单 collection + 命名向量 `dense` + `sparse` |
| 测试框架 | `memory_toolkits`（`memory_construction` → `memory_search` → `memory_evaluation`）|
| 硬件 | Apple M2 / 16GB / macOS 27 |

**关键前提：** 本机无任何可用 LLM API key，全部推理由本地 1.7B 模型承担。这直接决定了绝对准确率的天花板（见 §五）。

---

## 二、记忆构建结果

```
419 条消息  →  205 条记忆条目（写入耗时 39 分钟）
```

| 维度 | 分布 |
|---|---|
| 按层级 | L1 190 · L2 13 · L3 2 · L4 0 |
| 按视角 | factual 145 · relational 54 · state 6 |
| 语义主题（topic） | **99 个** |
| 压缩比 | 平均 14.6 条 L1 → 1 条 L2 |

**读法：**

- **三视角确实分化了**：factual/relational 占绝大多数，state 只有 6 条。这符合设计预期（LoCoMo 是日常闲聊，属性类陈述本就稀少），但也意味着 state 轨在本数据集上样本量极小。
- **在线聚类产生了 99 个主题**，419 条消息 → 205 条原子，说明提取阶段的去噪和合并起作用了（并非每条消息都变成一个记忆）。
- **分层压缩被真实触发**：13 条 L2 主题摘要 + 2 条 L3 阶段画像。L1→L2 平均 14.6:1，说明 `K1=5` 的水位线触发按预期工作，没有出现"反复压缩同一批 L1"的问题。
- **L4 未生成**：`K3=3` 要求至少 3 条 L3，本次只产出 2 条。这是阈值设置与数据规模的匹配问题，不是缺陷。

**L3 阶段画像样例**（验证抽象层确实在做抽象，而非拼接原文）：

> "Caroline and Melanie exhibit a stable pattern of deep, ongoing dialogue centered on personal expression, identity, and community engagement. Their conversations evolve over time but consistently revolve around the interplay between art, memory, and the complex..."

**L2 主题摘要样例**：

> "Melanie engaged with Caroline about her experience at a LGBTQ support group, expressing admiration for the impact of the group and inquiring about its effects."

---

## 三、检索与问答结果

**评测题目：** 152 题 = 原始 199 题 − 47 道对抗题。harness 的 `LoCoMo.filter_questions()` 会主动丢弃 `category_id=5`（adversarial）的题目，这是框架行为而非本次失败。逐题清单（中英对照，含金答案、预测、判定、证据是否召回）见 [`memory_toolkits/LayerMem_conv0_questions.md`](memory_toolkits/LayerMem_conv0_questions.md)。

| 类别 | 正确/总数 | 准确率 |
|---|---|---|
| temporal（时序） | 33/37 | **0.892** |
| single_hop（单跳） | 46/70 | 0.657 |
| multi_hop（多跳） | 21/32 | 0.656 |
| open_domain（开放域） | 7/13 | 0.538 |
| **总体** | **107/152** | **0.704** |

检索侧：平均召回 **9.6** 条/查询（min 1，max 11），**空召回 0/152**。

---

## 四、诊断：错在哪里

用 LoCoMo 自带的 ground-truth 证据标注（每题给出 `evidence: [D1:3]` 这类对话 ID，197/199 题有标注）来判定"证据是否被召回"，比用词面匹配可靠得多。判定口径：证据消息中超过一半的实词出现在检索上下文里。150 题可判定。

### 4.1 混淆矩阵

|  | 模型答对 | 模型答错 | 合计 |
|---|---|---|---|
| **证据被召回** | 87 | 31 | **118** |
| **证据漏召回** | 18 | 14 | **32** |
| **合计** | **107** | 45 | 152 |

- **检索召回率：118/150 = 78.7%**
- **召回时的答题准确率：87/118 = 73.7%**
- 漏召回时仍答对：18/32 = 56.2%（其他记忆或常识碰对）
- 整体：107/152 = **70.4%**（但见 4.3 —— 这个数字被评判模型高估了）

### 4.3 评判模型不可靠，真实准确率更低

答题与评判用的是同一个 1.7B 模型。对 107 个"正确"判定做机械审计（数字不符 / 是非相反这两类无需人工判断即可确认的矛盾），**至少 11 个是确凿的误判**：

| 问题 | 金答案 | 预测 | 判定 |
|---|---|---|---|
| How many children does Melanie have? | 3 | 2 | ✓ ← 数字不符 |
| Would Melanie be considered a member of the LGBTQ community? | Likely no | Yes. | ✓ ← 是非相反 |
| When did Melanie read the book "nothing is impossible"? | 2022 | 2023-10-20 | ✓ ← 年份不符 |
| How long has Melanie been creating art? | 7 years | 2 years | ✓ ← 数字不符 |
| When is Melanie's daughter's birthday? | 13 August | August 17. | ✓ ← 日期不符 |

（完整 11 条中还有：18th birthday 10年→5年、pottery workshop 日期、pride parade 日期、pride festival 2022→2023、practicing art 2016→3年、roadtrip 意愿 No→Yes。）

- 原始判定：107/152 = **70.4%**
- **扣除机械可证的误判后：96/152 = 63.2%（下界）**

这只扣掉了能机械识别的错误，**真实值应更低**——语义层面的误判（如"从祖国搬来"被判为回答了"Sweden"）无法用脚本发现。

**结论：所有绝对准确率应以 63.2% 为准，并理解为量级参考而非精确度量。** 要拿到可信数字，评判模型必须换成远强于答题模型的模型（论文里通常用 GPT-4 级做 judge）。

另一个值得注意的系统性失败模式：多道时序题模型给出的日期是**会话日期而非事件真正发生的日期**（如"Melanie 何时读的那本书" gold 2022、预测 2023-10-20）。这直接指向 `event_time` 的抽取质量——答案生成阶段看到的日期前缀就是记忆的 `event_time`，抽错就会被自信地答错。

### 4.2 结论（修正）

**检索与推理各占一半，两者都未饱和。** 早前版本的本节曾写"只要证据被召回模型 96% 能答对，瓶颈在检索"——那是把整体答对数（107）误当作"有证据时答对"的分子算出来的，**该结论不成立**。

按类别看会更清楚：

| 类别 | 证据召回率 | 答题准确率 | 说明 |
|---|---|---|---|
| multi_hop | 0.906 | 0.656 | 召回很好，**推理是瓶颈** |
| open_domain | 1.000 | 0.538 | 证据全都在，**推理是瓶颈** |
| temporal | 0.784 | 0.892 | 召回略欠，但召回后答得准 |
| single_hop | 0.700 | 0.657 | **两者都有问题** |

所以优化要分头做：`single_hop` 该提升召回（`theta_retrieval`、混合检索权重、关键词质量），`multi_hop`/`open_domain` 则受限于 1.7B 模型的推理能力，换更大的答题模型才有效。

**检索漏召回的例子：**

> Q: *When is Caroline's youth center putting on a talent show?*
> gold: September 2023 · 预测: 2023-08-28
> 证据原话是 *"We're putting together a talent show for the kids next month."*（D15:11）—— 这条没被召回，模型退而用会话日期作答。

> Q: *What did Melanie paint recently?*
> gold: sunset · 预测: "Recent." —— 相关绘画内容完全没被召回。

**推理错误的例子（证据确实在上下文里）：**

> Q: *Would Caroline likely have Dr. Seuss books on her bookshelf?*
> gold: Yes, since she collects classic children's books · 预测: No

> Q: *What do Melanie's kids like?*
> gold: dinosaurs, nature · 预测: camping

---

## 五、状态版本链的实际表现

这是 LayerMem 相对其他记忆系统的核心差异化设计，单独检查。

**构建出的版本链**（`Caroline / residence`）：

| 主体 | 属性 | 值 | 状态 | event_time |
|---|---|---|---|---|
| Caroline | residence | Helsinki | superseded | 2023-08-23 |
| Caroline | residence | local church | superseded | 2023-08-25 |
| Caroline | residence | Helsinki | superseded | 2023-10-13 |
| Caroline | residence | **Helsinki** | **current** | 2023-10-22 |
| Caroline | residence | new home | superseded | 2023-10-22 |
| Melanie | residence | Helsinki | current | 2023-10-21 |

**机制层面成立：** 同一 `(subject, attribute)` 下 6 个版本形成串形链，任意时刻只有一条 `current`，历史版本保留且按 `event_time` 排序（注意 2023-10-22 的两条：先写入的 "new home" 被同时间戳的 "Helsinki" 正确覆盖）。

**抽取层面暴露问题：** "local church"、"new home" 并不是居住地，却被抽成了 `residence` 的值。这是 **1.7B 模型的属性抽取质量**问题（它把"我在教堂/在新家"过度归一到 residence），不是版本链逻辑的问题 —— 链本身忠实地记录了模型给它的每一个值。

**对评测的影响：需要把「版本链」和「状态轨」分开看。**

早前版本的这一节只搜了 `Where does X live` / `now / currently / still` 两种模式，得出"LoCoMo 没有属性类问题"——**该结论是错的，搜索模式过窄**。放宽到生日、来源地、职业、婚姻、子女、宠物、数量等单值属性后：

| | 全量 1540 题 | conv0 152 题 |
|---|---|---|
| 单值属性类问题 | **221（14.4%）** | **31（20.4%）** |

所以 LoCoMo **确实有大量属性查询**——只是这些属性在对话中不发生值覆盖。例如 Q12 *Where did Caroline move from 4 years ago?*（gold: Sweden）和 Q13 *How long ago was Caroline's 18th birthday?*（gold: 10 years ago）都是典型的单值属性查询。

这暴露出 `LayerMem.md` 把**两件本可分离的事**绑在了一起：

| 区分维度 | 含义 | 生日 | 住址 |
|---|---|---|---|
| **存储语义** | 新值是否使旧值失效 | 否 | 是 |
| **查询语义** | 需要"一个正确的值"还是"一堆相关的记忆" | 需要值 | 需要值 |

- **版本链**（supersession + 历史回溯）只在值发生覆盖时才有意义 → LoCoMo 确实测不到，这一条成立。
- **状态轨**（精确查找、确定性单值、不参与重排）对**所有**单值属性都适用，与值是否变化无关 → LoCoMo 的 31 道题（20%）本该由它来答。

**实测印证了这个缺口：** conv0 的 31 道单值属性题准确率 **61.3%**，低于其余 121 题的 **72.7%**。原因很直接——系统只抽出了 6 条 state（2 个属性，都是 residence），生日、子女、宠物、职业全部落在了 factual 里，靠 ANN top-k 和其他 9 条记忆一起丢给模型，让它自己挑。而 state 轨正是为"只返回那一个值"设计的。

**结论（修正后）：**

1. **版本链的价值本次未被检验**——没有任何属性的值在对话中被覆盖。要验证需要 LongMemEval knowledge-update 子集或自建对照题。
2. **状态轨的价值本次是被部分检验到了，且表现不佳**（61.3% vs 72.7%）。这是一个可立刻行动的改进点：把 state 的抽取范围从"会被覆盖的属性"放宽到"所有单值属性"，让生日/子女数/宠物名这类问题走确定性查找而非 ANN 排序。conv0 就有 31 道题可以验证这个改动。

---

## 六、结论：验证了什么，没验证什么

### 已验证

1. **全链路可用**：419 条消息 → 三视角提取 → 在线聚类 → 分层压缩 → 混合检索 → 状态版本链 → 答案生成，端到端跑通，无空召回、无崩溃。
2. **分层压缩按设计工作**：数量驱动（而非日历驱动）触发，14.6:1 的压缩比，L2/L3 摘要确实在做抽象。
3. **版本链机制正确**：单一 current、历史可回溯、按 event_time 排序、同值确认不产生新版本（§五给出实例与单元测试）。
4. **在 1.7B 本地模型下达到 63.2%（下界，修正后）**，temporal 类 89.2%（未修正）；检索召回率 78.7%，召回后答题准确率 73.7%。

### 未验证 / 存疑

1. **版本链的收益未被本基准检验**（§五）—— LoCoMo 的属性质询中没有任何属性发生过值覆盖。这是最重要的缺口。
2. **绝对数值不可信**（§四.3）—— 判为正确的 107 题中至少 11 题可机械证伪，且答题与评判同源（同一个 1.7B），存在自我偏袒。63.2% 是下界而非真值。
3. **没有对照基线**。本次只跑了 LayerMem 自身，未跑 NaiveRAG / FullContext 等基线，因此得分缺少参照系。
4. **状态轨没被用起来**（§五）—— 31 道单值属性题（20%）只有 6 条 state 支撑，生日/子女/宠物/职业全落在 factual 里走 ANN 排序，准确率 61.3% 低于整体。
5. **属性抽取质量偏低**，normalizer 的同义词表覆盖有限；且 `event_time` 抽错会直接导致时序题答错（模型把会话日期当作事件日期）。
6. **仅 1 个对话（152 题）**，样本量小，类别准确率的置信区间较宽（如 open_domain 只有 13 题）。

### 下一步建议（按性价比排序）

1. **放宽 state 抽取范围到所有单值属性**：这是本次分析暴露出的最直接、最可验证的改进。让生日、子女数、宠物名、职业也进 state 轨走确定性查找，conv0 的 31 道题可直接量化收益。改动小（prompt + normalizer 词表），且正对当前最弱的一类题。
2. **换可靠的评判模型**：当前所有绝对数字都被 1.7B judge 污染。至少评判要用远强于答题的模型（哪怕只对抽样 50 题做人工/强模型复核），否则任何调参都无法判断是否真的变好。
3. **构造状态更新子集**：验证版本链的唯一手段。LoCoMo 全量 1540 题的属性质询中，值发生覆盖的是 0 道。可用 LongMemEval 的 knowledge-update 子集，或自建"值变更 + 追问当前值"的对照题。
4. **跑对照基线**：同 harness、同模型跑 NaiveRAG 或 FullContext，给得分一个参照系。
5. **分开优化两条瓶颈**：`single_hop` 召回率 0.70 偏低，调 `theta_retrieval` 与关键词策略；`multi_hop`/`open_domain` 召回率 0.91/1.00 但准确率 0.66/0.54，是模型推理能力所限，需换更大的答题模型（4B 仍能装进本机 16GB）。

---

## 七、复现方式

```bash
cd LayerMem

# 1. 环境（见 LayerMem_imple.md 附录 A）
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]" litellm
python3.13 -m venv .venv-serve && .venv-serve/bin/pip install mlx-lm

# 2. 模型与数据
python scripts/fetch_model.py Qwen/Qwen3-1.7B models/Qwen3-1.7B
curl -sL "https://ghproxy.net/https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" \
     -o benchmarks/locomo/locomo10.json

# 3. 启动本地 OpenAI 兼容服务（关键：关闭 Qwen3 思考模式）
export HF_HUB_OFFLINE=1
.venv-serve/bin/mlx_lm.server --model "$PWD/models/Qwen3-1.7B" \
    --host 127.0.0.1 --port 8080 --chat-template-args '{"enable_thinking": false}'

# 4. 一键跑通三阶段
./run_layermem_locomo.sh 0 1     # conv0；改成 0 10 则跑全部
```

查看构建出的记忆结构：

```bash
.venv/bin/python scripts/layermem_stats.py \
    "memory_toolkits/LayerMem_/Users/lyx/Projects/AgentMemory/LayerMem/models/Qwen3-1.7B/user_LoCoMo_locomo_0/qdrant" \
    user_LoCoMo_locomo_0
```

**环境注意：** 本机 `huggingface.co` 与 `github.com` 均不可达，模型走 `hf-mirror.com`、数据走 `ghproxy.net`；`scipy` 必须 ≥1.16（1.15.3 的 wheel 在 macOS 27 上无法加载）。
