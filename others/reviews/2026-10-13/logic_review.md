# LayerMem 逻辑审查（2026-10-13）

结论：当前实现不能视为“全部逻辑正确”。三路存储分工、基本状态版本查询和单次层级晋升成立，但聚类候选选择、异常恢复、重跑、更新以及评测存在可复现问题。

审查覆盖 `src/layermem` 的提取、缓冲、聚类、压缩、存储、检索、后端配置，以及 LayerMem 适配层、构建/检索/评测脚本。未审查其他 baseline 的完整实现。没有调用真实 LLM 或 embedding API；因此不对真实模型的抽取率、摘要保真度、属性消解质量作通过声明。本次未修改业务实现。

## 验证结果

| 检查 | 结果 | 解释 |
|---|---|---|
| 仓库测试：`pytest -q -m 'not integration'` | 收集阶段中断 | `test_memory_backends.py` 仍导入已移除的 `TextEmbedderHuggingface` |
| 排除上述文件后运行仓库测试 | 19 failed，34 passed | 失败主要依赖旧 Entry 类、旧来源字段和旧字典形式的 factual/relational 输出；不能全部当作当前业务缺陷 |
| 本次独立复现脚本 | 18 failed，3 passed | 失败项断言预期行为，展示下面的缺陷或语义差异；参数化产生多个失败，不代表 18 个独立缺陷 |

复现脚本：[test_logic_review.py](test_logic_review.py)。从 LayerMem 目录运行：

```bash
.venv/bin/python -m pytest -q others/reviews/2026-10-13/test_logic_review.py --tb=short
```

这些是保留问题现场的测试，当前应失败；位于 `others`，不加入项目默认 `tests` 收集路径。它们只使用内存数据库、pytest 临时目录和模拟服务。

## 高优先级发现

### 1. [P1] 已满主题遮蔽可加入主题，导致层级压缩碎片化

位置：[clusterer.py:50](../../../src/layermem/pipeline/clusterer.py#L50)。

当前先搜索最相似的 5 条节点，再逐个判断其主题是否达到容量。这 5 条可能全来自同一个已满主题，而不是 5 个可用主题。

复现：L1 已满主题 A 的节点与查询相似度为 1.0；未满主题 B 与查询相似度为 0.98，高于 0.75 阈值。搜索只返回 A 的前 5 条节点，代码跳过 A 后直接新建 C，完全没有检查 B。L2、L3 同样复现。

影响：相似的新记忆不断分散到小主题，难以累积到 K1/K2/K3；层级晋升次数和摘要内容因此失真。

建议：在候选检索前排除已满主题，或为可用主题建立独立候选索引；若采用扩大候选池，必须继续搜索到有效主题或明确耗尽候选，不能只固定扩大一个常数。

### 2. [P1] flush 在成功之前清空缓冲，失败后失去重试内容

位置：[api.py:74](../../../src/layermem/api.py#L74)。

`buffer.drain()` 先清空批次，然后执行提取、embedding、Qdrant 写入及 State 写入。任一环节抛错，原批次不会恢复，也没有待处理日志。

复现：分别让提取连续两次失败、让 embedding 失败；异常后 `pending_turns` 均为 0。再次 `flush()` 返回空计数，不能恢复该批数据。中途失败还可能留下部分 factual/relational 已写入、State 未写入的批次。

建议：保存可重试批次直到写入完成；为节点/批次建立稳定身份和完成进度，避免仅恢复整批后又重复写入已经成功的部分。

### 3. [P1] --rerun 实际追加旧库；存在任意记录又被当成完整构建

位置：[memory_construction.py:172](../../../memory_toolkits/memory_construction.py#L172)、[layermem.py:129](../../../memory_toolkits/memories/layers/layermem.py#L129)、[run_layermem_locomo.sh:42](../../../run_layermem_locomo.sh#L42)。

`--rerun` 只跳过 `load_memory()` 的提前返回，没有重建存储目录或创建独立运行命名空间。标准运行脚本每次都传入此参数。

复现：相同 trajectory 两次 `rerun=True`，记录数依次为 1、2，而不是 1、1。UUID 每次重新生成，写入不会覆盖旧节点，重复事实也会参与压缩。

另一方面，`load_memory()` 只检查总条数是否大于 0；没有“该 trajectory 已完整构建”的标识。构建中途失败留下部分数据后，默认重跑会直接跳过剩余输入。

建议：用独立 run ID/输入与配置指纹隔离构建；仅完整写入后设置完成标记。明确区分重新构建与断点续写，并显式关闭存储资源。

### 4. [P1] 评分使用子串，能把否定判决计为正确

位置：[memory_evaluation.py:154](../../../memory_toolkits/memory_evaluation.py#L154)。

`"yes" in content.lower() or "correct" in content.lower()` 对 `INCORRECT` 和 `No, the answer is not correct.` 都返回 True。对这两种模拟判决，实际评测函数都报告 100% 正确率。

LoCoMo 的 prompt 要求 JSON `label` 为 `CORRECT`/`WRONG`，而代码并未解析该字段；其他 prompt 又采用 Yes/No。标准合规的 `WRONG` 当前能判对，但一旦模型附带解释或输出不同的否定形式，就会虚增准确率。这里没有证明已有真实评测结果全部受影响。

建议：按对应 prompt 的输出协议解析，再用严格枚举匹配；无法解析的判决不能默认为正确。

## 其他已复现问题

### 5. [P2] 压缩没有完成记录：失败会漏晋升，重复调用会重复摘要

位置：[layer.py:63](../../../src/layermem/compression/layer.py#L63)、[layer.py:158](../../../src/layermem/compression/layer.py#L158)、[api.py:90](../../../src/layermem/api.py#L90)。

满主题不再接收新节点，只能保证正常单次写入不再触发该主题。它不能证明该主题已经成功生成了上层节点。

复现一：L1 达到 K1，写 L2 时发生一次异常；之后新节点进入另一个主题，原主题不会自动补压缩。K1=2 时四条输入只得到一条 L2。

复现二：对相同主题调用两次 `compress_topic()`，生成两条不同 ID 的 L2。`consumed` 参数未被使用，摘要没有源节点 ID，也无压缩完成标识。

建议：给晋升操作建立稳定身份、源节点关联及完成状态，使重试不会重复生成，并可恢复未完成晋升。

### 6. [P2] State 输出丢失 mention_time，原文相对时间失去正确锚点

位置：[retriever.py:186](../../../src/layermem/retrieval/retriever.py#L186)、[extraction.py](../../../src/layermem/prompts/extraction.py)。

State prompt 要求保留 memory 中的“去年”等原始表达，而 `_render_state()` 只输出 event_time，不输出 mention_time。

复现：2024-01-07 提及“去年住在 Helsinki”，event_time=2023-01-01，最后输出：

```text
2023-01-01, Alice residence: Helsinki. Alice lived in Helsinki last year.
```

输出已没有 2024-01-07。答题 prompt 又要求相对词基于记忆时间戳换算，因而可能把“去年”解释为 2022。复现证明时间信息在输出中丢失；未调用模型验证实际错答概率。

建议：明确标注事件时间和提及时间，或在输出时将相对表达转成有锚点的表达。广义时期映射出的 1 月 1 日也不应被描述为已知的精确搬家日期。

### 7. [P2，设计语义缺口] range 返回变更节点，不代表区间内有效状态

位置：[state_store.py:96](../../../src/layermem/storage/state_store.py#L96)、[LayerMem.md §2.4](../../LayerMem.md)。

复现：2022-01-01 起住北京，2024-01-01 起住上海；查询 2023 年 range 返回空，而在现有“后一节点使前值失效”的链语义下，2023 年有效值是北京。

代码符合文档的 `event_time >= lower AND event_time <= upper` 模板，也符合当前 planner 所写的“区间内节点”语义；问题是这与文档举例“去年住在哪里”的需求不同。因此不能只把这里归因为代码偏离文档。

建议：区分“区间内变更”与“区间内有效值”。后者需要区间起点之前最近的有效节点，再加区间内变更；一次 as-of 查询不能覆盖整个时期内多次迁移的情况。

### 8. [P2] factual/relational update 只改文本，不更新检索向量

位置：[api.py:143](../../../src/layermem/api.py#L143)、[vector_store.py:171](../../../src/layermem/storage/vector_store.py#L171)。

复现：把“喜欢 hiking”改为“plays chess”，返回成功，payload 已更新，但 dense 向量仍是 hiking 的向量；sparse 也没有重新计算。新内容可能检索不到，旧查询仍召回已修改文本。

此外代码未约束可修改字段；改 layer/topic/type 并不会执行相应迁移。更新或删除 L1 后，上层摘要也无来源关联可用于失效/重建。后两点为静态检查发现。

建议：内容更新重新 embedding 并 upsert dense/sparse；约束结构字段和明确已生成摘要的更新、删除语义。

### 9. [P2] Qdrant 孤立属性点占据 top-k，可能把同一 State 属性拆成两条链

位置：[state_chain.py:106](../../../src/layermem/storage/state_chain.py#L106)、[state_attribute.py:55](../../../src/layermem/storage/state_attribute.py#L55)、[state_store.py:150](../../../src/layermem/storage/state_store.py#L150)。

当前先取 top-k，再根据 SQLite `has_chain()` 丢弃孤立点。删除一条链的全部节点时，Qdrant 属性点没有一同移除；SQLite 写入失败也可能留下孤立点。

复现：5 个孤立点的相似度为 1.0，真实 residence 候选相似度为 0.95；top-5 全被丢弃后，home city 被错误地新建成独立链。即使 SQLite 中有正确候选，也无法在当前候选池中看到它。

建议：删除最后一个链节点时清理对应索引，并在消解时持续取足有效候选；单纯丢弃已召回的孤立点不充分。

### 10. [P2] 多说话人代词降级可能把 State 归给错误主体

位置：[normalizer.py:81](../../../src/layermem/pipeline/normalizer.py#L81)。

复现：已观察说话人 Alice、Bob，`subject("speaker_b")` 返回 Alice。`I`、`user` 等也一律映射到第一个历史说话人，未使用本条记忆的来源说话人。

这是模型未输出明确姓名时触发的降级缺陷；正常输出 Bob 不受影响。建议保留来源主体映射，或对无法确定的多说话人引用拒绝归属，不猜第一个人。

### 11. [P2] 构造 LayerMemory 会覆盖调用方注入的已连接 store

位置：[api.py:32](../../../src/layermem/api.py#L32)、[vector_store.py:105](../../../src/layermem/storage/vector_store.py#L105)。

构造函数无条件 `self.store.connect()`，而 connect 又无条件创建新 QdrantClient。

复现：预填一条记录的内存 VectorStore 传入 LayerMemory 后，查不到原记录，因为其客户端已被替换。注入的远程 client 也不会被保留；磁盘重复连接还可能遇到锁冲突。

建议：connect 幂等，且清楚规定注入客户端/存储的生命周期与关闭责任。

### 12. [P2] 未知历史时间仍被降级为 mention_time

位置：[extractor.py:68](../../../src/layermem/pipeline/extractor.py#L68)。

State prompt 明确禁止给无法确定时间的历史状态使用 mention_time；代码却对缺失 event_time 统一执行 `event or mention`。

复现：历史住所 Beijing、event_time=null、mention_time=2024-01-01，最终生成 event_time=2024-01-01 的有效节点。后续可覆盖真正的当前住所。正常遵循六字符串字段协议的模型输出不触发这个分支，但当前容错行为与 prompt 约束矛盾。

建议：明确验证 State 输出契约；对缺少合法历史 event_time 的候选报错/重试或拒绝入链，不凭降级值建立虚假的时序。

### 13. [P2] 对外暴露的 LLM 参数未传到请求

位置：[openai.py:58](../../../src/layermem/memory/llm/openai.py#L58)、[backends.py:55](../../../memory_toolkits/inference_utils/backends.py#L55)。

配置接收 temperature、max_tokens，但 Responses 调用只传 model、input。模拟客户端检查确认即使设置 temperature=0、max_tokens=123，请求也没有携带两者。评测端 Responses 分支同样忽略采样参数。

影响：以为温度/输出长度可控，实际取服务默认值；实验复现和成本配置不准确。应按后端能力转发参数或移除无效配置，不能宣称参数已生效。

## 已确认成立的部分与剩余限制

- factual 与 relational 分集合存储，每层按各自 entry_type 筛选；六个压缩 prompts 已分别接入。
- 用 K1=K2=K3=1 的本地控制用例，两类记忆都能顺序生成 L2、L3、L4，并保留下层节点。该用例不覆盖大库候选遮蔽问题。
- L4 生成不再读取 State。压缩器仍保留未使用的 `state_chain` 兼容参数，属于清理不彻底，并非仍有运行时耦合。
- State 的 SQLite 节点表和 Qdrant 属性索引职责符合 §2.4；Qdrant 仅向量化属性名，按 subject 过滤；State 查询通过固定参数化 SQL。
- 乱序历史插入、时区归一化、同一节点重试、属性高低阈值和中间 LLM 分支、SQLite 历史查询等已有测试通过。关闭 Qdrant 后，StateChain 查询仍可工作。
- “查询只用 SQLite”是 State 轨道的性质；完整 LayerMemory 初始化仍连接 Qdrant，混合检索中的 factual/relational 仍依赖 Qdrant，不能据此保证整个系统在 Qdrant 不可用时仍正常启动。
- 已移除的离线接口没有运行时代码调用，但 `others/LayerMem_imple.md` 仍列着 compactor、离线 L4 蒸馏和旧阈值。此前“相关文档已清理”的表述不完整。
- `theta_dup` 未用于当前去重逻辑；`rrf_k` 配置没有被检索合并读取（代码写死为 60）。这些配置会误导调参。
- 检索路由失败时虽把 tracks 设为全部，但 subject/attribute 缺失会使 State 轨道返回空，不能将此降级理解为真正完成了 State 查询。
- `close()` 不会自动 flush，构建/检索脚本也没有 finally 关闭 layer；使用者需要明确生命周期约定。
- 新 prompts 尚未经过真实模型摘要质量对比；层级数量增加不保证时间跨度更大或特征更稳定。

建议先修复聚类、批次持久化、重跑/完成标记和评分，再处理压缩重试、时间语义、更新索引及残留接口；同步重写过时测试，使全量测试恢复可运行。
