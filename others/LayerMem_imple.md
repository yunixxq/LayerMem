# LayerMem 实现细节

## 一、目录结构

```
layermem/
├── src/layermem/
│   ├── __init__.py
│   ├── api.py                     # 待建：LayerMemory 对外入口
│   ├── configs/
│   │   ├── __init__.py
│   │   └── config.py              # ✅ 已实现：EmbedderConfig、LLMConfig
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # 占位：待并入 configs/config.py 后删除
│   │   └── schema.py              # 占位：Entry 数据结构（见 §三 层级 1）
│   ├── memory/                    # ✅ 已实现：底层模型接入层
│   │   ├── __init__.py            # 仅导出工厂与配置，后端延迟导入
│   │   ├── embedder/
│   │   │   ├── __init__.py
│   │   │   ├── factory.py         # EmbedderFactory.from_config()
│   │   │   ├── huggingface.py     # 本地 SentenceTransformer / TEI API
│   │   │   └── openai.py          # OpenAI 兼容 Embeddings API
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── factory.py         # LLMFactory.from_config()
│   │       ├── utils.py           # 对话渲染、JSON 解析、schema 归一化
│   │       ├── openai.py          # OpenAI 兼容 Chat API
│   │       └── transformers.py    # 本地 Transformers 后端
│   ├── pipeline/                  # 写入路径
│   │   ├── __init__.py
│   │   ├── buffer.py              # 对话缓冲与批量触发
│   │   ├── extractor.py           # 三视角提取编排
│   │   ├── normalizer.py          # 主体/属性规范化（设计见 §七.1）
│   │   └── clusterer.py           # 待建：在线增量语义聚类
│   ├── storage/                   # 持久化
│   │   ├── __init__.py
│   │   ├── vector_store.py        # Qdrant 封装
│   │   └── state_chain.py         # State 版本链
│   ├── compression/               # 分层压缩
│   │   ├── __init__.py
│   │   ├── layer.py               # L1→L2→L3→L4 在线压缩与触发
│   │   └── compactor.py           # 待建：离线整理
│   ├── retrieval/                 # 检索路径
│   │   ├── __init__.py
│   │   ├── router.py              # 待建：检索规划器
│   │   └── retriever.py           # 待建：三轨检索 + 合并
│   └── prompts/
│       ├── __init__.py
│       ├── extraction.py          # ✅ 已实现：三视角提取 Prompt
│       ├── compression.py         # 占位：L1→L4 压缩 Prompt
│       ├── retrieval.py           # 占位：检索规划器 Prompt
│       └── compaction.py          # 待建：离线整理 Prompt
├── experiments/                   # 待建：run_layermem.py / run_baselines.py
├── docs/                          # 待建
└── tests/                         # 待建
```

## 二、模块详细规划

### 层级 0：工具与模型层

依赖关系：无外部依赖，最先实现。



#### 1. `memory/embedder/` ✅

**功能：** 将文本转为向量，供聚类和检索使用。

两种后端：

- `huggingface`：本地 `SentenceTransformer`；配置了 `base_url` 时改走 TEI 的 OpenAI 兼容接口；
- `openai`：OpenAI 或其他兼容 Embeddings API 的远程服务。

统一入口 `EmbedderFactory.from_config(config)`，接受 `EmbedderConfig`、 `{"model_name": ..., "configs": {...}}` 映射，或 `None`。具体实现提供 `embed(text)`（接受单条字符串或列表，按同样形状返回）和 `get_stats()`。

**实现要点：**

- 后端类按点号路径**延迟导入**，`import layermem.memory` 不会拖入 sentence-transformers / torch。
- 构造时对传入的 config 做副本（`dataclasses.replace`）：`model`、`embedding_dims` 的默认值填充只作用于副本，不会污染与 clusterer、vector store 共用的那份配置。
- `embedding_dims` 在本地后端由模型自省得到；在远程后端由**首次响应**的长度反推并记录——建 Qdrant collection 需要它，而 TEI 不提供维度查询接口。
- `pass_dimensions` 默认 `False`：只有少数服务商支持 OpenAI 的 `dimensions` 参数，vLLM / TEI 等会直接返回 400。
- `verify_ssl` 默认 `True`。LightMem 为访问自签名的内部端点全局关闭了证书校验，这里改为显式开关。

---

#### `memory/llm/` ✅

**功能：** 统一的 LLM 推理与记忆提取能力。

两种后端：`openai`（OpenAI 兼容 Chat Completions API）、`transformers`（本地 HuggingFace 因果语言模型）。统一入口 `LLMFactory.from_config(config)`，按 `LLMConfig.model_name` 选择，同样延迟导入（用远程 API 时不需要装 torch）。

两个后端都提供：

```python
generate_response(messages, ...) -> tuple[str, dict]   # 文本 + usage
extract_memories(conversation, system_prompt=None, current_date=None) -> dict
get_stats() -> dict
```

`extract_memories()` 使用 `prompts/extraction.py` 的 `EXTRACTION_PROMPT`，返回：

```json
{
  "factual": [], "relational": [], "state": [],
  "meta": {"usage": {...}, "raw_response": "...", "current_date": "..."}
}
```

**三个顶层列表是纯粹的记忆数据**，`usage` / `raw_response` 收在 `meta` 里，调用方可以直接遍历结果而无需特判元数据。

**双时间戳是硬性契约。** `memory/llm/utils.py` 的 `conversation_text()` 把对话渲染为 `[2024-03-15T14:30:00] Alice: 内容`，`latest_timestamp()` 取最后一个时间戳作为「今天」的锚点注入 prompt。没有这个前缀，LLM 既无法把「昨天」换算成绝对时间，也产不出 `mention_time`——而 State 版本链正是按 `event_time` 排序的。时间戳键名同时兼容 `timestamp` / `time_stamp` / `float_time_stamp`，说话人兼容 `speaker_name` / `speaker`，缺省回落到 `role`。

**解析容错：** `parse_json_object()` 容忍 markdown 围栏和前后散文（不实现 `response_format` 的服务端很常见）；`normalize_extraction()` 把缺失的类别补成空列表、丢弃非 dict 的脏条目，一条坏数据不会废掉整批。OpenAI 后端在 JSON 模式被服务端拒绝时会自动去掉 `response_format` 重试一次。

当前阶段只实现「原始对话 → 结构化记忆提取」，不含摘要、冲突判断、在线更新与离线整理。

---

#### `storage/vector_store.py`

**功能：** Qdrant 封装层，提供 CRUD、dense 检索、稀疏检索、payload 过滤。

**参考 LightMem：** [LightMem/src/lightmem/factory/retriever/embeddingretriever/qdrant.py](../LightMem/src/lightmem/factory/retriever/embeddingretriever/qdrant.py)（393 行，Qdrant 连接与 `_create_filter` 的写法可参考）。

**LayerMem 需要重新实现的接口（LightMem 的 filter 能力不够用）：**

```
upsert(entry: Entry, dense: np.ndarray, sparse: SparseVector)
delete(ids: List[str])
search(dense_vector, limit, entry_type=None, layer=None, score_threshold=None) -> List[(Entry, float)]
hybrid_search(dense_vector, sparse_vector, limit, entry_type=None, ...) -> List[(Entry, float)]
scroll(filter_conditions, limit=None) -> List[Entry]      # 精确过滤
get_by_id(entry_id) -> Entry
update_payload(entry_id, patch: dict)
```

**物理设计（BM25 方案已定：Qdrant 稀疏向量原生 hybrid）：**

- **每个视角一个 collection**：`<collection_name>_factual` / `_relational` / `_state`。按视角切分意味着查一类记忆只扫那一类的索引——state 链的精确 `(subject, attribute)` 查找与 factual 的 ANN 扫描互不干扰，各 collection 也可独立重建或调优。**层级（L1–L4）仍是 collection 内的 payload filter**，所以召回依然可以跨抽象层级。
- 每个 collection 内用**命名向量** `dense`（`VectorParams(size=embedding_dims, distance=Cosine)`）+ `sparse`（`SparseVectorParams(modifier=Modifier.IDF)`）。
- **不要为 BM25 引入额外模型依赖。** 因为 `modifier=IDF` 让服务端负责 IDF 计算，客户端只需给出**词频**。本地维护一个 tokenizer + `hash(token) % 2**20` 的索引映射即可，无需持久化词表（哈希冲突只是让两个罕见词共享 IDF，可接受）。若之后需要更强的稀疏编码，再换 fastembed 的 `SparseTextEmbedding("Qdrant/bm25")`。
- 查询用 `query_points(prefetch=[dense_prefetch, sparse_prefetch], query=FusionQuery(fusion=Fusion.RRF))`。**这意味着「单轨内 ANN+BM25 的 RRF」可以整体下推到 Qdrant 服务端**，客户端不必自己实现。
- payload 需要建索引的字段：`layer`、`topic_id`、`subject`、`attribute`、`status`（`entry_type` 由 collection 本身承载，不再需要过滤字段）。Qdrant 的 filter 没有 payload 索引会退化为全表扫描。
- **跨视角的代价**：不指定 `entry_type` 的查询要扇出到 3 个 collection。dense 分数（cosine）跨库可比，可以直接归并取 top-k；但 hybrid 的 RRF 是**名次分**，跨库不可比，因此扇出时按轮转（round-robin）交错而非按分数归并——即每个 collection 的 rank-1 先占位，再放 rank-2。调用方知道视角时应当显式传入。
- `scroll` 的 `limit` 是**每个 collection** 的上限而非全局上限，否则某个视角条目多时会把其他视角挤空（压缩和水位线扫描需要拿到全部匹配项）。
- 一个实现细节：Qdrant 的**嵌入式**客户端在给不含该点的 collection 打 payload 时会抛 `KeyError`，而服务端部署是 no-op。`update_payload` 扇出时需按"不在这个库里就继续找"处理，并把"哪个库都没有"作为异常信号报出来。

---

### 层级 1：数据结构（`core/schema.py`）

---

#### `Entry`

**参考 LightMem：** LightMem 没有统一的 Entry 数据类（其 `MemoryEntry` 是平铺的 dataclass），需要自己定义。

```python
@dataclass
class Entry:
    # 身份
    id: str                          # UUID
    entry_type: str                  # "factual" | "relational" | "state"
    layer: str                       # "L1" | "L2" | "L3" | "L4"

    # 内容
    memory: str                      # 记忆文本

    # 时间戳（双时间戳）
    float_mention_time: float        # 提及时间（对话时间，unix timestamp）
    float_event_time: Optional[float]# 事件发生时间（可为空）

    # 聚类归属
    topic_id: Optional[str]          # 所属语义主题 ID（L1/L2/L3 用）

    # State 专用
    speaker_id: Optional[str]        # 主体 ID
    attribute: Optional[str]         # 属性名
    attribute_value: Optional[str]   # 属性值
    status: Optional[str]            # "current" | "superseded"

    # 工具方法
    @classmethod
    def from_payload(cls, payload: dict) -> "Entry": ...
    def to_payload(self) -> dict: ...
```

**不需要的字段：** `valid_from`, `valid_to`, `superseded_by`, `parent_id`, `member_ids`（用 `topic_id` 反向查询替代）。

**排序键约定：** `float_event_time` 可为空，凡涉及「按时间排序」的逻辑（压缩读取顺序、版本链排序）一律使用 `float_event_time or float_mention_time`，不要直接比较裸的 `float_event_time`。

**建议增补的字段见 §七.6**（provenance、`mention_count`、`updated_at`、topic 节点的 `compressed_until`）。

---

### 层级 2：核心处理组件

依赖：层级 0 + 层级 1，按以下顺序实现。

---

#### `pipeline/buffer.py`

**功能：** 积累对话轮次，达到触发条件后将 batch 交给 extractor 处理。

**参考 LightMem：** 无直接对应组件，需自行实现。LightMem 是逐 session 调用 `add_memory()`，LayerMem 引入缓冲池批量处理。

**核心逻辑：**

```
内部维护 deque，每次对话 append 一条 {"role", "content", "timestamp"}
触发条件：轮次 >= buffer_turn_threshold（默认10）OR token 数 >= buffer_token_threshold（默认4096）
触发后：flush() → 返回当前 buffer 内容，清空 buffer
token 计数：简单 split 或 tiktoken，不需要精确
```

**输出格式必须包含 `timestamp`**：`memory/llm/utils.py::conversation_text()` 依赖它渲染时间前缀，缺了它整条双时间戳链路失效。若上游数据源只给 `time_stamp` / `float_time_stamp` 也能被识别，但 buffer 自身的契约统一用 `timestamp`。

---

#### `pipeline/extractor.py`

**功能：** 接收 buffer flush 出来的对话 batch，一次 LLM 调用提取三视角记忆原子。

**参考 LightMem：** [LightMem/src/lightmem/factory/memory_manager/openai.py](../LightMem/src/lightmem/factory/memory_manager/openai.py) 的 prompt 组织与 `_extract_with_prompt` 的并发批处理可参考；但 LightMem 是 flat / 双路（factual + relational）提取，LayerMem 一次调用同时产出三个视角。

**核心逻辑：**

- 调用 `LLMFactory.from_config(config.llm)` 得到后端，再调用 `extract_memories(batch, current_date=...)`。
- 产出即 `{factual, relational, state}` 三个列表（+ `meta`），不强制三类都有输出。
- 过滤无信息量内容由 prompt 承担（见 `prompts/extraction.py` 的 Rules 段）。
- 三视角共用一次调用，不需要并发——这是相对 LightMem 双路提取的成本优势。

**失败语义（见 §七.5）：** LLM 调用或 JSON 解析失败时，当前实现会抛异常，buffer 里的那批对话随之丢失。需要补重试与兜底。

---

#### `pipeline/normalizer.py`

**功能：** 把提取出的 `(subject, attribute)` 归一化到规范形式。

**参考 LightMem：** 无对应组件。

**为什么它是必需组件而不是可选优化：** State 版本链是按 `(subject, attribute)` **精确匹配**查询的（`get_current` / `get_at_time`），检索规划器输出的 `state_attr` 也要落在这个键空间里。若「住址」「居住地」「address」各自成链、「我」「用户」「Alice」各自成主体，那么版本链会碎片化成互不相干的短链，`get_current` 静默返回空——不报错，只是查不到。

**设计见 §七.1（待决策）。**

---

#### `pipeline/clusterer.py`（待建）

**功能：** 为新提取的 Factual/Relational 原子分配 `topic_id`（在线增量聚类）。

**参考 LightMem：** [LightMem/src/lightmem/factory/topic_segmenter/llmlingua_2.py](../LightMem/src/lightmem/factory/topic_segmenter/llmlingua_2.py) 可参考分段的接口形状；但 LightMem 的 TopicSegmenter 是**离线批量分割**（用 LLMLingua-2 做文本压缩辅助），LayerMem 需要的是**在线逐条聚类**。

**核心逻辑（在线增量，不是批量）：**

```
对新 entry 计算 embedding
在 vector_store 中查该 entry_type 的 L1 层已有 topic（取各 topic 的代表向量）
找相似度最高的 topic
  if max_sim >= theta_cluster：归入该 topic（entry.topic_id = 该 topic_id）
  else：创建新 topic（新 UUID），entry.topic_id = 新 topic_id
```

两个关键点（来自 `LayerMem.md` §2.2）：

1. 用「稳定 ID + 成员集合」而非「位置区间」描述主题集合，插入是 O(1)，不会级联更新。
2. **三个视角在各自独立的语义空间里聚类**，Factual 和 State 的内容不会因为语义相近被错误合并。

`theta_cluster` 需要在 dev set 上调，初始值 0.75。

---

#### `storage/state_chain.py`

**功能：** 处理 State 视角的新条目，维护版本链。

**参考 LightMem：** 无对应组件，完全自行实现。

**四分支逻辑（按 `event_time` 排序，不是按写入顺序）：**

```
新条目到达 (subject, attribute, value, event_time)
  ↓
（先经 normalizer 归一化 subject / attribute）
  ↓
查 Qdrant：该 (subject, attribute) 的所有版本
  ↓
  无历史版本 → ADD：直接写入 status=current
  有历史版本 →
    找 current 版本（float_event_time 最大且 status=current）
    if new_event_time >= current_event_time：
      语义相似度(new_value, current_value) >= theta_confirm？
        YES → CONFIRM：不写入，更新 mention_time 即可
        NO  → UPDATE：旧 current 改 superseded，新条目写入 current
    else（new_event_time < current_event_time）：
        HISTORICAL_INSERT：直接写入 status=superseded，不动 current
```

> 为什么按 `event_time` 排序：后来提到的可能是更早发生的事。用户先说自己现在住北京，之后又提到去年住上海——上海的正确位置在上海那一段，而不是链尾。

**对外接口：**

```python
handle(entry: Entry)   # 处理一条新 State 条目
get_current(subject, attribute) -> Entry
get_at_time(subject, attribute, target_ts) -> Entry
```

**边界情况见 §七.4（否定/终止状态、CONFIRM 是否推进时间、非标量值）。**

---

#### `compression/layer.py`

**功能：** 在线监听各 topic 的条目数量，触发时调用 LLM 生成上层摘要（L1→L2，L2→L3）。L3→L4 触发离线蒸馏。

**参考 LightMem：** [LightMem/src/lightmem/memory/lightmem.py](../LightMem/src/lightmem/memory/lightmem.py) 的 `text_summary` 可参考 prompt 组织，但 LightMem 的摘要是 **session 级**的，LayerMem 是 **topic 级**的分层摘要。

**触发时机：**

```
每次 Clusterer 完成 topic 分配后，检查该 topic 在当前层的条目数
  if count(topic_id, layer="L1") >= K1（默认5）：触发 L1→L2 压缩
  if count(entry_type, layer="L2") >= K2（默认5）：触发 L2→L3 压缩
  if count(entry_type, layer="L3") >= K3（默认3）：触发 L3→L4 蒸馏
```

> ⚠️ 上面的 `count` 条件有缺陷，且 L2/L3 的粒度尚未定论，**不要照抄实现**。见 §七.2（水位线）与 §七.3（层级粒度）。

**压缩逻辑：**

```
读取 topic 下所有 L1 条目（按 float_event_time or float_mention_time 排序）
LLM 生成摘要 → 写入 L2（带 topic_id + float_mention_time=now）
不删除 L1（L1 永远保留）
```

---

#### `compression/compactor.py`（待建）

**功能：** 离线定期整理 L2/L3/L4 的冗余内容，防止无限增长。

**参考 LightMem：** [lightmem.py:541](../LightMem/src/lightmem/memory/lightmem.py#L541) 的 `offline_update_all_entries(score_threshold, max_workers)` 的批处理与调用时机可参考，具体逻辑不同。

**三种整理（共用同一个 `_compact()` 核心函数，差异只在冗余发现方式）：**

```
L2 整理（within-topic）：
  触发：同 topic 的 L2 数量 >= l2_compact_count 或时间跨度 >= l2_compact_days
  操作：同 topic 所有 L2 → LLM 合并 → 写新 L2 → 删旧 L2

L3 整理（cross-topic 去重）：
  触发：同 entry_type 的 L3 总数 >= l3_compact_count
  操作：Union-Find 找语义相似对（sim >= l3_sim_threshold）→ 合并 → 删旧 L3

L4 整理（全局画像）：
  触发：L4 数量 >= l4_compact_count 或距上次整理 >= l4_compact_days
  操作：所有 L4 + State current 快照 → LLM 综合 → 写新 L4 → 删旧 L4
```

**调用方式：** 独立后台任务，不在写入路径上阻塞。

---

#### `retrieval/router.py`（待建）+ `retrieval/retriever.py`（待建）

拆成两个模块：`router.py` 负责**一次 LLM 调用**做检索规划，`retriever.py` 负责按轨道执行检索与合并。

**参考 LightMem：**

- [LightMem/src/lightmem/factory/retriever/embeddingretriever/qdrant.py](../LightMem/src/lightmem/factory/retriever/embeddingretriever/qdrant.py) — ANN 检索与 filter 写法
- ⚠️ LightMem **没有 BM25 实现**：[factory/retriever/contextretriever/bm25.py](../LightMem/src/lightmem/factory/retriever/contextretriever/bm25.py) 是 1 行的空壳（factory 里挂着 `"BM25": "...contextretriever.bm25.BM25"` 的映射，但类从未实现）。BM25 按 §三 `storage/vector_store.py` 的 Qdrant 稀疏向量方案自行实现。

**完整检索流程：**

```
1. 检索规划器（一次 LLM 调用，prompt 见 prompts/retrieval.py）
   输出：
     tracks:     [factual | relational | state]（可多选）
     keywords:   [关键词列表]（稀疏检索用）
     state_attr: 属性名（仅 state 时）
     subject:    主体（仅 state 时）          ← 见 §七.7
     time_ref:   时间表达 | null（仅 state 时）

2. 按轨道分流：
   Factual / Relational 轨：
     Hybrid 召回（dense + sparse，Qdrant 端 FusionQuery(RRF)，不加层级过滤）
     阈值过滤（见下方关于分数量纲的说明）
     取 top-K（K 是上限，不是目标）

   State 轨：
     不做 ANN，走版本链精确查询
     time_ref == null → get_current(subject, state_attr)
     time_ref != null → get_at_time(subject, state_attr, target_ts)
     至多返回 1 条

3. 多轨合并：
   Factual + Relational → RRF 重排序 → 取合并后 top-K
   State 结果直接附加，不参与重排

4. 返回纯文本列表（无前缀标记）：
   ["2024-03-15, 居住地: 芬兰",
    "用户对印象派艺术有持续热情",
    "Bob 祝贺了 Alice 被录取"]
```

**关于「不加层级过滤」：** L1–L4 的分层压缩同时编码了「时序稳定性」与「语义抽象度」，而查询文本的 embedding 天然落在这条「具体→抽象」的维度上（"用户上周做了什么" 偏具体，"用户喜欢什么风格" 偏抽象）。因此不需要预判该查哪一层，ANN 的距离计算会自动完成粒度对齐。这也正是 LayerMem 不做 TiMem 那种 simple/hybrid/complex 路由的原因——查询难度和存储层级并不等价。

**关于阈值：** dense 的 cosine 分数与稀疏检索的 BM25 分数**不在同一个量纲上**，用一个共同阈值过滤会误杀「关键词精确匹配但 embedding 一般」的条目。阈值只应作用于 dense 分支，或改用 RRF 融合后的名次截断。见 §七.8。

**多轨 RRF 合并（Factual + Relational）：**

```python
rrf_score = 1/(k + rank_factual) + 1/(k + rank_relational)   # k=60
```

（单轨内部 dense+sparse 的 RRF 由 Qdrant 的 `FusionQuery` 在服务端完成，客户端不重复实现。）

---

### 层级 3：对外入口

---

#### `api.py`（待建）

**功能：** `LayerMemory` 对外统一接口，编排所有核心组件。

**参考 LightMem：** [LightMem/src/lightmem/memory/lightmem.py](../LightMem/src/lightmem/memory/lightmem.py) 的 `LightMemory` 类接口设计可参考，内部实现完全不同。

> 注意：上一版文档把入口放在 `memory/__init__.py`，但 `memory/` 现在是**后端接入包**（embedder/llm），不适合再承载编排器。入口改为顶层 `src/layermem/api.py`。

```python
class LayerMemory:
    def __init__(self, config: LayerMemConfig): ...

    def add(self, messages: List[dict]) -> None:
        """
        主写入入口
        1. 写入 buffer（每条消息带 timestamp）
        2. buffer 触发 → extractor 提取三视角
        3. normalizer 归一化 subject / attribute
        4. Factual/Relational → clusterer 分配 topic → vector_store 写入
        5. State → state_chain.handle()
        6. compression/layer 检查是否需要压缩
        """

    def retrieve(self, query: str) -> List[str]:
        """主检索入口：router 规划 → retriever 三轨检索 → 返回记忆文本列表"""

    def compact(self) -> None:
        """手动触发离线整理（也可设为定时任务）→ compactor.run()"""
```

---

## 四、LightMem 组件复用对照表

> 所有「复用方式」都是**参考并重写**，不存在直接 import LightMem 的运行时对象。

| LayerMem 模块 | LightMem 对应组件 | 复用方式 |
|---|---|---|
| `memory/embedder/` | [factory/text_embedder/](../LightMem/src/lightmem/factory/text_embedder/) | 参考接口形状重写；配置改由 `configs/config.py` 提供 |
| `memory/llm/` | [factory/memory_manager/](../LightMem/src/lightmem/factory/memory_manager/) | 参考 prompt 组织与并发批处理；三视角 schema 自行设计 |
| `storage/vector_store.py` | [factory/retriever/embeddingretriever/qdrant.py](../LightMem/src/lightmem/factory/retriever/embeddingretriever/qdrant.py) | 参考 Qdrant 连接与 filter 构造，接口重新设计并新增稀疏向量 |
| `retrieval/retriever.py`（BM25 部分） | ⚠️ 无实现（`contextretriever/bm25.py` 是空壳） | 完全自行实现，走 Qdrant 稀疏向量 |
| `compression/layer.py` | [memory/lightmem.py](../LightMem/src/lightmem/memory/lightmem.py) 的 `text_summary` | 参考 prompt 组织；LightMem 是 session 级摘要，LayerMem 是 topic 级分层摘要 |
| `compression/compactor.py` | [lightmem.py:541 `offline_update_all_entries()`](../LightMem/src/lightmem/memory/lightmem.py#L541) | 参考调用时机与并发度，整理逻辑自行实现 |
| `pipeline/buffer.py` | 无 | 自行实现 |
| `pipeline/extractor.py` | `factory/memory_manager/` 的提取流程 | 参考 prompt 组织；LightMem 双路，LayerMem 一次调用三视角 |
| `pipeline/normalizer.py` | 无 | 完全自行实现 |
| `pipeline/clusterer.py` | [factory/topic_segmenter/](../LightMem/src/lightmem/factory/topic_segmenter/) | 仅参考接口形状；LightMem 是离线批量分割，LayerMem 是在线增量聚类 |
| `storage/state_chain.py` | 无 | 完全自行实现 |
| `core/schema.py` | 无统一 Entry 类 | 自行实现 |

---

## 五、实现顺序建议

```
第一批（依赖最少，先跑通 Qdrant 基础操作）：
  core/schema.py
  memory/embedder/          ✅ 已完成
  memory/llm/               ✅ 已完成
  storage/vector_store.py

第二批（核心写入路径）：
  prompts/extraction.py     ✅ 已完成
  pipeline/buffer.py
  pipeline/extractor.py
  prompts/compression.py
  pipeline/normalizer.py    ← 新增，必须在 state_chain 之前完成
  pipeline/clusterer.py
  storage/state_chain.py

第三批（分层压缩）：
  compression/layer.py
  compression/compactor.py
  prompts/compaction.py

第四批（检索路径）：
  prompts/retrieval.py
  retrieval/router.py
  retrieval/retriever.py

第五批（对外接口）：
  api.py
  configs/config.py 中的 LayerMemConfig 汇总
```

每批完成后写对应的单元测试，不要等全部实现完再测。

---

## 六、关键超参数

```python
@dataclass
class LayerMemConfig:
    # 子配置
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Buffer
    buffer_turn_threshold: int = 10
    buffer_token_threshold: int = 4096

    # 聚类
    theta_cluster: float = 0.75      # 归入已有 topic 的相似度下限

    # 去重（归属待定，见 §七.9）
    theta_dup: float = 0.92          # L1 写入时去重阈值

    # 压缩触发
    K1: int = 5                      # L1→L2 触发数量
    K2: int = 5                      # L2→L3 触发数量
    K3: int = 3                      # L3→L4 触发数量

    # 检索
    theta_retrieval: float = 0.6     # 相似度阈值（仅作用于 dense 分支，见 §七.8）
    top_k: int = 10                  # 检索上限

    # State
    theta_confirm: float = 0.90      # CONFIRM 分支的语义相似度阈值

    # 离线整理触发
    l2_compact_count: int = 3
    l2_compact_days: int = 60
    l3_compact_count: int = 5
    l3_sim_threshold: float = 0.85
    l4_compact_count: int = 3
    l4_compact_days: int = 90

    # 存储（新增）
    qdrant_path: str = "./data/qdrant"      # 或 url/host+port
    collection_name: str = "layermem"
    sparse_dim: int = 2 ** 20               # 稀疏向量哈希空间

    # 健壮性（新增）
    request_timeout: float = 60.0
    max_retries: int = 3
    llm_concurrency: int = 5
```

所有阈值都在 dev set 上调优后再用于 test，不使用默认值直接报 test 结果。

---

## 七、待决策与待实现

以下问题在 `LayerMem.md` 的设计意图里成立，但落到实现上还缺一个明确的答案。**每条都给出候选方案与需要定的量，实现前先定下来。**

### 七.1 `pipeline/normalizer.py` 的规范化策略

**问题：** State 版本链按精确 `(subject, attribute)` 匹配，`get_current(subject, attr)` 查不到就是静默返回空。属性名与主体名的任意措辞都会切碎版本链。

**候选：**

- (a) **受控词表**：每个 subject 维护一个 attribute 规范词表，首次遇到新属性时由 LLM 归一（如 "居住地" → `residence`），之后直接复用；prompt 里把已有词表回灌给提取器，让它优先选用已有名字。
- (b) **向量化解**：不要求字符串相等，改为对属性名做 embedding 相似度匹配（阈值 θ_attr）。灵活但对「住址」和「工作地址」这类近义属性容易误合并。
- (c) **混合**：字符串归一（大小写、单复数、同义词表）打底 + 相似度兜底。

**需要定：** 候选方案、θ_attr 取值（若用 b/c）、主体别名表（我 / 用户 / user_id）的维护方式。

### 七.2 压缩水位线

**问题：** 触发条件 `count(topic_id, layer="L1") >= K1` 在 L1 永不删除的前提下**恒为真**——一旦某个 topic 达到 K1，此后每次写入都会再次触发摘要，同一批 L1 被反复压缩，成本爆炸且 L2 出现重复条目。

**候选：** topic 节点上记一个 `compressed_until`（已压缩到的时间水位），触发条件改为「该 topic 中 `event_time > compressed_until` 的 L1 数量 >= K1」，压缩后推进水位。L2→L3 同理。

**需要定：** 水位存在 topic 节点的 payload 上还是单独的表；压缩时是只读增量条目还是每层重读全量（后者摘要更连贯但成本线性增长）。

### 七.3 层级粒度：L2 / L3 / L4 到底挂在哪一级

**问题：** 现有描述自相矛盾——`K1` 按 topic 触发，`K2`/`K3` 却按 `entry_type` 触发（跨 topic）；`core/schema.py` 的 `topic_id` 注释又写「L1/L2/L3 用」；而 `LayerMem.md` 把 L3 定义为「阶段画像 / 跨时间稳定特征」，听上去是跨 topic 的。另外 `K2` 与 `l2_compact_count` 数的是同一批 L2，两个阈值会互相打架。

**需要定：**

- L2 是否严格 per-topic（我认为是，它是「同一主题的多次事件的汇总」）？
- L3 是 per-topic 的阶段画像，还是 per-entry_type 的用户特征？若是后者，`topic_id` 就不该出现在 L3 上。
- L4 是**单例**（一个全局画像）还是每个 entry_type 一个？`l4_compact_count: 3` 的存在暗示 L4 可以有多个，但「稳定用户画像」的说法又像单例。
- `K2` 与 `l2_compact_count` 的职责如何划分（一个是压缩触发，一个是去重整理，但都在数 L2 的个数）。

### 七.4 State 链的边界情况

**需要定：**

- **否定/终止**：用户说「我不在北京了」「我已经离职了」——是写入 `value=null` 的新版本，还是给旧版本打 `status=ended`？`LayerMem.md` 只描述了值的覆盖，没描述状态的终止。
- **CONFIRM 分支是否推进时间**：用户 1 月说住北京，8 月又确认「我还在北京」，此时 `current` 版本的 `float_event_time` 要不要推进到 8 月？不推进的话，`get_at_time(6月)` 与 `get_at_time(9月)` 返回同一条，语义上没错；但推进后时间线更准确。需要明确取舍。
- **非标量值**：发色这类「新值覆盖旧值但旧值并未失效，只是变了」的属性，与住址这种互斥属性是否该分开处理。
- **同一 `event_time` 的多次提及**：去重与合并规则。

### 七.5 写入失败语义

**问题：** `extract_memories()` 在 LLM 调用失败或 JSON 解析失败时抛异常，buffer 里那批对话随之丢失——而且是静默的（调用方不处理异常就等于丢数据）。

**候选：** 保留 batch → 指数退避重试（`max_retries`）→ 多次失败后以**原文**落一条 L1 兜底条目（`entry_type=factual`，标记未结构化），保证「宁可存得粗糙，不可丢失」。

**需要定：** 兜底条目的标记字段、是否要单独的重试队列（跨进程重启后仍能补做）。

### 七.6 `Entry` 建议增补的字段

**候选：**

- `source_dialog_ids: List[str]` / `session_id` — provenance。L2/L3/L4 是 LLM 生成的摘要，出问题时唯一的追溯手段是回到原始对话。
- `mention_count: int` — 同一事实被提及的次数（对应 LightMem `MemoryEntry.hit_time`），可用于显著性排序与衰减。
- `updated_at: float` — 整理/压缩会重写条目，需要区分创建时间与修改时间。
- `compressed_until` 放在 topic 节点上（见 §七.2）。

**需要定：** 增补哪些、`topic_id` 在 L4 上是否保留。

### 七.7 检索规划器需要输出 `subject`

**问题：** `LayerMem.md` 的规划器输出只有 `state_attr` 和 `time_ref`，但 `get_current(subject, state_attr)` 需要主体。对单用户场景可以硬编码「User」，但提取出的 State 条目 subject 是多样化的（Alice / Bob / 用户）。

**候选：** (a) 规划器增加 `subject` 输出（与 `state_attr` 同级）；(b) State 轨改为「扫该 attribute 下所有 subject，各返回一条」。单用户数据集用 (a) 更省，多主体对话用 (b) 更稳。

**需要定：** 采用哪种；若选 (a)，`subject` 解析失败时的兜底。

### 七.8 Hybrid 检索的阈值作用于哪一段

**问题：** dense 的 cosine 与稀疏检索的 BM25 分数不可比，`LayerMem.md` 的「score < θ 直接丢弃」如果作用在融合后的混合分数上，会系统性误杀关键词命中。

**候选：** (a) 阈值只作用于 dense 分支，稀疏分支按名次取 top-N；(b) 两者都先取各自 top-N 再 RRF，最后只按名次截断到 K，完全不设分数阈值；(c) 各自归一化后再比较。

**需要定：** 方案；`theta_retrieval` 若保留则明确它只约束 dense。

### 七.9 `theta_dup` 的归属

**问题：** `theta_dup: 0.92 # L1 写入时去重阈值` 定义了但没有任何组件引用它——当前设计里 L1 写入路径只有 clusterer 的 topic 归属判断（`theta_cluster`）。

**候选：** (a) 挂到 clusterer 里作为二级判断：同一 topic 内若新原子与已有原子的相似度 >= `theta_dup`，视为重复提及，不新增条目而是累加 `mention_count`（配合 §七.6）；(b) 删除该参数。

**需要定：** 去重是「丢弃」还是「累加计数」，两者对时序压缩的信息量影响不同。

---

## 附录 A：开发环境

```bash
cd LayerMem
python3.13 -m venv .venv             # 需 3.10–3.13（见 pyproject requires-python）
.venv/bin/pip install -e ".[dev]"    # 全部依赖 + pytest
```

**两个已在本机验证过的环境坑：**

- **HuggingFace 直连不通**，模型下载必须走镜像：`export HF_ENDPOINT=https://hf-mirror.com`。本地 embedding 后端、`transformers` 后端，以及后续所有 `from_pretrained` / `load_dataset` 都受影响；不加会在连接阶段挂起后报错，看起来像代码 bug。
- **macOS 27 无法加载 scipy 1.15.3 的 arm64 wheel**（dyld 拒绝其 `__thread_bss` 段），会让 `import sentence_transformers` 直接失败。pyproject 已放宽为 `scipy>=1.16` 并附注释。

**测试：**

```bash
.venv/bin/pytest tests/ -m "not integration"                          # 快速用例，无需网络
HF_ENDPOINT=https://hf-mirror.com .venv/bin/pytest tests/ -m integration   # 含真实模型下载
```

当前覆盖 `memory/embedder/` 与 `memory/llm/`：prompt 渲染、对话时间戳渲染、JSON 容错解析、
两个工厂的后端选择、embedder 的批量/单条一致性、config 不被改写、LLM 的 JSON 模式降级与
token 统计，以及一次真实的本地 embedding 往返。其余模块实现时应同步补齐测试。
