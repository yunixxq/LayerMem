# LayerMem 评测框架

LayerMem 的评测复用 LightMem 的 `memory_toolkits`，其是一个用来横向对比各家记忆系统的 harness。我们把 LayerMem 作为新的 memory layer 注册进去，可以与 NaiveRAG / FullContext 等基线共用同一套数据加载与评分。


## 一、整体数据流
以 Locomo 的实现为例，**每个 trajectory 是一个独立评测单元**，构建和检索阶段通过独立存储实现隔离；评估阶段使用已经检索好的结果。比如 conv0 和 conv1 的记忆不会混在一起。
```
原始数据集（LoCoMo / LongMemEval JSON）
    ↓ read_raw_data() 
MemoryDataset
    ├─ trajectories[]                ✅ 对应 conversation 
    │     └─ sessions[] → messages[]
    └─ question_answer_pair_lists[]  ✅ 对应 qa

trajectory = Trajectory(  -- 评测单元构建
    sessions=sessions,
    metadata={
        "id": f"locomo_{sample_idx}",
        "speaker_a": speaker_a, 
        "speaker_b": speaker_b,  
    },
)
```

## 二、三阶段流水线

```
Stage 1  memory_construction.py   建记忆
            每场对话逐条 add_message() → save_memory() 落盘
            ↓  输出：qdrant中的记忆内容
Stage 2  memory_search.py         检索
            每题 retrieve(query, k)
            ↓  输出：{memory_type}_{model}_{dataset}_{top_k}_{start}_{end}.json
Stage 3  memory_evaluation.py     答题 + 评判
            ↓  输出：各子类准确率 + *_evaluation.json
```

三段**完全解耦**，但必须顺序执行。

### Stage 1 关键点

- **断点续跑**：`if not rerun and layer.load_memory(user_id): return`。已建过的用户自动跳过，不重复烧 LLM。
- **逐条写入**：`add_message({"role", "content"}, timestamp=session.get_string_timestamp())`，每条之间 `time.sleep(0.2)` 防风控。时间戳是**会话级**的，LoCoMo 的消息本身没有独立时间。
- **并行粒度是 trajectory（用户）级**，`_LOCK` 保护 layer 初始化。但 LayerMem 一场对话要几十次 LLM 调用，实测 `--num-workers 1` 更稳妥。
- **token 监控**用 monkey-patch 实现：运行时把 layer 的 LLM 调用方法替换成带计数的 wrapper，`with` 块退出后自动还原，**不改任何 baseline 源码**。统计按 `(model_name, op_type)` 分桶，结果写入 `token_cost.json`。

### Stage 2 关键点

- `layer.retrieve()` 必须返回 `List[{"used_content": ...}]`——下游评分只读这个字段，且**断言非空**。
- `dataset.filter_questions()` 在这里做题目过滤（LoCoMo 剔除对抗题）。
- 输出文件名编码了全部关键参数，方便对比。**注意 `model` 若是本地路径，这个"文件名"实际是嵌套目录**，shell glob 匹配不到，得用 `find`。

### Stage 3 关键点

- **答题**：把 `used_content` 拼成 context，用数据集对应的 prompt 调 QA 模型（LoCoMo 的 prompt 特别提示注意时间戳）。
- **评判**：再用一个模型对比预测与金答案，按类别分组统计准确率。
- **判分是子串匹配**（`"correct" in content`），而 `incorrect` 包含 `correct`，会把"答错"误判为"答对"，系统性高估准确率。详见 §六。

## 三、LayerMem 的接入点

### 核心入口

[src/layermem/api.py](../src/layermem/api.py) 的 `class LayerMemory`，可以将其理解为是 LayerMem 记忆存储系统的统一调用入口，具体功能由不同的模块实现。编排全部组件：

```text
add(messages)            写入（进 buffer，达阈值自动 flush）
flush()                  强制把 buffer 里的对话提取入库
retrieve(query, k)       检索，返回记忆文本列表
retrieve_detailed(query) 同上，但返回 RetrievalResult（含 tracks、entries）
compact() / close() / stats()
```

它是**纯库，没有 CLI**，必须由外部驱动——所以需要适配层。

### 适配层

[memories/layers/layermem.py](../memory_toolkits/memories/layers/layermem.py) 的 `LayerMemLayer`，实现 harness 要求的 `BaseMemoryLayer` 接口，并注册进 [memories/__init__.py](../memory_toolkits/memories/__init__.py) 的两张表（`MEMORY_LAYERS_MAPPING` / `CONFIG_MAPPING`）。

文件里有**两个** `LayerMemConfig`：1️⃣ 对外接口适配层(pydantic，字段名对齐 harness 其它 baseline，如 `retriever_name_or_path`、`llm_backend`)和 2️⃣ LayerMemConfig 本身的核心实现(dataclass，LayerMem 超参 `K1`、`theta_cluster`…)。import 时用 `as CoreConfig` 来进行区分区分。`__init__` 里实现逐字段搬运：

```
retriever_name_or_path  →  embedder=EmbedderConfig(model=…)
llm_backend / llm_model →  llm=LLMConfig(model_name=…, base_url=…)
save_dir                →  qdrant_path=os.path.join(save_dir, "qdrant")
K1/K2/K3/theta_*        →  同名直传
```

## 四、完整链路

```text
./run_layermem_locomo.sh                       ← 一键入口
  └─ memory_construction.py --memory-type LayerMem
       └─ MEMORY_LAYERS_MAPPING["LayerMem"]     ← 注册表查找（惰性加载）
            └─ LayerMemLayer(config)            ← 适配层
                 └─ LayerMemory(core)           ← ★ 核心入口
                      ├─ DialogBuffer            ← 写入路径起点
                      ├─ Extractor → Clusterer / StateChain → VectorStore
                      ├─ LayerCompressor
                      ├─ RetrievalRouter → Retriever   ← 读取路径
                      └─ VectorStore (Qdrant: _factual / _relational / _state)
```

## 五、运行

```bash
cd LayerMem

# 1. 环境
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]" litellm
python3.13 -m venv .venv-serve && .venv-serve/bin/pip install mlx-lm   # 用本地模型时

# 2. 模型与数据
.venv/bin/python scripts/fetch_model.py Qwen/Qwen3-1.7B models/Qwen3-1.7B
mkdir -p benchmarks/locomo && curl -sL \
  "https://ghproxy.net/https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" \
  -o benchmarks/locomo/locomo10.json

# 3. 本地模型服务（启用 OpenAI 兼容接口，提取/答题/评判共用）
export HF_HUB_OFFLINE=1
.venv-serve/bin/mlx_lm.server --model "$PWD/models/Qwen3-1.7B" \
    --host 127.0.0.1 --port 8080 --chat-template-args '{"enable_thinking": false}'
#   ↑ enable_thinking 必须关：Qwen3 默认在  thinking 块里推理，会耗尽 completion
#     预算，导致提取阶段拿不到 JSON

# 4. 一键跑三段
./run_layermem_locomo.sh 0 1      # conv0；改成 0 10 则跑全部 10 场
```

产物全部落在 `memory_toolkits/LayerMem_/…/`，已在 `.gitignore` 中，重建即可。查看记忆结构：

```bash
.venv/bin/python scripts/layermem_stats.py <qdrant目录> user_LoCoMo_locomo_0
```