# LayerMem 分层记忆：设计与实现

LayerMem 旨在给 Agent 提供基于语义-时序结构化的三视角长期记忆系统。

## 1. 现存问题

现有长期记忆系统（MemTree / StructMem / TiMem / VikingMem / LightMem）存在共同的根本缺陷：

1. **仅考虑“时序”或“语义”的单一视角**，导致记忆检索时无法同时捕捉时序和语义二者信息。
2. **把所有记忆当作“同质的文本片段”处理**，导致针对同一属性事实的新信息与旧信息并列存储，检索时 固定的 Top-K 可能召回矛盾内容，需要交由 LLM 自行判断查询所需的属性事实。
   
```text
问题1会导致一些按照时序组合的不相关的语义主题的记忆被错误合并，产生记忆碎片化和语义丢失；
问题2会导致状态更新类查询的系统性失败，即记忆系统中并列存在针对同一属性（eg. 住址）的多次描述。

用户 1月："我住在杭州"
用户 8月："我搬到芬兰了"
查询 9月："用户现在住在哪里？"

现有系统：根据 Top-k 的固定检索原则多条记录可能同时被召回，LLM 需要猜哪条是当前的最新值 → 概率性正确
```

## 2. 设计架构

### 2.1 三个视角

我们观察到，记忆在本质上是**异质的**：

| 类型 | 例子 | 本质 | 新旧关系 |
|---|---|---|---|
| 事实 | "我昨天看了哪吒2" | 不可变事件 | 新事件不使旧事件失效，二者之间是可以共存的 |
| 关系 | "小李是我好友" | 持续性关系 | 多段关系可并行共存 |
| 状态 | "我住在北京" | 可被覆盖的属性值 | 新值使旧值失效（互斥） |

因此，三类异质信息必须用**不同的数据结构分别建模**，同样在检索召回的时候也应当根据查询需求从不同视角类别的记忆库中进行召回，以确保最终喂给LLM的记忆片段**最小最精确**。
> ps. 虽然我们是需要提取出来三种不同语义，但是我们只需要对原始的输入流进行一次处理即可，即使用一个完整的Prompts实现三视角的提取，并将提取的内容分路存入向量数据库Qdrant中

### 2.2 两个维度

<u>**维度1. 语义轴（横轴）**</u>
对于语义轴，我们进行相同主题的聚类实现，由于用户的对话可能是具有跳跃性的，相同主题的内容可能并不是按顺序出现的（**Ref**. VikingMem3.1最后一段），因此我们需要将**相同主题的内容进行聚类**，以便于在时序轴上实现压缩精炼时实现的是对同一主题内容的摘要提取。

```text
作用：决定新生成的记忆原子“归属哪个主题节点”
机制：在线增量聚类（dialogbuffer->L1写入时实时聚类，不是事后批量进行组合）

对于从 dialogbuffer 中提取出来的新记忆原子 Node
  → 计算 embedding
  → 与 L1 已有语义节点做相似度匹配
  → 归入相似度最大的节点所在的 member_ids（稳定 ID 集合）中
  → 若所有节点的相似度 < θ，则创建新语义主题集合并将该原子加入

关键设计：
  ① 用"稳定 ID + 成员集合"而非"位置区间"来描述一个语义主题集合
     → 插入新原子 O(1)，不用位置区间标识主题集合导致级联更新
  ② 三个视角类别的原子在各自独立的语义空间中进行聚类
     → 不同视角 Factual 和 State 的内容不会因语义相近被错误合并
```

<u>**维度2. 时序轴（纵轴）**</u>
对于时序轴，我们不选择像 TiMem 那样按照 Session/Daily/Weekly...的固定时间间隔进行压缩，而是选择**在线增量压缩**，即按照同一主题集合中的原子数量进行压缩，当该集合中的原子数量达到一定阈值时，将原子压缩成一条记录，进入下一层，形成**语义时序分层压缩结构**。

```text
作用：决定记忆原子"如何随时间演化"
机制：Factual&Relational时序分层压缩 + State实现版本链构建 

驱动力：数量驱动（内容自适应）而非时间驱动（日历刻度）
  → 话痨用户：压缩触发频繁，节点均匀
  → 沉默用户：慢慢积累，不进行无意义的摘要提取
  → 比 TiMem 的日历驱动更合理
  对 TiMem 的预设定时间边界实现了合理优化

Factual / Relational 路：L1-L4 分层时序摘要压缩
  → dialogbuffer → L1 → L2 → L3 → L4
  → 触发阈值：Li中同一语义主题的Node数量count>=K_count
  → L1. 单条时间原子[多次发生的具体事件] 写入时根据语义向量在线聚簇到语义节点
  → L2. 主题摘要节点[主题级别的行为模式] 对于单条记忆的第一次聚类主题摘要
  → L3. 阶段画像节点[跨时间稳定特征] 描述跨时间的稳定特性
  → L4. 稳定用户画像 构建长期较为稳定的用户个人画像
* 层级越高---时序维度-时间跨度长，越稳定；粒度维度-细节越少，抽象程度越高
* 层级越低---时序维度-越近期，越偶发；粒度维度-细节越丰富，越具体

State 路：版本链
  → 定义：主体的某个属性当前仅存在唯一有效值，eg. 我 -- 住址 -- 北京
  → 同属性新值出现，旧值 status=superseded，构成串形版本链：北京(superseded) → 上海(superseded) → 芬兰(current)
  → 当前版本指针 current_ptr 始终唯一指向当前有效值
  → 历史版本保留（支持历史回溯，eg. 询问我去年住在哪里，需要根据住址这个属性找到该属性链上的时间为去年的旧值）
```

> 1. 对于时序分层的 L2-L4层，为了防止其无限增长的可能性，当时间跨度达到预设阈值/Li中的节点数量超过阈值后，交由LLM进行去重整理，防止类似的主题摘要/用户画像描述重复出现造成记忆冗余。
> 2. 无论是执行四层分层时序摘要压缩的 Factual/Relational 路，还是维护版本链的 State 路，都需要维护“**双时间戳**”来提升记忆精确性，即event_time（发生时间）+ mention_time（提及时间），在prompt中需要显式说明。（**Ref.** StructMem Prompt）
> 3. State路，对于新达到的已有属性的新值，并不一定是“最新状态”，例如可能存在之前说的是我现在住在北京，后面又提到我去年住在上海，因此版本链的构建是需要按照双时间戳中的**event_time**进行排序的。

### 2.3 缓冲池与提取

为了更好地维护我们的记忆架构且尽量减少对于LLM的调用产生延迟&Token开销，对于新来的对话我们会选择将其放入 **dialog buffer** 中，累计达到一定的数量后再进入记忆库中。

```text
缓冲池：不断积累用户与Agent的对话消息，达到触发条件后经LLM提取写入记忆库
  → 触发条件：对话轮次达到阈值 10 或 token 数达到阈值 4096 后
  → 特点：批量处理，避免逐句调用 LLM，大幅度降低成本开销
Extraction prompt：一次 LLM 调用生成三个视角的记忆原子
  → 三视角并行处理：需要同时包含三视角记忆类型的提取原则
  → 不过度提取：不能硬性规定必须三类记忆都提取，需根据当前的输入流自适应提取
  → 无信息量内容过滤：过滤掉无信息量内容，如“嗯”“好的”、开场白等
```

### 2.4 版本链 State_chain 的具体实现

首先，区分一下我们的版本链与**时序数据库**(Temporal Database)的区别，时序数据库存储的是“**数值随时间的采样**”，严格按照时间顺序发生&存储，而版本链存储的是事实生命周期，即“**事实随时间的版本**”，且并不一定是严格按照顺序发生&存储的，大多数情况下是一种“**语义时间**”，eg. 我现在住在xxx；我去年搬到了xxx。

**1. 数据模型**

- 链：`(subject, attribute)` 唯一确定一条状态链。
- 节点：标量 `value` + `event_time`（用于实现链内排序的键）。
- 无显式有效区间：失效时间由「后一节点的 event_time」隐含表达。
  
**2. 存储分工**

| 组件 | 职责 |
|---|---|
| SQLite | 权威存储：链节点 + 全部查询 |
| Qdrant | 写入时属性消解（向量相似度召回候选），非权威 |

关联关系：Qdrant payload 存 `(subject, attribute)` 用于回链定位；**查询永远只走 SQLite，Qdrant 不参与查询**。

**2. SQLite Schema**

```sql
CREATE TABLE state_chain (
  id             INTEGER PRIMARY KEY,   -- 节点身份（插入序）
  subject        TEXT NOT NULL,         -- 主体名称，eg. 刘艳雪
  attribute      TEXT NOT NULL,         -- 主体的唯一属性，eg. 住址
  value          TEXT NOT NULL,         -- 属性的标量值，eg. 杭州 / 北京
  content      TEXT NOT NULL,           -- 状态摘要：“我下个月可能搬去上海”
  event_time     REAL NOT NULL,         -- 时间：用于排序键
  mention_time   REAL NOT NULL,         -- 审计：何时被提到
);
CREATE INDEX idx_chain ON state_chain (subject, attribute, event_time);
```

idx_chain 的示例：
```text
(李彦学, 城市, 2024-01-05)
(李彦学, 城市, 2025-03-01)
(李彦学, 城市, 2025-06-15)   ← 同一 (subject, attribute) 的行聚在一起，按时间有序
(李彦学, 公司, 2019-07-01)
(李彦学, 公司, 2022-03-01)
(张三,   城市, 2023-11-11)
...
```

* **链是查询语义，而非物理结构**：链 = WHERE subject+attribute ORDER BY event_time 的连续段；无需指针/链表，索引树上天然相邻。
* **B+tree 复合索引 = 有序即存储**：idx_chain 让同链节点按 (subject, attribute, event_time) 物理有序；当查询命中索引最左前缀(subject+attribute)时可以直接读索引。

## 4. Qdrant 属性消解索引

Qdrant 不参与查询，只用于**属性消解(attribute resolution)+索引(Index)**：
> <u>属性消解</u>：判断新遇到的属性是否已经在库里面存在相似表示，存在则直接合入，否则新建；
> <u>索引</u>：辅助结构/旁路结构，而非权威存储，用于快速找到相似属性便于回链(SQLite)定位。

首先，明确 Qdrant 的三层数据组织：

```text
Qdrant
 └── collection（集合）          ≈ SQLite 里的“表”
       └── point（点）           ≈ 表里的“行” point_id 是每个point的唯一标识符，类似于 SQLite 表中的主键
             ├── vector          ≈ 行的"向量列"——参与相似度计算
             └── payload         ≈ 行的"普通列"（JSON）——不参与相似度，可过滤/返回
```

因此，根据我们的需求，我们需要创建一个名为 `state_attribute` 的集合，每个 point 代表“某个主题的某个属性名”：

```text
collection: state_attribute
point id:   hash(subject + attribute)     -- point的唯一标识符
vector:     embed(attribute)              -- 只向量化属性名
payload:    { subject, attribute }        -- 过滤条件 + 回链定位
```

规则：

- 检索必须按 `filter={subject}` 过滤，**不跨主体合并**。
- 相同 `(subject, attribute)` 多次写入 → upsert 覆盖，不产生重复 point。
- 只向量化属性名是确保向量相似度纯粹反映“属性名的语义接近程度”，**不引入主体语义**。

## 5. 写入流程

在获取了 LLM 对于原始语句片段提取出来的 state 相关内容后，执行 **upsert(update+insert)** 实现对于状态链的更新：

```text
upsert_state(subject, attribute, value, content, event_time, mention_time)
  │
  ├─ ① 属性消解（Qdrant 检索候选）→ ② 判定（阈值/LLM）→ ③ 写入 SQLite
  │                                             │
  │                        ┌─────────── 合并 ───┴─── 新建 ───────────┐
  │                        ↓                                          ↓
  │               INSERT 进已有链                          INSERT 首节点 + Qdrant upsert
  └─────────────────────────────────────────────────────────────────

upsert_state(subject, attribute, value, content, event_time, mention_time):
  1. 属性消解：Qdrant 搜索 embed(attribute)，filter subject → top-k
  2. 判定：
       sim(top1) ≥ θ_high (0.9)         → 直接合并至top1的这个属性(无需再次调用 LLM)
       θ_low ≤ sim < θ_high             → 将 topk 的属性以及新的属性一起喂给 LLM 确认后再决策
       无候选 或 sim < θ_low             → 直接新建链
  3. 写入：
       合并 → INSERT 节点（ORDER BY event_time 自然排序；
              尾部 = 新状态，中间 = 历史补录）
       新建 → INSERT 首节点 + Qdrant upsert 属性 point
```

## 6. 查询（仅 SQLite）

注意，查询语句不能由 LLM 生成，而是使用**预定义的 SQLite 语句模版**，以确保一致的架构格式(schema format)并减少潜在的幻觉风险(hallucinations) --- ref. Zep

* 首先，对于一个 LLM 判定需要查询 State 版本链的查询，**LLM 需要输出必需参数指标**：

```json
{
  "query_type": "current | range | all",
  "subject": "李彦学",
  "attribute": "住址",
  "time_lower": null,
  "time_upper": "2023-10-22T00:00:00Z"
}
```

* 其次，**预定义一个 SQL 模版**，支持三种查询：1. 当前状态；2. 时间区间；3. 全部历史。eg. A现在居住在哪里？/ A去年居住在哪里？/ A 一共住过哪些地方？
  
```sql
SELECT subject, attribute, value, content, event_time
FROM state_chain
WHERE subject = ?          -- ① subject（必填）
  AND attribute = ?        -- ② attribute（必填）
  AND event_time <= ?      -- ③ time_upper（可空 = 不限）
  AND event_time >= ?      -- ④ time_lower（可空 = 不限）
ORDER BY event_time ASC|DESC   -- ⑤ 白名单
LIMIT ?;                   -- ⑥ current = 1，其余 NULL
```

* 最后，将 LLM 解析得到的查询参数填入模版，执行查询。

```py
# LLM 输出 → SQL 参数的映射（服务端做，LLM 不碰 WHERE 文本）
def build_query(params: dict):
    assert params["query_type"] in ("current", "range", "all")
    order = {"current": "DESC", "range": "ASC", "all": "ASC"}[params["query_type"]]
    limit = 1 if params["query_type"] == "current" else None
    return (
        SQL_TEMPLATE.replace("ASC|DESC", order), 
        [
            params["subject"],
            params["attribute"], 
            params["time_upper"],                      
            params["time_lower"],                      
            limit,                                      
        ],
    )
```

### 2.5 检索

现有分层记忆系统（TiMem、MemTree）面临一个共同困境：查询者不知道记忆库里有什么，更不知道某个主题积累到了哪一层。 TiMem 用 simple/hybrid/complex 三级分类路由到不同层，本质上是在用查询难度预判存储层级——但这两件事并不等价，路由错误时要么**召回的细节不足**，要么**召回了过时的泛化画像**。

而我们的layerMem的检索逻辑基于一个结构性观察：L1-L4的分层压缩在时序-语义双轴的实现下同时编码了两个维度--时序稳定性与语义抽象度，且两个维度在embedding空间里天然对齐：
```text
L1：具体事件，偶发，embedding 偏具体
L2：主题摘要，反复出现，embedding 偏模式
L3：稳定特征，跨时间，embedding 偏抽象
L4：综合画像，全局，embedding 最抽象

查询文本的 embedding 也落在这个具体→抽象的维度上：
  "用户上周做了什么" → 偏具体 → 自然靠近 L1
  "用户喜欢什么风格" → 偏抽象 → 自然靠近 L3
```
因此，不需要预先决定查哪层——问题本身的抽象程度和对应存储层的抽象程度共享同一个语义空间，ANN 的距离计算会自动完成粒度对齐。

```text
查询
  ↓
检索规划器（一次 LLM 调用）
  输出：
    tracks:     ["factual" | "relational" | "state"]  // 可多选
    keywords:   ["关键词1", ...]                       // BM25 用，3-6个
    state_attr: "属性名"                               // 仅 tracks 含 state 时
    time_ref:   "去年" | "三个月前" | 具体时间点 | null   // 仅 tracks 含 state 时
  ↓
按轨道并行分流：

  ── Factual 轨（如需）──────────────────────────────
    Hybrid 召回（ANN + BM25），不考虑层次/主题，宽泛召回候选集
    相似度阈值过滤（score < θ 的条目直接丢弃）
    剩余条目取 top-K（K 是上限，不是目标），提前过滤是为了不用为了选 topk 而引入非相关记忆
    → 简单单跳查询：阈值过滤后可能只剩 1-2 条，直接返回
    → 复杂多跳查询：阈值过滤后剩余较多，再截断到 K
    
  ── Relational 轨（如需）──────────────────────────
    同 Factual 轨，流程完全一致
    → top-k

  ── State 轨（如需）──────────────────────────────
    不做 ANN，走版本链精确查询：
    if time_ref == null：
      get_current(subject, state_attr)
      → 返回 float_event_time 最大的 status=current 版本
    else：
      解析 time_ref → target_timestamp
      get_at_time(subject, state_attr, target_timestamp)
      → 返回 float_event_time ≤ target_timestamp 的最新版本
    → 至多 1 条，确定性结果，不参与后续重排
    → 返回的内容需要组织为“时间 + 主体 + 属性 + 唯一标量值”的完整格式
  ↓
多轨合并（如多轨触发）：
    Factual top-K + Relational top-K
      → RRF 重排序（消除两轨分数量纲差异）
      → 取合并后 top-K
    State 结果直接附加到末尾，不参与重排
  ↓
将所有记忆条目输入至LLM，生成最终答案
```

> 1. 我们其实在Facutal和Relational的检索流程中使用<u>**数据驱动**</u>的相似度分布，隐式实现了TiMem检索规划器试图完成的功能（区分simple/hybrid/complex以召回不同数量的记忆），同时我们避免了检索规划器可能出错的风险。
> 2. State 轨和另外两轨的根本区别在于：Factual/Relational 是“找相关的”，State 是"找正确的"。前者的答案是一个排序列表，后者的答案是一个确定性的值（**<u>在给定时间点上该属性只有一个有效版本</u>**）。这也是为什么 State 轨不需要参与重排——它的结果不需要和其他条目竞争相似度分数。