# State 版本链实现说明

以 `LayerMem.md` §2.4 为准。`LayerMem_imple.md` 中把节点保存在 Qdrant、使用
`CONFIRM` 丢弃重复值及维护 `current/superseded` 的方案不再适用于 State 路径。

## 写入

`Extractor._state_atoms` 创建 `StateEntry`；`LayerMemory._write_state` 保留模型
返回的 `memory`，并交给 `StateChain.handle`：

1. 相同主体下已知的属性或已确认别名直接定位原链。
2. 新属性以 `embed(attribute)` 检索 Qdrant 的属性候选，始终按 `subject` 过滤。
3. 相似度 ≥ `state_attribute_high`（默认 0.90）合并；低于
   `state_attribute_low`（初始默认 0.75）或无候选时新建；中间区间调用 LLM，
   只接受候选属性名或 `null`。候选数由 `state_attribute_top_k`（默认 5）控制。
4. 节点写入 SQLite；新链另建一个 Qdrant 属性 point。

阈值比较针对属性名，和属性值是否相同无关。再次提及同值也写入新节点，保留
它的 `mention_time` 和 `memory`；同一 `entry_id` 的写入重试不会重复插入。
`theta_confirm` 仅作为旧配置兼容字段保留，已经不控制 State 写入。

属性只做大小写和空白归一化，保留多词属性及中文；不强制 snake_case。
已确认的别名存入 SQLite，读取别名时也无需查询 Qdrant。

## 存储

- SQLite：`state_chain` 保存节点，索引为 `(subject, attribute, event_time)`。
  `id` 是整数插入序；新增 `entry_id` 唯一列保留公开的 UUID，支持更新、删除和
  重试。对象的 `memory` 对应表中的 `content`。
- 对外时间使用标准化 UTC ISO 字符串；SQLite 用 `REAL` Unix 时间保存双时间戳。
- Qdrant：物理集合为 `<collection_name>_state_attribute`，避免不同用户/集合
  混用索引；每点 payload 只有 `{subject, attribute}`，UUID5 确定性生成 point ID。
  Factual/Relational 仍各用自己的向量集合，State 节点不再放入向量集合。
- 默认 SQLite 文件在 `qdrant_path` 内，按 `collection_name` 派生独立文件名；
  可通过核心配置 `state_db_path` 指定其它文件。

## 查询

`StateChain.query(subject, attribute, query_type, time_lower, time_upper)` 只查询
SQLite，SQL 的排序方向和 LIMIT 由服务端白名单选择，参数使用占位符绑定。

| 类型 | 返回内容 |
|---|---|
| `current` | 最新一条；提供 `time_upper` 时为该时间之前（含该时间）的最新值 |
| `range` | `event_time` 落在闭区间内的节点，升序排列 |
| `all` | 全部节点，升序排列；未提供上下界时不限制时间 |

范围查询遵循 §2.4 的节点时间过滤语义，不自动插入区间之前的前驱节点。
如果要查询某日有效值，使用 `get_at_time` 或 `current + time_upper`。
空时间界限表示不限制；空下界不能当作 0；不设旧实现的 1000 节点截断。

`event_time` 决定链内位置；相同时依次用 `mention_time`、整数插入序打破平局。
无需持久化 `current/superseded` 或显式有效区间。后提及的旧状态不会覆盖链尾。

检索规划器接收 SQLite 中现有链的属性目录，用于输出正确的主体和规范属性名，
返回 `current/range/all` 及绝对时间界限，不生成 SQL。缺少主体或属性时不任意
选择其它主体。多节点查询结果通过 `RetrievalResult.state_entries` 返回。

## 旧库与验证

`LayerMemory` 初始化时一次性导入已有 `<collection_name>_state` 节点，保留旧
Qdrant 集合，不删除源数据；SQLite 中记录迁移标记，重试按 UUID 去重。不会
重新调用 LLM 抽取。丢失双时间戳的旧节点会明确报错，不用当前时间替换。

属性索引缺失时，下一次写入根据 SQLite 重建缺失的属性 point；读取不要求
Qdrant 可用。统计、更新、删除和关闭入口同时处理新的 SQLite 存储。

离线测试：`.venv/bin/python -m pytest -q tests/test_state_chain_sqlite.py`。
