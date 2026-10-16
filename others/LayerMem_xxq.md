配置导入完整调用链：
- `LayerMem.json` 中存在的字段，优先使用 JSON 值。
- JSON 中缺少的字段，由 toolkit 层的 `LayerMemConfig` 默认值补齐。
- toolkit 层再把这些值传给核心 `config.py` 中的 `LayerMemConfig`，构造 `core`。
- 最后通过 `LayerMemory(core)` 创建 `DialogBuffer` 等组件。
```text
--config-path configs/LayerMem.json
        ↓
memory_construction.py 读取 JSON
        ↓
toolkits/memories/layers/layermem.py::LayerMemConfig
        ↓
LayerMemLayer.__init__() 构造 src/layermem/configs/config.py::LayerMemConfig
        ↓
self.memory = LayerMemory(core)

```
---
extract执行抽取时的List[Message]的Message所对应的dict结构，这个消息结构是由locomo_message_preprocessor所处理得到的：
```text
{
    "role": "user",
    "content": "消息内容",
    "timestamp": "2020-05-01T10:00:00",
    "speaker_name": "Alice",
    "session_id": "locomo_session_0_1"
}
```

其中：

- `role`：角色，例如 `user`、`assistant`
- `content`：消息正文
- `timestamp`：这条消息所属的对话时间
- `speaker_name`：说话人的真实姓名，可选
- `session_id`：来源会话 ID

## 查询构建的测试

```bash
../.venv/bin/python memory_extraction.py \
  --dataset-path ../benchmarks/locomo/locomo10.json \
  --config-path configs/LayerMem.json \
  --llm-profile gpt-4o-mini \
  --start-idx 0 \
  --end-idx 1 \
  --output-dir ../extraction_results/gpt-4o-mini-prompt-test
```