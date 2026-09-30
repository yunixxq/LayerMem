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
