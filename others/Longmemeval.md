### LongMemEval

```text
longmemeval_s_cleaned.json
└── 样本数组（500 条）
    ├── question_id
    │   └── 样本唯一 ID，如 "6a1eabeb"
    │
    ├── question_type
    │   └── 问题类别
    │       ├── "multi-session"
    │       ├── "temporal-reasoning"
    │       ├── "knowledge-update"
    │       ├── "single-session-user"
    │       ├── "single-session-assistant"
    │       └── "single-session-preference"
    │
    ├── question
    │   └── 测试问题，如
    │       "What was my personal best time in the charity 5K run?"
    │
    ├── question_date
    │   └── 提问时间，如 "2023/06/25 (Sun) 13:22"
    │
    ├── answer
    │   └── 标准答案，如 "25 minutes and 50 seconds (or 25:50)"
    │
    ├── answer_session_ids
    │   └── 包含答案依据的会话 ID 列表
    │
    ├── haystack_dates
    │   └── 历史会话时间列表
    │
    ├── haystack_session_ids
    │   └── 历史会话 ID 列表
    │
    └── haystack_sessions
        └── 历史会话列表
            ├── 第 1 个 session
            │   ├── {"role": "user", "content": "..."}
            │   ├── {"role": "assistant", "content": "..."}
            │   └── ...
            ├── 第 2 个 session
            └── ...
```