"""Inspect buffered LoCoMo extraction without clustering, storage or compression."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import os
import tempfile

from config_loader import load_memory_config
from layermem.configs.config import LLMConfig
from layermem.memory.llm import LLMFactory, conversation_text
from layermem.pipeline.buffer import DialogBuffer
from layermem.pipeline.extractor import Extractor
from layermem.prompts.extraction import (
    FACTUAL_RELATIONAL_EXTRACTION_PROMPT,
    STATE_EXTRACTION_PROMPT,
)


class RecordingLLM:
    """Record the same calls used by Extractor, including its retry."""

    def __init__(self, llm):
        self.llm = llm
        self.attempts = []

    def extract_memories(self, batch):
        try:
            result = self.llm.extract_memories(batch)
        except Exception as exc:
            # Avoid copying service error messages that may contain credentials.
            self.attempts.append({"success": False, "error_type": type(exc).__name__})
            raise
        self.attempts.append({"success": True, "result": result})
        return result


def extract_batches(trajectories, llm, turn_threshold=10, token_threshold=4096):
    recorder = RecordingLLM(llm)
    extractor = Extractor(recorder)
    for trajectory in trajectories:
        buffer = DialogBuffer(turn_threshold, token_threshold)
        batch_index = 0

        def flush(reason):
            nonlocal batch_index
            tokens = buffer.pending_tokens
            batch = buffer.drain()
            recorder.attempts = []
            try:
                atoms = extractor.extract(batch)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{trajectory.metadata.get('id')} / batch {batch_index + 1}: "
                    "无法调用 LLM 完成记忆抽取（已尝试两次），停止测试。"
                ) from exc
            batch_index += 1
            # Keep the exact text representation sent to the extraction prompt.
            # This makes the log directly comparable with the dialog buffer and
            # preserves message number, mention_time, speaker, and content in one
            # human-readable field.
            rendered_dialog = conversation_text(batch)
            return {
                "trajectory_id": trajectory.metadata.get("id"),
                "batch_index": batch_index,
                "trigger": reason,
                "input_tokens_estimate": tokens,
                "messages": batch,
                "dialog_buffer": rendered_dialog,
                "prompts": {
                    "factual_relational": FACTUAL_RELATIONAL_EXTRACTION_PROMPT + "\n\n" + rendered_dialog,
                    "state": STATE_EXTRACTION_PROMPT + "\n\n" + rendered_dialog,
                },
                "attempts": recorder.attempts,
                "atoms": {kind: [asdict(entry) for entry in entries] for kind, entries in atoms.items()},
            }

        for session in trajectory:
            for message in session:
                buffer.add({
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp,
                    "speaker_name": message.metadata.get("name") or message.metadata.get("speaker_name"),
                    "session_id": session.metadata.get("id"),
                })
                if buffer.should_flush():
                    yield flush("threshold")
        if buffer.pending_turns:
            yield flush("trajectory_end")


def write_review(record, output):
    output.write(f"## {record['trajectory_id']} / batch {record['batch_index']}\n\n")
    output.write(f"状态：抽取成功；触发：{record['trigger']}；尝试次数：{len(record['attempts'])}\n\n")
    output.write("### 输入消息\n\n")
    output.write("以下是 DialogBuffer 传给抽取器的完整文本表示（包含 message 序号、mention_time、speaker 和 content）。\n\n")
    # Use a fenced block so message content containing Markdown is preserved
    # verbatim and the review log exactly matches record["dialog_buffer"].
    output.write("```text\n")
    output.write(record["dialog_buffer"])
    output.write("\n```\n\n")
    for kind, atoms in record["atoms"].items():
        output.write(f"### {kind}（{len(atoms)}）\n\n")
        for atom in atoms:
            for line in json.dumps(atom, ensure_ascii=False, indent=2).splitlines():
                output.write(f"    {line}\n")
            output.write("\n")


def save_batches(path, records):
    """Replace the JSON snapshot only after the complete array is written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".batches-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(records, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-path", default=str(Path(__file__).parent / "configs/LayerMem.json"))
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=1, help="Exclusive trajectory end index.")
    parser.add_argument("--output-dir", required=True, help="Directory for batches.json and review.md.")
    args = parser.parse_args()
    try:
        config = load_memory_config(args.config_path, args.llm_profile)
        if config is None:
            raise ValueError("A config is required")
        turns = config.get("buffer_turn_threshold", 10)
        tokens = config.get("buffer_token_threshold", 4096)
        if turns <= 0 or tokens <= 0:
            raise ValueError("Buffer thresholds must be positive")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    from memories.datasets.locomo import LoCoMo
    dataset = LoCoMo.read_raw_data(args.dataset_path)
    if not 0 <= args.start_idx < args.end_idx <= len(dataset.trajectories):
        parser.error(f"Require 0 <= start-idx < end-idx <= {len(dataset.trajectories)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "batches.json", output_dir / "review.md"]
    llm = LLMFactory.from_config(LLMConfig(
        model_name=config.get("llm_backend", "openai"),
        api_type=config.get("llm_api_type", "responses"),
        model=config["llm_model"],
        base_url=config.get("llm_base_url"),
        api_key=config.get("llm_api_key"),
        max_tokens=config.get("llm_max_tokens", 1024),
        temperature=config.get("llm_temperature", 0.0),
    ))
    records = []
    counts = {kind: 0 for kind in ("factual", "relational", "state")}
    try:
        save_batches(paths[0], records)
        with paths[1].open("w", encoding="utf-8") as review:
            review.write(f"# 抽取检查：{config['llm_model']}\n\n")
            review.write("仅检查 Extractor 返回的原子；未聚类、入库或构建状态链。\n\n")
            review.write("mention_time 和 source_dialog_ids 来自每条记忆的来源消息；未知 event_time 使用该消息的 mention_time。\n\n")
            for record in extract_batches(dataset.trajectories[args.start_idx:args.end_idx], llm, turns, tokens):
                records.append(record)
                save_batches(paths[0], records)
                write_review(record, review)
                review.flush()
                for kind in counts:
                    counts[kind] += len(record["atoms"][kind])
                print(f"batch {len(records)}: atoms={sum(len(v) for v in record['atoms'].values())}")
    finally:
        llm.client.close()
    print(f"Saved to {output_dir}: batches={len(records)}, atoms={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
