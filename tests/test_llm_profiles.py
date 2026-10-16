"""Profile selection and evaluation transport, without external services."""
import json
import sys
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

TOOLKITS = Path(__file__).resolve().parents[1] / "memory_toolkits"
sys.path.insert(0, str(TOOLKITS))
from config_loader import load_memory_config
from inference_utils.backends import OpenAIClient, OpenAIClientPool


def test_repository_profiles_share_settings():
    path = str(TOOLKITS / "configs/LayerMem.json")
    gpt = load_memory_config(path, "gpt-4o-mini")
    assert load_memory_config(path) == gpt
    assert gpt["llm_api_type"] == "responses"
    assert "llm_profiles" not in gpt


def test_new_profile_requires_only_config_changes(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"buffer_turn_threshold": 7, "llm_model": "old",
        "llm_profiles": {"new-model": {"llm_model": "new", "llm_api_type": "responses"}}}))
    result = load_memory_config(str(path), "new-model")
    assert result == {"buffer_turn_threshold": 7, "llm_model": "new", "llm_api_type": "responses"}
    with pytest.raises(ValueError, match="Select --llm-profile"):
        load_memory_config(str(path))
    with pytest.raises(ValueError, match="Select --llm-profile"):
        load_memory_config(str(path), "typo")


def test_legacy_config_and_invalid_selection(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"llm_model": "legacy"}')
    assert load_memory_config(str(path)) == {"llm_model": "legacy"}
    assert load_memory_config(None) is None
    with pytest.raises(ValueError):
        load_memory_config(None, "gpt-4o-mini")
    with pytest.raises(ValueError):
        load_memory_config(str(path), "gpt-4o-mini")


def test_evaluation_pool_routes_responses_and_postprocesses():
    pool = OpenAIClientPool(["test"], ["https://example.invalid/v1"],
                            model="gpt-4o-mini", api_type="responses")
    client = pool.client_pool[0]
    with patch.object(client.responses, "create", return_value=SimpleNamespace(
        status="completed", error=None, output_text="yes",
    )) as create, patch.object(client.chat.completions, "create") as chat:
        output = pool([[{"role": "user", "content": "Judge this answer"}]],
                      post_processor=str.upper, temperature=0.0)
    assert output["processed_content"] == "YES"
    assert create.call_args.kwargs == {"model": "gpt-4o-mini", "input": "USER:\nJudge this answer"}
    chat.assert_not_called()
    client.close()


def test_evaluation_chat_completions_transport():
    client = OpenAIClient(api_key="test")
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])
    with patch.object(client.chat.completions, "create", return_value=completion) as create:
        assert client.get_text_generation_output([], model="test-model", stream=False)["content"] == "yes"
    assert create.call_args.kwargs["model"] == "test-model"
    client.close()


def test_evaluation_does_not_score_failed_responses():
    client = OpenAIClient(api_key="test", api_type="responses")
    with patch.object(client.responses, "create", return_value=SimpleNamespace(status="incomplete")):
        with pytest.raises(ValueError, match="did not complete"):
            client.get_text_generation_output([])
    client.close()


@pytest.mark.parametrize("arguments,profile", [([], "gpt-4o-mini"), (["0", "2", "gpt-4o-mini"], "gpt-4o-mini")])
def test_shell_passes_profile_to_all_stages(tmp_path, monkeypatch, arguments, profile):
    root = TOOLKITS.parent
    script = tmp_path / "run_layermem_locomo.sh"
    shutil.copy(root / script.name, script)
    toolkit = tmp_path / "memory_toolkits"
    (toolkit / "configs").mkdir(parents=True)
    shutil.copy(TOOLKITS / "config_loader.py", toolkit)
    shutil.copy(TOOLKITS / "configs/LayerMem.json", toolkit / "configs")
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.argv = sys.argv[1:]
if sys.argv[0] == '-':
    exec(sys.stdin.read())
else:
    with open('calls.jsonl', 'a') as target:
        target.write(json.dumps(sys.argv) + '\\n')
    if sys.argv[0] == 'memory_search.py':
        output = Path(sys.argv[sys.argv.index('--output-path') + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('[]')
''')
    executable.chmod(0o755)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    result = subprocess.run(["/bin/bash", str(script), *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (toolkit / "calls.jsonl").read_text().splitlines()]
    assert [call[0] for call in calls] == ["memory_construction.py", "memory_search.py", "memory_evaluation.py"]
    for call in calls:
        assert call[call.index("--llm-profile") + 1] == profile
    search_output = calls[1][calls[1].index("--output-path") + 1]
    assert calls[2][calls[2].index("--search-results-path") + 1] == search_output
    assert profile in search_output
