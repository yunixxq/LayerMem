"""Load shared settings and flatten a selected LLM profile."""
import json
from pathlib import Path
from typing import Any, Optional


def load_memory_config(path: Optional[str], llm_profile: Optional[str] = None) -> Optional[dict[str, Any]]:
    if path is None:
        if llm_profile is not None:
            raise ValueError("--llm-profile requires --config-path")
        return None
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("Memory config must be a JSON object")
    profiles = config.pop("llm_profiles", None)
    default = config.pop("default_llm_profile", None)
    if profiles is None:
        if llm_profile is not None:
            raise ValueError("--llm-profile requires llm_profiles in the config")
        return config
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("llm_profiles must be a nonempty object")
    selected = llm_profile if llm_profile is not None else default
    if not isinstance(selected, str) or selected not in profiles:
        raise ValueError(f"Select --llm-profile from: {', '.join(profiles)} (got {selected!r})")
    profile = profiles[selected]
    if not isinstance(profile, dict) or not profile.get("llm_model"):
        raise ValueError(f"LLM profile {selected!r} must be an object with llm_model")
    if any(not key.startswith("llm_") for key in profile):
        raise ValueError(f"LLM profile {selected!r} may contain only llm_* settings")
    return {**config, **profile}
