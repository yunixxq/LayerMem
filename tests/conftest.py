"""Keep the test session off the network.

The embedding model is expected to be in the local HuggingFace cache; without
these, sentence-transformers probes the Hub for optional side files and the
resulting connection error looks like a code failure.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
