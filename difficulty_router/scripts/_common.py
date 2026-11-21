import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_model_args(p):
    p.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id (default: Qwen/Qwen3-4B)")
    p.add_argument("--dtype", default="bfloat16", help="bfloat16 (Ampere+) or float16 (e.g. T4)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4, help="AnyJev HFBackend batch size; keep small for long prompts")
    p.add_argument("--max-prompt-tokens", type=int, default=12000)
    p.add_argument("--prior", default="batch", help="L0 prior estimator: batch (default) | none | content_free")
