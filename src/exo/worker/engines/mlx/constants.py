# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5

# KV cache backend selection: "default", "mlx_quantized", "turboquant", "turboquant_adaptive"
KV_CACHE_BACKEND: str = os.environ.get("EXO_KV_CACHE_BACKEND", "default")

# TurboQuant settings (only used when KV_CACHE_BACKEND starts with "turboquant")
TURBOQUANT_K_BITS: int = int(os.environ.get("EXO_TQ_K_BITS", "3"))
TURBOQUANT_V_BITS: int = int(os.environ.get("EXO_TQ_V_BITS", "4"))
TURBOQUANT_FP16_LAYERS: int = int(os.environ.get("EXO_TQ_FP16_LAYERS", "2"))

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
