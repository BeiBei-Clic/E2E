"""ELF continuous flow-matching denoiser, ported from ELF-pytorch.

See ``docs/elf_sr_implementation_plan.md`` Step 1. The denoiser operates in the
frozen expression-encoder embedding space (``text_encoder_dim``); ``vocab_size``
is bound to ``len(equation_words)`` at construction in later steps.
"""

from .model import ELF, ELF_models
from .layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
)

__all__ = [
    "ELF", "ELF_models",
    "Attention", "BottleneckTextProj", "FinalLayer", "RMSNorm", "SwiGLUFFN",
    "TextRotaryEmbeddingFast", "TimestepEmbedder",
]
