import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_meanflow.sit import VisionRotaryEmbeddingFast


def test_pixel_rotary_embedding_uses_defined_pi():
    embedding = VisionRotaryEmbeddingFast(dim=8, freqs_for="pixel", pt_seq_len=2)

    assert embedding.freqs_cos.shape[0] == 4
    assert embedding.freqs_cos.shape[1] > 0
    assert embedding.freqs_sin.shape == embedding.freqs_cos.shape
    assert math.isfinite(float(embedding.freqs_cos[0, 0]))
