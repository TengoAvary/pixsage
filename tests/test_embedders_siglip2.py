from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_siglip2_cache() -> bool:
    """Skip unless the model is downloaded (don't pull weights in CI)."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    target = "models--google--siglip2-so400m-patch14-384"
    return os.path.isdir(os.path.join(cache, target))


pytestmark = pytest.mark.skipif(
    not (_has_cuda() and _has_siglip2_cache()),
    reason="SigLIP2 smoke test requires CUDA and a cached model",
)


def test_image_and_text_embeddings_share_space():
    from pixsage.embedders.siglip2 import SigLIP2Embedder

    e = SigLIP2Embedder()
    e.load("cuda")

    cat_img = Image.new("RGB", (224, 224), color=(180, 130, 70))   # placeholder
    img_vecs = e.embed_image([cat_img])
    text_vecs = e.embed_text(["a brown cat sitting"])

    assert img_vecs.shape == (1, e.info.dim)
    assert text_vecs.shape == (1, e.info.dim)
    assert img_vecs.dtype == np.float32
    assert text_vecs.dtype == np.float32

    # L2-normalized
    assert abs(np.linalg.norm(img_vecs[0]) - 1.0) < 1e-3
    assert abs(np.linalg.norm(text_vecs[0]) - 1.0) < 1e-3


def test_text_text_similarity_makes_sense():
    from pixsage.embedders.siglip2 import SigLIP2Embedder

    e = SigLIP2Embedder()
    e.load("cuda")
    a, b, c = e.embed_text(["a leopard seal", "a leopard", "an emperor penguin"])

    # "a leopard seal" should be closer to "a leopard" than to "an emperor penguin"
    assert float(a @ b) > float(a @ c)
