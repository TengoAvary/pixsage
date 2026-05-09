from __future__ import annotations

import numpy as np
from PIL import Image

from pixsage.embedders.mock import MockEmbedder


def test_image_embedding_shape_and_dtype():
    e = MockEmbedder(dim=8)
    img = Image.new("RGB", (32, 32), color="red")
    vecs = e.embed_image([img])
    assert vecs.shape == (1, 8)
    assert vecs.dtype == np.float32


def test_text_embedding_shape_and_dtype():
    e = MockEmbedder(dim=8)
    vecs = e.embed_text(["a leopard seal"])
    assert vecs.shape == (1, 8)
    assert vecs.dtype == np.float32


def test_l2_normalized_output():
    e = MockEmbedder(dim=16)
    vecs = e.embed_image([Image.new("RGB", (32, 32))])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5


def test_deterministic_for_same_input():
    e = MockEmbedder(dim=8)
    a = e.embed_text(["leopard seal"])
    b = e.embed_text(["leopard seal"])
    np.testing.assert_array_equal(a, b)


def test_different_text_different_vector():
    e = MockEmbedder(dim=8)
    a = e.embed_text(["leopard seal"])
    b = e.embed_text(["emperor penguin"])
    assert not np.array_equal(a, b)


def test_batched_embedding_matches_single_calls():
    e = MockEmbedder(dim=8)
    imgs = [Image.new("RGB", (32, 32), c) for c in ("red", "green", "blue")]
    batched = e.embed_image(imgs)
    one_at_a_time = np.vstack([e.embed_image([img]) for img in imgs])
    np.testing.assert_array_equal(batched, one_at_a_time)
