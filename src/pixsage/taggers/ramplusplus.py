from __future__ import annotations

from PIL import Image

from pixsage.taggers.base import Tag, TagResult

MODEL_VERSION = "ram_plus_swin_large_14m"


class RamPlusPlusTagger:
    name = "ram++"
    model_version = MODEL_VERSION

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import os

        import torch  # noqa: F401  (ensures torch is available)
        from ram import get_transform
        from ram.models import ram_plus

        self._device = device
        # Image size 384 is the standard RAM++ training resolution.
        self._transform = get_transform(image_size=384)
        # Load the public checkpoint. Users may need to download the .pth and pass via env.
        ckpt = os.environ.get("PIXSAGE_RAM_CKPT", "ram_plus_swin_large_14m.pth")
        model = ram_plus(pretrained=ckpt, image_size=384, vit="swin_l")
        model.eval()
        self._model = model.to(device)

    def tag(self, image: Image.Image) -> TagResult:
        return self.tag_batch([image])[0]

    def tag_batch(self, images: list[Image.Image]) -> list[TagResult]:
        import torch

        if not images:
            return []
        batch = torch.stack([self._transform(img) for img in images]).to(self._device)
        with torch.no_grad():
            tag_strings, _ = self._model.generate_tag(batch)
        out: list[TagResult] = []
        for tags_string in tag_strings:
            labels = [s.strip() for s in tags_string.split("|") if s.strip()]
            tags = [Tag(name=lbl, confidence=1.0, hierarchy=None, source="ram++") for lbl in labels]
            out.append(TagResult(tags=tags, caption=None))
        return out
