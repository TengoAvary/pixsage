from __future__ import annotations

from pathlib import Path

from PIL import Image

from pixsage.taggers.base import Tag, TagResult

MODEL_VERSION = "ram_plus_swin_large_14m"
DEFAULT_CKPT_PATH = Path.home() / ".cache" / "pixsage" / f"{MODEL_VERSION}.pth"


def resolve_ram_ckpt() -> str:
    """Return the RAM++ checkpoint path: env override wins, else default cache."""
    import os
    return os.environ.get("PIXSAGE_RAM_CKPT") or str(DEFAULT_CKPT_PATH)


class RamPlusPlusTagger:
    name = "ram++"
    model_version = MODEL_VERSION

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import torch  # noqa: F401  (ensures torch is available)
        from ram import get_transform
        from ram.models import ram_plus

        self._device = device
        # Image size 384 is the standard RAM++ training resolution.
        self._transform = get_transform(image_size=384)
        ckpt = resolve_ram_ckpt()
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
