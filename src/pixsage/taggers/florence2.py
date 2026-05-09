from __future__ import annotations

from PIL import Image

from pixsage.taggers.base import Tag, TagResult

MODEL_ID = "microsoft/Florence-2-large"
MODEL_VERSION = MODEL_ID  # use HF model id; encodes the version


class Florence2Tagger:
    name = "florence2"
    model_version = MODEL_VERSION

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self._device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        self._model.eval()

    def tag(self, image: Image.Image) -> TagResult:
        caption = self._run_prompt(image, "<MORE_DETAILED_CAPTION>")
        regions = self._run_prompt(image, "<DENSE_REGION_CAPTION>")
        # Florence-2's post_process returns a dict like {"<DENSE_REGION_CAPTION>": {"bboxes": [...], "labels": [...]}}
        labels = self._extract_labels(regions)
        tags = [Tag(name=lbl.strip(), confidence=1.0, hierarchy=None, source="florence2") for lbl in labels if lbl.strip()]
        # De-dupe by lower-cased name, preserve order:
        seen = set()
        unique_tags: list[Tag] = []
        for t in tags:
            key = t.name.lower()
            if key not in seen:
                seen.add(key)
                unique_tags.append(t)
        caption_text = caption.get("<MORE_DETAILED_CAPTION>") if isinstance(caption, dict) else (str(caption) if caption else None)
        return TagResult(tags=unique_tags, caption=caption_text)

    def _run_prompt(self, image: Image.Image, task: str):
        import torch
        inputs = self._processor(text=task, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(text, task=task, image_size=(image.width, image.height))

    def _extract_labels(self, regions) -> list[str]:
        if isinstance(regions, dict):
            payload = regions.get("<DENSE_REGION_CAPTION>", regions)
            if isinstance(payload, dict):
                return list(payload.get("labels", []))
        return []
