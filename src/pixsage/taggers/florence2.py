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
        # Florence-2's HF modeling file does an unconditional `import flash_attn`
        # at module load. flash_attn has no Windows wheels and requires CUDA dev
        # tools to build. We register a stub before from_pretrained so the
        # import succeeds; we then pass `attn_implementation="eager"` so the
        # code path that would actually call flash_attn is never taken.
        import importlib.machinery
        import sys
        import types

        def _stub_module(name: str) -> types.ModuleType:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            return m

        if "flash_attn" not in sys.modules:
            stub = _stub_module("flash_attn")
            stub.__version__ = "0.0.0-stub"
            sys.modules["flash_attn"] = stub
            sys.modules["flash_attn.bert_padding"] = _stub_module("flash_attn.bert_padding")
            sys.modules["flash_attn.flash_attn_interface"] = _stub_module(
                "flash_attn.flash_attn_interface"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self._device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self._model.eval()

    def tag(self, image: Image.Image) -> TagResult:
        caption = self._run_prompt(image, "<MORE_DETAILED_CAPTION>")
        # Pull tags from both DENSE_REGION_CAPTION (region phrases like "boy in
        # red jacket") and OD (single-word categories like "willow", "person").
        # Different images favor one or the other; landscapes especially tend
        # to return nothing from DENSE_REGION_CAPTION.
        labels: list[str] = []
        for task in ("<DENSE_REGION_CAPTION>", "<OD>"):
            labels.extend(self._extract_labels(self._run_prompt(image, task), task))
        # De-dupe by lower-cased name, preserve first occurrence.
        seen: set[str] = set()
        unique_tags: list[Tag] = []
        for lbl in labels:
            name = lbl.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_tags.append(Tag(name=name, confidence=1.0, hierarchy=None, source="florence2"))
        caption_text = caption.get("<MORE_DETAILED_CAPTION>") if isinstance(caption, dict) else (str(caption) if caption else None)
        return TagResult(tags=unique_tags, caption=caption_text)

    def _run_prompt(self, image: Image.Image, task: str):
        import torch
        inputs = self._processor(text=task, images=image, return_tensors="pt").to(self._device)
        # Match pixel_values dtype to the model's dtype (fp16 on CUDA, fp32 on CPU/MPS).
        model_dtype = next(self._model.parameters()).dtype
        pixel_values = inputs["pixel_values"].to(model_dtype)
        with torch.no_grad():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(text, task=task, image_size=(image.width, image.height))

    def _extract_labels(self, result, task: str) -> list[str]:
        if isinstance(result, dict):
            payload = result.get(task, result)
            if isinstance(payload, dict):
                return [str(lbl) for lbl in payload.get("labels", [])]
        return []
