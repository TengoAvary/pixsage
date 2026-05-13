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
        return self.tag_batch([image])[0]

    def tag_batch(self, images: list[Image.Image]) -> list[TagResult]:
        if not images:
            return []
        caption_results = self._run_prompt_batch(images, "<MORE_DETAILED_CAPTION>")
        dense_results = self._run_prompt_batch(images, "<DENSE_REGION_CAPTION>")
        od_results = self._run_prompt_batch(images, "<OD>")

        out: list[TagResult] = []
        for img, cap, dense, od in zip(images, caption_results, dense_results, od_results):
            labels: list[str] = []
            labels.extend(self._extract_labels(dense, "<DENSE_REGION_CAPTION>"))
            labels.extend(self._extract_labels(od, "<OD>"))
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
            caption_text = (
                cap.get("<MORE_DETAILED_CAPTION>") if isinstance(cap, dict)
                else (str(cap) if cap else None)
            )
            out.append(TagResult(tags=unique_tags, caption=caption_text))
        return out

    def _run_prompt_batch(self, images: list[Image.Image], task: str) -> list:
        import torch
        n = len(images)
        inputs = self._processor(
            text=[task] * n, images=list(images), return_tensors="pt", padding=True
        ).to(self._device)
        # Match pixel_values dtype to the model's dtype (fp16 on CUDA, fp32 on CPU/MPS).
        model_dtype = next(self._model.parameters()).dtype
        pixel_values = inputs["pixel_values"].to(model_dtype)
        with torch.no_grad():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
            )
        texts = self._processor.batch_decode(generated, skip_special_tokens=False)
        return [
            self._processor.post_process_generation(t, task=task, image_size=(img.width, img.height))
            for t, img in zip(texts, images)
        ]

    def _extract_labels(self, result, task: str) -> list[str]:
        if isinstance(result, dict):
            payload = result.get(task, result)
            if isinstance(payload, dict):
                return [str(lbl) for lbl in payload.get("labels", [])]
        return []
