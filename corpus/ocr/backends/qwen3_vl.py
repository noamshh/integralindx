"""
Qwen3-VL backend that talks to a vLLM OpenAI-compatible server.

Start the server on the remote GPU box like:

    vllm serve Qwen/Qwen3-VL-32B-Instruct \\
        --port 8000 \\
        --max-model-len 32768 \\
        --gpu-memory-utilization 0.92 \\
        --limit-mm-per-prompt image=4

Then point this backend at it via base_url (default http://localhost:8000/v1).
Guided JSON decoding is requested via `response_format` with the schema from
corpus.ocr.prompts. The model is forced to emit conformant JSON, so the only
failure modes are (a) the server isn't running, (b) the model couldn't read
the page at all, in which case it emits {"integrals": []}.
"""

import base64
import dataclasses
import io
import json
import logging
from typing import List, Optional

from PIL import Image

from corpus.ocr.backends.base import OcrBackend
from corpus.ocr.models import OcrIntegralRecord, OcrSourceMetadata
from corpus.ocr.prompts import (
    INTEGRAL_PAGE_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class Qwen3VLBackend(OcrBackend):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-32B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        source_hint: Optional[str] = None,
    ):
        from openai import OpenAI  # lazy import; openai is in requirements.txt

        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._source_hint = source_hint

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def extract_from_page(
        self, image: Image.Image, source: OcrSourceMetadata
    ) -> List[OcrIntegralRecord]:
        image_url = _image_to_data_url(image)
        user_text = build_user_prompt(self._source_hint)

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "integral_page_extraction",
                    "schema": INTEGRAL_PAGE_SCHEMA,
                    "strict": True,
                },
            },
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = resp.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"backend returned non-JSON despite guided decoding: {e}; raw={raw[:300]}")
            return []

        integrals = payload.get("integrals", []) or []
        records: List[OcrIntegralRecord] = []
        for idx, item in enumerate(integrals):
            # Each record gets its own source copy so per-integral `label`
            # (e.g. G&R "2.124.1") doesn't leak across records sharing a page.
            rec_source = dataclasses.replace(source, label=item.get("label") or source.label)
            rec = OcrIntegralRecord(
                id=OcrIntegralRecord.make_id(source.source_id, source.page_number, idx),
                integrand_latex=item.get("integrand_latex", ""),
                integrand_sympy=item.get("integrand_sympy", ""),
                variable=item.get("variable", "x"),
                lower_limit_latex=item.get("lower_limit_latex"),
                upper_limit_latex=item.get("upper_limit_latex"),
                lower_limit_sympy=item.get("lower_limit_sympy"),
                upper_limit_sympy=item.get("upper_limit_sympy"),
                integral_type=item.get("integral_type", "indefinite"),
                solution_latex=item.get("solution_latex"),
                solution_sympy=item.get("solution_sympy"),
                has_solution=bool(item.get("has_solution", False)),
                constants=item.get("constants", []) or [],
                conditions=item.get("conditions"),
                source=rec_source,
                ocr_model=self._model,
                ocr_prompt_version=PROMPT_VERSION,
                raw_model_output=raw,
            )
            records.append(rec)
        return records
