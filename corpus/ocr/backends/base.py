"""Abstract base class for OCR backends. A backend takes a rendered page image
plus its source metadata and returns a list of OcrIntegralRecord objects."""

from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from corpus.ocr.models import OcrIntegralRecord, OcrSourceMetadata


class OcrBackend(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier for the underlying model (e.g. 'Qwen/Qwen3-VL-32B-Instruct').
        Stored on every record's `ocr_model` field for provenance."""

    @property
    @abstractmethod
    def prompt_version(self) -> str:
        """Prompt version string stored on every record's `ocr_prompt_version`."""

    @abstractmethod
    def extract_from_page(
        self, image: Image.Image, source: OcrSourceMetadata
    ) -> List[OcrIntegralRecord]:
        """Run extraction on a single page image. Records returned have their
        `source`, `ocr_model`, `ocr_prompt_version`, and `raw_model_output` fields
        populated; the orchestrator fills in the canonicalization fields."""
