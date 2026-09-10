"""
OCR extraction orchestrator.

Iterates rendered pages of a PDF, hands each one to a backend, canonicalizes
the resulting `integrand_sympy` strings to the same hash scheme as
`IntegralFormula` (so OCR records can later be joined with the existing
IntegrandGroup index), and appends to a JSONL file.

Resumable: pages whose `page_image_sha256` already appears in the output JSONL
are skipped, so an SSH disconnect / OOM kill mid-run can be picked up where it
left off just by re-invoking the same command.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from corpus.ocr.backends import get_backend
from corpus.ocr.models import OcrIntegralRecord, OcrSourceMetadata
from corpus.ocr.pdf_renderer import render_pdf_pages
from src.utils.integrand_canonicalization import canonicalize_integrand

logger = logging.getLogger(__name__)


def _load_completed_page_hashes(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                src = obj.get("source") or {}
                h = src.get("page_image_sha256")
                if h:
                    done.add(h)
            except json.JSONDecodeError:
                continue
    return done


def _serialize_record(rec: OcrIntegralRecord) -> str:
    d = asdict(rec)  # OcrSourceMetadata is a dataclass, asdict handles nesting
    return json.dumps(d, ensure_ascii=False)


def _append(out_path: Path, rec: OcrIntegralRecord) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(_serialize_record(rec) + "\n")


def run_extraction(
    pdf_path: Path,
    source_meta_base: Dict[str, Any],
    out_path: Path,
    renders_dir: Path,
    page_range: Optional[Tuple[int, int]] = None,
    dpi: int = 200,
    backend_name: str = "qwen3_vl",
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> int:
    """Run extraction. Returns the total number of records written this run.

    source_meta_base: dict of OcrSourceMetadata fields shared across all pages
        of this PDF (source_type, source_id, source_title, source_url, license,
        author_*, extra, ...). page_number/page_image_path/page_image_sha256
        are filled in per-page by the orchestrator.
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    renders_dir = Path(renders_dir)

    backend = get_backend(backend_name, **(backend_kwargs or {}))
    logger.info(f"using backend {backend_name} (model={backend.model_id}, prompt={backend.prompt_version})")

    completed = _load_completed_page_hashes(out_path)
    if completed:
        logger.info(f"resuming — {len(completed)} pages already in {out_path}")

    written = 0
    for page_num, png_path, png_sha in render_pdf_pages(
        pdf_path, renders_dir, dpi=dpi, page_range=page_range
    ):
        if png_sha in completed:
            logger.info(f"skip page {page_num} (already in output)")
            continue

        source = OcrSourceMetadata(
            **source_meta_base,
            page_number=page_num,
            page_image_path=str(png_path),
            page_image_sha256=png_sha,
        )

        with Image.open(png_path) as img:
            img.load()
            try:
                records = backend.extract_from_page(img, source)
            except Exception as e:
                logger.exception(f"backend failed on page {page_num}: {e}")
                continue

        for rec in records:
            if rec.integrand_sympy:
                try:
                    canonical, hsh = canonicalize_integrand(rec.integrand_sympy)
                    rec.integrand_canonical = canonical
                    rec.integrand_hash = hsh
                except Exception as e:
                    logger.warning(f"canonicalize failed on {rec.id}: {e}")
            _append(out_path, rec)
            written += 1

        logger.info(f"page {page_num}: {len(records)} records (total written: {written})")

    return written
