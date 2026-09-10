"""
Thin PyMuPDF wrapper that renders PDF pages to PNG and yields (page_number,
png_path, png_sha256). Resumable: if a target PNG already exists with the same
content hash, it is not re-rendered.

Page numbering is 1-based to match how the user / the PDF viewer numbers pages.
"""

import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_pdf_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    page_range: Optional[Tuple[int, int]] = None,
) -> Iterator[Tuple[int, Path, str]]:
    """Render selected pages of `pdf_path` to PNGs in `out_dir`.

    page_range is an inclusive 1-based (start, end) tuple. If None, render all
    pages. Yields (page_number, png_path, png_sha256) per rendered page.
    """
    import pymupdf  # imported lazily — base requirements.txt installs it

    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    doc = pymupdf.open(pdf_path)
    try:
        total = doc.page_count
        start_1b = 1 if page_range is None else max(1, page_range[0])
        end_1b = total if page_range is None else min(total, page_range[1])
        for page_num in range(start_1b, end_1b + 1):
            out_path = out_dir / f"page_{page_num:04d}.png"
            if out_path.exists():
                yield page_num, out_path, _sha256_file(out_path)
                continue
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))
            sha = _sha256_file(out_path)
            logger.info(f"rendered {pdf_path.name} page {page_num} -> {out_path.name} ({pix.width}x{pix.height})")
            yield page_num, out_path, sha
    finally:
        doc.close()
