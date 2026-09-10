"""
End-to-end test for the OCR pipeline using a mock backend. Runs entirely on
CPU — no GPU, no vLLM server. Validates that:

  * the orchestrator wires up the backend, PDF renderer, and JSONL writer
  * `integrand_canonical` / `integrand_hash` get populated via the same
    canonicalize_integrand() the existing pipeline uses
  * resumption works (rerunning skips pages already in output)
  * per-record `source.label` is correctly assigned (no shared-mutation bug)
  * the hash matches what an IntegralFormula would compute for the same
    integrand_sympy string (forward compatibility with IntegrandGroup dedup)
"""

import json
from pathlib import Path
from typing import List

import pytest
from PIL import Image

from corpus.ocr.backends import OcrBackend, register_backend
from corpus.ocr.backends.registry import _BUILTIN
from corpus.ocr.extraction_pipeline import run_extraction
from corpus.ocr.models import OcrIntegralRecord, OcrSourceMetadata
from src.utils.integrand_canonicalization import canonicalize_integrand


class _MockBackend(OcrBackend):
    """Returns two hand-crafted integrals for any input page. The model output
    is deterministic so we can assert exact fields."""

    @property
    def model_id(self) -> str:
        return "mock/test-backend-v0"

    @property
    def prompt_version(self) -> str:
        return "test-v1"

    def extract_from_page(
        self, image: Image.Image, source: OcrSourceMetadata
    ) -> List[OcrIntegralRecord]:
        import dataclasses

        raw_json = '{"integrals": [/* mock */]}'
        items = [
            {
                "label": "test.1",
                "integrand_latex": "x^{2}",
                "integrand_sympy": "x**2",
                "variable": "x",
                "integral_type": "indefinite",
                "has_solution": True,
                "solution_latex": "\\frac{x^{3}}{3}",
                "solution_sympy": "x**3/3",
            },
            {
                "label": "test.2",
                "integrand_latex": "e^{-x^{2}}",
                "integrand_sympy": "exp(-x**2)",
                "variable": "x",
                "integral_type": "definite",
                "lower_limit_latex": "0",
                "upper_limit_latex": "\\infty",
                "lower_limit_sympy": "0",
                "upper_limit_sympy": "oo",
                "has_solution": True,
                "solution_latex": "\\frac{\\sqrt{\\pi}}{2}",
                "solution_sympy": "sqrt(pi)/2",
            },
        ]
        records = []
        for idx, item in enumerate(items):
            rec_source = dataclasses.replace(source, label=item["label"])
            records.append(
                OcrIntegralRecord(
                    id=OcrIntegralRecord.make_id(source.source_id, source.page_number, idx),
                    integrand_latex=item["integrand_latex"],
                    integrand_sympy=item["integrand_sympy"],
                    variable=item["variable"],
                    lower_limit_latex=item.get("lower_limit_latex"),
                    upper_limit_latex=item.get("upper_limit_latex"),
                    lower_limit_sympy=item.get("lower_limit_sympy"),
                    upper_limit_sympy=item.get("upper_limit_sympy"),
                    integral_type=item["integral_type"],
                    solution_latex=item.get("solution_latex"),
                    solution_sympy=item.get("solution_sympy"),
                    has_solution=item["has_solution"],
                    source=rec_source,
                    ocr_model=self.model_id,
                    ocr_prompt_version=self.prompt_version,
                    raw_model_output=raw_json,
                )
            )
        return records


@pytest.fixture
def mock_backend_registered():
    register_backend("mock", lambda **_: _MockBackend())
    yield
    _BUILTIN.pop("mock", None)


@pytest.fixture
def gr_pdf() -> Path:
    p = Path("data/raw/books/gradshteyn_ryzhik.pdf")
    if not p.exists():
        pytest.skip(f"G&R PDF not present at {p}; copy it first")
    return p


def test_pipeline_end_to_end(tmp_path: Path, mock_backend_registered, gr_pdf: Path):
    out_path = tmp_path / "ocr_out.jsonl"
    renders_dir = tmp_path / "renders"

    written = run_extraction(
        pdf_path=gr_pdf,
        source_meta_base=dict(
            source_type="book",
            source_id="gradshteyn_ryzhik",
            source_title="G&R Test",
            source_url=None,
            license=None,
            author_name=None,
            author_profile_url=None,
            author_id=None,
            extra={"edition": "test"},
        ),
        out_path=out_path,
        renders_dir=renders_dir,
        page_range=(100, 101),
        dpi=120,
        backend_name="mock",
    )

    assert written == 4  # 2 pages × 2 mock integrals
    assert out_path.exists()

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4

    # Canonicalization populated correctly
    for r in records:
        assert r["integrand_canonical"] is not None
        assert r["integrand_hash"] is not None
        assert len(r["integrand_hash"]) == 12

    # Forward-compat with IntegrandGroup: the hash on the record matches what
    # canonicalize_integrand returns for the same sympy string directly.
    r0 = records[0]
    expected_canonical, expected_hash = canonicalize_integrand(r0["integrand_sympy"])
    assert r0["integrand_hash"] == expected_hash
    assert r0["integrand_canonical"] == expected_canonical

    # Per-record source.label was assigned without cross-record leakage.
    labels = [r["source"]["label"] for r in records]
    assert labels == ["test.1", "test.2", "test.1", "test.2"]

    # Source metadata extras + page hashing survived JSON round-trip.
    for r in records:
        assert r["source"]["source_type"] == "book"
        assert r["source"]["source_id"] == "gradshteyn_ryzhik"
        assert r["source"]["page_image_sha256"]
        assert r["source"]["extra"] == {"edition": "test"}
        assert r["ocr_model"] == "mock/test-backend-v0"
        assert r["ocr_prompt_version"] == "test-v1"


def test_pipeline_resumes(tmp_path: Path, mock_backend_registered, gr_pdf: Path):
    out_path = tmp_path / "ocr_out.jsonl"
    renders_dir = tmp_path / "renders"
    src_base = dict(
        source_type="book",
        source_id="gradshteyn_ryzhik",
        source_title=None, source_url=None, license=None,
        author_name=None, author_profile_url=None, author_id=None,
        extra={},
    )

    first = run_extraction(
        pdf_path=gr_pdf, source_meta_base=src_base, out_path=out_path,
        renders_dir=renders_dir, page_range=(100, 100), dpi=120, backend_name="mock",
    )
    assert first == 2  # one page × two mock integrals

    second = run_extraction(
        pdf_path=gr_pdf, source_meta_base=src_base, out_path=out_path,
        renders_dir=renders_dir, page_range=(100, 100), dpi=120, backend_name="mock",
    )
    assert second == 0  # already done — nothing new appended

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_round_trip_record_from_dict(tmp_path: Path, mock_backend_registered, gr_pdf: Path):
    """OcrIntegralRecord.from_dict() correctly rebuilds the dataclass from JSONL."""
    out_path = tmp_path / "ocr_out.jsonl"
    renders_dir = tmp_path / "renders"

    run_extraction(
        pdf_path=gr_pdf,
        source_meta_base=dict(
            source_type="book", source_id="gradshteyn_ryzhik",
            source_title=None, source_url=None, license=None,
            author_name=None, author_profile_url=None, author_id=None, extra={},
        ),
        out_path=out_path, renders_dir=renders_dir,
        page_range=(100, 100), dpi=120, backend_name="mock",
    )

    line = out_path.read_text(encoding="utf-8").splitlines()[0]
    rec = OcrIntegralRecord.from_dict(json.loads(line))
    assert isinstance(rec.source, OcrSourceMetadata)
    assert rec.integrand_sympy == "x**2"
    assert rec.source.label == "test.1"
