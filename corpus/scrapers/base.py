from typing import List, Optional
from abc import ABC, abstractmethod
import time
import requests

from corpus.formula_models import RawFormula


class BaseScraper(ABC):
    def __init__(self, config: Optional[dict] = None, session: Optional[requests.Session] = None):
        self.config = config or {}
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.config.get("user_agent") })
        self.source_name = self.config.get("id", "unknown")

    def discover(self, entry_urls: List[str]) -> List[str]:
        """Returns list of relevant URLs to fetch, given the entry URLs.  By default, return entry URLs as-is"""
        return list(entry_urls or [])

    def fetch(self, page_url: str, timeout: int = 20, rate_delay: float = 0.0) -> str:
        "Fetch the raw content from the given URL. By default, returns html content."
        resp = self.session.get(page_url, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        if rate_delay and rate_delay > 0:
            time.sleep(rate_delay)
        return html

    @abstractmethod
    def extract(self, raw_html: str) -> List[RawFormula]:
        """Extract latex content from the given raw content."""
        pass

    def create_formula(self, raw_latex: str, latex_origin: str, provenance: dict) -> RawFormula:
        rf = RawFormula(
            id=RawFormula.make_id(self.source_name, raw_latex),
            source=self.source_name,
            source_url="",
            raw_latex=raw_latex,
            latex_origin=latex_origin,
            provenance=provenance
        )
        rf.compute_checksum(field_name="raw_latex")
        return rf
