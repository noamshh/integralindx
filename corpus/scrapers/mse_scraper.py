import time
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlencode
import requests
import yaml

from corpus.scrapers.base import BaseScraper
from corpus.formula_models import RawFormula
from src.utils.latex_patterns import DISPLAY_MATH_PATTERNS
from src.utils.paths import get_paths

paths = get_paths()
API_KEYS = paths['config']['api_keys']

class MSEScraper(BaseScraper):
    def __init__(self, config: Optional[dict] = None, session: Optional[requests.Session] = None):
        super().__init__(config=config, session=session)
        self.api_base = config.get("api_base")
        self.site = config.get("site")
        api_settings = config.get("api_settings", {}) if config else {}
        self.tags = api_settings.get("tags")
        self.max_pages = api_settings.get("max_pages")
        self.page_size = api_settings.get("page_size")
        self.sort_order = api_settings.get("sort_order")
        self.pages_per_run = api_settings.get("pages_per_run")
        with open(API_KEYS, "r") as f:
            api_keys = yaml.safe_load(f)
            self.api_key = api_keys["mse"]["api_key"]

    def discover(self, entry_urls: List[str]) -> List[str]:
        api_urls = []
        for tag in self.tags:
            for page in range(1, self.pages_per_run + 1):
                params = {
                    "order": "desc",
                    "sort": self.sort_order,
                    "tagged": tag,
                    "site": self.site,
                    "page": page,
                    "pagesize": self.page_size,
                    "filter": "withbody"
                }
                if self.api_key:
                    params["key"] = self.api_key
                questions_url = f"{self.api_base}/questions?" + urlencode(params)
                api_urls.append(questions_url)
        return api_urls

    def fetch(self, api_url: str, timeout: int = 30, rate_delay: float = 0.1) -> Union[Dict[str, Any], str]:
        try:
            resp = self.session.get(api_url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()   # StackExchange API returns JSON
            if "error_message" in data:
                raise Exception(f"API Error: {data['error_message']}")
            quota_remaining = data.get("quota_remaining", 0)
            if quota_remaining < 100:
                print(f"Warning: API quota running low: {quota_remaining} requests remaining")
            # respect API rate limits
            rate_delay = self.config.get("rate_delay", 0.1)
            if rate_delay > 0:
                time.sleep(rate_delay)
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch from StackExchange API: {e}")

    def extract(self, api_data: Union[Dict[str, Any], str]) -> tuple[List[RawFormula], Dict[str, Any]]:
        """
        Extract latex formulas from StackExchange API response
        Returns:
            tuple of (formulas, answers_data) where answers_data is dict for saving as blob
        """
        if isinstance(api_data, str):
            return [], {}
        if not isinstance(api_data, dict):
            return [], {}
        results: List[RawFormula] = []
        seen = set()
        question_ids = []
        answers_data = {}

        # process questions and answers
        for item in api_data.get("items", []):
            item_type = "question" if "title" in item else "answer"
            item_id = item.get("question_id", item.get("answer_id", 0))
            if item_type == "question":
                question_ids.append(item_id)
            # extract from title (questions only)
            if item_type == "question" and "title" in item:
                title_formulas = self._extract_latex_from_text(
                    item["title"], item_id, f"{item_type}_title", seen
                )
                results.extend(title_formulas)
            # extract from body
            if "body" in item:
                body_formulas = self._extract_latex_from_text(
                    item["body"], item_id, f"{item_type}_body", seen
                )
                results.extend(body_formulas)
        # extract from answers
        if question_ids:
            ids_str = ";".join(map(str, question_ids))
            answers_url = f"{self.api_base}/questions/{ids_str}/answers?site={self.site}&filter=withbody"
            if self.api_key:
                answers_url += f"&key={self.api_key}"
            try:
                answers_resp = self.session.get(answers_url)
                answers_resp.raise_for_status()
                answers_data = answers_resp.json()
                for answer in answers_data.get("items", []):
                    answer_id = answer.get("answer_id", 0)
                    if "body" in answer:
                        answer_formulas = self._extract_latex_from_text(
                            answer["body"], answer_id, "answer_body", seen
                        )
                        results.extend(answer_formulas)
            except Exception as e:
                print(f"Failed to fetch answers: {e}")
        return results, answers_data

    def _extract_latex_from_text(self, text: str, item_id: int, origin: str, seen: set) -> List[RawFormula]:
        results = []
        for i, pattern in enumerate(DISPLAY_MATH_PATTERNS):
            for match in pattern.finditer(text):
                latex_content = match.group(1).strip()
                if latex_content and latex_content not in seen:
                    seen.add(latex_content)
                    provenance = {
                        "extractor": "mse",
                        "item_id": item_id,
                        "origin": origin,
                        "math_type": "display",
                        "pattern_index": i
                    }
                    lf = self.create_formula(
                        raw_latex=latex_content,
                        latex_origin=f"mse_{origin}_display",
                        provenance=provenance
                    )
                    results.append(lf)
        return results

    def create_formula(self, raw_latex: str, latex_origin: str, provenance: dict) -> RawFormula:
        item_id = provenance.get("item_id", 0)
        origin = provenance.get("origin", "")
        if "question" in origin:
            source_url = f"https://math.stackexchange.com/q/{item_id}"
        elif "answer" in origin:
            source_url = f"https://math.stackexchange.com/a/{item_id}"
        else:
            source_url = f"https://math.stackexchange.com/q/{item_id}"
        lf = RawFormula(
            id=RawFormula.make_id(self.source_name, raw_latex),
            source=self.source_name,
            source_url=source_url,
            raw_latex=raw_latex,
            latex_origin=latex_origin,
            provenance=provenance
        )
        lf.compute_checksum(field_name="raw_latex")
        return lf
