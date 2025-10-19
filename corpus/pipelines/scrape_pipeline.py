import json, hashlib, base64
from typing import Tuple, Optional
from pathlib import Path
import logging

from corpus.formula_models import RawFormula
from corpus.pipeline_utils import LOGGING_FORMAT, append_jsonl, load_yaml_configs, load_existing_checksums
from corpus.scrapers.mse_scraper import MSEScraper
from src.utils.provenance import now_iso, sha256_bytes
from src.utils.paths import get_paths

paths = get_paths()
RAW_PATH = paths['data']['raw']['root']
BLOBS_PATH = paths['data']['raw']['blobs']
FORMULA_PATH = paths['data']['raw']['formulas']
RAW_PATH.mkdir(parents=True, exist_ok=True)
BLOBS_PATH.mkdir(parents=True, exist_ok=True)
FORMULA_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

SCRAPER_MAP = {
    'mse': MSEScraper,
}


def _load_scraped_urls(source_id: str) -> set:
    seen_file = RAW_PATH / f"{source_id}_urls.txt"
    if not seen_file.exists():
        return set()
    with seen_file.open('r', encoding='utf-8') as f:
        return {line.strip() for line in f}

def _save_scraped_url(source_id: str, url: str):
    seen_file = RAW_PATH / f"{source_id}_urls.txt"
    with seen_file.open('a', encoding='utf-8') as f:
        f.write(url + "\n")

def _is_text_bytes(b: bytes) -> bool:
    if b'\x00' in b:
        return False
    try:
        b.decode('utf-8')
        return True
    except Exception:
        return False

def _save_blob_as_text(base_dir: Path, base_name: str, content: bytes, url: str = None,
                      content_type: Optional[str] = None, overwrite: bool = True) -> Tuple[str, str]:
    """Saves content as: utf-8 if text,  base64 if binary"""
    base_dir.mkdir(parents=True, exist_ok=True)
    checksum = sha256_bytes(content)
    if _is_text_bytes(content):
        text = content.decode('utf-8', errors='replace')
        fname = f"{base_name}.txt"
        path = base_dir / fname
        if not path.exists() or overwrite:
            path.write_text(text, encoding='utf-8')
        encoding = "utf-8"
    else:
        b64 = base64.b64encode(content).decode('ascii')
        fname = f"{base_name}.b64.txt"
        path = base_dir / fname
        if not path.exists() or overwrite:
            path.write_text(b64, encoding='ascii')
        encoding = "base64"
    manifest = {
        "url": url,
        "saved_at": now_iso(),
        "content_type": content_type or "",
        "encoding": encoding,
        "path": str(path),
        "blob_checksum": checksum
    }
    meta_path = base_dir / f"{base_name}.meta.json"
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path), str(meta_path)

def main():
    config_dir = paths['config']['root']
    configs = load_yaml_configs(config_dir)
    total_extracted = 0
    total_saved = 0
    for source_cfg in configs:
        source_id = source_cfg.get('id')
        scraper_cls = SCRAPER_MAP.get(source_id)
        logger.info(f"=== Scraping for source '{source_id}' ===")
        scraper = scraper_cls(config=source_cfg)
        entry_urls = source_cfg.get('entry_urls', [])
        urls = scraper.discover(entry_urls)
        out_path = FORMULA_PATH / f"{source_id}.jsonl"
        existing_checksums = load_existing_checksums(out_path)
        seen_urls = _load_scraped_urls(source_id)
        for url in urls:
            if url in seen_urls:
                logger.info(f"Skipping {url} (already scraped)")
                continue
            logger.info(f"Fetching: {url}")
            try:
                raw = scraper.fetch(url)
            except Exception as e:
                logger.info(f"Failed to fetch: {url} -> {e}")
                continue
            page_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            base_name = f"{source_id}_{page_hash}"
            if isinstance(raw, dict):
                content_bytes = json.dumps(raw, indent=2).encode('utf-8')
                content_type = "application/json"
            else:
                content_bytes = raw.encode('utf-8')
                content_type = "text/html"
            blob_path, meta_path = _save_blob_as_text(
                BLOBS_PATH, base_name, content_bytes,
                url=url, content_type=content_type, overwrite=True
            )
            try:
                result = scraper.extract(raw)
                if isinstance(result, tuple):
                    extracted, answers_data = result
                else:
                    extracted = result
                    answers_data = {}
            except Exception as e:
                logger.warning(f"failed to extract: {url} -> {e}")
                continue
            if not isinstance(extracted, list):
                logger.warning(f"extractor for {source_id} did not return a list; skipping")
                continue

            answer_blob_paths = {}
            if answers_data and answers_data.get('items'):
                answer_blobs_dir = BLOBS_PATH.parent / 'answer_blobs'
                answer_blobs_dir.mkdir(parents=True, exist_ok=True)
                for answer in answers_data.get('items', []):
                    answer_id = answer.get('answer_id')
                    if answer_id:
                        answer_blob_name = f"answer_{answer_id}.json"
                        answer_blob_path = answer_blobs_dir / answer_blob_name
                        with open(answer_blob_path, 'w', encoding='utf-8') as af:
                            json.dump({'items': [answer]}, af, indent=2, ensure_ascii=False)
                        answer_blob_paths[answer_id] = str(answer_blob_path)

            saved = 0
            for lf in extracted:
                if not isinstance(lf, RawFormula):
                    logger.warning("non-RawFormula item returned by extractor; skipping")
                    continue
                lf.source_url = url
                lf.blob_paths.setdefault("raw_blob", str(blob_path))
                lf.blob_paths.setdefault("raw_manifest", str(meta_path))
                provenance = lf.provenance or {}
                origin = provenance.get('origin', '')
                if 'answer' in origin:
                    item_id = provenance.get('item_id')
                    if item_id and item_id in answer_blob_paths:
                        lf.blob_paths['answer_blob'] = answer_blob_paths[item_id]
                if not lf.checksum:
                    lf.compute_checksum()
                cs = lf.checksum
                if cs and cs in existing_checksums:
                    continue
                append_jsonl(str(out_path), lf)
                existing_checksums.add(cs)
                saved += 1
            _save_scraped_url(source_id, url)
            total_extracted += len(extracted)
            total_saved += saved
            logger.info(f"extracted {len(extracted)} formulas; saved {saved} new records (source={source_id})")
        logger.info(f"=== pipeline finished: {total_saved} new formulas out of {total_extracted} extracted ===")

if __name__ == '__main__':
    main()
