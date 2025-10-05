from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()

def checksum_of_str(s: str) -> str:
    return "sha256:" + hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def load_jsonl(path: Path):
    with path.open('rt', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Path, items):
    with path.open('wt', encoding='utf8') as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')