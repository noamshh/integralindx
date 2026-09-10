#!/usr/bin/env python3
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive"
LAST_FETCH_FILE = LOGS_DIR / ".last_fetch"
SEARCH_QUERIES_FILE = LOGS_DIR / "search_queries.log"
FLY_APP_NAME = "integralindx"

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub('', text)

def parse_log_timestamp(line: str) -> datetime | None:
    clean_line = strip_ansi(line)
    match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', clean_line)
    if match:
        return datetime.fromisoformat(match.group(1).replace('Z', '+00:00'))
    return None

def fetch_logs() -> list[str]:
    try:
        result = subprocess.run(
            ['flyctl', 'logs', '--app', FLY_APP_NAME, '--no-tail'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"error fetching logs: {result.stderr}")
            return []
        return result.stdout.strip().split('\n')
    except subprocess.TimeoutExpired:
        print("error: flyctl logs command timed out")
        return []
    except FileNotFoundError:
        print("error: flyctl command not found. install Fly CLI first.")
        return []

def get_last_fetch_time() -> datetime | None:
    if not LAST_FETCH_FILE.exists():
        return None
    try:
        timestamp_str = LAST_FETCH_FILE.read_text().strip()
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, OSError):
        return None

def save_last_fetch_time(timestamp: datetime):
    LAST_FETCH_FILE.write_text(timestamp.isoformat())

def is_search_query_log(line: str) -> bool:
    return "query='" in line or 'query="' in line

def main():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"fetching logs from {FLY_APP_NAME}...")
    raw_logs = fetch_logs()

    if not raw_logs:
        print("no logs fetched")
        return 0

    last_fetch = get_last_fetch_time()
    if last_fetch:
        print(f"last fetch: {last_fetch.isoformat()}")
    else:
        print("first run - saving all available logs")

    new_logs = []
    search_query_logs = []
    latest_timestamp = None

    for raw_line in raw_logs:
        if not raw_line.strip():
            continue

        timestamp = parse_log_timestamp(raw_line)
        if timestamp:
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp

            if last_fetch is None or timestamp > last_fetch:
                clean_line = strip_ansi(raw_line)
                new_logs.append(clean_line)

                if is_search_query_log(clean_line):
                    search_query_logs.append(clean_line)

    if not new_logs:
        print("no new logs since last fetch")
        return 0

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    archive_file = ARCHIVE_DIR / f"{timestamp_str}.txt"

    time_range = f"{new_logs[0][:19]} to {new_logs[-1][:19]}" if len(new_logs) > 1 else new_logs[0][:19]
    header = f"""# Fly.io logs saved at {now.isoformat()}
# App: {FLY_APP_NAME}
# Log count: {len(new_logs)}
# Time range: {time_range}
# Search queries in this batch: {len(search_query_logs)}

"""

    archive_file.write_text(header + '\n'.join(new_logs), encoding='utf-8')
    print(f"saved {len(new_logs)} logs to {archive_file.relative_to(PROJECT_ROOT)}")

    if search_query_logs:
        if SEARCH_QUERIES_FILE.exists():
            with SEARCH_QUERIES_FILE.open('a', encoding='utf-8') as f:
                f.write('\n' + '\n'.join(search_query_logs))
        else:
            header_combined = f"# Search query logs - started {now.isoformat()}\n\n"
            SEARCH_QUERIES_FILE.write_text(header_combined + '\n'.join(search_query_logs), encoding='utf-8')
        print(f"appended {len(search_query_logs)} search queries to {SEARCH_QUERIES_FILE.relative_to(PROJECT_ROOT)}")

    if latest_timestamp:
        save_last_fetch_time(latest_timestamp)
        print(f"updated last fetch time: {latest_timestamp.isoformat()}")

    return 0

if __name__ == "__main__":
    exit(main())
