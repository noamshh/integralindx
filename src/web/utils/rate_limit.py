import time
from collections import OrderedDict, deque

from fastapi import HTTPException, Request

WINDOW_SECONDS = 60
MAX_PER_CLIENT = 30
MAX_GLOBAL = 300
MAX_TRACKED_CLIENTS = 4096

_clients: "OrderedDict[str, deque]" = OrderedDict()
_global: deque = deque()


def _client_key(request: Request) -> str:
    # Fly and HF Spaces both put the real address at the head of X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _trim(hits: deque, now: float) -> None:
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.popleft()


def rate_limit(request: Request) -> None:
    now = time.monotonic()

    _trim(_global, now)
    if len(_global) >= MAX_GLOBAL:
        raise HTTPException(status_code=429, detail="Server busy, try again shortly")

    key = _client_key(request)
    hits = _clients.get(key)
    if hits is None:
        # X-Forwarded-For is client-supplied, so cap how many buckets it can create.
        if len(_clients) >= MAX_TRACKED_CLIENTS:
            _clients.popitem(last=False)
        hits = _clients[key] = deque()
    _clients.move_to_end(key)

    _trim(hits, now)
    if len(hits) >= MAX_PER_CLIENT:
        raise HTTPException(status_code=429, detail="Too many requests")

    hits.append(now)
    _global.append(now)
