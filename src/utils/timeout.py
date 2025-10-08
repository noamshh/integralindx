import gc
import logging
import threading
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

class TimeoutError(Exception):
    pass

def with_timeout(func: Callable[..., T], timeout_seconds: float = 20) -> Callable[..., T]:
    def wrapper(*args, **kwargs):
        result = [None]
        exception = [None]
        finished = [False]
        def target():
            try:
                result[0] = func(*args, **kwargs)
                finished[0] = True
            except Exception as e:
                exception[0] = e
                finished[0] = True
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        if not finished[0] and thread.is_alive():
            logger.warning(f"Operation timed out after {timeout_seconds} seconds (thread may continue in background)")
            gc.collect()
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        if exception[0]:
            raise exception[0]
        return result[0]
    return wrapper
