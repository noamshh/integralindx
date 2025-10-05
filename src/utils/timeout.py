"""
Timeout utilities for long-running operations.

Provides thread-based timeout wrapper for functions that may hang
(particularly SymPy operations like sympify, simplify, subs).
"""
import gc
import logging
import threading
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout"""
    pass


def with_timeout(func: Callable[..., T], timeout_seconds: float = 20) -> Callable[..., T]:
    """
    Execute function with timeout using threading with aggressive cleanup.

    Args:
        func: Function to execute with timeout
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Wrapped function that raises TimeoutError if execution exceeds timeout

    Raises:
        TimeoutError: If function execution exceeds timeout_seconds

    Example:
        >>> def slow_operation(x):
        ...     return simplify(x)
        >>> result = with_timeout(slow_operation, 5)(expr)
    """
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
