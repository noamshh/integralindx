"""Backend factory. The orchestrator only ever sees `OcrBackend`, never a
concrete class — so swapping models is a config change. To add a new backend:
implement `OcrBackend` in `corpus/ocr/backends/<name>.py`, then either call
`register_backend(name, factory)` or extend `_BUILTIN` below."""

from typing import Any, Callable, Dict

from corpus.ocr.backends.base import OcrBackend


_REGISTRY: Dict[str, Callable[..., OcrBackend]] = {}


def register_backend(name: str, factory: Callable[..., OcrBackend]) -> None:
    _REGISTRY[name] = factory


def _qwen3_vl_factory(**kwargs: Any) -> OcrBackend:
    from corpus.ocr.backends.qwen3_vl import Qwen3VLBackend
    return Qwen3VLBackend(**kwargs)


_BUILTIN: Dict[str, Callable[..., OcrBackend]] = {
    "qwen3_vl": _qwen3_vl_factory,
}


def get_backend(name: str, **kwargs: Any) -> OcrBackend:
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if name in _BUILTIN:
        return _BUILTIN[name](**kwargs)
    raise KeyError(
        f"unknown OCR backend: {name!r}. "
        f"available: {sorted(set(_BUILTIN) | set(_REGISTRY))}"
    )
