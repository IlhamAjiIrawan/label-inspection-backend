"""
app/ocr/executor.py — Executor OCR terkontrol (R6).

- OCR_EXECUTOR: ThreadPoolExecutor(max_workers=OCR_WORKERS).
- BoundedSemaphore(OCR_QUEUE_MAX): batasi antrean agar tidak OOM;
  submit_ocr raise QueueFull bila penuh → caller return 503 + Retry-After.
- Jangan naikkan uvicorn --workers (store/rate-limit di memori per proses).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from app import config


def _workers() -> int:
    try:
        return max(1, int(getattr(config, "OCR_WORKERS", 2) or 2))
    except Exception:
        return 2


def _queue_max() -> int:
    try:
        return max(1, int(getattr(config, "OCR_QUEUE_MAX", 20) or 20))
    except Exception:
        return 20


OCR_EXECUTOR = ThreadPoolExecutor(max_workers=_workers(), thread_name_prefix="ocr")
_slots = threading.BoundedSemaphore(_queue_max())


class QueueFull(Exception):
    """Antrean OCR penuh — caller harus return 503 + Retry-After."""


def submit_ocr(fn, *args, **kwargs):
    """Submit job OCR dengan batas antrean. Raise QueueFull bila penuh."""
    if not _slots.acquire(blocking=False):
        raise QueueFull("OCR queue full")
    try:
        fut = OCR_EXECUTOR.submit(fn, *args, **kwargs)
    except Exception:
        try:
            _slots.release()
        except Exception:
            pass
        raise
    fut.add_done_callback(lambda _f: _slots.release())
    return fut


def queue_depth_approx() -> dict:
    """Estimasi untuk /health (aditif, best-effort)."""
    try:
        # BoundedSemaphore tidak ekspos count; aproksimasi via _value (CPython).
        val = getattr(_slots, "_value", None)
        qmax = _queue_max()
        if isinstance(val, int):
            used = max(0, qmax - val)
            return {"queue_max": qmax, "queue_used_approx": used, "queue_free_approx": val}
    except Exception:
        pass
    return {"queue_max": _queue_max()}
