"""Quản lý các phương pháp phát hiện AI.

* ``prob_mean`` (mặc định): Qwen2.5-1.5B-Instruct + prob_mean, hỗ trợ văn bản dài.
* ``fastgpt``  (cũ)       : roberta-large-mnli + Fast-DetectGPT + bộ phân loại
                             models/model_gradient.pkl (giữ nguyên như trước).

Mô hình được nạp lười (lần đầu dùng) để server không phải giữ cả hai mô hình
trong bộ nhớ nếu chỉ dùng một. Phương pháp mặc định được nạp sẵn khi chạy
``runserver`` (giống hành vi cũ).
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable

DEFAULT_METHOD = os.environ.get("DETECT_DEFAULT_METHOD", "prob_mean")


def _load_prob_mean():
    from detectgptmodel.probmean import DEFAULT_MODEL, ProbMeanDetector
    return ProbMeanDetector(model=os.environ.get("PROB_MEAN_MODEL", DEFAULT_MODEL))


def _load_fastgpt():
    from detectgptmodel.fastgpt import GPTChecker
    return GPTChecker(model="roberta-large-mnli")


METHODS: dict[str, dict[str, Any]] = {
    "prob_mean": {
        "label": "Prob-Mean (Qwen2.5-1.5B-Instruct)",
        "description": "Xác suất trung bình của token dưới Qwen2.5-1.5B-Instruct; hỗ trợ văn bản dài theo từng đoạn.",
        "loader": _load_prob_mean,
        "long_text": True,
    },
    "fastgpt": {
        "label": "Fast-DetectGPT (RoBERTa)",
        "description": "Phương pháp cũ: Fast-DetectGPT trên roberta-large-mnli + bộ phân loại.",
        "loader": _load_fastgpt,
        "long_text": False,
    },
}

_instances: dict[str, Any] = {}
_lock = threading.Lock()


def get_detector(method: str):
    if method not in METHODS:
        raise KeyError(method)
    if method not in _instances:
        with _lock:
            if method not in _instances:
                loader: Callable[[], Any] = METHODS[method]["loader"]
                _instances[method] = loader()
    return _instances[method]


def list_methods() -> list[dict]:
    return [
        {
            "id": key,
            "label": m["label"],
            "description": m["description"],
            "long_text": m["long_text"],
            "default": key == DEFAULT_METHOD,
            "loaded": key in _instances,
        }
        for key, m in METHODS.items()
    ]


def preload_default() -> None:
    # Chỉ nạp khi chạy server, không nạp khi migrate/test
    if "runserver" in sys.argv:
        get_detector(DEFAULT_METHOD)
