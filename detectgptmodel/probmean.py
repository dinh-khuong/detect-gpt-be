"""Phương pháp phát hiện văn bản AI dựa trên ``prob_mean``.

prob_mean = trung bình xác suất mà mô hình ngôn ngữ gán cho token thật:

    prob_mean = mean_t  p(x_t | x_<t)          (= ll.exp().mean())

Văn bản do AI sinh ra thường "dễ đoán" với LLM nên prob_mean cao hơn văn bản
người viết. Mô hình chấm điểm: Qwen/Qwen2.5-1.5B-Instruct (không dùng chat
template, đưa thẳng văn bản thô vào giống lúc trích đặc trưng trong notebook).

Văn bản dài được cắt thành nhiều đoạn (chunk) ~512 token (giống dữ liệu huấn
luyện ~300-600 token/đoạn). Mỗi đoạn được chấm kèm 128 token ngữ cảnh của đoạn
trước (không tính điểm phần ngữ cảnh), sau đó gộp lại:
    * prob_mean tổng = trung bình theo số token của các đoạn
    * nll_mean       = -mean_t log p(x_t | x_<t)   (cross-entropy)
    * ppl            = exp(nll_mean)               (perplexity)
    * base_*         = 7 đặc trưng của mô hình cũ (ttr, punct, sentence_std, polarity,
                       subjectivity, flesch, gunning_fog) — xem text_features.py
    * score (0-100)  = P(AI | các đặc trưng trên) * 100 từ mô hình hồi quy logistic
    * ai_fraction    = tỉ lệ token nằm trong các đoạn bị đánh giá là AI

Mô hình hồi quy logistic (models/prob_mean_logreg.pkl) là mô hình trong Colab
Binoculars.ipynb (SimpleImputer -> StandardScaler -> LogisticRegression(C=0.3,
class_weight="balanced"), đặc trưng perf_* và base_*), tạo bằng
``python -m detectgptmodel.train_probmean_classifier``. Đổi đường dẫn bằng biến
môi trường PROB_MEAN_CLASSIFIER.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass

import math

import joblib
import numpy as np
import torch as tch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .text_features import BASE_FEATURES, base_features

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Mô hình hồi quy logistic phân loại theo prob_mean (xem docstring).
PERF_KEYS = ("prob_mean", "nll_mean", "ppl")
SUPPORTED_FEATURES = {f"perf_{k}" for k in PERF_KEYS} | {f"base_{k}" for k in BASE_FEATURES}
DEFAULT_CLASSIFIER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "prob_mean_logreg.pkl"
)

# Độ dài đoạn: dữ liệu huấn luyện có trung bình ~530-560 token / đoạn.
DEFAULT_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 128          # đoạn cuối ngắn hơn mức này sẽ gộp vào đoạn trước
MIN_RELIABLE_TOKENS = 64        # văn bản ngắn hơn -> cảnh báo độ tin cậy thấp
DEFAULT_MAX_TOKENS = 200_000    # giới hạn an toàn cho văn bản rất dài
# Số token ngữ cảnh (của đoạn trước) đưa thêm vào trước mỗi đoạn, không được chấm
# điểm, để token đầu đoạn không bị mất ngữ cảnh (sliding window có chồng lấn).
DEFAULT_CONTEXT_TOKENS = 128


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Chunk:
    index: int
    tok_start: int
    tok_end: int
    char_start: int
    char_end: int


_SENT_END = re.compile(r"[.!?…。\n]")


class ProbMeanDetector:
    """Chấm điểm AI bằng prob_mean trên Qwen2.5-1.5B-Instruct, hỗ trợ văn bản dài."""

    name = "prob_mean"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        classifier_path: str | None = None,
        chunk_tokens: int | None = None,
        max_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        print("load model: " + model)
        self.model_name = model
        self.device = tch.device(device or ("cuda" if tch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        path = classifier_path or os.environ.get("PROB_MEAN_CLASSIFIER", DEFAULT_CLASSIFIER)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy mô hình hồi quy '{path}'. Chạy: python -m detectgptmodel.train_probmean_classifier"
            )
        bundle = joblib.load(path)
        self.classifier = bundle["model"]
        # Tên cột trong CSV huấn luyện, vd ["perf_prob_mean", "perf_nll_mean", "perf_ppl"]
        self.features: list[str] = bundle.get("features") or [bundle["feature"]]
        unknown = [f for f in self.features if f not in SUPPORTED_FEATURES]
        if unknown:
            raise ValueError(f"Mô hình hồi quy dùng đặc trưng chưa hỗ trợ: {unknown}")
        self.cv_auc = bundle.get("cv_auc")
        self.uses_base = any(f.startswith("base_") for f in self.features)
        # Chỉ có ngưỡng 1 chiều khi mô hình dùng đúng 1 đặc trưng prob_mean
        self.threshold = None
        if self.features == ["perf_prob_mean"]:
            scaler, clf = self.classifier[1], self.classifier[-1]
            self.threshold = float(scaler.mean_[0] - clf.intercept_[0] * scaler.scale_[0] / clf.coef_[0, 0])
        self.chunk_tokens = chunk_tokens or _env_int("PROB_MEAN_CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS)
        self.max_tokens = max_tokens or _env_int("PROB_MEAN_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        self.context_tokens = _env_int("PROB_MEAN_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS)
        default_bs = 4 if self.device.type == "cuda" else 1
        self.batch_size = batch_size or _env_int("PROB_MEAN_BATCH_SIZE", default_bs)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = tch.bfloat16 if self.device.type == "cuda" else tch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        # Một mô hình dùng chung cho nhiều request -> tránh chạy song song trên GPU.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ chunking
    def _tokenize(self, text: str):
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def _make_chunks(self, text: str, offsets: list[tuple[int, int]]) -> list[Chunk]:
        """Cắt theo token, ưu tiên cắt ở cuối câu gần nhất để không cắt ngang câu."""
        n = len(offsets)
        size = self.chunk_tokens
        bounds: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + size, n)
            if end < n:
                # lùi tối đa 25% kích thước đoạn để tìm ranh giới câu
                for j in range(end, max(start + size * 3 // 4, start + 1), -1):
                    s, e = offsets[j - 1]
                    if _SENT_END.search(text[s:e]):
                        end = j
                        break
            bounds.append((start, end))
            start = end

        # đoạn cuối quá ngắn -> gộp vào đoạn trước
        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MIN_CHUNK_TOKENS:
            last = bounds.pop()
            bounds[-1] = (bounds[-1][0], last[1])

        chunks = []
        for i, (ts, te) in enumerate(bounds):
            chunks.append(Chunk(i, ts, te, offsets[ts][0], offsets[te - 1][1]))
        return chunks

    # ------------------------------------------------------------------ scoring
    @tch.no_grad()
    def _token_logprobs(self, batch_ids: list[list[int]], n_score: list[int]) -> list[tch.Tensor]:
        """Log-xác suất của token thật cho ``n_score`` token cuối của từng chuỗi
        (phần trước đó chỉ làm ngữ cảnh). Token đầu chuỗi không có xác suất."""
        pad = self.tokenizer.pad_token_id
        max_len = max(len(x) for x in batch_ids)
        ids = tch.full((len(batch_ids), max_len), pad, dtype=tch.long)
        mask = tch.zeros((len(batch_ids), max_len), dtype=tch.long)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = tch.tensor(x, dtype=tch.long)
            mask[i, : len(x)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)

        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        out = []
        for i, x in enumerate(batch_ids):
            n = len(x)
            if n < 2:
                out.append(tch.empty(0))
                continue
            k = min(n_score[i], n - 1)
            lg = logits[i, n - 1 - k : n - 1].float()     # [k, vocab]
            tgt = ids[i, n - k : n]
            ll = lg.gather(1, tgt[:, None]).squeeze(1) - tch.logsumexp(lg, dim=-1)
            out.append(ll.cpu())
        return out

    @staticmethod
    def _features(ll: tch.Tensor) -> dict[str, float]:
        """Đặc trưng giống _per_model_feats trong notebook (ll = log p của token thật)."""
        nll_mean = float(-ll.mean())
        return {
            "prob_mean": float(ll.exp().mean()),
            "nll_mean": nll_mean,
            "ppl": math.exp(nll_mean),
        }

    def _score(self, feats: dict[str, float], base: dict[str, float]) -> float:
        x = np.array([[feats[f[5:]] if f.startswith("perf_") else base[f[5:]] for f in self.features]])
        return float(self.classifier.predict_proba(x)[0, 1]) * 100.0

    def predict(self, text: str) -> dict:
        text = text.strip()
        ids, offsets = self._tokenize(text)
        truncated = False
        if len(ids) > self.max_tokens:
            ids, offsets = ids[: self.max_tokens], offsets[: self.max_tokens]
            truncated = True

        chunks = self._make_chunks(text, offsets)
        probs: list[tch.Tensor] = []
        with self._lock:
            for b in range(0, len(chunks), self.batch_size):
                batch = chunks[b : b + self.batch_size]
                seqs = [ids[max(0, c.tok_start - self.context_tokens) : c.tok_end] for c in batch]
                probs.extend(self._token_logprobs(seqs, [c.tok_end - c.tok_start for c in batch]))

        chunk_results = []
        total_probs = []
        for c, p in zip(chunks, probs):
            if p.numel() == 0:
                continue
            feats = self._features(p)
            base = base_features(text[c.char_start : c.char_end]) if self.uses_base else {}
            score = self._score(feats, base)
            total_probs.append(p)
            chunk_results.append({
                "index": c.index,
                "start": c.char_start,
                "end": c.char_end,
                "n_tokens": c.tok_end - c.tok_start,
                **feats,
                "base": base,
                "score": score,
                "is_ai": score >= 50.0,
            })

        if not total_probs:
            raise ValueError("Văn bản quá ngắn để phân tích (cần ít nhất 2 token).")

        all_ll = tch.cat(total_probs)
        feats = self._features(all_ll)
        # Đặc trưng cũ phụ thuộc độ dài (ttr, sentence_std...) -> lấy trung bình theo
        # số token của các đoạn, để giữ cùng thang với dữ liệu huấn luyện (~300-600 token)
        base = {}
        if self.uses_base:
            w = np.array([r["n_tokens"] for r in chunk_results], dtype=float)
            base = {k: float(np.average([r["base"][k] for r in chunk_results], weights=w)) for k in BASE_FEATURES}
        all_probs = all_ll.exp()
        n_scored = sum(r["n_tokens"] for r in chunk_results)
        ai_tokens = sum(r["n_tokens"] for r in chunk_results if r["is_ai"])
        n_tokens = len(ids)

        warnings = []
        if n_tokens < MIN_RELIABLE_TOKENS:
            warnings.append(f"Văn bản ngắn ({n_tokens} token), kết quả có độ tin cậy thấp. Nên dùng >= {MIN_RELIABLE_TOKENS * 2} token.")
        if truncated:
            warnings.append(f"Văn bản quá dài, chỉ phân tích {self.max_tokens} token đầu tiên.")

        return {
            "method": self.name,
            "model": self.model_name,
            "score": self._score(feats, base),
            **feats,
            "base": base,
            "prob_std": float(all_probs.std()) if all_probs.numel() > 1 else 0.0,
            "threshold": self.threshold,
            "features": self.features,
            "cv_auc": self.cv_auc,
            "n_tokens": n_tokens,
            "n_chunks": len(chunk_results),
            "ai_fraction": ai_tokens / n_scored if n_scored else 0.0,
            "truncated": truncated,
            "warnings": warnings,
            "chunks": chunk_results,
        }
