"""Đặc trưng văn bản của mô hình cũ (FastGptDetect.get_base_features), dùng lại cho prob_mean.

Giữ đúng công thức cũ (cùng các hàm trong fastgpt.py):
    ttr           Type-Token Ratio trên token của tokenizer roberta-large-mnli
    punct         số dấu '-' / số ký tự
    sentence_std  độ lệch chuẩn độ dài câu (số từ)
    polarity, subjectivity   TextBlob (chỉ có ý nghĩa với tiếng Anh)
    flesch, gunning_fog      textstat (công thức tiếng Anh)

Tên cột trong CSV huấn luyện có tiền tố ``base_`` (vd base_flesch).
"""

from __future__ import annotations

from functools import lru_cache

import textstat
from textblob import TextBlob

from .fastgpt import calc_punctuation, calculate_sentence_length_std, calculate_ttr

BASE_FEATURES = ("ttr", "punct", "sentence_std", "polarity", "subjectivity", "flesch", "gunning_fog")
TTR_TOKENIZER = "roberta-large-mnli"


@lru_cache(maxsize=1)
def _ttr_tokenizer():
    from transformers import RobertaTokenizer
    return RobertaTokenizer.from_pretrained(TTR_TOKENIZER)


def base_features(text: str) -> dict[str, float]:
    tb = TextBlob(text)
    # verbose=False: văn bản dài hơn 512 token chỉ dùng để đếm, không đưa vào mô hình
    ids = _ttr_tokenizer()(text, verbose=False).input_ids
    return {
        "ttr": float(calculate_ttr(ids)),
        "punct": calc_punctuation(text, "-") / max(len(text), 1),
        "sentence_std": float(calculate_sentence_length_std(text)),
        "polarity": float(tb.polarity),
        "subjectivity": float(tb.subjectivity),
        "flesch": float(textstat.flesch_reading_ease(text)),
        "gunning_fog": float(textstat.gunning_fog(text)),
    }
