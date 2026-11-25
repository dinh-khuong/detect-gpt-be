"""Tính đặc trưng cũ (base_*) cho 3 tập dữ liệu huấn luyện của prob_mean.

Các file scores*.csv chỉ chứa đặc trưng, không chứa văn bản, nên script này dựng
lại văn bản theo đúng các bước trong Colab Binoculars.ipynb (cùng seed, cùng cách
cắt theo token của Qwen) và kiểm tra khớp từng dòng qua ``n_tokens`` và ``ttr``
của CSV trước khi tính đặc trưng.

    human   : trieunh/Vietnamese_literature_VuTrongPhung (collections_chunked)
    ai      : trieunh/vietnamese.human.ai.news.sets (train), cắt theo độ dài người viết
    ai_high : "high prelexity.csv" (cột chunk)

Chạy (từ detect-gpt-be, cần mạng để tải 2 dataset HF, không cần GPU):

    python -m detectgptmodel.build_base_features --data ../data --high "../high prelexity.csv"

Kết quả: base_scores.csv, base_scores_ai.csv, base_ai_high_pll_scores.csv trong --data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from transformers import AutoTokenizer

from .text_features import base_features

QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
HUMAN_DS = "hf://datasets/trieunh/Vietnamese_literature_VuTrongPhung/collections_chunked.parquet"
AI_DS = "hf://datasets/trieunh/vietnamese.human.ai.news.sets/train.parquet"
SEED = 42


def rebuild_texts(tok, high_csv: str) -> dict[str, list[str]]:
    def n_tok(t):
        return len(tok(t, add_special_tokens=False).input_ids)

    def truncate_text(text, k):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc.input_ids) <= k:
            return text
        return text[: enc.offset_mapping[k - 1][1]]

    human = pd.read_parquet(HUMAN_DS)
    human["n_tokens"] = [n_tok(t) for t in human["content"]]

    ai = pd.read_parquet(AI_DS).rename(columns={"sail-Sailor2-8B-Chat-generated_text": "content"})[["content"]]
    # Bước 1 (cell 5): độ dài theo phân phối chuẩn bị chặn
    mean, std = human["n_tokens"].mean(), human["n_tokens"].std()
    lo, hi = human["n_tokens"].min(), human["n_tokens"].max()
    n = 2000
    tl = truncnorm.rvs((lo - mean) / std, (hi - mean) / std, loc=mean, scale=std,
                       size=n, random_state=SEED).round().astype(int)
    ai = ai[:n].copy()
    ai["content"] = [truncate_text(t, k) for t, k in zip(ai["content"].astype(str), tl)]
    # Bước 2 (cell 7): gán độ dài lấy từ phân bố người viết, bỏ văn bản quá ngắn
    rng = np.random.default_rng(SEED)
    full = ai["content"].astype(str).apply(n_tok).to_numpy()
    tl = rng.choice(human["n_tokens"].to_numpy(), size=len(ai), replace=True)
    ai["content"] = [truncate_text(t, k) for t, k in zip(ai["content"].astype(str), tl)]
    ai = ai[full >= tl].reset_index(drop=True)

    return {
        "scores.csv": list(human["content"]),
        "scores_ai.csv": list(ai["content"]),
        "ai_high_pll_scores.csv": list(pd.read_csv(high_csv)["chunk"]),
    }


def verify(tok, texts: list[str], ref: pd.DataFrame, name: str) -> None:
    """So n_tokens và ttr (token Qwen, như _lexical_feats) với CSV gốc."""
    assert len(texts) == len(ref), f"{name}: {len(texts)} văn bản nhưng CSV có {len(ref)} dòng"
    for i, t in enumerate(texts):
        ids = tok(t, truncation=True, max_length=4096).input_ids
        if len(ids) != ref["n_tokens"][i] or abs(len(set(ids)) / len(ids) - ref["ttr"][i]) > 1e-9:
            raise ValueError(f"{name}: dòng {i} không khớp CSV gốc")
    print(f"{name}: {len(texts)} dòng khớp CSV gốc")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="thư mục chứa scores*.csv")
    ap.add_argument("--high", required=True, help='đường dẫn "high prelexity.csv"')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(QWEN)
    for name, texts in rebuild_texts(tok, args.high).items():
        verify(tok, texts, pd.read_csv(args.data / name), name)
        feats = pd.DataFrame([base_features(t) for t in texts]).add_prefix("base_")
        feats.insert(0, "idx", range(len(texts)))
        out = args.data / f"base_{name}"
        feats.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
