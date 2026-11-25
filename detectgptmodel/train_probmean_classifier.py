"""Huấn luyện mô hình hồi quy logistic cho phương pháp prob_mean.

Tái hiện đúng mô hình trong Colab ``Binoculars.ipynb`` (cell "PHÂN TÍCH CV"):

    make_model(C=0.3) = SimpleImputer(median) -> StandardScaler
                        -> LogisticRegression(C=0.3, class_weight="balanced")

thí nghiệm "Human vs cả hai". Đặc trưng mặc định: perf_prob_mean (top 1 trong
Colab), perf_nll_mean (mean NLL) và perf_ppl (perplexity). Mô hình cuối được
fit trên toàn bộ dữ liệu.

Chạy (từ thư mục detect-gpt-be):

    python -m detectgptmodel.train_probmean_classifier \\
        --human ../data/scores.csv \\
        --ai ../data/scores_ai.csv ../data/ai_high_pll_scores.csv

Kết quả: models/prob_mean_logreg.pkl (pipeline sklearn, nạp bằng joblib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["perf_prob_mean", "perf_nll_mean", "perf_ppl"]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "prob_mean_logreg.pkl"


def make_model(C: float = 0.3):
    # giống hệt make_model trong Colab
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="+", required=True, help="CSV văn bản người viết (nhãn 0)")
    ap.add_argument("--ai", nargs="+", required=True, help="CSV văn bản AI (nhãn 1)")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                    help="cột đặc trưng (hỗ trợ perf_prob_mean, perf_nll_mean, perf_ppl)")
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    feats = args.features
    parts = [pd.read_csv(f)[feats].assign(y_ai=0) for f in args.human]
    parts += [pd.read_csv(f)[feats].assign(y_ai=1) for f in args.ai]
    df = pd.concat(parts, ignore_index=True)
    X, y = df[feats].values, df["y_ai"].values
    print(f"Đặc trưng: {feats}")
    print(f"Human: {(y == 0).sum()}  AI: {(y == 1).sum()}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    aucs, accs = [], []
    for tr, te in cv.split(X, y):
        m = make_model(args.C).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        accs.append((m.predict(X[te]) == y[te]).mean())
    print(f"CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"CV Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    model = make_model(args.C).fit(X, y)
    coef = model[-1].coef_[0]
    print("Hệ số (đặc trưng đã chuẩn hoá):", {f: round(float(c), 4) for f, c in zip(feats, coef)},
          "intercept:", round(float(model[-1].intercept_[0]), 4))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats, "C": args.C,
                 "cv_auc": float(np.mean(aucs)), "cv_acc": float(np.mean(accs))}, args.out)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
