"""Huấn luyện mô hình hồi quy logistic cho phương pháp prob_mean.

Tái hiện đúng mô hình trong Colab ``Binoculars.ipynb`` (cell "PHÂN TÍCH CV"):

    make_model(C=0.3) = SimpleImputer(median) -> StandardScaler
                        -> LogisticRegression(C=0.3, class_weight="balanced")

thí nghiệm "Human vs cả hai", bộ đặc trưng "top 1 (perf_prob_mean)"
(CV AUC 0.998 ± 0.001). Mô hình cuối được fit trên toàn bộ dữ liệu.

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

FEATURE = "perf_prob_mean"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "prob_mean_logreg.pkl"


def make_model(C: float = 0.3):
    # giống hệt make_model trong Colab
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def decision_threshold(model) -> float:
    """Giá trị prob_mean tại đó P(AI) = 0.5 (ranh giới quyết định của mô hình)."""
    scaler, clf = model[1], model[-1]
    return float(scaler.mean_[0] - clf.intercept_[0] * scaler.scale_[0] / clf.coef_[0, 0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="+", required=True, help="CSV văn bản người viết (nhãn 0)")
    ap.add_argument("--ai", nargs="+", required=True, help="CSV văn bản AI (nhãn 1)")
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    parts = [pd.read_csv(f)[[FEATURE]].assign(y_ai=0) for f in args.human]
    parts += [pd.read_csv(f)[[FEATURE]].assign(y_ai=1) for f in args.ai]
    df = pd.concat(parts, ignore_index=True)
    X, y = df[[FEATURE]].values, df["y_ai"].values
    print(f"Human: {(y == 0).sum()}  AI: {(y == 1).sum()}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    aucs = [roc_auc_score(y[te], make_model(args.C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            for tr, te in cv.split(X, y)]
    print(f"CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    model = make_model(args.C).fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"Ngưỡng quyết định (P(AI)=0.5): {FEATURE} = {decision_threshold(model):.4f}")
    print(f"Accuracy trên toàn bộ dữ liệu: {acc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature": FEATURE, "C": args.C,
                 "cv_auc": float(np.mean(aucs))}, args.out)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
