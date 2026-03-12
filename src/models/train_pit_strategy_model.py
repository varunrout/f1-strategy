"""
train_pit_strategy_model.py
===========================
Domain 3 — Pit-Stop Timing Prediction Model.

Trains an XGBoost classifier that predicts, for each lap, whether it is
a good pit lap ("1") or not ("0"), using the XT strategy features built
by build_xt_features.py.

Usage
-----
    python -m src.models.train_pit_strategy_model \
        [--data-dir DATA_DIR] \
        [--model-dir MODEL_DIR] \
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature & target config
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "tyre_age",
    "laps_to_go",
    "deg_rate_ms_per_lap",
    "projected_deg_loss_ms",
    "pit_urgency_score",
    "undercut_score",
    "sc_score",
    "strategy_score",
]

LABEL_COL = "in_pit_window"   # binary 0/1

COMPOUND_ENCODER = LabelEncoder()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(db_xt: Path) -> pd.DataFrame:
    """
    Join strategy_scores + pit_window_features + undercut_scores
    from features_xt.db into a single feature matrix.
    """
    if not db_xt.exists():
        logger.warning("%s not found — returning empty DataFrame", db_xt)
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_xt))
    try:
        ss = pd.read_sql_query("SELECT * FROM strategy_scores", conn)
        pw = pd.read_sql_query("SELECT * FROM pit_window_features", conn)
        uc = pd.read_sql_query(
            "SELECT session_id, driver, lap_number, undercut_score FROM undercut_scores",
            conn,
        )
    except Exception as exc:
        logger.error("Failed to load tables: %s", exc)
        return pd.DataFrame()
    finally:
        conn.close()

    if ss.empty or pw.empty:
        return pd.DataFrame()

    merge_keys = ["session_id", "driver", "lap_number"]
    df = pw.merge(ss[merge_keys + ["undercut_score", "sc_score", "strategy_score"]],
                  on=merge_keys, how="left")
    if not uc.empty:
        df = df.merge(uc, on=merge_keys, how="left", suffixes=("", "_uc"))
        # Prefer undercut_score from undercut_scores table if present
        if "undercut_score_uc" in df.columns:
            df["undercut_score"] = df["undercut_score_uc"].fillna(
                df.get("undercut_score", 0)
            )
            df.drop(columns=["undercut_score_uc"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns (X, y, groups) where groups = session_id (for CV splitting).
    """
    df = df.copy()

    # Encode compound
    if "compound" in df.columns:
        df["compound_enc"] = COMPOUND_ENCODER.fit_transform(
            df["compound"].fillna("UNKNOWN")
        )
        if "compound_enc" not in FEATURE_COLS:
            FEATURE_COLS.append("compound_enc")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X = df[available_features]
    y = df[LABEL_COL].astype(int) if LABEL_COL in df.columns else pd.Series(
        np.zeros(len(df), dtype=int)
    )
    groups = df["session_id"] if "session_id" in df.columns else pd.Series(
        np.zeros(len(df), dtype=int)
    )
    return X, y, groups


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    seed: int = 42,
) -> tuple[CalibratedClassifierCV, dict]:
    """
    Train XGBoost + Platt calibration with group-aware cross-validation.

    Returns (fitted_model, metrics_dict).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )

    model = CalibratedClassifierCV(xgb, cv=3, method="sigmoid")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: dict = {}
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
        metrics["avg_precision"] = round(average_precision_score(y_test, y_prob), 4)
    else:
        logger.warning("Only one class in test set — skipping ROC/AP metrics")
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    metrics["feature_names"] = list(X.columns)

    return model, metrics


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap_importance(
    model: CalibratedClassifierCV,
    X: pd.DataFrame,
    n_samples: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Compute mean |SHAP| importance for the inner XGBoost estimator.
    Returns None if shap is not installed.
    """
    try:
        import shap  # noqa: PLC0415

        # Unwrap calibrated classifier to get XGBClassifier
        inner = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(inner)
        sample = X.sample(min(n_samples, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample)
        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return importance
    except ImportError:
        logger.info("shap not installed — skipping SHAP importance")
        return None
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    model: CalibratedClassifierCV,
    metrics: dict,
    shap_df: Optional[pd.DataFrame],
    model_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "pit_strategy_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", model_path)

    metrics_path = model_dir / "pit_strategy_metrics.json"
    metrics_serialisable = {
        k: v for k, v in metrics.items() if k != "classification_report"
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_serialisable, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    report_path = model_dir / "pit_strategy_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics.get("classification_report", {}), f, indent=2)

    if shap_df is not None:
        shap_path = model_dir / "pit_strategy_shap_importance.csv"
        shap_df.to_csv(shap_path, index=False)
        logger.info("SHAP importance saved → %s", shap_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(data_dir: Path, model_dir: Path) -> None:
    db_xt = data_dir / "features_xt.db"

    logger.info("Loading training data from %s …", db_xt)
    df = load_training_data(db_xt)

    if df.empty:
        logger.error(
            "No training data found. Run build_xt_features.py first to "
            "populate features_xt.db."
        )
        return

    logger.info("Preparing features (%d rows) …", len(df))
    X, y, groups = prepare_features(df)

    logger.info(
        "Class balance — pit window: %d / %d (%.1f%%)",
        y.sum(),
        len(y),
        100 * y.mean(),
    )

    logger.info("Training XGBoost pit-strategy classifier …")
    model, metrics = train(X, y, groups)

    roc = metrics.get("roc_auc")
    ap = metrics.get("avg_precision")
    if roc is not None:
        logger.info("ROC-AUC: %.4f  |  Avg Precision: %.4f", roc, ap)

    logger.info("Computing SHAP feature importance …")
    shap_df = compute_shap_importance(model, X)

    logger.info("Saving artefacts …")
    save_artefacts(model, metrics, shap_df, model_dir)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Domain 3 pit-stop timing prediction model."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    run(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
