"""
Domain 1 – Tyre Degradation: XGBoost quantile regression model.

Trains four models on the ML feature matrix:

1. ``main``  – standard MSE objective (point estimate)
2. ``q10``   – lower-bound quantile (10th percentile)
3. ``q50``   – median quantile (50th percentile)
4. ``q90``   – upper-bound quantile (90th percentile)

Usage
-----
    python -m src.models.train_degradation_model
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from src.utils.paths import DOMAIN1_MODELS, DOMAIN1_SILVER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# XGBoost base hyperparameters (non-quantile model)
BASE_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

FEATURE_COLS = [
    "compound_enc",
    "stint_length",
    "track_temp_proxy",
    "consistency_cv",
    "rolling_deg_rate",
    "compound_x_temp",
    "compound_x_stint",
    "peak_lap",
]
TARGET_COL = "target_deg_rate"


# ---------------------------------------------------------------------------
# 1. Load features
# ---------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    """Load ML features from the silver domain1 layer.

    Returns
    -------
    pd.DataFrame
        Feature matrix with at least FEATURE_COLS + [TARGET_COL].

    Raises
    ------
    FileNotFoundError
        If the features Parquet has not yet been built.
    """
    path = DOMAIN1_SILVER / "ml_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"ml_features.parquet not found: {path}\n"
            "Run `python -m src.features.feature_builder` first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded ml_features: %d rows × %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------

def prepare_train_test_split(
    features_df: pd.DataFrame,
    n_test_events: int = 4,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Time-based train/test split: last ``n_test_events`` rounds → test set.

    Parameters
    ----------
    features_df:
        Full feature matrix.
    n_test_events:
        Number of most-recent rounds to hold out for evaluation.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    df = features_df.dropna(subset=[TARGET_COL]).copy()
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    if "round_number" in df.columns:
        sorted_rounds = sorted(df["round_number"].unique())
        test_rounds = sorted_rounds[-n_test_events:]
        is_test = df["round_number"].isin(test_rounds)
    else:
        # Fallback: random 20% split
        is_test = np.random.default_rng(42).random(len(df)) > 0.8

    train = df[~is_test].fillna(0)
    test = df[is_test].fillna(0)

    X_train = train[available_features]
    y_train = train[TARGET_COL]
    X_test = test[available_features]
    y_test = test[TARGET_COL]

    log.info(
        "Train: %d rows, Test: %d rows (%d features)",
        len(X_train), len(X_test), len(available_features),
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Train XGBoost quantile model
# ---------------------------------------------------------------------------

def train_xgboost_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float | None = None,
) -> XGBRegressor:
    """Train an XGBoost regression model.

    Parameters
    ----------
    X_train, y_train:
        Training data.
    quantile:
        When ``None``, uses standard ``'reg:squarederror'`` objective.
        When a float in (0, 1), uses ``'reg:quantileerror'`` with
        ``quantile_alpha=quantile``.

    Returns
    -------
    XGBRegressor
        Fitted model.
    """
    params = BASE_PARAMS.copy()
    if quantile is None:
        params["objective"] = "reg:squarederror"
        label = "MSE"
    else:
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = quantile
        label = f"Q{int(quantile * 100)}"

    log.info("Training XGBoost (%s) on %d samples …", label, len(X_train))
    model = XGBRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, verbose=False)

    log.info("Training complete (%s).", label)
    return model


# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    quantile: float | None = None,
) -> dict[str, float]:
    """Evaluate a fitted model on the test set.

    Returns
    -------
    dict
        ``mae``, ``rmse``, and optionally ``coverage`` (for quantile models).
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    metrics: dict[str, float] = {"mae": mae, "rmse": rmse}

    if quantile is not None:
        # Coverage: fraction of actuals that fall at or below the predicted quantile.
        # For a well-calibrated Q-alpha model this should equal alpha.
        coverage = float(np.mean(y_test.to_numpy() <= y_pred))
        metrics["coverage"] = coverage
        metrics["target_coverage"] = quantile

    log.info(
        "Evaluation (Q%s): MAE=%.4f  RMSE=%.4f%s",
        "MSE" if quantile is None else int(quantile * 100),
        mae,
        rmse,
        f"  Coverage={metrics.get('coverage', ''):.3f}" if quantile is not None else "",
    )
    return metrics


# ---------------------------------------------------------------------------
# 5. Save model
# ---------------------------------------------------------------------------

def save_model(model: XGBRegressor, model_name: str) -> Path:
    """Persist a fitted XGBRegressor in JSON format.

    Parameters
    ----------
    model:
        Fitted XGBRegressor instance.
    model_name:
        Base name (without extension), e.g. ``'degradation_q90'``.

    Returns
    -------
    Path
        Path of the saved model file.
    """
    DOMAIN1_MODELS.mkdir(parents=True, exist_ok=True)
    out = DOMAIN1_MODELS / f"{model_name}.json"
    model.save_model(str(out))
    log.info("Model saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Degradation model training starting ===")

    features_df = load_features()
    X_train, y_train, X_test, y_test = prepare_train_test_split(features_df)

    if X_train.empty:
        log.error("Training set is empty. Exiting.")
        return

    all_metrics: dict[str, dict] = {}

    # Model 1: MSE (point estimate)
    model_main = train_xgboost_quantile(X_train, y_train, quantile=None)
    all_metrics["main"] = evaluate_model(model_main, X_test, y_test, quantile=None)
    save_model(model_main, "degradation_main")

    # Models 2-4: quantile regression
    for q in (0.10, 0.50, 0.90):
        model_q = train_xgboost_quantile(X_train, y_train, quantile=q)
        label = f"q{int(q * 100):02d}"
        all_metrics[label] = evaluate_model(model_q, X_test, y_test, quantile=q)
        save_model(model_q, f"degradation_{label}")

    # Persist metrics
    DOMAIN1_MODELS.mkdir(parents=True, exist_ok=True)
    metrics_path = DOMAIN1_MODELS / "model_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(all_metrics, fh, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    log.info("=== Degradation model training complete ===")


if __name__ == "__main__":
    main()
