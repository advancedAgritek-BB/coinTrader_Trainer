import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import yaml

from data_loader import fetch_trade_logs
from feature_engineering import make_features
from trainers.regime_lgbm import train_regime_lgbm
from evaluation import simulate_signal_pnl
from registry import ModelRegistry


def load_cfg(path: str) -> dict:
    """Load YAML configuration file and return a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--cfg", default="cfg.yaml", help="Config file path")
    parser.add_argument("--start-ts", help="Start timestamp ISO format")
    parser.add_argument("--end-ts", help="End timestamp ISO format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    params = cfg.get("regime_lgbm", {})

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

    # Import and invoke LightGBM GPU wheel helper
    try:
        from lightgbm_gpu_build import build_and_upload_lightgbm_wheel
    except Exception:  # pragma: no cover - helper may not be available during tests
        build_and_upload_lightgbm_wheel = None
    if build_and_upload_lightgbm_wheel is not None:
        build_and_upload_lightgbm_wheel(url, key)

    end_ts = pd.to_datetime(args.end_ts) if args.end_ts else datetime.utcnow()
    start_ts = (
        pd.to_datetime(args.start_ts)
        if args.start_ts
        else end_ts - timedelta(days=7)
    )

    df = fetch_trade_logs(start_ts, end_ts)
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    df = make_features(df)
    if "target" not in df.columns:
        raise ValueError("Data must contain a 'target' column for training")

    X = df.drop(columns=["target"])
    y = df["target"]

    model, metrics = train_regime_lgbm(X, y, params, use_gpu=True)

    preds = model.predict(X)
    sharpe = simulate_signal_pnl(df, (preds >= 0.5).astype(int))
    eval_metrics = {"sharpe": sharpe}

    registry = ModelRegistry(url, key)
    registry.upload(model, "regime_model", {**metrics, **eval_metrics})


if __name__ == "__main__":
    main()
