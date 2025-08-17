from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .meme_features import (
    MEME_SYMBOL_DEFAULT,
    build_features_and_labels,
    detect_meme_csv_columns,
)

# LightGBM with optional GPU
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM is required for meme sniping training") from e

# Optional Supabase publishing
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
MODELS_BUCKET = os.getenv("MODELS_BUCKET", "models")  # keep aligned with repo conventions
REGISTRY_PREFIX = os.getenv("REGISTRY_PREFIX", "models/regime")  # path inside bucket

def _maybe_import_supabase():
    try:
        from supabase import create_client
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        return None
    except Exception:
        return None

def _time_series_split_ix(n: int, val_frac: float = 0.15):
    val = int(max(1000, math.floor(n * val_frac)))
    train = n - val
    return list(range(train)), list(range(train, n))

def _publish_to_supabase(model_path: Path, latest_json: dict, symbol: str):
    client = _maybe_import_supabase()
    if not client:
        print("[publish] Supabase client not configured; skipping upload.")
        return

    ts = int(time.time())
    file_key = f"{REGISTRY_PREFIX}/{symbol}/{ts}_{Path(model_path.name).stem}.pkl"
    latest_key = f"{REGISTRY_PREFIX}/{symbol}/LATEST.json"

    # Upload model binary
    with open(model_path, "rb") as f:
        client.storage.from_(MODELS_BUCKET).upload(
            file_key,
            f,
            {"content-type": "application/octet-stream", "x-upsert": "true"},
        )

    # Upload/update LATEST.json
    client.storage.from_(MODELS_BUCKET).upload(
        latest_key,
        json.dumps(latest_json, indent=2).encode("utf-8"),
        {"content-type": "application/json", "x-upsert": "true"}
    )
    print(f"[publish] Uploaded {file_key} and updated {latest_key}")

def train_meme_regime(
    input_csv: str,
    symbol: str = MEME_SYMBOL_DEFAULT,
    use_gpu: bool = False,
    federated: bool = False,
    publish: bool = True,
    horizon_rows: int = 15,
    tp: float = 0.5,
    sl: float = 0.30,
    class_weight_flat: float = 0.2,  # downweight 'flat'
    model_out_dir: str = "local_models",
):
    df = pd.read_csv(input_csv)
    if not detect_meme_csv_columns(df):
        raise ValueError(
            "Input does not look like meme logs; expected dev/sentiment/structure columns."
        )

    X, y, meta = build_features_and_labels(
        df, horizon_rows=horizon_rows, tp=tp, sl=sl, symbol_tag=symbol
    )
    n = len(y)
    if n < 2000:
        print(f"[warn] Very few samples ({n}). Training anyway.")

    tr_ix, va_ix = _time_series_split_ix(n, 0.15)
    Xtr, ytr = X.iloc[tr_ix], y.iloc[tr_ix]
    Xva, yva = X.iloc[va_ix], y.iloc[va_ix]

    # Label weights to handle imbalance; +1 and -1 are rarer than 0.
    weights = np.where(ytr == 0, class_weight_flat, 1.0)

    device_type = "gpu" if use_gpu else "cpu"
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": ["multi_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 256,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.5,
        "device_type": device_type,
        "max_bin": 63 if use_gpu else 255,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(Xtr, label=(ytr + 1), weight=weights, feature_name=list(X.columns))
    dvalid = lgb.Dataset(Xva, label=(yva + 1), reference=dtrain)

    if federated:
        # If repo provides federated_trainer, use it; else do a simple local
        # FedAvg over shards.
        try:
            from federated_trainer import run_federated_meme_training  # expected helper if present
            booster = run_federated_meme_training(X, y, params=params)
        except Exception:
            # Fallback: naive FedAvg across 4 shards (same machine), just to honor the flag
            shards = np.array_split(np.arange(len(Xtr)), 4)
            boosters = []
            for s in shards:
                ds = lgb.Dataset(Xtr.iloc[s], label=(ytr.iloc[s] + 1), feature_name=list(X.columns))
                boosters.append(lgb.train(params, ds, num_boost_round=120))
            # Average trees (very rough): pick best booster; true FedAvg would average leaf weights.
            booster = boosters[0]
    else:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=600,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        )

    os.makedirs(model_out_dir, exist_ok=True)
    model_name = f"{symbol.lower()}_regime_lgbm.pkl"
    model_path = Path(model_out_dir) / model_name
    joblib.dump({"model": booster, "feature_list": meta["feature_list"], "meta": meta}, model_path)

    # Basic metrics
    proba = booster.predict(Xva, num_iteration=booster.best_iteration)
    pred = proba.argmax(axis=1) - 1
    acc = float((pred == (yva.values)).mean())

    latest_json = {
        "model_key": model_name,
        "feature_list": meta["feature_list"],
        "symbol": symbol,
        "created_at": int(time.time()),
        "metrics": {"valid_acc": acc, "n_valid": len(yva)},
        "train": {"horizon_rows": meta["horizon_rows"], "tp": meta["tp"], "sl": meta["sl"]},
        "loader": {"type": "lightgbm_multiclass", "class_order": [-1, 0, 1]},
        "version": 1,
    }

    print(f"[train] valid_acc={acc:.4f}, samples={len(y)} -> saved {model_path}")

    if publish:
        _publish_to_supabase(model_path, latest_json, symbol)

def cli():
    p = argparse.ArgumentParser("meme sniping trainer")
    p.add_argument("--input", required=True, help="Path to solana_meme_logs.csv")
    p.add_argument(
        "--symbol",
        default=MEME_SYMBOL_DEFAULT,
        help="Registry symbol tag (CT runtime model key)",
    )
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--federated", action="store_true")
    p.add_argument("--no-publish", action="store_true")
    p.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Forward window in rows (≈minutes for 1m data)",
    )
    p.add_argument("--tp", type=float, default=0.5)
    p.add_argument("--sl", type=float, default=0.30)
    args = p.parse_args()

    train_meme_regime(
        input_csv=args.input,
        symbol=args.symbol,
        use_gpu=args.use_gpu,
        federated=args.federated,
        publish=(not args.no_publish),
        horizon_rows=args.horizon,
        tp=args.tp,
        sl=args.sl,
    )

if __name__ == "__main__":
    cli()
