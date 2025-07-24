"""Utilities for environment setup tasks like building LightGBM."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import yaml

from data_loader import fetch_trade_logs
from feature_engineering import make_features
from trainers.regime_lgbm import train_regime_lgbm
from evaluation import simulate_signal_pnl
from registry import ModelRegistry

logger = logging.getLogger(__name__)


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
    except ImportError as exc:  # pragma: no cover - helper may not be available during tests
        logger.warning("GPU wheel helper unavailable: %s", exc)
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

import glob
import os
import platform
import subprocess
from pathlib import Path

from supabase import create_client


def ensure_lightgbm_gpu(supabase_url: str, supabase_key: str, script_path: str | None = None) -> bool:
    """Ensure a GPU-enabled LightGBM build and upload wheels to Supabase.

    When running on Windows and LightGBM with GPU support is not available,
    ``build_lightgbm_gpu.ps1`` is invoked via ``subprocess`` to build the
    wheel. All wheels produced are uploaded to the ``wheels`` bucket in
    Supabase using ``create_client``.

    Parameters
    ----------
    supabase_url : str
        Supabase project URL.
    supabase_key : str
        Service role key or other API key with storage access.
    script_path : str, optional
        Path to the PowerShell build script. Defaults to ``build_lightgbm_gpu.ps1``
        located next to this module.

    Returns
    -------
    bool
        ``True`` if a build was performed and wheels uploaded, ``False`` if
        skipped because GPU-enabled LightGBM was already installed or the
        platform is not Windows.
    """

    if platform.system() != "Windows":
        return False

    try:
        import lightgbm as lgb

        lgb.train(
            {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0},
            lgb.Dataset([[1.0]], label=[0]),
            num_boost_round=1,
        )
        return False
    except Exception:
        pass

    script = Path(script_path or Path(__file__).with_name("build_lightgbm_gpu.ps1"))
    subprocess.run([
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ], check=True)

    wheel_dir = script.with_name("LightGBM").joinpath("python-package", "dist")
    wheels = glob.glob(str(wheel_dir / "*.whl"))

    sb = create_client(supabase_url, supabase_key)
    bucket = sb.storage.from_("wheels")
    for whl in wheels:
        with open(whl, "rb") as fh:
            bucket.upload(os.path.basename(whl), fh)
    return True
