"""Utilities for environment setup tasks like building LightGBM."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import platform
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from supabase import create_client

from data_loader import fetch_trade_logs, _get_redis_client
from evaluation import full_strategy_eval
from feature_engineering import make_features
from registry import ModelRegistry
from trainers.regime_lgbm import train_regime_lgbm

try:  # optional dependency
    import pyopencl as cl
except Exception as exc:  # pragma: no cover - pyopencl may be absent
    cl = None  # type: ignore
    logging.getLogger(__name__).warning("pyopencl not available: %s", exc)

load_dotenv()


logger = logging.getLogger(__name__)


def load_cfg(path: str) -> dict:
    """Load YAML configuration file and return a dictionary with defaults."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("default_window_days", 7)
    cfg.setdefault("backtest", {"slippage": 0.005, "costs": 0.002})
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--cfg", default="cfg.yaml", help="Config file path")
    parser.add_argument("--start-ts", help="Start timestamp ISO format")
    parser.add_argument("--end-ts", help="End timestamp ISO format")
    parser.add_argument("--table", default="ohlc_data", help="Supabase table name")
    parser.add_argument(
        "--feature-cache-key",
        help="Redis key for caching generated features",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    params = cfg.get("regime_lgbm", {})

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    if check_clinfo_gpu():
        params.setdefault("device_type", "gpu")
        params.setdefault("gpu_platform_id", 0)
        params.setdefault("gpu_device_id", 0)
        use_gpu = True
    else:
        params["device_type"] = "cpu"
        logger.warning("GPU not detected; falling back to CPU")
        use_gpu = False

    # Import and invoke LightGBM GPU wheel helper
    try:
        from lightgbm_gpu_build import build_and_upload_lightgbm_wheel
    except (
        ImportError
    ) as exc:  # pragma: no cover - helper may not be available during tests
        logger.warning("GPU wheel helper unavailable: %s", exc)
        build_and_upload_lightgbm_wheel = None
    if build_and_upload_lightgbm_wheel is not None:
        build_and_upload_lightgbm_wheel(url, key)
        verify_opencl()

    end_ts = pd.to_datetime(args.end_ts) if args.end_ts else datetime.utcnow()
    window = cfg.get("default_window_days", 7)
    start_ts = (
        pd.to_datetime(args.start_ts)
        if args.start_ts
        else end_ts - timedelta(days=window)
    )

    df = fetch_trade_logs(start_ts, end_ts, table=args.table)
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    redis_client = _get_redis_client() if args.feature_cache_key else None
    df = make_features(
        df,
        redis_client=redis_client,
        cache_key=args.feature_cache_key,
    )
    if "target" not in df.columns:
        raise ValueError("Data must contain a 'target' column for training")

    X = df.drop(columns=["target"])
    y = df["target"]

    model, metrics = train_regime_lgbm(X, y, params, use_gpu=use_gpu)

    preds = model.predict(X)

    if preds.ndim > 1:
        pred_labels = preds.argmax(axis=1) - 1
    else:
        pred_labels = (preds >= 0.5).astype(int)

    bt_params = cfg.get("backtest", {})
    eval_metrics = simulate_signal_pnl(df, pred_labels, **bt_params)

    registry = ModelRegistry(url, key)
    registry.upload(model, "regime_model", {**metrics, **eval_metrics})


if __name__ == "__main__":
    main()


def check_clinfo_gpu() -> bool:
    """Return ``True`` if ``clinfo`` reports a GPU device."""
    exe = shutil.which("clinfo") or shutil.which("rocminfo")
    if not exe:
        logger.warning("clinfo not found; skipping GPU check")
        return False
    try:
        result = subprocess.run([exe], capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - subprocess errors are unlikely
        logger.warning("failed to run %s: %s", exe, exc)
        return False
    output = result.stdout + result.stderr
    if "GPU" in output.upper():
        return True
    logger.warning("No GPU device detected via clinfo")
    return False


def ensure_lightgbm_gpu(
    supabase_url: str, supabase_key: str, script_path: str | None = None
) -> bool:
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

    logger.info("Checking existing LightGBM GPU support")
    try:
        import lightgbm as lgb

        lgb.train(
            {"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0},
            lgb.Dataset([[1.0]], label=[0]),
            num_boost_round=1,
        )
        return False
    except Exception:
        pass

    script = Path(script_path or Path(__file__).with_name("build_lightgbm_gpu.ps1"))
    logger.info("Running %s", script)
    subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        check=True,
    )

    wheel_dir = script.with_name("LightGBM").joinpath("python-package", "dist")
    wheels = glob.glob(str(wheel_dir / "*.whl"))

    sb = create_client(supabase_url, supabase_key)
    bucket = sb.storage.from_("wheels")
    for whl in wheels:
        with open(whl, "rb") as fh:
            bucket.upload(os.path.basename(whl), fh)
        logger.info("Uploaded %s to Supabase", os.path.basename(whl))
    return True


def verify_opencl():
    """Delegates to :func:`opencl_utils.verify_opencl`."""
    from opencl_utils import verify_opencl as _verify

    return _verify()
