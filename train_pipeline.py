"""Utilities for environment setup tasks like building LightGBM."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from config import load_config
from supabase import create_client

from data_loader import fetch_trade_logs, _get_redis_client
from evaluation import simulate_signal_pnl
from sklearn.utils import resample
from feature_engineering import make_features
from utils import validate_schema
from registry import ModelRegistry
from trainers.regime_lgbm import train_regime_lgbm

try:  # optional dependency
    import pyopencl as cl
except ImportError as exc:  # pragma: no cover - pyopencl may be absent
    cl = None  # type: ignore
    logging.getLogger(__name__).warning("pyopencl not available: %s", exc)


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
    parser.add_argument(
        "--no-generate-target",
        dest="generate_target",
        action="store_false",
        help="Do not create the target column automatically",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    params = cfg.get("regime_lgbm", {})

    cfg_env = load_config()
    url = cfg_env.supabase_url
    key = cfg_env.supabase_key or cfg_env.supabase_service_key
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    if check_clinfo_gpu():
        params.setdefault("device", "opencl")
        params.setdefault("gpu_platform_id", 0)
        params.setdefault("gpu_device_id", 0)
        params.pop("device_type", None)
        use_gpu = True
    else:
        params["device"] = "cpu"
        params.pop("device_type", None)
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
    validate_schema(df, ["ts"])

    cache_key = getattr(args, "feature_cache_key", None)
    redis_client = _get_redis_client() if cache_key else None

    feature_kwargs = {}
    if redis_client is not None:
        feature_kwargs.update({"redis_client": redis_client, "cache_key": cache_key})

    df = make_features(df, generate_target=args.generate_target, **feature_kwargs)
    if "target" not in df.columns:
        raise ValueError("Data must contain a 'target' column for training")

    # Balance labels by oversampling each class
    try:
        counts = df["target"].value_counts()
        max_count = counts.max()
        if len(counts) > 1 and max_count > 0:
            frames = [
                resample(g, replace=True, n_samples=max_count, random_state=42)
                for _, g in df.groupby("target")
            ]
            df = (
                pd.concat(frames)
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
    except Exception:
        logging.exception("Failed to balance labels")

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
    """Return ``True`` if an AMD GPU OpenCL device is available."""
    if cl is None:
        logger.warning("pyopencl not installed; skipping GPU check")
        return False

    try:
        platforms = cl.get_platforms()
    except Exception as exc:  # pragma: no cover - OpenCL query failures are rare
        logger.warning("OpenCL detection failed: %s", exc)
        return False

    for plat in platforms:
        names = [getattr(plat, "name", "")]
        try:
            devices = plat.get_devices()
        except Exception:
            devices = []
        names.extend(getattr(dev, "name", "") for dev in devices)
        for name in names:
            lname = str(name).lower()
            if "amd" in lname or "radeon" in lname:
                return True

    logger.warning("No AMD GPU device found")
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
            {"device": "opencl", "gpu_platform_id": 0, "gpu_device_id": 0},
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


def verify_lightgbm_gpu(params: dict) -> bool:
    """Return ``True`` if LightGBM can train using the GPU with ``params``."""
    try:
        from sklearn.datasets import make_classification
        import lightgbm as lgb

        X, y = make_classification(n_samples=10, n_features=5, n_informative=3, random_state=0)
        dataset = lgb.Dataset(X, label=y)
        lgb.train(params, dataset, num_boost_round=1)
        return True
    except Exception as exc:  # pragma: no cover - depends on local hardware
        logger.warning("LightGBM GPU verification failed: %s", exc)
        return False
