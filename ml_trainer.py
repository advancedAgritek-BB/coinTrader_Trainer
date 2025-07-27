"""Command line interface for running coinTrader training tasks."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import platform
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from prometheus_client import Gauge, start_http_server

import numpy as np
import pandas as pd
import yaml
from config import load_config

import historical_data_importer
from data_import import download_historical_data, insert_to_supabase
from train_pipeline import check_clinfo_gpu, verify_lightgbm_gpu

logger = logging.getLogger(__name__)

# expose model accuracy metrics via Prometheus
accuracy_gauge = Gauge("model_accuracy", "Model accuracy")
if __name__ == "__main__":
    start_http_server(8000)

try:
    from trainers.regime_lgbm import train_regime_lgbm
except ImportError:  # pragma: no cover - optional during tests
    train_regime_lgbm = None  # type: ignore


try:  # pragma: no cover - optional dependency
    from federated_trainer import train_federated_regime
except ImportError:  # pragma: no cover - missing during testing
    try:  # pragma: no cover - federated trainer may be optional
        from trainers.federated import train_federated_regime
    except ImportError:  # pragma: no cover - trainer not available
        train_federated_regime = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import federated_fl
except ImportError:  # pragma: no cover - module may be missing
    federated_fl = None  # type: ignore

TRAINERS = {
    "regime": (train_regime_lgbm, "regime_lgbm"),
}


def load_cfg(path: str) -> Dict[str, Any]:
    """Load YAML configuration with sensible defaults."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    for key in ("regime_lgbm", "federated_regime"):
        section = cfg.get(key)
        if isinstance(section, dict):
            section.setdefault("device_type", "gpu")
    cfg.setdefault("backtest", {"slippage": 0.005, "costs": 0.002})
    return cfg


def _make_dummy_data(n: int = 200) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic dataset for local testing."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "price": rng.random(n) * 100,
            "high": rng.random(n) * 100,
            "low": rng.random(n) * 100,
        }
    )
    return df, pd.Series(rng.integers(0, 2, size=n))


def _start_rocm_smi_monitor() -> subprocess.Popen | None:
    """Start a ``rocm-smi`` monitor and log its output."""
    if platform.system() == "Windows":
        logging.info("ROCm SMI monitor not supported on Windows")
        return None
    cmd = ["rocm-smi", "--showuse", "--interval", "1"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - command missing
        logging.warning("Failed to start rocm-smi monitor: %s", exc)
        return None

    def _forward() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            logging.info("rocm-smi: %s", line.rstrip())

    threading.Thread(target=_forward, daemon=True).start()
    logging.info("Started rocm-smi monitor: %s", " ".join(cmd))
    return proc


def _launch_rgp(pid: int) -> None:
    """Launch AMD RGP if available, otherwise print the command."""
    cmd = ["rgp.exe", "--process", str(pid)]
    exe = shutil.which("rgp.exe")
    if exe:
        try:
            subprocess.Popen([exe, "--process", str(pid)])
            print(f"AMD RGP launched: {' '.join(cmd)}")
            return
        except (
            FileNotFoundError,
            OSError,
        ) as exc:  # pragma: no cover - unexpected failures
            logging.warning("Failed to launch AMD RGP: %s", exc)
    print(" ".join(cmd))


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="coinTrader trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("task", help="Task to train, e.g. 'regime'")
    train_p.add_argument("--cfg", default="cfg.yaml", help="Config file path")
    train_p.add_argument("--use-gpu", action="store_true", help="Enable GPU training")
    train_p.add_argument(
        "--gpu-platform-id", type=int, default=None, help="OpenCL platform id"
    )
    train_p.add_argument(
        "--gpu-device-id", type=int, default=None, help="OpenCL device id"
    )
    train_p.add_argument(
        "--swarm",
        action="store_true",
        help="Run hyperparameter swarm search before training",
    )
    train_p.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna hyperparameter search before training",
    )
    train_p.add_argument(
        "--federated",
        action="store_true",
        help="Use federated learning (regime task only)",
    )
    train_p.add_argument(
        "--true-federated",
        action="store_true",
        help="Use Flower-based federated learning",
    )
    train_p.add_argument("--start-ts", help="Data start timestamp (ISO format)")
    train_p.add_argument("--end-ts", help="Data end timestamp (ISO format)")
    train_p.add_argument("--table", default="ohlc_data", help="Supabase table name")
    train_p.add_argument(
        "--profile-gpu", action="store_true", help="Log GPU utilisation via rocm-smi"
    )

    csv_p = sub.add_parser("import-csv", help="Import historical CSV data")
    csv_p.add_argument("csv", help="CSV file path")
    csv_p.add_argument("--symbol", required=True, help="Trading pair symbol")
    csv_p.add_argument("--start-ts", help="Start timestamp (ISO)")
    csv_p.add_argument("--end-ts", help="End timestamp (ISO)")
    csv_p.add_argument(
        "--table",
        help="Supabase table name (defaults to historical_prices_<symbol>)",
        default=None,
    )

    import_p = sub.add_parser(
        "import-data", help="Download historical data and insert to Supabase"
    )
    import_p.add_argument(
        "--source-url", required=True, help="HTTP endpoint for historical data"
    )
    import_p.add_argument("--symbol", required=True, help="Trading pair symbol")
    import_p.add_argument(
        "--start-ts", required=True, help="Data start timestamp (ISO)"
    )
    import_p.add_argument("--end-ts", required=True, help="Data end timestamp (ISO)")
    import_p.add_argument(
        "--output-file", required=True, help="File to store downloaded data"
    )
    import_p.add_argument(
        "--batch-size", type=int, default=1000, help="Insert batch size"
    )

    args = parser.parse_args()

    if args.command == "import-data":
        df = download_historical_data(
            args.source_url,
            output_file=args.output_file,
            symbol=args.symbol,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )
        insert_to_supabase(df, batch_size=args.batch_size)
        return

    if args.command == "import-csv":
        df = historical_data_importer.download_historical_data(
            args.csv,
            symbol=args.symbol,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )
        cfg_env = load_config()
        url = cfg_env.supabase_url
        key = cfg_env.supabase_service_key or cfg_env.supabase_key
        if not url or not key:
            raise SystemExit("SUPABASE_URL and service key must be set")
        table = args.table or f"historical_prices_{args.symbol.lower()}"
        historical_data_importer.insert_to_supabase(
            df,
            url=url,
            key=key,
            table=table,
            symbol=args.symbol,
        )
        return

    cfg = load_cfg(args.cfg)

    if args.command != "train":
        raise SystemExit("Unknown command")

    if args.task not in TRAINERS:
        raise SystemExit(f"Unknown task: {args.task}")

    trainer_fn, cfg_key = TRAINERS[args.task]

    if args.federated:
        if args.task != "regime":
            raise SystemExit("--federated only supported for 'regime' task")
        if train_federated_regime is None:
            raise SystemExit("Federated training not supported")
        trainer_fn = train_federated_regime
        cfg_key = "federated_regime"

    if trainer_fn is None and not args.federated:
        raise SystemExit(f"Trainer '{args.task}' not available, install LightGBM")

    params = cfg.get(cfg_key, {}).copy()

    if args.swarm and args.optuna:
        raise SystemExit("--optuna and --swarm cannot be used together")

    # GPU parameter overrides
    if args.use_gpu:
        params["device_type"] = "gpu"
    if args.gpu_platform_id is not None:
        params["gpu_platform_id"] = args.gpu_platform_id
    if args.gpu_device_id is not None:
        params["gpu_device_id"] = args.gpu_device_id

    if check_clinfo_gpu() and verify_lightgbm_gpu(params):
        params.setdefault("device_type", "gpu")
        params.setdefault("gpu_platform_id", 0)
        params.setdefault("gpu_device_id", 0)
        use_gpu_flag = True
    else:
        params["device_type"] = "cpu"
        logger.warning("GPU not detected; falling back to CPU")
        use_gpu_flag = False

    # Swarm optimisation
    if args.swarm:
        try:
            import swarm_sim
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "--swarm requires the 'swarm_sim' module to be installed"
            ) from exc
        end_ts = datetime.utcnow()
        start_ts = end_ts - timedelta(days=7)
        swarm_params = asyncio.run(
            swarm_sim.run_swarm_search(start_ts, end_ts, table=args.table)
        )
        if isinstance(swarm_params, dict):
            params.update(swarm_params)

    # Optuna optimisation
    if args.optuna:
        try:
            import optuna_search as optuna_mod
        except ImportError:
            try:
                import optuna_optimizer as optuna_mod  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SystemExit(
                    "--optuna requires the 'optuna_optimizer' module to be installed"
                ) from exc

        window = cfg.get("default_window_days", 7)
        defaults = cfg.get("optuna", {})

        run_func = optuna_mod.run_optuna_search
        sig = inspect.signature(run_func)
        param_names = set(sig.parameters.keys())
        if {"start", "end"}.issubset(param_names):
            start = datetime.utcnow() - timedelta(days=window)
            end = datetime.utcnow()
            result = run_func(start, end, table=args.table, **defaults)
        else:
            result = run_func(window, table=args.table, **defaults)

        if inspect.iscoroutine(result):
            result = asyncio.run(result)

        if isinstance(result, dict):
            params.update(result)

    monitor_proc = None
    if args.profile_gpu:
        if platform.system() != "Windows":
            monitor_proc = _start_rocm_smi_monitor()
        _launch_rgp(os.getpid())

    # Training dispatch
    try:
        if args.true_federated:
            if federated_fl is None:
                raise SystemExit("True federated training not supported")
            if not args.start_ts or not args.end_ts:
                raise SystemExit("--true-federated requires --start-ts and --end-ts")
            federated_fl.start_server(
                args.start_ts,
                args.end_ts,
                config_path=args.cfg,
                params_override=params,
                table=args.table,
            )
            return

        if args.federated:
            if not args.start_ts or not args.end_ts:
                raise SystemExit("--federated requires --start-ts and --end-ts")
            model, metrics = asyncio.run(
                trainer_fn(
                    args.start_ts,
                    args.end_ts,
                    config_path=args.cfg,
                    params_override=params,
                    table=args.table,
                )
            )
        else:
            X, y = _make_dummy_data()
            model, metrics = trainer_fn(
                X,
                y,
                params,
                use_gpu=use_gpu_flag,
                profile_gpu=args.profile_gpu,
            )  # type: ignore[arg-type]

        print("Training completed. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        accuracy_gauge.set(metrics.get("accuracy", 0))
    finally:
        if monitor_proc:
            monitor_proc.terminate()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    asyncio.run(main())
