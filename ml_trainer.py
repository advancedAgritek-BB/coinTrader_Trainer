"""Command line interface for running coinTrader training tasks."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple
import os
from dotenv import load_dotenv
import subprocess


import numpy as np
import pandas as pd
import yaml

load_dotenv()

try:
    from trainers.regime_lgbm import train_regime_lgbm
except Exception:  # pragma: no cover - optional during tests
    train_regime_lgbm = None  # type: ignore

from data_import import download_historical_data, insert_to_supabase
import historical_data_importer

try:  # pragma: no cover - optional dependency
    from federated_trainer import train_federated_regime
except Exception:  # pragma: no cover - missing during testing
    try:  # pragma: no cover - federated trainer may be optional
        from trainers.federated import train_federated_regime
    except Exception:  # pragma: no cover - trainer not available
        train_federated_regime = None  # type: ignore

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


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="coinTrader trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("task", help="Task to train, e.g. 'regime'")
    train_p.add_argument("--cfg", default="cfg.yaml", help="Config file path")
    train_p.add_argument("--use-gpu", action="store_true", help="Enable GPU training")
    train_p.add_argument("--gpu-platform-id", type=int, default=None, help="OpenCL platform id")
    train_p.add_argument("--gpu-device-id", type=int, default=None, help="OpenCL device id")
    train_p.add_argument("--swarm", action="store_true", help="Run hyperparameter swarm search before training")
    train_p.add_argument("--federated", action="store_true", help="Use federated learning (regime task only)")
    train_p.add_argument("--start-ts", help="Data start timestamp (ISO format)")
    train_p.add_argument("--end-ts", help="Data end timestamp (ISO format)")
    train_p.add_argument("--profile-gpu", action="store_true", help="Profile GPU usage with AMD RGP")

    csv_p = sub.add_parser("import-csv", help="Import historical CSV data")
    csv_p.add_argument("csv", help="CSV file path")
    csv_p.add_argument("--start-ts", help="Start timestamp (ISO)")
    csv_p.add_argument("--end-ts", help="End timestamp (ISO)")
    csv_p.add_argument(
        "--table",
        default="historical_prices",
        help="Supabase table name",
    )

    import_p = sub.add_parser("import-data", help="Download historical data and insert to Supabase")
    import_p.add_argument("--source-url", required=True, help="HTTP endpoint for historical data")
    import_p.add_argument("--symbol", required=True, help="Trading pair symbol")
    import_p.add_argument("--start-ts", required=True, help="Data start timestamp (ISO)")
    import_p.add_argument("--end-ts", required=True, help="Data end timestamp (ISO)")
    import_p.add_argument("--output-file", required=True, help="File to store downloaded data")
    import_p.add_argument("--batch-size", type=int, default=1000, help="Insert batch size")

    args = parser.parse_args()

    if args.command == "import-data":
        df = download_historical_data(
            args.source_url,
            args.symbol,
            args.start_ts,
            args.end_ts,
            batch_size=args.batch_size,
            output_file=args.output_file,
        )
        insert_to_supabase(df, batch_size=args.batch_size)
        return

    if args.command == "import-csv":
        df = historical_data_importer.download_historical_data(
            args.csv,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise SystemExit("SUPABASE_URL and service key must be set")
        historical_data_importer.insert_to_supabase(df, url, key, table=args.table)
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

    params = cfg.get(cfg_key, {}).copy()

    # GPU parameter overrides
    if args.use_gpu:
        params["device_type"] = "gpu"
    if args.gpu_platform_id is not None:
        params["gpu_platform_id"] = args.gpu_platform_id
    if args.gpu_device_id is not None:
        params["gpu_device_id"] = args.gpu_device_id

    # Swarm optimisation
    if args.swarm:
        try:
            import swarm_sim
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit("--swarm requires the 'swarm_sim' module to be installed") from exc
        end_ts = datetime.utcnow()
        start_ts = end_ts - timedelta(days=7)
        swarm_params = asyncio.run(swarm_sim.run_swarm_search(start_ts, end_ts))
        if isinstance(swarm_params, dict):
            params.update(swarm_params)

    # Training dispatch
    if args.profile_gpu:
        cmd = ["rgp.exe", "--process", str(os.getpid())]
        try:
            subprocess.Popen(cmd)
            print("Started AMD RGP profiler:", " ".join(cmd))
        except Exception:
            print("GPU profiling enabled. Run: {}".format(" ".join(cmd)))

    if args.federated:
        if not args.start_ts or not args.end_ts:
            raise SystemExit("--federated requires --start-ts and --end-ts")
        model, metrics = trainer_fn(  # type: ignore[assignment]
            args.start_ts,
            args.end_ts,
            config_path=args.cfg,
            params_override=params,
        )
    else:
        X, y = _make_dummy_data()
        model, metrics = trainer_fn(
            X,
            y,
            params,
            use_gpu=args.use_gpu,
            profile_gpu=args.profile_gpu,
        )  # type: ignore[arg-type]

    print("Training completed. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
