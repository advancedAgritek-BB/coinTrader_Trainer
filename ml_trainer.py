"""Command line interface for running coinTrader training tasks."""

import argparse
import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

from trainers.regime_lgbm import train_regime_lgbm

try:  # pragma: no cover - optional dependency
    from federated_trainer import train_federated_regime
except Exception:  # pragma: no cover - during tests this may not be available
    try:
        from trainers.federated import train_federated_regime
    except Exception:
        train_federated_regime = None


TRAINERS = {
    "regime": (train_regime_lgbm, "regime_lgbm"),
}


def load_cfg(path: str) -> dict:
    """Load YAML configuration file and apply defaults."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    regime_cfg = cfg.get("regime_lgbm")
    if isinstance(regime_cfg, dict):
        regime_cfg.setdefault("device_type", "gpu")

    fed_cfg = cfg.get("federated_regime")
    if isinstance(fed_cfg, dict):
        fed_cfg.setdefault("device_type", "gpu")

    return cfg


def _make_dummy_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "price": rng.random(n) * 100,
        "high": rng.random(n) * 100,
        "low": rng.random(n) * 100,
    })
    X = df
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


def main() -> None:  # pragma: no cover - exercised via CLI tests
    parser = argparse.ArgumentParser(description="coinTrader trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("task")
    train_p.add_argument("--cfg", default="cfg.yaml")
    train_p.add_argument("--use-gpu", action="store_true")
    train_p.add_argument("--gpu-platform-id", type=int, default=None)
    train_p.add_argument("--gpu-device-id", type=int, default=None)
    train_p.add_argument("--swarm", action="store_true")
    train_p.add_argument("--federated", action="store_true")
    train_p.add_argument("--start-ts")
    train_p.add_argument("--end-ts")

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    if args.command == "train":
        if args.task not in TRAINERS:
            raise SystemExit(f"Unknown task: {args.task}")
        trainer_fn, cfg_key = TRAINERS[args.task]

        if args.federated and args.task == "regime":
            if train_federated_regime is None:
                raise SystemExit("Federated training not supported")
            trainer_fn = train_federated_regime
            params = cfg.get("federated_regime", {}).copy()
        else:
            params = cfg.get(cfg_key, {}).copy()

        if args.use_gpu:
            params["device_type"] = "gpu"
        if args.gpu_platform_id is not None:
            params["gpu_platform_id"] = args.gpu_platform_id
        if args.gpu_device_id is not None:
            params["gpu_device_id"] = args.gpu_device_id

        if args.swarm:
            try:
                import swarm_sim
            except Exception as exc:
                raise SystemExit(
                    "--swarm requires the 'swarm_sim' module to be installed"
                ) from exc
            end_ts = datetime.utcnow()
            start_ts = end_ts - timedelta(days=7)
            swarm_params = asyncio.run(swarm_sim.run_swarm_search(start_ts, end_ts))
            if isinstance(swarm_params, dict):
                params.update(swarm_params)

        X, y = _make_dummy_data()
        trainer_fn(X, y, params, use_gpu=args.use_gpu)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
