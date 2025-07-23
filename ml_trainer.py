"""Command line interface for running coinTrader training tasks."""

import argparse
import yaml
import numpy as np
import pandas as pd

from trainers.regime_lgbm import train_regime_lgbm
try:  # pragma: no cover - federated trainer may be optional
    from trainers.federated import train_federated_regime
except Exception:  # pragma: no cover - during testing trainer might be missing
    train_federated_regime = None
try:  # Federated training may be optional
    from trainers.federated_regime import train_federated_regime
except Exception:  # pragma: no cover - trainer not available
    train_federated_regime = None
from trainers.regime_lgbm import train_regime_lgbm, train_federated_regime

TRAINERS = {
    "regime": (train_regime_lgbm, "regime_lgbm"),
}

def load_cfg(path: str) -> dict:
    """Load configuration from a YAML file and apply defaults.

    Parameters
    ----------
    path : str
        Path to a ``.yaml`` or ``.yml`` configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary with defaults applied.  If the file
        is empty an empty dictionary is returned.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Ensure LightGBM trainer defaults to GPU when not specified in the config
    regime_cfg = cfg.get("regime_lgbm")
    if isinstance(regime_cfg, dict):
        regime_cfg.setdefault("device_type", "gpu")

    fed_cfg = cfg.get("federated_regime")
    if isinstance(fed_cfg, dict):
        fed_cfg.setdefault("device_type", "gpu")

    return cfg

def _make_dummy_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic dataset for demonstration purposes."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "price": rng.random(n) * 100,
        "high": rng.random(n) * 100,
        "low": rng.random(n) * 100,
    })
    X = df
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y

def main() -> None:
    """Entry point for the ``coinTrainer`` command line interface."""
    parser = argparse.ArgumentParser(description="coinTrader trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("task", help="Task to train, e.g. 'regime'")
    train_p.add_argument("--cfg", default="cfg.yaml", help="Config file path")
    train_p.add_argument("--use-gpu", action="store_true", help="Enable GPU training")
    train_p.add_argument("--gpu-platform-id", type=int, default=None, help="OpenCL platform id")
    train_p.add_argument("--gpu-device-id", type=int, default=None, help="OpenCL device id")
    train_p.add_argument("--federated", action="store_true", help="Use federated trainer")
    train_p.add_argument("--federated", action="store_true", help="Use federated training")
    train_p.add_argument(
        "--federated",
        action="store_true",
        help="Use federated learning when training the 'regime' task",
    )

    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    if args.command == "train":
        if args.task not in TRAINERS:
            raise SystemExit(f"Unknown task: {args.task}")
        trainer_fn, cfg_key = TRAINERS[args.task]
        if args.federated and args.task == "regime" and train_federated_regime is not None:
            trainer_fn = train_federated_regime
            cfg_key = "regime_lgbm"
        if args.federated:
            if train_federated_regime is None:
                raise SystemExit("Federated training not supported")
            trainer_fn = train_federated_regime
            params = cfg.get("federated_regime", {})
        else:
            params = cfg.get(cfg_key, {})
        if args.federated and args.task == "regime":
            trainer_fn = train_federated_regime
        params = cfg.get(cfg_key, {})
        if args.gpu_platform_id is not None:
            params["gpu_platform_id"] = args.gpu_platform_id
        if args.gpu_device_id is not None:
            params["gpu_device_id"] = args.gpu_device_id
        X, y = _make_dummy_data()
        model, metrics = trainer_fn(X, y, params, use_gpu=args.use_gpu)
        print("Training completed. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
