import argparse
import yaml
import numpy as np
import pandas as pd

from trainers.regime_lgbm import train_regime_lgbm

TRAINERS = {
    "regime": (train_regime_lgbm, "regime_lgbm"),
}

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

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

def main() -> None:
    parser = argparse.ArgumentParser(description="coinTrader trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("task", help="Task to train, e.g. 'regime'")
    train_p.add_argument("--cfg", default="cfg.yaml", help="Config file path")

    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    if args.command == "train":
        if args.task not in TRAINERS:
            raise SystemExit(f"Unknown task: {args.task}")
        trainer_fn, cfg_key = TRAINERS[args.task]
        params = cfg.get(cfg_key, {})
        X, y = _make_dummy_data()
        model, metrics = trainer_fn(X, y, params)
        print("Training completed. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
