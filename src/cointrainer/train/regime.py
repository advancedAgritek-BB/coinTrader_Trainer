"""Training entry point for the regime model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional


def run(
    symbol: str,
    horizon: str,
    use_gpu: bool,
    optuna: bool,
    federated: bool,
    true_federated: bool,
    config: str | None,
) -> None:
    """Train the regime classifier and store it via the registry.

    Parameters
    ----------
    symbol:
        Trading pair identifier used in the model path.
    horizon:
        Candle horizon string such as ``"15m"``.
    use_gpu:
        Enable GPU feature generation and model training.
    optuna:
        When ``True`` hyperparameters are tuned with Optuna.
    federated / true_federated:
        Select experimental federated training modes. These options are
        mutually exclusive and a :class:`ValueError` is raised when both are
        ``True``.
    config:
        Optional path to a YAML configuration file containing LightGBM
        parameters under ``regime_lgbm``.
    """

    if federated and true_federated:
        raise ValueError("--federated and --true-federated cannot be used together")

    import pickle
    import numpy as np
    import pandas as pd
    import yaml

    from cointrainer.features.build import make_features
    from cointrainer.registry import save_model

    params: dict = {}
    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        params = cfg.get("regime_lgbm", {})

    rng = pd.date_range("2023-01-01", periods=64, freq=horizon)
    price = np.linspace(100, 101, len(rng))
    df = pd.DataFrame({
        "ts": rng,
        "price": price,
        "high": price + 1,
        "low": price - 1,
    })
    df = make_features(df, use_gpu=use_gpu)
    X = df.drop(columns=["target"])
    y = df["target"]

    if true_federated:
        from trainers.federated import train_federated_regime as _true_fed

        model, _ = _true_fed(df, None, params, use_gpu=use_gpu)
    elif federated:
        from trainers.regime_lgbm import train_federated_regime as _fed

        model, _ = _fed(X, y, params, use_gpu=use_gpu)
    else:
        from trainers.regime_lgbm import train_regime_lgbm

        model, _ = train_regime_lgbm(X, y, params, use_gpu=use_gpu, tune=optuna)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    key = f"models/regime/{symbol}/{ts}_model.pkl"
    blob = pickle.dumps(model)
    save_model(key, blob, {"symbol": symbol, "horizon": horizon})
    print(f"models/regime/{symbol}/LATEST.json")


__all__ = ["run"]
