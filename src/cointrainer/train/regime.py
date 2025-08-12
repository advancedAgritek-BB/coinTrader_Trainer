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
from cointrainer.registry import save_model
from cointrainer.features.build import build_features, make_labels


def _load_data(data: pd.DataFrame | str) -> pd.DataFrame:
    """Return ``data`` as a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def train_regime_model(
    data: pd.DataFrame | str,
    *,
    use: dict | None = None,
    params: dict | None = None,
    horizon: str = "1",
    thresholds: dict | None = None,
    use_gpu: bool = True,
    model_name: str = "regime_model",
    registry: Any | None = None,
    symbol: str = "BTCUSDT",
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier for market regimes and save it."""

    df = _load_data(data)
    X, feat_meta = build_features(df, use=use or {}, params=params or {})
    y, label_meta = make_labels(df, horizon=horizon, thresholds=thresholds or {})
    X, y = X.align(y, join="inner")

    lgb_params: dict[str, Any] = {"objective": "binary"}
    if use_gpu:
        lgb_params["device"] = "gpu"

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X, y)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    key = f"models/regime/{symbol}/{ts}_model.pkl"
    blob = pickle.dumps(model)
    save_model(key, blob, {"symbol": symbol, "horizon": horizon})
    print(f"models/regime/{symbol}/LATEST.json")


__all__ = ["run"]
    metadata = {**feat_meta, **label_meta}
    save_model(key, blob, metadata)
    return model


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for training the regime model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to CSV containing training data")
    parser.add_argument("--model-name", default="regime_model", help="Name used when uploading")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args(argv)

    train_regime_model(args.data, use_gpu=not args.cpu, model_name=args.model_name)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["train_regime_model", "main"]
