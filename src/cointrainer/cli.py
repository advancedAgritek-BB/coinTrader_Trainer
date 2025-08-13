from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _cmd_train_regime(args: argparse.Namespace) -> None:
    from cointrainer.train import regime as regime_mod

    regime_mod.run(
        symbol=args.symbol,
        horizon=args.horizon,
        use_gpu=args.use_gpu,
        optuna=args.optuna,
        federated=args.federated,
        true_federated=args.true_federated,
        config=args.config,
    )


def _cmd_import_data_supabase(args: argparse.Namespace) -> None:

    from cointrainer.data.loader import fetch_trade_logs

    df = fetch_trade_logs(args.start, args.end, args.symbol, table=args.table)
    df.to_parquet(args.out)


def _cmd_import_csv(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_csv(args.file)
    df.to_parquet(args.out)


def _cmd_import_csv7(args: argparse.Namespace) -> None:
    from cointrainer.io.csv7 import read_csv7

    df = read_csv7(args.file)
    prefix = Path(args.out) if args.out else Path(f"{args.symbol}_1m")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    norm_csv = prefix.with_suffix(".normalized.csv")
    df.to_csv(norm_csv, index=True)
    print(
        f"Wrote normalized CSV: {norm_csv}  rows={len(df):,}  range={df.index[0]} .. {df.index[-1]}"
    )
    try:
        df.to_parquet(prefix.with_suffix(".parquet"))
        print(f"Wrote Parquet:       {prefix.with_suffix('.parquet')}")
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"Parquet not written (install pyarrow or fastparquet): {e}")


def _cmd_csv_train(args: argparse.Namespace) -> None:
    import pandas as pd

    from cointrainer.train.local_csv import (
        FEATURE_LIST,
        TrainConfig,
        _maybe_publish_registry,
        _save_local,
        make_features,
        make_labels,
        train_from_csv7,
    )

    cfg = TrainConfig(
        symbol=args.symbol,
        horizon=args.horizon,
        hold=args.hold,
        publish_to_registry=args.publish,
    )
    # Try CSV7 path first
    try:
        train_from_csv7(args.file, cfg)
        print("Training completed from CSV7 source.")
        return
    except Exception:
        # second chance: try normalized CSV (OHLCV header)
        df = pd.read_csv(args.file, parse_dates=[0], index_col=0)
        df.index.name = "ts"
        df = df.sort_index()
        X = make_features(df).dropna()
        y = make_labels(df.loc[X.index, "close"], cfg.horizon, cfg.hold).dropna()
        m = y.index.intersection(X.index)
        X = X.loc[m]
        y = y.loc[m]
        model = None
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                objective="multiclass",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ).fit(X, y)
        except Exception as e:  # pragma: no cover - optional dependency
            raise SystemExit(
                f"LightGBM not available for normalized CSV path: {e}"
            ) from e
        # Save local + optional registry
        meta = {
            "schema_version": "1",
            "feature_list": FEATURE_LIST,
            "label_order": [-1, 0, 1],
            "horizon": f"{cfg.horizon}m",
            "thresholds": {"hold": cfg.hold},
            "symbol": cfg.symbol,
        }
        path = _save_local(model, cfg, meta)
        try:
            import io

            import joblib

            buf = io.BytesIO()
            joblib.dump(model, buf)
            _maybe_publish_registry(buf.getvalue(), meta, cfg)
        except Exception:
            pass
        print(f"Training completed from normalized CSV. Model: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cointrainer")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_sub = train_parser.add_subparsers(dest="trainer")

    regime = train_sub.add_parser("regime")
    regime.add_argument("--symbol", default="BTCUSDT")
    regime.add_argument("--optuna", action="store_true")
    regime.add_argument("--use-gpu", action="store_true")
    group = regime.add_mutually_exclusive_group()
    group.add_argument("--federated", action="store_true")
    group.add_argument("--true-federated", action="store_true")
    regime.add_argument("--horizon", default="15m")
    regime.add_argument("--config")
    regime.set_defaults(func=_cmd_train_regime)

    import_data = subparsers.add_parser("import-data")
    import_sub = import_data.add_subparsers(dest="source")

    supabase = import_sub.add_parser("supabase")
    supabase.add_argument("--table", default="trade_logs")
    supabase.add_argument("--start")
    supabase.add_argument("--end")
    supabase.add_argument("--symbol")
    supabase.add_argument("--out", required=True)
    supabase.set_defaults(func=_cmd_import_data_supabase)

    import_csv = subparsers.add_parser("import-csv")
    import_csv.add_argument("--file", required=True)
    import_csv.add_argument("--out", required=True)
    import_csv.set_defaults(func=_cmd_import_csv)

    import_csv7 = subparsers.add_parser(
        "import-csv7",
        help="Ingest headerless 7-col CSV (ts,o,h,l,c,v,trades)",
    )
    import_csv7.add_argument("--file", required=True, help="Path to the source CSV")
    import_csv7.add_argument("--symbol", default="XRPUSD")
    import_csv7.add_argument(
        "--out", default=None, help="Output prefix (e.g., data\\XRPUSD_1m)"
    )
    import_csv7.set_defaults(func=_cmd_import_csv7)

    csv_train = subparsers.add_parser(
        "csv-train",
        help="Train a regime model from CSV7 or normalized CSV",
    )
    csv_train.add_argument(
        "--file",
        required=True,
        help="CSV7 (7-col) or normalized CSV with OHLCV(+trades)",
    )
    csv_train.add_argument("--symbol", default="XRPUSD")
    csv_train.add_argument("--horizon", type=int, default=15)
    csv_train.add_argument("--hold", type=float, default=0.0015)
    csv_train.add_argument(
        "--publish", action="store_true", help="Publish to registry if configured"
    )
    csv_train.set_defaults(func=_cmd_csv_train)

    args = parser.parse_args(argv)

    if args.version:
        try:
            print(version("cointrader-trainer"))
        except PackageNotFoundError:
            print("0.1.0")
        return

    if hasattr(args, "func"):
        args.func(args)
        return

    parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
