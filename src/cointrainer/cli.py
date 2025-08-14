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
        _fit_model,
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
        outdir=Path(args.outdir),
        write_predictions=args.write_predictions,
        device_type=args.device_type,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        max_bin=args.max_bin,
        gpu_use_dp=args.gpu_use_dp,
        n_jobs=args.n_jobs,
    )
    print(
        f"[train] device={cfg.device_type} max_bin={cfg.max_bin} "
        f"gpu_platform_id={cfg.gpu_platform_id if cfg.gpu_platform_id is not None else -1} "
        f"gpu_device_id={cfg.gpu_device_id if cfg.gpu_device_id is not None else -1}"
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
            model = _fit_model(X, y, cfg)
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


def _cmd_csv_train_batch(args: argparse.Namespace) -> None:
    """Train regime models from a directory of CSV files."""
    import json
    from pathlib import Path

    import pandas as pd

    from cointrainer.train.local_csv import (
        FEATURE_LIST,
        TrainConfig,
        _fit_model,
        _maybe_publish_registry,
        _save_local,
        make_features,
        make_labels,
        train_from_csv7,
    )
    from cointrainer.utils.batch import (
        derive_symbol,
        is_csv7,
        is_normalized_csv,
        iter_csv_files,
    )

    files = iter_csv_files(args.folder, glob=args.glob, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No CSV files matched {args.glob} in {args.folder}")

    summary = []
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for f in files:
        symbol = derive_symbol(f, mode=args.symbol_from, fixed=args.symbol)
        cfg = TrainConfig(
            symbol=symbol,
            horizon=args.horizon,
            hold=args.hold,
            publish_to_registry=args.publish,
            outdir=outdir,
            device_type=args.device_type,
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            max_bin=args.max_bin,
            gpu_use_dp=args.gpu_use_dp,
            n_jobs=args.n_jobs,
        )
        print(
            f"[train] device={cfg.device_type} max_bin={cfg.max_bin} "
            f"gpu_platform_id={cfg.gpu_platform_id if cfg.gpu_platform_id is not None else -1} "
            f"gpu_device_id={cfg.gpu_device_id if cfg.gpu_device_id is not None else -1}"
        )
        item = {"file": str(f), "symbol": symbol, "status": "ok", "model": None, "error": None}
        try:
            if is_csv7(f):
                train_from_csv7(f, cfg, limit_rows=args.limit_rows)
                item["model"] = str(outdir / f"{symbol.lower()}_regime_lgbm.pkl")
            elif is_normalized_csv(f):
                df = pd.read_csv(f, parse_dates=[0], index_col=0).sort_index()
                if args.limit_rows:
                    df = df.tail(int(args.limit_rows))
                X = make_features(df).dropna()
                y = make_labels(df.loc[X.index, "close"], cfg.horizon, cfg.hold).dropna()
                m = y.index.intersection(X.index)
                X = X.loc[m]
                y = y.loc[m]
                model = _fit_model(X, y, cfg)
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
                item["model"] = str(path)
            else:
                raise RuntimeError(
                    "Unrecognized CSV format (not CSV7 and not normalized OHLCV)."
                )
        except Exception as e:  # pragma: no cover - defensive
            item["status"] = "error"
            item["error"] = str(e)
        summary.append(item)
        print(
            f"[{item['status'].upper()}] {symbol} <- {f}  "
            f"{('model='+item['model']) if item['model'] else ''} "
            f"{('err='+item['error']) if item['error'] else ''}"
        )

    (outdir / "batch_train_summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for s in summary if s["status"] == "ok")
    fail = len(summary) - ok
    print(
        f"\nBatch finished: {ok} ok / {fail} failed. Summary: {outdir / 'batch_train_summary.json'}"
    )


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
    csv_train.add_argument(
        "--outdir", default="local_models", help="Where to write model/predictions"
    )
    csv_train.add_argument(
        "--write-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write predictions CSV alongside model (disable with --no-write-predictions)",
    )
    csv_train.add_argument(
        "--device-type", default="gpu", choices=["cpu", "gpu", "cuda"]
    )
    csv_train.add_argument("--gpu-platform-id", type=int, default=None)
    csv_train.add_argument("--gpu-device-id", type=int, default=None)
    csv_train.add_argument("--max-bin", type=int, default=63)
    csv_train.add_argument("--gpu-use-dp", action="store_true")
    csv_train.add_argument("--n-jobs", type=int, default=None)
    csv_train.set_defaults(func=_cmd_csv_train)

    csv_train_batch = subparsers.add_parser(
        "csv-train-batch",
        help="Train a model for each CSV in a folder (CSV7 or normalized).",
    )
    csv_train_batch.add_argument(
        "--folder", required=True, help="Folder containing CSV files"
    )
    csv_train_batch.add_argument(
        "--glob", default="*.csv", help="Glob pattern (default: *.csv)"
    )
    csv_train_batch.add_argument(
        "--recursive", action="store_true", help="Recurse into subfolders"
    )
    csv_train_batch.add_argument(
        "--symbol-from",
        default="filename",
        choices=["filename", "parent", "fixed"],
        help="How to derive symbol",
    )
    csv_train_batch.add_argument(
        "--symbol", default=None, help="If --symbol-from fixed, use this value"
    )
    csv_train_batch.add_argument(
        "--horizon", type=int, default=15, help="Label horizon in bars"
    )
    csv_train_batch.add_argument(
        "--hold", type=float, default=0.0015, help="Hold band (e.g., 0.0015)"
    )
    csv_train_batch.add_argument(
        "--publish", action="store_true", help="Publish to registry if configured"
    )
    csv_train_batch.add_argument(
        "--outdir", default="local_models", help="Where to write model/predictions"
    )
    csv_train_batch.add_argument(
        "--limit-rows", type=int, default=None,
        help="Limit to last N rows per file (optional)",
    )
    csv_train_batch.add_argument(
        "--device-type", default="gpu", choices=["cpu", "gpu", "cuda"]
    )
    csv_train_batch.add_argument("--gpu-platform-id", type=int, default=None)
    csv_train_batch.add_argument("--gpu-device-id", type=int, default=None)
    csv_train_batch.add_argument("--max-bin", type=int, default=63)
    csv_train_batch.add_argument("--gpu-use-dp", action="store_true")
    csv_train_batch.add_argument("--n-jobs", type=int, default=None)
    csv_train_batch.set_defaults(func=_cmd_csv_train_batch)

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
