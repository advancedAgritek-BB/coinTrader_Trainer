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
        publish=args.publish,
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
            key = _maybe_publish_registry(buf.getvalue(), meta, cfg)
            if key:
                print(f"[publish] Uploaded: {key}")
                print(f"[publish] Pointer:  {key.rsplit('/',1)[0]}/LATEST.json")
            elif cfg.publish_to_registry:
                print("[publish] Skipped (no registry configured or --publish not set)")
        except Exception as e:
            print(f"[publish] ERROR: {e}")
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
        item = {
            "file": str(f),
            "symbol": symbol,
            "status": "ok",
            "model": None,
            "pointer": None,
            "error": None,
        }
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
                    key = _maybe_publish_registry(buf.getvalue(), meta, cfg)
                    if key:
                        print(f"[publish] Uploaded: {key}")
                        print(f"[publish] Pointer:  {key.rsplit('/',1)[0]}/LATEST.json")
                    elif cfg.publish_to_registry:
                        print("[publish] Skipped (no registry configured or --publish not set)")
                except Exception as e:
                    print(f"[publish] ERROR: {e}")
                item["model"] = str(path)
            else:
                raise RuntimeError(
                    "Unrecognized CSV format (not CSV7 and not normalized OHLCV)."
                )
        except Exception as e:  # pragma: no cover - defensive
            item["status"] = "error"
            item["error"] = str(e)
        prefix = f"models/regime/{symbol}"
        if args.publish:
            item["pointer"] = f"{prefix}/LATEST.json"
        summary.append(item)
        print(
            f"[{item['status'].upper()}] {symbol} <- {f}  "
            f"model={item['model']}  pointer={item['pointer'] or '-'}"
        )

    (outdir / "batch_train_summary.json").write_text(json.dumps(summary, indent=2))
    ok_count = sum(s["status"] == "ok" for s in summary)
    fail_count = len(summary) - ok_count
    print(f"\nBatch finished: {ok_count} ok / {fail_count} failed.")
    print(f"Summary: {outdir / 'batch_train_summary.json'}")


def _cmd_csv_train_aggregate(args: argparse.Namespace) -> None:
    from cointrainer.train.global_model import GlobalTrainConfig, train_aggregate

    cfg = GlobalTrainConfig(
        horizon=args.horizon,
        hold=args.hold,
        outdir=Path(args.outdir),
        publish_to_registry=args.publish,
        global_symbol=args.global_symbol,
        per_pair=args.per_pair,
        limit_rows_per_file=args.limit_rows_per_file,
        cap_rows_per_pair=args.cap_rows_per_pair,
        max_total_rows=args.max_total_rows,
        downsample_flat=args.downsample_flat,
        device_type=args.device_type,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        max_bin=args.max_bin,
        gpu_use_dp=args.gpu_use_dp,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    train_aggregate(Path(args.folder), args.glob, args.recursive, cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cointrainer")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="cmd")

    train_parser = sub.add_parser("train")
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
    regime.add_argument(
        "--publish", action="store_true", help="Publish to registry if configured"
    )
    regime.set_defaults(func=_cmd_train_regime)

    import_data = sub.add_parser("import-data")
    import_sub = import_data.add_subparsers(dest="source")

    supabase = import_sub.add_parser("supabase")
    supabase.add_argument("--table", default="trade_logs")
    supabase.add_argument("--start")
    supabase.add_argument("--end")
    supabase.add_argument("--symbol")
    supabase.add_argument("--out", required=True)
    supabase.set_defaults(func=_cmd_import_data_supabase)

    import_csv = sub.add_parser("import-csv")
    import_csv.add_argument("--file", required=True)
    import_csv.add_argument("--out", required=True)
    import_csv.set_defaults(func=_cmd_import_csv)

    import_csv7 = sub.add_parser(
        "import-csv7",
        help="Ingest headerless 7-col CSV (ts,o,h,l,c,v,trades)",
    )
    import_csv7.add_argument("--file", required=True, help="Path to the source CSV")
    import_csv7.add_argument("--symbol", default="XRPUSD")
    import_csv7.add_argument(
        "--out", default=None, help="Output prefix (e.g., data\\XRPUSD_1m)"
    )
    import_csv7.set_defaults(func=_cmd_import_csv7)

    csv_train = sub.add_parser(
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

    csv_train_batch = sub.add_parser(
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

    csv_train_agg = sub.add_parser(
        "csv-train-aggregate",
        help="Aggregate many CSVs into one training run (global or per-pair).",
    )
    csv_train_agg.add_argument(
        "--folder", required=True, help="Folder containing CSV files"
    )
    csv_train_agg.add_argument(
        "--glob", default="*.csv", help="Glob pattern (default: *.csv)"
    )
    csv_train_agg.add_argument(
        "--recursive", action="store_true", help="Recurse into subfolders"
    )
    csv_train_agg.add_argument(
        "--per-pair", action="store_true", help="Train one model per pair"
    )
    csv_train_agg.add_argument(
        "--global-symbol", default="GLOBAL", help="Symbol for global model"
    )
    csv_train_agg.add_argument(
        "--limit-rows-per-file", type=int, default=None,
        help="Tail rows per file (optional)",
    )
    csv_train_agg.add_argument(
        "--cap-rows-per-pair", type=int, default=None,
        help="Cap rows per pair (optional)",
    )
    csv_train_agg.add_argument(
        "--max-total-rows", type=int, default=None,
        help="Max total rows across all files",
    )
    csv_train_agg.add_argument(
        "--downsample-flat", type=float, default=None,
        help="Fraction of y==0 rows to keep",
    )
    csv_train_agg.add_argument(
        "--horizon", type=int, default=15, help="Label horizon in bars"
    )
    csv_train_agg.add_argument(
        "--hold", type=float, default=0.0015, help="Hold band (e.g., 0.0015)"
    )
    csv_train_agg.add_argument(
        "--publish", action="store_true", help="Publish to registry if configured"
    )
    csv_train_agg.add_argument(
        "--outdir", default="local_models", help="Where to write model(s)"
    )
    csv_train_agg.add_argument(
        "--device-type", default="gpu", choices=["cpu", "gpu", "cuda"]
    )
    csv_train_agg.add_argument("--gpu-platform-id", type=int, default=None)
    csv_train_agg.add_argument("--gpu-device-id", type=int, default=None)
    csv_train_agg.add_argument("--max-bin", type=int, default=63)
    csv_train_agg.add_argument("--gpu-use-dp", action="store_true")
    csv_train_agg.add_argument("--n-jobs", type=int, default=0)
    csv_train_agg.add_argument("--random-state", type=int, default=42)
    csv_train_agg.set_defaults(func=_cmd_csv_train_aggregate)

    ab = sub.add_parser(
        "autobacktest",
        help="Continuous loop: retrain+backtest on updates, optional publish.",
    )
    ab.add_argument("--file", required=True)
    ab.add_argument("--symbol", required=True)
    ab.add_argument("--horizon", type=int, default=15)
    ab.add_argument("--hold", type=float, default=0.0015)
    ab.add_argument("--open-thr", type=float, default=0.55)
    ab.add_argument("--interval-sec", type=int, default=900)
    ab.add_argument("--limit-rows", type=int, default=1000000)
    ab.add_argument("--fee-bps", type=float, default=2.0)
    ab.add_argument("--slip-bps", type=float, default=0.0)
    ab.add_argument("--device-type", default="gpu", choices=["cpu","gpu","cuda"])
    ab.add_argument("--max-bin", type=int, default=63)
    ab.add_argument("--n-jobs", type=int, default=0)
    ab.add_argument("--publish", action="store_true")
    ab.add_argument("--outdir", default="out/autobacktest")
    op = sub.add_parser(
        "optimize",
        help="Optimize training + signal policy using Optuna (fallback to grid if Optuna missing).",
    )
    op.add_argument("--file", required=True)
    op.add_argument("--symbol", required=True)
    op.add_argument("--n-trials", type=int, default=100)
    op.add_argument("--folds", type=int, default=4)
    op.add_argument("--val-len", type=int, default=100000)
    op.add_argument("--gap", type=int, default=500)
    op.add_argument("--limit-rows", type=int, default=800000)
    op.add_argument("--fee-bps", type=float, default=2.0)
    op.add_argument("--slip-bps", type=float, default=0.0)
    op.add_argument("--device-type", default="gpu", choices=["cpu","gpu","cuda"])
    op.add_argument("--max-bin", type=int, default=63)
    op.add_argument("--n-jobs", type=int, default=0)
    op.add_argument("--seed", type=int, default=42)
    op.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (e.g., sqlite:///out/opt/studies/SYMBOL.db)",
    )
    op.add_argument(
        "--study-name", default=None, help="Study name (defaults to SYMBOL)"
    )
    op.add_argument("--outdir", default="out/opt")
    op.add_argument(
        "--publish-best",
        action="store_true",
        help="Retrain best on full data and publish to Supabase",
    )
    bt = sub.add_parser("backtest", help="Backtest a trained model on a CSV (normalized or CSV7).")
    bt.add_argument("--file", required=True)
    bt.add_argument("--symbol", required=True)
    bt.add_argument("--model", default=None, help="Path to local .pkl model")
    bt.add_argument(
        "--from-registry",
        action="store_true",
        help="Load model from Supabase (LATEST.json)",
    )
    bt.add_argument("--open-thr", type=float, default=0.55)
    bt.add_argument("--close-thr", type=float, default=None)
    bt.add_argument("--fee-bps", type=float, default=2.0)
    bt.add_argument("--slip-bps", type=float, default=0.0)
    bt.add_argument("--position", default="gated", choices=["gated","sized"])
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--outdir", default="out/backtests")

    # registry-smoke
    rs = sub.add_parser(
        "registry-smoke",
        help="Upload & download a tiny test blob to verify credentials/bucket.",
    )
    rs.add_argument(
        "--symbol", default=None, help="Symbol (default from CT_SYMBOL or XRPUSD)"
    )

    # registry-list
    rl = sub.add_parser(
        "registry-list",
        help="List objects under models/regime/<SYMBOL> in Storage.",
    )
    rl.add_argument("--symbol", default=None)

    # registry-pointer
    rp = sub.add_parser(
        "registry-pointer",
        help="Print LATEST.json for models/regime/<SYMBOL>.",
    )
    rp.add_argument("--symbol", default=None)

    args = parser.parse_args(argv)

    if args.version:
        try:
            print(version("cointrader-trainer"))
        except PackageNotFoundError:
            print("0.1.0")
        return

    if args.cmd == "autobacktest":
        from pathlib import Path

        from cointrainer.backtest.continuous import loop_autobacktest

        loop_autobacktest(
            csv_path=Path(args.file),
            symbol=args.symbol.upper(),
            horizon=args.horizon,
            hold=args.hold,
            open_thr=args.open_thr,
            interval_sec=args.interval_sec,
            limit_rows=args.limit_rows,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            device_type=args.device_type,
            max_bin=args.max_bin,
            n_jobs=args.n_jobs,
            publish=args.publish,
            outdir=Path(args.outdir),
        )
    if args.cmd == "optimize":
        from pathlib import Path

        from cointrainer.backtest.optimize import optimize_grid
        from cointrainer.backtest.optuna_opt import (
            OptunaConfig,
            optimize_optuna,
            publish_best_model,
        )

        csv_path = Path(args.file)
        symbol = args.symbol.upper()
        outdir = Path(args.outdir)

        try:
            cfg = OptunaConfig(
                n_trials=args.n_trials,
                n_folds=args.folds,
                val_len=args.val_len,
                gap=args.gap,
                limit_rows=args.limit_rows,
                fee_bps=args.fee_bps,
                slip_bps=args.slip_bps,
                device_type=args.device_type,
                max_bin=args.max_bin,
                n_jobs=args.n_jobs,
                seed=args.seed,
                storage=args.storage,
                study_name=(args.study_name or f"{symbol}_study"),
                publish_best=args.publish_best,
            )
            res = optimize_optuna(csv_path, symbol, outdir=outdir, cfg=cfg)
            print("[optimize] best:", res["best"])
            print("[optimize] leaderboard:", res["leaderboard_path"])

            if args.publish_best:
                key = publish_best_model(csv_path, symbol, outdir, res["best"])
                print(f"[optimize] Published best model to: {key}")
            return
        except RuntimeError:
            print("[optimize] Optuna not available → falling back to grid search")
            res = optimize_grid(
                csv_path=csv_path,
                symbol=symbol,
                horizons=[15, 30, 60],
                holds=[0.001, 0.0015, 0.002, 0.003],
                open_thrs=[0.52, 0.55, 0.58],
                position_modes=["gated", "sized"],
                fee_bps=args.fee_bps,
                slip_bps=args.slip_bps,
                device_type=args.device_type,
                max_bin=args.max_bin,
                n_jobs=args.n_jobs,
                limit_rows=args.limit_rows,
                outdir=outdir,
            )
            print("[optimize-grid] best:", res["best"])
            print("[optimize-grid] leaderboard:", res["leaderboard"])
            return
    if args.cmd == "backtest":
        from pathlib import Path

        from cointrainer.backtest.run import backtest_csv
        prefix = None
        if args.from_registry:
            prefix = f"models/regime/{args.symbol.upper()}"
        res = backtest_csv(
            path=Path(args.file),
            symbol=args.symbol.upper(),
            model_local=Path(args.model) if args.model else None,
            model_registry_prefix=prefix,
            outdir=Path(args.outdir),
            open_thr=args.open_thr,
            close_thr=args.close_thr,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            position_mode=args.position,
            start=args.start,
            end=args.end,
        )
        print("[backtest] summary:", res["stats"])
        return

    if args.cmd == "registry-smoke":
        import os
        import time

        from cointrainer.registry import _get_bucket, _get_client

        symbol = (args.symbol or os.getenv("CT_SYMBOL") or "XRPUSD").upper()
        cli = _get_client()
        bucket = _get_bucket()
        prefix = f"models/regime/{symbol}"
        path = f"{prefix}/__smoke__.bin"
        blob = f"smoke-{int(time.time())}".encode("utf-8")  # noqa: UP012
        print(f"Uploading {bucket}/{path}")
        cli.storage.from_(bucket).upload(
            path,
            blob,
            {"contentType": "application/octet-stream", "upsert": "true"},
        )
        print(f"Listing {bucket}/{prefix}")
        print(cli.storage.from_(bucket).list(prefix=prefix))
        print(f"Downloading {bucket}/{path}")
        out = cli.storage.from_(bucket).download(path)
        print("Downloaded bytes:", len(out))
        return

    if args.cmd == "registry-list":
        import os

        from cointrainer.registry import _get_bucket, _get_client

        symbol = (args.symbol or os.getenv("CT_SYMBOL") or "XRPUSD").upper()
        cli = _get_client()
        bucket = _get_bucket()
        prefix = f"models/regime/{symbol}"
        print(cli.storage.from_(bucket).list(prefix=prefix))
        return

    if args.cmd == "registry-pointer":
        import json
        import os

        from cointrainer.registry import load_pointer

        symbol = (args.symbol or os.getenv("CT_SYMBOL") or "XRPUSD").upper()
        prefix = f"models/regime/{symbol}"
        meta = load_pointer(prefix)
        print(json.dumps(meta, indent=2))
        return

    if hasattr(args, "func"):
        args.func(args)
        return

    parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
