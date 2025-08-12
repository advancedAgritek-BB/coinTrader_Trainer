import argparse
from importlib.metadata import PackageNotFoundError, version


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

    args = parser.parse_args(argv)

    if args.version:
        try:
            print(version("cointrader-trainer"))
        except PackageNotFoundError:
            print("0.0.0")
        return

    if hasattr(args, "func"):
        args.func(args)
        return

    parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
