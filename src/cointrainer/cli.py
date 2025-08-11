import argparse
from importlib.metadata import PackageNotFoundError, version


def main() -> None:
    parser = argparse.ArgumentParser(prog="cointrainer")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train")
    subparsers.add_parser("import-data")

    args = parser.parse_args()

    if args.version:
        try:
            print(version("cointrader-trainer"))
        except PackageNotFoundError:
            print("0.0.0")
        return

    if args.command in {"train", "import-data"}:
        print("coming in P6")
        return

    parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
