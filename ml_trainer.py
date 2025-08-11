"""Command line interface for running coinTrader training tasks."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import platform
import shutil
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Optional imports used by the interactive menu.
# These modules may not be available during testing so we guard them with
# ``try`` blocks and fall back to ``None`` if missing.  The interactive menu
# checks for ``None`` before invoking any functionality.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional utility
    from tools.backtest_strategies.backtest import backtest
except Exception:  # pragma: no cover - missing dependency
    backtest = None  # type: ignore[misc]

try:  # pragma: no cover - optional signal model
    from crypto_bot.ml_signal_model import train_from_csv as train_signal_model
except Exception:  # pragma: no cover - module may be absent
    train_signal_model = None  # type: ignore

try:  # pragma: no cover - optional LightGBM helpers
    from tools.train_meta_selector import main as train_meta_selector
    from tools.train_regime_model import main as train_regime_model
    from tools.train_fallback_model import main as train_fallback_model
except Exception:  # pragma: no cover - training scripts may be missing
    train_meta_selector = train_regime_model = train_fallback_model = None  # type: ignore

try:  # pragma: no cover - optional RL trainers
    import rl.rl as _rl_mod
    ppo_train = getattr(_rl_mod, "train", getattr(_rl_mod, "train_ppo", None))
except Exception:  # pragma: no cover - RL module may be missing
    ppo_train = None  # type: ignore

try:  # pragma: no cover - contextual bandit trainer
    from rl.strategy_selector import train as bandit_train
except Exception:  # pragma: no cover - selector module may be missing
    bandit_train = None  # type: ignore

try:  # pragma: no cover - optional RL helpers
    from rl.train import run as rl_train
    from rl.evaluate import run as rl_evaluate
except Exception:  # pragma: no cover - RL modules may be absent
    rl_train = rl_evaluate = None  # type: ignore

try:  # pragma: no cover - optional scheduling helper
    from utils.token_registry import schedule_retrain
except Exception:  # pragma: no cover - module not available
    schedule_retrain = None  # type: ignore

from prometheus_client import Gauge, start_http_server

import numpy as np
import pandas as pd
import yaml

from cointrainer.data import importers
from cointrainer.train.pipeline import check_clinfo_gpu, verify_lightgbm_gpu
from cointrainer.registry import ModelRegistry

logger = logging.getLogger(__name__)

# expose model accuracy metrics via Prometheus
accuracy_gauge = Gauge("model_accuracy", "Model accuracy")
if __name__ == "__main__":
    start_http_server(8000)

try:
    from trainers.regime_lgbm import train_regime_lgbm
except ImportError:  # pragma: no cover - optional during tests
    train_regime_lgbm = None  # type: ignore


try:  # pragma: no cover - optional dependency
    from federated_trainer import train_federated_regime
except ImportError:  # pragma: no cover - missing during testing
    try:  # pragma: no cover - federated trainer may be optional
        from trainers.federated import train_federated_regime
    except ImportError:  # pragma: no cover - trainer not available
        train_federated_regime = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import federated_fl
except ImportError:  # pragma: no cover - module may be missing
    federated_fl = None  # type: ignore

TRAINERS = {
    "regime": (train_regime_lgbm, "regime_lgbm"),
}


def load_cfg(path: str) -> Dict[str, Any]:
    """Load YAML configuration with sensible defaults."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    for key in ("regime_lgbm", "federated_regime"):
        section = cfg.get(key)
        if isinstance(section, dict):
            section.setdefault("device", "opencl")
    cfg.setdefault("backtest", {"slippage": 0.005, "costs": 0.002})
    return cfg


def _make_dummy_data(n: int = 200) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic dataset for local testing."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "open": rng.random(n) * 100,
            "high": rng.random(n) * 100,
            "low": rng.random(n) * 100,
            "close": rng.random(n) * 100,
        }
    )
    return df, pd.Series(rng.integers(0, 2, size=n))


def _start_rocm_smi_monitor() -> subprocess.Popen | None:
    """Start a ``rocm-smi`` monitor and log its output."""
    if platform.system() == "Windows":
        logging.info("ROCm SMI monitor not supported on Windows")
        return None
    cmd = ["rocm-smi", "--showuse", "--interval", "1"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - command missing
        logging.warning("Failed to start rocm-smi monitor: %s", exc)
        return None

    def _forward() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            logging.info("rocm-smi: %s", line.rstrip())

    threading.Thread(target=_forward, daemon=True).start()
    logging.info("Started rocm-smi monitor: %s", " ".join(cmd))
    return proc


def _launch_rgp(pid: int) -> None:
    """Launch AMD RGP if available, otherwise print the command."""
    cmd = ["rgp.exe", "--process", str(pid)]
    exe = shutil.which("rgp.exe")
    if exe:
        try:
            subprocess.Popen([exe, "--process", str(pid)])
            print(f"AMD RGP launched: {' '.join(cmd)}")
            return
        except (
            FileNotFoundError,
            OSError,
        ) as exc:  # pragma: no cover - unexpected failures
            logging.warning("Failed to launch AMD RGP: %s", exc)
    print(" ".join(cmd))


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="coinTrader trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("task", help="Task to train, e.g. 'regime'")
    train_p.add_argument("--cfg", default="cfg.yaml", help="Config file path")
    train_p.add_argument("--use-gpu", action="store_true", help="Enable GPU training")
    train_p.add_argument(
        "--gpu-platform-id", type=int, default=None, help="OpenCL platform id"
    )
    train_p.add_argument(
        "--gpu-device-id", type=int, default=None, help="OpenCL device id"
    )
    train_p.add_argument(
        "--swarm",
        action="store_true",
        help="Run hyperparameter swarm search before training",
    )
    train_p.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna hyperparameter search before training",
    )
    train_p.add_argument(
        "--federated",
        action="store_true",
        help="Use federated learning (regime task only)",
    )
    train_p.add_argument(
        "--true-federated",
        action="store_true",
        help="Use Flower-based federated learning",
    )
    train_p.add_argument("--start-ts", help="Data start timestamp (ISO format)")
    train_p.add_argument("--end-ts", help="Data end timestamp (ISO format)")
    train_p.add_argument("--table", default="ohlc_data", help="Supabase table name")
    train_p.add_argument(
        "--profile-gpu", action="store_true", help="Log GPU utilisation via rocm-smi"
    )

    csv_p = sub.add_parser("import-csv", help="Import historical CSV data")
    csv_p.add_argument("csv", help="CSV file path")
    csv_p.add_argument("--symbol", required=True, help="Trading pair symbol")
    csv_p.add_argument("--start-ts", help="Start timestamp (ISO)")
    csv_p.add_argument("--end-ts", help="End timestamp (ISO)")
    csv_p.add_argument(
        "--table",
        help="Supabase table name (defaults to historical_prices_<symbol>)",
        default=None,
    )

    import_p = sub.add_parser(
        "import-data", help="Download historical data and insert to Supabase"
    )
    import_p.add_argument(
        "--source-url", required=True, help="HTTP endpoint for historical data"
    )
    import_p.add_argument("--symbol", required=True, help="Trading pair symbol")
    import_p.add_argument(
        "--start-ts", required=True, help="Data start timestamp (ISO)"
    )
    import_p.add_argument("--end-ts", required=True, help="Data end timestamp (ISO)")
    import_p.add_argument(
        "--output-file", required=True, help="File to store downloaded data"
    )
    import_p.add_argument(
        "--batch-size", type=int, default=1000, help="Insert batch size"
    )

    args = parser.parse_args()

    if args.command == "import-data":
        df = importers.download_historical_data(
            args.source_url,
            output_file=args.output_file,
            symbol=args.symbol,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )
        importers.insert_to_supabase(df, batch_size=args.batch_size)
        return

    if args.command == "import-csv":
        df = importers.download_historical_data(
            args.csv,
            symbol=args.symbol,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise SystemExit("SUPABASE_URL and service key must be set")
        table = args.table or f"historical_prices_{args.symbol.lower()}"
        importers.insert_to_supabase(
            df,
            url=url,
            key=key,
            table=table,
            symbol=args.symbol,
        )
        return

    cfg = load_cfg(args.cfg)

    if args.command != "train":
        raise SystemExit("Unknown command")

    if args.task not in TRAINERS:
        raise SystemExit(f"Unknown task: {args.task}")

    trainer_fn, cfg_key = TRAINERS[args.task]

    federated_enabled = False
    if args.federated:
        if args.task != "regime":
            raise SystemExit("--federated only supported for 'regime' task")
        if train_federated_regime is None:
            logger.warning(
                "Federated training not supported; continuing locally"
            )
        else:
            trainer_fn = train_federated_regime
            cfg_key = "federated_regime"
            federated_enabled = True

    if trainer_fn is None and not args.federated:
        raise SystemExit(f"Trainer '{args.task}' not available, install LightGBM")

    params = cfg.get(cfg_key, {}).copy()

    if args.swarm and args.optuna:
        raise SystemExit("--optuna and --swarm cannot be used together")

    # GPU parameter overrides
    if args.use_gpu:
        params["device"] = "opencl"
    if args.gpu_platform_id is not None:
        params["gpu_platform_id"] = args.gpu_platform_id
    if args.gpu_device_id is not None:
        params["gpu_device_id"] = args.gpu_device_id

    if check_clinfo_gpu() and verify_lightgbm_gpu(params):
        params.setdefault("device", "opencl")
        params.setdefault("gpu_platform_id", 0)
        params.setdefault("gpu_device_id", 0)
        params.pop("device_type", None)
        use_gpu_flag = True
    else:
        params["device"] = "cpu"
        params.pop("device_type", None)
        logger.warning("GPU not detected; falling back to CPU")
        use_gpu_flag = False

    # Swarm optimisation
    if args.swarm:
        try:
            import swarm_sim
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("Swarm optimization unavailable; skipping")
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("Swarm optimization unavailable: %s", exc)
        else:
            end_ts = datetime.utcnow()
            start_ts = end_ts - timedelta(days=7)
            swarm_params = asyncio.run(
                swarm_sim.run_swarm_search(start_ts, end_ts, table=args.table)
            )
            if isinstance(swarm_params, dict):
                params.update(swarm_params)

    # Optuna optimisation
    if args.optuna:
        optuna_mod = None
        try:
            import optuna_search as optuna_mod
        except ImportError:
            try:
                import optuna_optimizer as optuna_mod  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                logger.warning("Optuna optimization unavailable: %s", exc)
                optuna_mod = None

        if optuna_mod:
            window = cfg.get("default_window_days", 7)
            defaults = cfg.get("optuna", {})
            run_func = optuna_mod.run_optuna_search
            sig = inspect.signature(run_func)
            param_names = set(sig.parameters.keys())
            if {"start", "end"}.issubset(param_names):
                start = datetime.utcnow() - timedelta(days=window)
                end = datetime.utcnow()
                result = run_func(start, end, table=args.table, **defaults)
            else:
                result = run_func(window, table=args.table, **defaults)

            if inspect.iscoroutine(result):
                result = asyncio.run(result)

            if isinstance(result, dict):
                params.update(result)

    monitor_proc = None
    if args.profile_gpu:
        if platform.system() != "Windows":
            monitor_proc = _start_rocm_smi_monitor()
            _launch_rgp(os.getpid())
        else:
            logger.warning("ROCm profiling tools are unavailable on Windows")

    # Training dispatch
    try:
        if args.true_federated:
            if federated_fl is None or not getattr(federated_fl, "_HAVE_FLWR", False):
                logger.warning(
                    "True federated training requires the 'flwr' package. Install it with 'pip install flwr'"
                )
                return
            if not args.start_ts or not args.end_ts:
                raise SystemExit("--true-federated requires --start-ts and --end-ts")
            federated_fl.start_server(
                args.start_ts,
                args.end_ts,
                config_path=args.cfg,
                params_override=params,
                table=args.table,
            )
            return

        if federated_enabled:
            if not args.start_ts or not args.end_ts:
                raise SystemExit("--federated requires --start-ts and --end-ts")
            model, metrics = asyncio.run(
                trainer_fn(
                    args.start_ts,
                    args.end_ts,
                    config_path=args.cfg,
                    params_override=params,
                    table=args.table,
                )
            )
        else:
            X, y = _make_dummy_data()
            model, metrics = trainer_fn(
                X,
                y,
                params,
                use_gpu=use_gpu_flag,
                profile_gpu=args.profile_gpu,
            )  # type: ignore[arg-type]

        print("Training completed. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        accuracy_gauge.set(metrics.get("accuracy", 0))
    finally:
        if monitor_proc:
            monitor_proc.terminate()


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

# cached objects used across menu actions
_CACHED_PARAMS: Dict[str, Any] = {}
_CACHED_DATA: Tuple[pd.DataFrame, pd.Series] | None = None
_MENU_USE_GPU: bool = False


def get_params(cfg_path: str = "cfg.yaml", task: str = "regime") -> Dict[str, Any]:
    """Load and cache training parameters for ``task``.

    Parameters are loaded from ``cfg_path`` and stored globally so subsequent
    menu actions can reuse them without re-reading the file.
    """

    global _CACHED_PARAMS
    cfg = load_cfg(cfg_path)
    trainer = TRAINERS.get(task)
    key = trainer[1] if trainer else task
    _CACHED_PARAMS = cfg.get(key, {})
    print("Parameters loaded")
    return _CACHED_PARAMS


def prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare a small dummy dataset for experimentation."""

    global _CACHED_DATA
    _CACHED_DATA = _make_dummy_data()
    print(f"Prepared dataset with {_CACHED_DATA[0].shape[0]} rows")
    return _CACHED_DATA


def train_model() -> None:
    """Train the default regime model on dummy data."""

    params = _CACHED_PARAMS or get_params()
    data = _CACHED_DATA or prepare_data()
    trainer_fn, _ = TRAINERS.get("regime", (None, "regime_lgbm"))
    if trainer_fn is None:
        print("Trainer not available")
        return
    model, metrics = trainer_fn(
        data[0],
        data[1],
        params,
        use_gpu=False,
        profile_gpu=False,
    )  # type: ignore[arg-type]
    print("Training completed. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    accuracy_gauge.set(metrics.get("accuracy", 0))


def _generate_csv() -> str:
    """Generate a CSV of simulated trades via :func:`backtest`."""
    if backtest is None:
        raise RuntimeError("backtest module not available")
    df, _ = _make_dummy_data()
    df.index = pd.date_range("2020", periods=len(df), freq="H")
    backtest(df, ["ml"])
    return "simulated_trades.csv"


def _train_signal_model() -> None:
    if train_signal_model is None:
        print("Signal model training not available")
        return
    try:
        csv_path = _generate_csv()
        kwargs = {}
        if "use_gpu" in inspect.signature(train_signal_model).parameters:
            kwargs["use_gpu"] = _MENU_USE_GPU
        model = train_signal_model(csv_path, **kwargs)
        ModelRegistry().upload(model, "signal_model")
        print("Signal model uploaded")
    except Exception as exc:  # pragma: no cover - external deps
        print(f"Signal model training failed: {exc}")


def _train_meta_selector() -> None:
    if train_meta_selector is None:
        print("Meta selector training not available")
        return
    try:
        path = _generate_csv()
        train_meta_selector(path, use_gpu=_MENU_USE_GPU)
        print("Meta selector uploaded")
    except Exception as exc:  # pragma: no cover - external deps
        print(f"Meta selector training failed: {exc}")


def _train_regime_model_menu() -> None:
    if train_regime_model is None:
        print("Regime model training not available")
        return
    try:
        path = _generate_csv()
        train_regime_model(path, use_gpu=_MENU_USE_GPU)
        print("Regime model uploaded")
    except Exception as exc:  # pragma: no cover
        print(f"Regime model training failed: {exc}")


def _train_ppo_selector() -> None:
    if ppo_train is None:
        print("PPO trainer not available")
        return
    try:
        env = None
        try:
            import gymnasium as gym  # type: ignore
            env = gym.make("CartPole-v1")
        except Exception:
            print("gymnasium unavailable; skipping")
            return
        model = ppo_train(env, use_gpu=_MENU_USE_GPU)
        ModelRegistry().upload(model, "ppo_selector")
        print("PPO selector uploaded")
    except Exception as exc:  # pragma: no cover - external deps
        print(f"PPO training failed: {exc}")


def _train_bandit_selector() -> None:
    if bandit_train is None:
        print("Bandit trainer not available")
        return
    try:
        path = _generate_csv()
        bandit_train(path, use_gpu=_MENU_USE_GPU)
        print("Bandit selector uploaded")
    except Exception as exc:  # pragma: no cover - external deps
        print(f"Bandit training failed: {exc}")


def _train_fallback() -> None:
    if train_fallback_model is None:
        print("Fallback model training not available")
        return
    try:
        path = _generate_csv()
        train_fallback_model(path)
        print("Fallback model uploaded")
    except Exception as exc:  # pragma: no cover - external deps
        print(f"Fallback model training failed: {exc}")


def _run_backtest() -> None:
    """Invoke the optional backtest helper if available."""

    if backtest is None:
        print("Backtest module not available")
        return
    try:
        backtest()
    except Exception as exc:  # pragma: no cover - depends on external module
        print(f"Backtest failed: {exc}")


def _rl_train() -> None:
    if rl_train is None:
        print("RL training module not available")
        return
    try:  # pragma: no cover - optional
        rl_train()
    except Exception as exc:  # pragma: no cover - depends on external module
        print(f"RL training failed: {exc}")


def _rl_evaluate() -> None:
    if rl_evaluate is None:
        print("RL evaluation module not available")
        return
    try:  # pragma: no cover - optional
        rl_evaluate()
    except Exception as exc:  # pragma: no cover - depends on external module
        print(f"RL evaluation failed: {exc}")


def schedule() -> None:
    """Schedule a future retraining job if the helper exists."""

    if schedule_retrain is None:
        print("schedule_retrain unavailable")
        return
    try:  # pragma: no cover - optional
        schedule_retrain()
        print("Retraining scheduled")
    except Exception as exc:  # pragma: no cover - depends on external module
        print(f"Scheduling failed: {exc}")


def show_menu() -> None:
    """Display a basic interactive menu for common training tasks."""

    actions = {
        "1": ("Load parameters", lambda: get_params()),
        "2": ("Prepare data", prepare_data),
        "3": ("Train model", train_model),
        "4": ("Backtest strategy", _run_backtest),
        "5": ("RL training", _rl_train),
        "6": ("RL evaluation", _rl_evaluate),
        "7": ("Schedule retrain", schedule),
        "8": ("Exit", None),
        "9": ("Train logistic regression signal model", _train_signal_model),
        "10": ("Train meta selector", _train_meta_selector),
        "11": ("Train regime model", _train_regime_model_menu),
        "12": ("Train PPO RL selector", _train_ppo_selector),
        "13": ("Train contextual bandit selector", _train_bandit_selector),
        "14": ("Train fallback LightGBM model", _train_fallback),
    }

    while True:
        print("\ncoinTrader trainer menu:")
        for key, (desc, _) in actions.items():
            print(f"{key}. {desc}")
        choice = input("Select option: ").strip()
        action = actions.get(choice)
        if not action:
            print("Invalid choice")
            continue
        if choice == "8":
            print("Goodbye")
            break
        func = action[1]
        if func:
            func()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(description="coinTrader training menu")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU training")
    args = parser.parse_args()
    _MENU_USE_GPU = args.use_gpu
    show_menu()
