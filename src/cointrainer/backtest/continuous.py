from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional

from cointrainer.backtest.run import backtest_csv
from cointrainer.backtest.optimize import _train_one
from cointrainer.registry import save_model  # optional publish
import io, joblib

def loop_autobacktest(
    csv_path: Path,
    symbol: str,
    horizon: int,
    hold: float,
    open_thr: float,
    interval_sec: int = 900,
    limit_rows: int | None = 1_000_000,
    fee_bps: float = 2.0,
    slip_bps: float = 0.0,
    device_type: str = "gpu",
    max_bin: int = 63,
    n_jobs: int | None = 0,
    publish: bool = False,
    outdir: Path = Path("out/autobacktest"),
):
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"models/regime/{symbol}"

    last_mtime = 0.0
    while True:
        try:
            mtime = csv_path.stat().st_mtime
            if mtime <= last_mtime:
                time.sleep(interval_sec)
                continue
            last_mtime = mtime

            # 1) train on latest slice
            model_path = _train_one(csv_path, symbol, horizon, hold, outdir, device_type, max_bin, n_jobs, limit_rows)

            # 2) backtest with policy
            res = backtest_csv(csv_path, symbol, model_local=model_path, outdir=outdir / "backtests",
                               open_thr=open_thr, fee_bps=fee_bps, slip_bps=slip_bps, position_mode="gated")

            # 3) optional publish
            if publish:
                blob = io.BytesIO()
                joblib.dump(joblib.load(model_path), blob)
                ts = time.strftime("%Y%m%d-%H%M%S")
                key = f"{prefix}/{ts}_regime_lgbm.pkl"
                meta = {
                    "feature_list": ["ema_8","ema_21","rsi_14","atr_14","roc_5","obv"],
                    "label_order": [-1,0,1],
                    "horizon": f"{horizon}m",
                    "thresholds": {"hold": hold, "open_thr": open_thr},
                    "symbol": symbol,
                }
                save_model(key, blob.getvalue(), meta)

            # 4) write summary snapshot locally
            ts = time.strftime("%Y%m%d-%H%M%S")
            (outdir / f"summary_{symbol}_{ts}.json").write_text(json.dumps(res["stats"], indent=2))
            print(f"[autobacktest] {symbol} {ts}  {res['stats']}")
        except KeyboardInterrupt:
            print("[autobacktest] stopped")
            break
        except Exception as e:
            print("[autobacktest] ERROR:", e)
        time.sleep(interval_sec)
