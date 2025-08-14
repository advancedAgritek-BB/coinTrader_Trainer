# coinTrader Trainer

Release **0.1.0** of the coinTrader Trainer package.

Install from PyPI:

```bash
pip install cointrader-trainer==0.1.0
```

For development use the editable install and import as usual:

```bash
pip install -e .
import cointrainer
```
Installable via `pip install -e ".[train]"` for training or a base `pip install .` for runtime-only use.
[![CI](https://github.com/OWNER/coinTrader_Trainer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OWNER/coinTrader_Trainer/actions/workflows/ci.yml)

Now installable via `pip install -e .` and importable via `import cointrainer`.

coinTrader Trainer forms the model training component of the broader
`coinTrader2.0` trading application.  The live system records executed
orders and market snapshots in a Supabase project.  This repository
contains utilities that pull those historical trade logs, generate
technical indicators and fit machine learning models such as the
LightGBM ``regime_lgbm`` classifier.  Trained models are uploaded back
to Supabase so that the ``coinTrader2.0`` process can load them for
real-time decisions.

The project includes helpers for fetching data from Supabase,
engineering features and running training pipelines.  Results are
optionally persisted to Supabase Storage and the ``models`` table.

## Requirements

* Python 3.9 or newer
* networkx for the swarm simulation
* backtrader for backtesting strategies
* ``SUPABASE_URL`` and credentials for the Supabase project. Data reads
  use ``SUPABASE_USER_EMAIL`` with ``SUPABASE_PASSWORD`` (or ``SUPABASE_JWT``),
  while uploads continue to rely on ``SUPABASE_SERVICE_KEY`` (or
  ``SUPABASE_KEY``).
* ``PARAMS_BUCKET`` and ``PARAMS_TABLE`` control where swarm optimisation
  parameters are uploaded. Defaults are ``agent_params`` for both.
* Optional: ``numba`` for faster CPU feature generation and ``jax`` to enable
  GPU acceleration when available.
* Optional: a GPU-enabled LightGBM build for faster training. A helper script
  is provided to compile and upload wheels.
* Feature generation always relies on
  [Numba](https://numba.pydata.org/) on the CPU. When
  ``jax`` is installed and ``use_gpu=True`` is passed to
  ``cointrainer.features.build.make_features`` compatible GPUs are used to accelerate
  calculations.
* Optional: set ``REDIS_URL`` (or ``REDIS_TLS_URL``) to cache trade log queries
  in Redis.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"  # omit [train] for runtime only
cointrainer train regime --symbol BTCUSDT --horizon 15m [--optuna] [--use-gpu]
```

### CSV7 ingest & local training (1-minute bars)

Our CSV7 format is headerless with 7 columns:
ts, open, high, low, close, volume, trades

```kotlin
where `ts` is epoch seconds (some files may be ms; the importer auto-detects).
```

**Ingest → normalized CSV (and Parquet if available):**
```bash
cointrainer import-csv7 --file ./XRPUSD_1.csv --symbol XRPUSD --out ./data/XRPUSD_1m
# writes: data/XRPUSD_1m.normalized.csv (+ .parquet if pyarrow/fastparquet installed)
```

Train a quick regime model:
```bash
# from CSV7 (direct):
cointrainer csv-train --file ./XRPUSD_1.csv --symbol XRPUSD --horizon 15 --hold 0.0015
# or from normalized CSV:
cointrainer csv-train --file ./data/XRPUSD_1m.normalized.csv --symbol XRPUSD --horizon 15 --hold 0.0015
```

Outputs:

* local_models/xrpusd_regime_lgbm.pkl — local model
* local_models/xrpusd_predictions.csv — actions + scores for inspection

If you pass `--publish` and your registry environment is configured, the trainer will also upload the model and update `LATEST.json` under `models/regime/<SYMBOL>/`.

_On Windows, use `python -m cointrainer` if the `cointrainer` command is not found._

### Batch training from a folder of CSVs

If you have many CSVs (either headerless CSV7 or normalized OHLCV CSVs), run:

```bash
# Train every *.csv in the folder (non-recursive)
cointrainer csv-train-batch --folder ./datasets --glob "*.csv" --symbol-from filename --horizon 15 --hold 0.0015

# Options:
# --recursive           # search subfolders
# --symbol-from parent  # derive symbol from parent folder name
# --symbol-from fixed --symbol XRPUSD  # force a symbol for all files
# --limit-rows 250000   # only train on the last 250k rows of each file
# --publish             # publish each model to the registry if env is set
```

Results:

Models in local_models/<symbol>_regime_lgbm.pkl

Summary in local_models/batch_train_summary.json

## Backtesting & Auto‑optimization

### One-off backtest
```bash
cointrainer backtest --file ./data/XRPUSD_1m.normalized.csv --symbol XRPUSD \
  --model ./local_models/xrpusd_regime_lgbm.pkl \
  --open-thr 0.55 --fee-bps 2 --slip-bps 0 --position gated
```
Outputs reports to `out/backtests/<SYMBOL>/`.

On Windows:

```powershell
python -m cointrainer.cli backtest `
  --file .\data\XRPUSD_1m.normalized.csv `
  --symbol XRPUSD `
  --model .\local_models\xrpusd_regime_lgbm.pkl `
  --open-thr 0.55 --fee-bps 2 --position gated
```

### Grid search optimizer
```bash
cointrainer optimize --file ./data/XRPUSD_1m.normalized.csv --symbol XRPUSD \
  --horizons 15 30 60 --holds 0.001 0.0015 0.002 --open-thrs 0.52 0.55 0.58 \
  --positions gated sized --fee-bps 2 --slip-bps 0
```
Creates `out/opt/<SYMBOL>/leaderboard.csv` and `best.json`.

### Continuous autobacktest
```bash
# retrain + backtest every 15 minutes, publish latest model/pointer
cointrainer autobacktest --file ./data/XRPUSD_1m.normalized.csv --symbol XRPUSD \
  --horizon 15 --hold 0.0015 --open-thr 0.55 --interval-sec 900 --publish
```

GPU: add `--device-type gpu --max-bin 63 --n-jobs 0` to the commands above for Radeon acceleration.

### Runtime usage

```python
from crypto_bot.regime.api import predict
pred = predict(df_features)
print(pred.action, pred.score)
```

## Integration with coinTrader2.0

``coinTrader2.0`` streams live trading data into the ``trade_logs`` table in
Supabase.  The trainer consumes those logs to build supervised training sets.
When ``train_pipeline.py`` or ``ml_trainer.py`` runs, it queries the desired
time range from Supabase, engineers features with ``cointrainer.features.build.make_features``
and trains the ``regime_lgbm`` model.  After training the resulting model is
uploaded back to Supabase where the trading application can fetch it for live
predictions.

Runtime integrations can rely on a stable facade for regime predictions:

```python
from crypto_bot.regime.facade import predict
action = predict(features_df).action
```

Pin the package to guarantee compatibility with coinTrader2.0:

```
cointrader-trainer>=0.1.0
```

## Artifacts & Registry

* Artifacts at ``models/regime/{SYMBOL}/{YYYYMMDD-HHMMSS}_regime_lgbm.pkl``
* Pointer at ``models/regime/{SYMBOL}/LATEST.json``
* ``--publish`` writes both.

Trained models are saved using a standard layout so the trainer and runtime
agree on how to locate them.  Each run writes the pickled model to the artifact
path and updates ``LATEST.json`` with metadata:

```json
{
  "key": "models/regime/BTCUSDT/20250811-153000_regime_lgbm.pkl",
  "schema_version": "1",
  "feature_list": ["rsi_14", "atr_14", "ema_8", "ema_21"],
  "label_order": [-1, 0, 1],
  "horizon": "15m",
  "thresholds": {"hold": 0.55},
  "hash": "sha256:...hex..."
}
```

``coinTrader2.0`` resolves the latest model by reading ``LATEST.json`` and then
downloading the referenced pickle.  If a hash is present it is verified before
the model is deserialised.

## Publishing models to Supabase Storage

Environment variables (use a **service role** key for write access):

```bash
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<SERVICE_ROLE_KEY>
CT_MODELS_BUCKET=models     # create this bucket in Supabase Storage
CT_SYMBOL=XRPUSD            # default symbol for training, optional
```

Train and publish a model:

```bash
cointrainer csv-train --file ./data/XRPUSD_1m.normalized.csv --symbol XRPUSD --horizon 15 --hold 0.0015 --publish
```

If successful, you will see console lines like:

```
[publish] Uploaded: models/regime/XRPUSD/2025..._regime_lgbm.pkl
[publish] Pointer:  models/regime/XRPUSD/LATEST.json
```

Diagnostics:

```
cointrainer registry-smoke --symbol XRPUSD      # upload/list/download a tiny blob
cointrainer registry-list  --symbol XRPUSD      # list objects under the prefix
cointrainer registry-pointer --symbol XRPUSD    # print LATEST.json
```

Debug logging (optional):

```bash
export CT_REGISTRY_DEBUG=1   # prints registry operations during save/load
```

## How to verify end‑to‑end (PowerShell)

```powershell
# 1) Set env with a service role key and bucket name
$env:SUPABASE_URL  = "https://<your-project-ref>.supabase.co"
$env:SUPABASE_KEY  = "<SERVICE_ROLE_KEY>"
$env:CT_MODELS_BUCKET = "models"
$env:CT_SYMBOL     = "XRPUSD"
$env:CT_REGISTRY_DEBUG = "1"   # optional

# 2) Publish a single model
python -m cointrainer.cli csv-train `
  --file .\data\XRPUSD_1m.normalized.csv `
  --symbol $env:CT_SYMBOL `
  --horizon 15 `
  --hold 0.0015 `
  --publish

# Expected console:
# [publish] Uploaded: models/regime/XRPUSD/<timestamp>_regime_lgbm.pkl
# [publish] Pointer:  models/regime/XRPUSD/LATEST.json
# [registry] upload models/... (debug lines if CT_REGISTRY_DEBUG=1)

# 3) Show the pointer
python -m cointrainer.cli registry-pointer --symbol $env:CT_SYMBOL

# 4) Batch publish all CSVs in a folder (GPU enabled if you added those flags)
python -m cointrainer.cli csv-train-batch `
  --folder .\my_csvs `
  --glob "*.csv" `
  --symbol-from filename `
  --horizon 15 `
  --hold 0.0015 `
  --publish
```

## Environment

Set environment variables so the runtime can locate models:

* ``CT_SYMBOL`` – trading pair such as ``BTCUSDT`` (default: ``BTCUSDT``)
* ``CT_MODELS_BUCKET`` – bucket or local directory containing artifacts (default: ``models``)
* ``CT_REGIME_PREFIX`` – object prefix under the bucket (default: ``models/regime``)
* ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` if Supabase storage is used

## Troubleshooting

If no artifact is available the runtime falls back to an embedded model
stored in ``fallback_b64.txt`` and queues a retrain via
``cointrainer.train.enqueue``.

## Contributing

Run the linters and tests before submitting a PR:

```bash
ruff .
pytest
vulture .
```
Continuous integration runs the same checks.
