# coinTrader Trainer

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

Trained models are saved using a standard layout so the trainer and runtime
agree on how to locate them.  Each run writes the pickled model to

```
models/regime/{SYMBOL}/{YYYYMMDD-HHMMSS}_regime_lgbm.pkl
```

and updates a pointer file ``models/regime/{SYMBOL}/LATEST.json`` with metadata:

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

## Environment

Set environment variables so the runtime can locate models:

* ``CT_SYMBOL`` – trading pair such as ``BTCUSDT``
* ``CT_MODELS_BUCKET`` – bucket or local directory containing artifacts
* ``CT_REGIME_PREFIX`` – object prefix under the bucket
* ``SUPABASE_URL`` and ``SUPABASE_KEY`` if Supabase storage is used

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
