# coinTrader Trainer

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
  ``feature_engineering.make_features`` compatible GPUs are used to accelerate
  calculations.
* Optional: set ``REDIS_URL`` (or ``REDIS_TLS_URL``) to cache trade log queries
  in Redis.

## Integration with coinTrader2.0

``coinTrader2.0`` streams live trading data into the ``trade_logs`` table in
Supabase.  The trainer consumes those logs to build supervised training sets.
When ``train_pipeline.py`` or ``ml_trainer.py`` runs, it queries the desired
time range from Supabase, engineers features with ``feature_engineering.make_features``
and trains the ``regime_lgbm`` model.  After training the resulting model is
uploaded back to Supabase where the trading application can fetch it for live
predictions.

The trading bot must be able to import modules from this repository.
Either install the package or add the project path to ``PYTHONPATH`` so that
``from coinTrader_Trainer import ...`` works. An easy approach is to export the
repository location when launching the bot:

```bash
export PYTHONPATH=/path/to/coinTrader_Trainer
```

Otherwise the bot will display a "Machine learning unavailable" warning because
it cannot import the trainer code.

## Installation

Create a virtual environment and install the required Python
dependencies.  A ``requirements.txt`` file is expected in the
repository, but you may also install the packages manually.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs all required packages including the
[Flower](https://flower.ai) library used for federated learning and
[Backtrader](https://www.backtrader.com/) for running strategy backtests.

Windows users should activate the environment with:

```powershell
./.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same
session to temporarily allow script execution. Alternatively, switch to
Command Prompt and run `./.venv\Scripts\activate`.

or in Command Prompt:

```cmd
./.venv\Scripts\activate
```

After activation run:

```bash
pip install -r requirements.txt
pip install -e .
```

The requirements file now includes [Flower](https://flower.ai) for
running true federated learning experiments and
[Backtrader](https://www.backtrader.com/) for offline backtesting.

* Install [Backtrader](https://www.backtrader.com/) to run strategy
  backtests:

  ```bash
  pip install backtrader
  ```

On **Windows**, `pyopencl` requires a C/C++ compiler.  Install
[Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
or another compiler before running the command above.  If you plan to
train only on the CPU you can remove `pyopencl` from `requirements.txt`.

If you update the repository at a later date, run the installation
command again so new dependencies such as ``pyyaml``, ``networkx``, ``backtrader`` or ``requests`` are installed.
For GPU-accelerated feature engineering install ``numba`` and ``jax``.
Depending on your hardware choose the CUDA or ROCm build of JAX. For
CUDA GPUs:

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For AMD GPUs:

```bash
pip install --upgrade "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
```


If you prefer to install packages individually:

```bash
pip install pandas numpy lightgbm scikit-learn supabase tenacity pyarrow pytz networkx requests backtrader
```

## Configuration

Copy `.env.example` to `.env` and populate your Supabase credentials. The
optional `PARAMS_BUCKET` and `PARAMS_TABLE` variables default to
`agent_params`. After editing the file run `python bootstrap_env.py` to
verify the configuration is complete.

```bash
cp .env.example .env
# edit .env with your credentials
python bootstrap_env.py
```

Training modules now live under the `cointrainer` package inside `src/`. Legacy root-level modules remain for backward compatibility but are deprecated and will be removed after the transition.

Modules can now be imported as normal, for example:

```python
from cointrainer.data import loader as data_loader
```

### Trade Log Fetching and Caching

``fetch_trade_logs`` provides a simple synchronous interface for retrieving
the historical trade logs recorded by ``coinTrader2.0``.  The trading
application writes every executed order to the ``trade_logs`` table in
Supabase.  Pass UTC ``datetime`` objects for ``start_ts`` and ``end_ts``—
naive timestamps are interpreted as UTC.  The optional ``symbol`` argument
filters rows to a specific pair.  When a ``cache_path`` is supplied the
function will read from the Parquet file if it exists and write new
results back to this location, avoiding repeated network requests.
If ``REDIS_URL`` or ``REDIS_TLS_URL`` is configured, results are additionally
cached in Redis as Parquet bytes.

Use ``max_rows`` to limit the number of rows returned:

```python
df = data_loader.fetch_trade_logs(start_ts, end_ts, symbol="BTC", max_rows=1000)
```

When Redis is available, trade logs are also cached in Redis using a key
derived from the time range and symbol. The loader falls back to
``localhost`` on port ``6379`` when ``REDIS_HOST`` or ``REDIS_PORT`` are not
specified. Cached data expires after ``REDIS_TTL`` seconds (default
``86400``). Passing ``cache_features=True`` to ``fetch_trade_logs`` will
compute ``make_features`` on the returned data and store the result under
``features_<key>`` in Redis. Subsequent calls with the same parameters return
this cached feature set directly.

Set ``cache_only=True`` to instruct ``fetch_trade_logs`` to return data only if
the corresponding key already exists in Redis. When the cache is populated no
database request is made; otherwise the data is fetched and the cache
refreshed.

```python
df = data_loader.fetch_trade_logs(start_ts, end_ts, symbol="BTC", cache_only=True)
```

### Async Data Fetching

`data_loader` now provides asynchronous helpers for retrieving trade logs
without blocking the event loop. `fetch_all_rows_async` retrieves every row in a
table, `fetch_table_async` pages through a table, and `fetch_data_range_async`
fetches rows between two timestamps. Each function returns a
``pandas.DataFrame`` and must be awaited:

These helpers provide the historical market data used for training.  By
querying the same Supabase tables populated by ``coinTrader2.0`` you can
reconstruct any period of market activity and produce datasets for model
tuning.

```python
import asyncio
from datetime import datetime, timedelta
from coinTrader_Trainer.data_loader import fetch_data_range_async

end = datetime.utcnow()
start = end - timedelta(days=1)
df = asyncio.run(
    fetch_data_range_async("trade_logs", start.isoformat(), end.isoformat())
)
```

Because the functions are asynchronous, callers must run them in an `asyncio`
event loop.  Inside existing async code simply use ``await fetch_data_range_async(...)``.

### Feature Engineering Options

The ``make_features`` function now accepts several parameters to customise the
technical indicators that are produced:
* ``rsi_period`` – lookback window for the relative strength index (default ``14``)
* ``atr_window`` – average true range window (default ``3``)
* ``volatility_window`` – period used to compute log return volatility (default ``20``)
* ``ema_short_period`` – window for the short-term EMA (default ``12``)
* ``ema_long_period`` – window for the long-term EMA (default ``26``)
* ``bb_window`` – lookback period for Bollinger Bands (default ``20``)
* ``bb_std`` – standard deviation multiplier for Bollinger Bands (default ``2``)
* ``mom_window`` – momentum calculation window (default ``10``)
* ``adx_window`` – lookback window for the Average Directional Index (default ``14``)
* ``obv_window`` – period used to smooth On‑Balance Volume (default ``20``)
* ``generate_target`` – create the ``target`` column when missing (default ``True``)

Installing [TA‑Lib](https://ta-lib.org/) is recommended for more accurate
technical indicator implementations.

Feature generation now always relies on Numba running on the CPU. When
``jax`` is installed, ``make_features`` can accelerate calculations on
compatible CUDA or ROCm/HIP GPUs by passing ``use_gpu=True``. The internal
call to ``_compute_features_pandas`` still sets ``use_numba=True`` so the
indicator calculations remain optimised on the CPU.

When the [`modin[ray]`](https://modin.org/) package is available, you can
set ``use_modin=True`` to distribute the pandas workload across CPU cores.

The execution time for feature generation and model training is logged
at ``INFO`` level automatically so no extra flag is required.

The ``target`` column produced during training is now a three-class label:
``1`` for long, ``0`` for flat and ``-1`` for short. These classes are
derived from the future return over a configurable horizon, with the
threshold for long/short defined in ``cfg.yaml``.

### Training Pipeline

The training pipeline reads options from ``cfg.yaml``. The ``default_window_days``
setting determines how many days of data to load when ``--start-ts`` is omitted.
Other defaults such as ``learning_rate`` and ``num_boost_round`` for the
LightGBM trainer live under the ``regime_lgbm`` section, while feature
engineering parameters like ``rsi_period`` and ``volatility_window`` reside under
``features``. Swarm search settings are grouped beneath ``swarm``.
Configuration also includes an ``optuna`` section controlling hyperparameter
tuning—``n_trials`` defaults to ``100`` and ``direction`` is ``minimize``.
Adjust these values to customise the standard training behaviour.

Input data is validated to contain ``ts`` and ``price`` columns after any
renaming. Missing columns raise a ``ValueError``.

## GPU Setup

LightGBM wheels from PyPI do not include GPU support. Use the provided
PowerShell script ``build_lightgbm_gpu.ps1`` to compile LightGBM with OpenCL
and upload the resulting wheel automatically. Run the script from a Windows
environment with Visual Studio Build Tools installed:

```powershell
pwsh ./build_lightgbm_gpu.ps1
```

The script places the wheel under ``python-package/dist`` and pushes it to your
package index. When ``ml_trainer.py --use-gpu`` is invoked and no GPU-enabled
wheel is installed, the script will run automatically to build and install it.

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake .. -DUSE_OPENCL=ON
make -j$(nproc)
cd ../python-package
python setup.py install --precompile
```

Alternatively LightGBM can be compiled during installation using ``pip``.

```bash
pip install lightgbm --config-settings=cmake_args="-DUSE_OPENCL=ON"
```

See the [GPU tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
for full instructions. Install the AMD driver (or ROCm) so your GPU appears when
running `clinfo`.

Once LightGBM is built with GPU support you can train using the CLI:

```bash
python ml_trainer.py train regime --use-gpu
```

When `--use-gpu` is supplied, ``ml_trainer.py`` injects default GPU
settings into the LightGBM parameters automatically.  If you call the
training code directly you can set the device yourself after loading the
configuration:

```python
import yaml
with open("cfg.yaml") as f:
    params = yaml.safe_load(f)["regime_lgbm"]
params.setdefault("device", "opencl")
```

### ROCm on Windows for RX 7900 XTX

ROCm on Windows remains **experimental** and LightGBM relies on the OpenCL backend. Install the latest Adrenalin driver from AMD with WSL 2 support, then use an administrator PowerShell prompt to enable WSL and install Ubuntu:

```powershell
wsl --install
```

After rebooting, download the Radeon driver for WSL from AMD and install it on
Windows. Inside the Linux environment add the ROCm apt repository and install
the packages:

```bash
sudo apt update
sudo apt install rocm-dev rocm-smi rocm-utils
```

Restart WSL and verify that an OpenCL device is available by running `clinfo`.

## Running the CLI

Model training can be launched via the ``ml_trainer.py`` command line
interface.  Use ``--help`` to see all available options.

```bash
python ml_trainer.py --help
```

Use ``--no-generate-target`` if your dataset already includes a ``target``
column and you do not want the pipeline to create one.

To import historical trades from a CSV and upload them to Supabase run:

```bash
python ml_trainer.py train regime
python ml_trainer.py import-data \
  --source-url https://example.com/data.csv \
  --symbol BTC \
  --start-ts 2024-01-01T00:00:00Z \
  --end-ts 2024-01-02T00:00:00Z \
  --output-file trades.parquet \
  --batch-size 1000
```

If you already have a CSV on disk, pass its path instead of a URL:

```bash
python ml_trainer.py import-data \
  --source-url ./trades.csv \
  --symbol BTC \
  --start-ts 2024-01-01T00:00:00Z \
  --end-ts 2024-01-02T00:00:00Z \
  --output-file trades.parquet
```

### Federated Training

Passing the ``--federated`` flag runs a local federated **simulation**.  The
trainer splits the dataset into ``num_clients`` partitions and trains a model on
each in memory. Only the resulting weights are aggregated, so the individual
rows never mix. Client training now runs in **separate processes** for better
CPU utilisation. Set ``use_processes=False`` to disable multiprocessing during
testing or on low‑resource machines.

To run a true distributed federated session across machines, pass
``--true-federated`` instead.  This launches a Flower server and multiple
Flower clients that exchange parameters over the network while keeping each
participant's data local.
The trainer exposes two flavours of federated learning.  ``--federated``
starts a **local simulation** where the dataset is split into several
partitions on a single machine.  Each partition is trained separately and
the resulting models are averaged into a ``FederatedEnsemble``.  The number
of simulated clients and rounds defaults to ``10`` and ``20`` respectively in
the ``federated_regime`` section of ``cfg.yaml``.

``--true-federated`` launches a real Flower-based session.  Each Flower
client loads its own data and only the model weights are exchanged with the
server.  Use this option when coordinating training across multiple
machines.

If the `flwr` package is not installed, the CLI logs a warning and continues without that feature.

Example local simulation:

```bash
python ml_trainer.py train regime --federated \
  --start-ts 2024-01-01T00:00:00Z \
  --end-ts 2024-01-02T00:00:00Z
```

Example Flower-based session:

```bash
python ml_trainer.py train regime --true-federated \
  --start-ts 2024-01-01T00:00:00Z \
  --end-ts 2024-01-02T00:00:00Z
```

The aggregated model is written to ``federated_model.pkl`` and uploaded
to the ``models`` bucket in Supabase just like other trained models.

Programmatic access is also available via
``federated_trainer.train_federated_regime``:

```python
from coinTrader_Trainer import federated_trainer

ensemble, metrics = federated_trainer.train_federated_regime(
    "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z",
    use_processes=False,  # disable multiprocessing when resources are limited
)
```
By default each client is trained in a separate process. Pass
``use_processes=False`` to fall back to threading when spawning processes is
undesirable.

For the Flower-based variant, call ``federated_fl.start_server`` on the
aggregation host and ``federated_fl.start_client`` on each participant:

```
from coinTrader_Trainer import federated_fl

# server side
federated_fl.start_server(num_rounds=20)

# client side
federated_fl.start_client(
    "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"
)
```

When ``SUPABASE_URL`` and credentials are present, the resulting ensemble is
uploaded to the ``models`` bucket automatically.
If the upload fails for any reason, the error now propagates so that
calling code can handle it appropriately.

### Optuna Hyperparameter Tuning

``ml_trainer.py`` also integrates [Optuna](https://optuna.org) for automated
learning rate search. Enable tuning with the ``--optuna`` flag. The optimiser
performs 50 trials by default before training the final model.

```bash
python ml_trainer.py train regime --optuna
```

If the `optuna_search` or `optuna_optimizer` package is not installed, the CLI logs a warning and continues without that feature.

### Importing Historical Data

Use the ``import-data`` command to load a CSV of trade logs from a URL or
local path and upload the rows to Supabase. Specify the desired time window:

```bash
python ml_trainer.py import-data \
  --source-url https://example.com/data.csv \
  --symbol BTC \
  --start-ts 2023-01-01T00:00:00Z \
  --end-ts 2023-01-02T00:00:00Z \
  --output-file trades.parquet \
  --batch-size 1000
```

To process a local file, replace the URL with the file path. The command writes
the parsed rows to the given output file before inserting them into Supabase.
Files downloaded from CryptoDataDownload include a banner line at the top of the
CSV. ``import-data`` now detects and skips this line automatically even when
reading from a local path.

Columns like ``Volume USDT`` from CryptoDataDownload are automatically renamed
to ``volume_usdt`` when using ``import-csv``. Asset-specific variants such as
``Volume XRP`` are renamed to ``volume_xrp``.

``import-data`` expects trade log entries with the same columns that the live
system stores in ``trade_logs``.  For importing OHLCV datasets use the
``import-csv`` command instead.  It reads a CSV of candle data and inserts the
rows into the requested table:

```bash
python ml_trainer.py import-csv ./prices.csv --symbol BTC --table historical_prices
```

### Creating the `historical_prices` table

The import tools expect a table named `historical_prices` to exist in your
Supabase project. Symbol-specific tables such as `historical_prices_btc` are
created by cloning this schema. The importer writes the raw columns (`unix`,
`date`, `symbol`, `open`, `high`, `low`, `close`, `volume_xrp`, `volume_usdt`,
`tradecount`) and relies on the database to populate a `timestamp` column. One
possible schema is:

```sql
CREATE TABLE historical_prices (
    unix BIGINT,
    date TIMESTAMPTZ,
    symbol TEXT,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume_xrp DOUBLE PRECISION,
    volume_usdt DOUBLE PRECISION,
    tradecount INTEGER,
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (
        COALESCE(to_timestamp(unix / 1000), date)
    ) STORED PRIMARY KEY
);
```

### Maintaining the `ohlc_data` partitions

The `ohlc_data` table is range partitioned on the `ts` column. Ingestion scripts
such as `kraken_fetch.py` depend on each partition enforcing uniqueness so that
`(symbol, ts)` is globally unique. When preparing future months create the next
partition and add the constraint:

```sql
CREATE TABLE IF NOT EXISTS ohlc_data_2025_01
    PARTITION OF ohlc_data
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
ALTER TABLE ohlc_data_2025_01 ADD UNIQUE (symbol, ts);
```

An automated job can handle this step. With `pg_cron` installed you might
schedule a function each month to create the upcoming partition:

```sql
SELECT cron.schedule(
    'create_ohlc_partition',
    '0 0 25 * *',
    $$
    DO $$
    DECLARE
        start_ts date := date_trunc('month', now()) + interval '1 month';
        end_ts   date := start_ts + interval '1 month';
        name     text := format('ohlc_data_%s', to_char(start_ts, 'YYYY_MM'));
    BEGIN
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF ohlc_data FOR VALUES FROM (%L) TO (%L);',
            name, start_ts, end_ts
        );
        EXECUTE format('ALTER TABLE %I ADD UNIQUE (symbol, ts);', name);
    END;
    $$;
    $$
);
```


## GPU Training

LightGBM must be built with OpenCL/ROCm support to train on AMD GPUs such
as a Radeon RX 7900 XTX. Before building, open a Windows or WSL shell and
run ``clinfo`` to verify that the GPU is detected. The helper script
``build_lightgbm_gpu.ps1`` also invokes ``clinfo`` and aborts if no
OpenCL device is available. Clone the LightGBM repository and build with
``-DUSE_OPENCL=ON``:

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake -DUSE_OPENCL=ON ..
make -j$(nproc)
```

Once installed, enable GPU mode with the ``--use-gpu`` flag when invoking
``ml_trainer.py``.  Optional ``--gpu-platform-id`` and ``--gpu-device-id``
can be provided to select a specific OpenCL device.

```bash
python ml_trainer.py train regime --use-gpu --gpu-device-id 0
```

Pass ``--profile-gpu`` to monitor utilisation. This integration is
available only on **Linux**.
On Linux the CLI launches ``rocm-smi --showuse --interval 1`` and
attempts to start [AMD RGP](https://gpuopen.com/rgp/) with
``rgp.exe --process <PID>``. If RGP is not found the command is printed so
you can start the profiler manually.
On Windows the flag merely checks that an OpenCL device is detected via
``clinfo``—no ``rocm-smi`` metrics are collected.

After installation, test training with a large dataset to verify the
OpenCL driver remains stable under load.

### PyTorch ROCm environment variables

When using the reinforcement-learning helpers with PyTorch on AMD hardware
ensure the proper ROCm environment variables are set so that PyTorch can
detect your GPU.  The most common settings are:

```bash
export PYTORCH_ROCM_ARCH=gfx1030  # adjust to match your GPU
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

Set these before installing or running PyTorch.  On multi-GPU systems you may
also need to restrict visibility using ``HIP_VISIBLE_DEVICES`` or
``ROCR_VISIBLE_DEVICES``.

## Running Tests

Tests are executed with ``pytest``.  After installing dependencies,
run:

```bash
pytest
```

## Scheduled Training

Model training also runs automatically each night via GitHub Actions.
The job defined in
[`\.github/workflows/train.yml`](\.github/workflows/train.yml) executes at
3\:00 UTC on a macOS runner.  To use it, set the repository secrets
`SUPABASE_URL`, `SUPABASE_SERVICE_KEY` and `TELEGRAM_TOKEN`.

## Automatic Model Uploads

The `train_regime_lgbm` function uploads the trained model to Supabase
Storage when `SUPABASE_URL` and either `SUPABASE_SERVICE_KEY` or
`SUPABASE_KEY` are present in the environment. Uploaded artifacts are
stored in the `models` bucket and recorded in the `models` table.

Similarly, the swarm simulation uploads the best parameter set to the bucket
specified by `PARAMS_BUCKET` and logs a row in `PARAMS_TABLE`.

## Supabase Security

When row level security policies restrict direct table access, aggregated
statistics can be retrieved via the `aggregate-trades` Edge Function. The helper
`fetch_trade_aggregates` wraps this call and returns the JSON result as a
`pandas.DataFrame`:

```python
from datetime import datetime, timedelta
from coinTrader_Trainer.data_loader import fetch_trade_aggregates

end = datetime.utcnow()
start = end - timedelta(days=1)
df = fetch_trade_aggregates(start, end, symbol="BTC")
```

Set `SUPABASE_URL` and a service key in your environment to authenticate before
invoking the function.
Enable row level security on the core tables so that users can only access
their own data:

```sql
ALTER TABLE trade_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE models ENABLE ROW LEVEL SECURITY;
```

Policies then control who can read and write:

```sql
CREATE POLICY "User can read own trades"
  ON trade_logs FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Admins can upload models"
  ON models FOR INSERT TO authenticated
  WITH CHECK (auth.role() = 'admin');
```

Create an index on `user_id` in `trade_logs` so permission checks remain fast.
Model uploads still use `SUPABASE_SERVICE_KEY`, but data reads authenticate with
`SUPABASE_USER_EMAIL` and `SUPABASE_PASSWORD` (or `SUPABASE_JWT`).

### Fetching Kraken OHLC Data

`kraken_fetch.py` downloads recent candles from the Kraken API and writes them to
Supabase. By default rows are inserted into the `trade_logs` table. Set the
`KRAKEN_TABLE` environment variable or pass `--table` on the command line to use
a different destination:

```bash
python kraken_fetch.py --table ohlc_raw
```

### CCXT OHLC Imports

Another helper script relies on the [ccxt](https://github.com/ccxt/ccxt)
library to pull daily candles from other exchanges. It is usually scheduled with
cron so new rows are inserted every day:

```cron
0 4 * * * python ccxt_fetch_all.py --exchange binance --timeframe 1d --delay-sec 1 --table ohlc_data
```

The destination table defaults to the value of the `CCXT_TABLE` environment
variable (`ohlc_data` if unset). The CLI accepts `--exchange`, `--timeframe`,
`--delay-sec` and `--table` arguments to control what data is fetched and where
it is stored.

## Swarm Scenario Simulation

`swarm_sim.py` explores how multiple trading strategies perform when run in parallel. The simulator uses `networkx` to build a graph of market scenarios and evaluate parameter combinations across the nodes. Invoke it via the command line:

```bash
python ml_trainer.py --swarm
```

If the `swarm_sim` package is not installed, the CLI logs a warning and continues without that feature.

The script searches for optimal LightGBM parameters during the simulation. Once complete, those values are passed into `train_regime_lgbm` so subsequent training runs start from the tuned configuration.

## Backtesting

The project includes a minimal helper built on
[Backtrader](https://www.backtrader.com/) for evaluating prediction
signals. Default options such as starting ``cash`` and ``commission`` are
defined in ``cfg.yaml`` under the ``backtest`` section.

```python
from coinTrader_Trainer.evaluation import run_backtest, simulate_signal_pnl
import pandas as pd
import numpy as np

df = pd.read_parquet("ohlc.parquet")
preds = np.load("predictions.npy")
final_equity = run_backtest(df, preds)
metrics = simulate_signal_pnl(df, preds)
```

``run_backtest`` returns the final portfolio value while
``simulate_signal_pnl`` reports a dictionary of metrics for the same
signals. Those metrics include Sharpe squared, Sharpe, Sortino, maximum
drawdown, win rate, Calmar ratio and profit factor.
``simulate_signal_pnl`` computes Sharpe/Sortino along with drawdown,
win rate and other trade statistics for the same signals.

## Training Menu

The ``ml_trainer.py`` module provides a small interactive menu for
common training tasks. Launch it with ``python ml_trainer.py`` and
optionally pass ``--use-gpu`` to enable GPU-accelerated routines. The
menu offers entries for training various models including a logistic
regression signal model, LightGBM meta selector, LightGBM regime model,
PPO reinforcement-learning selector, contextual bandit selector and a
fallback LightGBM model. Each option first generates a CSV of simulated
trades using ``tools.backtest_strategies.backtest`` and uploads the
resulting model artefact to Supabase via :class:`ModelRegistry`.

## Monitoring and Automation

To monitor the trainer during long running jobs install the Prometheus client and run Grafana:

```bash
pip install prometheus_client
docker run -p 3000:3000 grafana/grafana
```

`ml_trainer.py` exposes metrics on port `8000`. Configure a new Grafana data source of type `Prometheus` pointing at `http://localhost:8000` and create a dashboard to visualise training loss or other statistics.

For automated retraining you can schedule the trainer via `cron`. The example below performs a weekly run with parameter optimisation enabled:

```cron
0 0 * * 0 python /path/to/ml_trainer.py train regime --optuna
```

Make sure the job inherits your Supabase credentials such as `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (or `SUPABASE_KEY`) and any other variables referenced by `ml_trainer.py`.


## Running the ML Training Application

* Activate the virtual environment:

  ```bash
  source .venv/bin/activate
  ```

* Execute `ml_trainer.py` to launch the menu-driven trainer and optionally enable GPU routines:

  ```bash
  python ml_trainer.py --use-gpu
  ```

* Use backtesting scripts such as `lstm_bot.py` or `mean_bot.py` to bootstrap training data.

* Schedule recurring retraining via `token_registry.py`.

* `crypto_bot.regime.regime_classifier` embeds a base64-encoded model as a fallback in case no parameters are available.


## Maintenance

Use `tools/cleanup_old_files.py` to remove stale files based on their last modification date. By default it scans the repository root, but any directory can be supplied:

```bash
python tools/cleanup_old_files.py            # remove files older than today from repo root
python tools/cleanup_old_files.py /path/to/data  # specify a custom directory
```

To run the cleanup automatically, schedule it with cron:

```cron
0 0 * * * python /path/to/coinTrader_Trainer/tools/cleanup_old_files.py /path/to/data
```
