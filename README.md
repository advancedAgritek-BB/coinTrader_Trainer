# coinTrader Trainer

coinTrader Trainer is a small set of utilities for building machine
learning models for crypto trading strategies.  The project includes
helpers to fetch trading data from Supabase, generate technical
indicators, train models (e.g. a LightGBM regime classifier) and
store results back into Supabase storage.

## Installation

Create a virtual environment and install the required Python
dependencies.  A ``requirements.txt`` file is expected in the
repository, but you may also install the packages manually.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you update the repository at a later date, run the installation
command again so new dependencies such as ``pyyaml`` are installed.
For GPU-accelerated feature engineering install
[`cudf`](https://rapids.ai/). The package requires CUDA
and is not included in ``requirements.txt`` by default:

```bash
pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
```

If you prefer to install packages individually:

```bash
pip install pandas numpy lightgbm scikit-learn supabase tenacity pyarrow pytz
```

With the new ``src/`` layout install the package in editable mode so
Python can resolve imports:

```bash
pip install -e .
```

Modules can then be imported as normal.  For example:

```python
from coinTrader_Trainer import data_loader
```

### Trade Log Fetching and Caching

``fetch_trade_logs`` provides a simple synchronous interface for
downloading trade logs for a specific trading pair.  Pass UTC ``datetime``
objects for ``start_ts`` and ``end_ts``—naive timestamps are interpreted as
UTC.  The optional ``symbol`` argument filters rows to that pair.  When a
``cache_path`` is supplied the function will read from the Parquet file if
it exists and write new results back to this location, avoiding repeated
network requests.

### Async Data Fetching

`data_loader` now provides asynchronous helpers for retrieving trade logs
without blocking the event loop. `fetch_all_rows_async` retrieves every row in a
table, `fetch_table_async` pages through a table, and `fetch_data_range_async`
fetches rows between two timestamps. Each function returns a
``pandas.DataFrame`` and must be awaited:

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
* ``atr_period`` – average true range window (default ``3``)
* ``volatility_period`` – period used to compute log return volatility (default ``20``)
* ``ema_periods`` – list of exponential moving average periods to generate

GPU acceleration is possible when the `cudf` package is installed.  Pass
``use_gpu=True`` to ``make_features`` to switch to GPU-backed DataFrame
operations.

Set ``log_time=True`` to print the total processing time for feature
generation.

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
cmake .. -DUSE_GPU=1
make -j$(nproc)
cd ../python-package
python setup.py install --precompile
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
params.setdefault("device_type", "gpu")
```

## Running the CLI

Model training can be launched via the ``ml_trainer.py`` command line
interface.  Use ``--help`` to see all available options.

```bash
python ml_trainer.py --help
```

A typical training run might look like:

```bash
python ml_trainer.py --input data.csv --output model.pkl
```

## GPU Training

LightGBM must be built with OpenCL/ROCm support to train on AMD GPUs such
as a Radeon RX 7900 TX.  Clone the LightGBM repository and build with the
``USE_GPU`` flag enabled:

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake -DUSE_GPU=1 ..
make -j$(nproc)
```

Once installed, enable GPU mode with the ``--use-gpu`` flag when invoking
``ml_trainer.py``.  Optional ``--gpu-platform-id`` and ``--gpu-device-id``
can be provided to select a specific OpenCL device.

```bash
python ml_trainer.py train regime --use-gpu --gpu-device-id 0
```

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
