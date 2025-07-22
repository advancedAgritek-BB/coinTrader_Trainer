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

If you prefer to install packages individually:

```bash
pip install pandas numpy lightgbm scikit-learn supabase tenacity
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

### Async Data Fetching

`data_loader` now provides asynchronous helpers for retrieving trade logs
without blocking the event loop. `fetch_table_async` pages through a table while
`fetch_data_range_async` fetches rows between two timestamps. Both functions
return a ``pandas.DataFrame`` and must be awaited:

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

## GPU Setup

LightGBM wheels from PyPI do not include GPU support. Build from source and
enable the GPU backend:

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
for full instructions.  For an AMD Radeon RX 7900 TX install ROCm or the
appropriate OpenCL drivers so the GPU is recognized by `rocminfo` or `clinfo`.

Once LightGBM is built with GPU support you can train using the CLI:

```bash
python ml_trainer.py train regime --use-gpu
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
