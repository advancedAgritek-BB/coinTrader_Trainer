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

## Running Tests

Tests are executed with ``pytest``.  After installing dependencies,
run:

```bash
pytest
```
