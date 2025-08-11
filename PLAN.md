# Refactor Plan

- **Current structure**
  - Mix of runtime (`crypto_bot/` packages) and training utilities (top-level scripts, `tools/`, `trainers/`).
  - Data loading is split across `data_loader.py`, `data_import.py`, `historical_data_importer.py`, and `utils/data_loader.py`.
  - Models are persisted and retrieved through `registry.py` and `crypto_bot/regime/regime_classifier.py`.
  - Entry points include `ml_trainer.py`, `train_pipeline.py`, and scripts in `tools/`.
  - Tests live under `tests/` but depend on many optional services.

- **Goals**
  - Separate runtime prediction code from training utilities.
  - Reorganize repository into a `src/` layout and publish as an installable package.
  - Provide a minimal runtime API for coinTrader2.0: `crypto_bot.regime.api.predict(df)`.
  - Unify data access and feature engineering.
  - Supply a command line interface `cointrainer` for training and data import jobs.
  - Prune unused or obsolete modules, strengthen tests, and set up continuous integration.

- **Steps**
  1. **src layout**
     - Move all importable code into `src/cointrainer/`.
     - Keep runtime pieces (`crypto_bot/regime`, minimal helpers) separate from training modules (`trainers/`, feature engineering, evaluation).
     - Update `pyproject.toml` with package metadata and console scripts.
  2. **Runtime API**
     - Create `src/cointrainer/crypto_bot/regime/api.py` exposing `predict(df)` that wraps model loading from `regime_classifier`.
     - Ensure coinTrader2.0 imports only this API; document deprecation of direct access to `regime_classifier`.
  3. **Data loading consolidation**
     - Merge logic from `data_loader.py`, `utils/data_loader.py`, `data_import.py`, and `historical_data_importer.py` into a single `cointrainer.data` module with synchronous and asynchronous helpers and consistent caching.
     - Remove duplicated functions and adjust imports throughout.
  4. **Training package**
     - Gather model training code (`train_regime_model.py`, `tools/train_meta_selector.py`, etc.) under `cointrainer.training`.
     - Split runtime-only components from training-only modules; isolate heavy dependencies behind optional extras.
     - Replace scripts with callable functions to ease testing.
  5. **CLI**
     - Introduce `cointrainer` CLI using `typer` or `argparse` with subcommands:
       - `cointrainer train regime|meta|signal [options]`
       - `cointrainer import supabase|csv [options]`
       - `cointrainer backtest …`
     - Wire CLI commands to consolidated data and training modules.
  6. **Model registry & persistence**
     - Keep `ModelRegistry` but relocate to `cointrainer.registry` and use it from all trainers.
     - Standardize saving/loading conventions under `models/` bucket.
  7. **Cleanup**
     - Remove deprecated or unused files (e.g., `bootstrap_env.py`, `bandit_train.py`, `federated_fl.py`, legacy fetch scripts) after verifying they are not referenced.
     - Drop duplicated utilities in `utils/` once functionality is moved.
  8. **Testing**
     - Add unit tests for the new API, data loader, and CLI.
     - Reduce reliance on network calls by mocking Supabase/Redis.
     - Ensure existing tests are updated to new locations.
  9. **CI**
     - Add GitHub Actions workflow running lint (`ruff` or `flake8`) and `pytest` on push and pull requests.
     - Cache dependencies to keep builds fast.
  10. **Documentation**
      - Update README with installation instructions (`pip install cointrainer`) and usage examples for CLI and runtime API.
      - Document how coinTrader2.0 should import `crypto_bot.regime.api.predict`.

