# Agents in the ML Training Application for coinTrader2.0

This document outlines the machine learning agents integrated into the training suite. Each agent handles a specialized task, from signal classification to reinforcement learning. Models are trained via `ml_trainer.py`, persisted locally and in Supabase buckets, and many support GPU acceleration with ROCm on an AMD Radeon RX 7900 XTX.

## 1. Signal Classification Agent
- **File:** `crypto_bot/ml_signal_model.py`
- **Purpose:** Logistic regression classifier producing buy/sell signals.
- **Training:** `train_from_csv` uses ElasticNet regularization and GridSearchCV (`C`, `l1_ratio`). Evaluated with ROC-AUC and F1.
- **Integration:** Triggered through the `/train_model` Flask endpoint or the console menu. Bootstraps with backtested data (e.g., `lstm_bot.py`).
- **Persistence:** Saves model & scaler to `.pkl`; optionally uploaded to Supabase.
- **GPU Usage:** CPU-only via scikit-learn.

## 2. Meta-Selector Agent
- **File:** `tools/train_meta_selector.py`
- **Purpose:** LightGBM regressor selecting optimal strategies using trade statistics (win rate, Sharpe, etc.).
- **Training:** Builds features from trade history/backtests, trains with GPU-enabled LightGBM, early stopping and GridSearchCV (e.g., `learning_rate`, `num_leaves`). Evaluated with MSE/R2.
- **Integration:** Invoked from `ml_trainer.py`; bootstraps with simulated trades (e.g., `mean_bot.py`).
- **Persistence:** `meta_selector_lgbm.pkl` uploaded to Supabase.
- **GPU Usage:** `device='gpu'` for ROCm acceleration.

## 3. Regime Classification Agent
- **Files:** `tools/train_regime_model.py` (primary), `tools/train_fallback_model.py` (fallback), `regime_classifier.py` (loader)
- **Purpose:** LightGBM multi-class classifier detecting market regimes (low/medium/high volatility).
- **Training:** Uses trade logs/CSVs, focuses on high-volatility pairs. GPU LightGBM with early stopping and GridSearchCV (`num_leaves`, `n_estimators`). Evaluated via F1-macro and ROC-AUC (ovr). Fallback trainer produces a small synthetic model embedded as base64.
- **Integration:** `token_registry` can trigger asynchronous retraining; `regime_classifier.py` loads from Supabase or embedded model and schedules recovery on failure.
- **Persistence:** Primary `.pkl` stored in Supabase; fallback embedded in code.
- **GPU Usage:** `device='gpu'` for ROCm acceleration.

## 4. RL Strategy Selector Agent
- **File:** `rl/rl.py`
- **Purpose:** PPO-based reinforcement learning agent choosing strategies from trade history.
- **Training:** Custom Gym environment from CSVs, PPO with entropy coefficient, clip range annealing, and vectorized envs. Saved as `rl_selector.zip`.
- **Integration:** Option in `ml_trainer.py`; bootstraps with simulated trades from coinTrader2.0 strategies.
- **Persistence:** `.zip` optionally uploaded to Supabase.
- **GPU Usage:** `device='cuda'` for ROCm-accelerated training.

## 5. Contextual-Bandit Selector Agent
- **File:** `rl/strategy_selector.py`
- **Purpose:** LinUCB bandit selecting strategies per regime, balancing exploration/exploitation.
- **Training:** Aggregates mean PnL by regime/strategy from `strategy_pnl.csv`; computes UCB scores and saves mean/params to `.npy`.
- **Integration:** Called from `ml_trainer.py`; bootstraps with backtested PnL (e.g., `lstm_bot.py`).
- **Persistence:** `.npy` files, with optional base64 or Supabase upload.
- **GPU Usage:** CPU-based (NumPy).

## 6. LSTM Strategy Agent
- **File:** `crypto_bot/strategy/lstm_bot.py`
- **Purpose:** LSTM model predicting sequences for trading strategies.
- **Training:** External trainer with PyTorch (`AdamW`, early stopping, windowed sequences). Saves `state_dict` to `.pth`.
- **Integration:** Used in backtesting (`tools/backtest_strategies.py`) and can be retrained via the app.
- **Persistence:** `.pth` model; may be uploaded to Supabase.
- **GPU Usage:** `model.to('cuda')` for ROCm training/inference.

## 7. Mean-Reversion Strategy Agent
- **File:** `crypto_bot/strategy/mean_bot.py`
- **Purpose:** Scores mean-reversion setups via Ornstein-Uhlenbeck process.
- **Training:** Fits OU parameters (`mu`, `theta`, `sigma`) with `least_squares`; validates stationarity via ADF test and Hurst exponent. Saves to `.npy`.
- **Integration:** Used in backtesting to supply PnL for other agents.
- **Persistence:** `.npy` files; optional Supabase upload.
- **GPU Usage:** CPU-based (SciPy/NumPy).

## Automation and Fallback Mechanisms
- **Retraining:** `utils/token_registry.py` schedules retraining (e.g., regime model) when new tokens appear, optionally via cron in `ml_trainer.py`.
- **Fallbacks:** Embedded base64 models (e.g., in `regime_classifier.py`) keep agents functional if Supabase is unavailable and trigger recovery loops.
- **Bootstrapping:** All agents can train on simulated data from `tools/backtest_strategies.py`, leveraging coinTrader2.0 strategies on local historical CSVs.

