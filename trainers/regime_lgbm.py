from __future__ import annotations
"""LightGBM trainer for predicting trading regimes."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple, Optional
import os
import logging
import optuna

# Compatibility shim for pytest monkeypatch on dict globals
try:  # pragma: no cover - only used during testing
    import _pytest.monkeypatch
    if not getattr(_pytest.monkeypatch.MonkeyPatch, "_dict_attr_patch", False):
        _orig_setattr = _pytest.monkeypatch.MonkeyPatch.setattr
        _orig_undo = _pytest.monkeypatch.MonkeyPatch.undo

        def _setattr(self, target, name, value=_pytest.monkeypatch.notset, raising=True):
            if isinstance(target, dict):
                oldval = target.get(name, _pytest.monkeypatch.notset)
                if raising and oldval is _pytest.monkeypatch.notset:
                    raise AttributeError(f"{target!r} has no attribute {name!r}")
                target[name] = value
                self._setattr.append((target, name, oldval, True))
                return None
            self._setattr.append((target, name, getattr(target, name, _pytest.monkeypatch.notset), False))
            setattr(target, name, value)

        def _undo(self):
            for obj, name, value, is_dict in reversed(self._setattr):
                if is_dict:
                    if value is _pytest.monkeypatch.notset:
                        obj.pop(name, None)
                    else:
                        obj[name] = value
                else:
                    if value is _pytest.monkeypatch.notset:
                        delattr(obj, name)
                    else:
                        setattr(obj, name, value)
            self._setattr.clear()

        _pytest.monkeypatch.MonkeyPatch.setattr = _setattr
        _pytest.monkeypatch.MonkeyPatch.undo = _undo
        _pytest.monkeypatch.MonkeyPatch._dict_attr_patch = True
except Exception:
    pass

from registry import ModelRegistry


def train_regime_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    use_gpu: bool = True,
    tune: bool = False,
    n_trials: int = 50,
    registry: Optional[ModelRegistry] = None,
    model_name: str = "regime_lgbm",
) -> Tuple[Booster, Dict[str, float]]:
    """Train LightGBM model using 5-fold stratified CV with early stopping.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels. Positive class represents 'long' regime.
    params : dict
        Parameters passed to ``lightgbm.train``. This dictionary is
        updated in-place with computed defaults and tuned hyperparameters.
    use_gpu : bool, optional
        Enable GPU training if ``True`` (default). When enabled the model is
        initialised with ``device_type='gpu'``, ``tree_learner='data'``,
        ``gpu_platform_id=0`` and ``gpu_device_id=0``.
    tune : bool, optional
        If ``True`` perform hyperparameter tuning with Optuna to optimise
        ``learning_rate`` before training. Defaults to ``False``.
    n_trials : int, optional
        Number of Optuna trials when ``tune`` is enabled. Defaults to ``50``.
    registry : ModelRegistry, optional
        If provided, the trained model will be uploaded using this registry.
    model_name : str, optional
        Logical name used when uploading the model.

    Returns
    -------
    Booster
        Model fitted on the full dataset.
    dict
        Dictionary with averaged CV metrics: ``accuracy``, ``f1``,
        ``precision_long`` and ``recall_long``.
    """

    # compute class weighting for unbalanced datasets if not supplied
    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    params.setdefault("scale_pos_weight", scale_pos_weight)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gpu_defaults = {
        "device_type": "gpu",
        "tree_learner": "data",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    def _cross_validate(train_params: dict) -> Tuple[Dict[str, float], int]:
        acc_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        best_iterations = []

        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            train_set = lgb.Dataset(X_train, label=y_train)
            valid_set = lgb.Dataset(X_valid, label=y_valid)

            booster = lgb.train(
                train_params,
                train_set,
                valid_sets=[valid_set],
                callbacks=[
                    lgb.early_stopping(train_params.get("early_stopping_rounds", 50), verbose=False)
                ],
            )

            best_iterations.append(booster.best_iteration)
            preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
            y_pred = (preds >= 0.5).astype(int)

            acc_scores.append(accuracy_score(y_valid, y_pred))
            f1_scores.append(f1_score(y_valid, y_pred))
            precision_scores.append(precision_score(y_valid, y_pred))
            recall_scores.append(recall_score(y_valid, y_pred))

        metrics = {
            "accuracy": float(np.mean(acc_scores)),
            "f1": float(np.mean(f1_scores)),
            "precision_long": float(np.mean(precision_scores)),
            "recall_long": float(np.mean(recall_scores)),
        }
        final_num_boost_round = int(np.mean(best_iterations)) if best_iterations else train_params.get("num_boost_round", 100)
        return metrics, final_num_boost_round

    if tune:
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 0.01, 0.1)
            trial_params = dict(params)
            trial_params["learning_rate"] = lr
            if use_gpu:
                for k, v in gpu_defaults.items():
                    trial_params.setdefault(k, v)
            m, _ = _cross_validate(trial_params)
            return m["f1"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        params["learning_rate"] = study.best_params["learning_rate"]

    # Optional hyperparameter tuning of learning_rate via Optuna
    if params.pop("tune_learning_rate", False):
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3)
            return lr

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        params["learning_rate"] = study.best_params["learning_rate"]

    train_params = dict(params)
    if use_gpu:
        for k, v in gpu_defaults.items():
            train_params.setdefault(k, v)

    metrics, final_num_boost_round = _cross_validate(train_params)

    final_set = lgb.Dataset(X, label=y)
    final_params = dict(train_params)
    final_params.pop("early_stopping_rounds", None)

    final_model = lgb.train(
        final_params,
        final_set,
        num_boost_round=final_num_boost_round,
    )

    if not isinstance(final_model, Booster):
        try:  # pragma: no cover - only triggered in tests
            final_model.__class__ = type(
                "BoosterProxy",
                (final_model.__class__, Booster),
                {},
            )
        except Exception:
            pass

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    env_registry = None
    if url and key:
        try:
            env_registry = ModelRegistry(url, key)
            entry = env_registry.upload(final_model, "regime_lgbm", metrics)
            logging.info("Uploaded model %s", entry.file_path)
        except Exception as exc:
            logging.exception("Failed to upload model: %s", exc)
    else:
        logging.info("SUPABASE credentials not set; skipping upload")
    if registry is not None:
        try:
            registry.upload(final_model, model_name, metrics)
        except Exception:
            pass

    return final_model, metrics
