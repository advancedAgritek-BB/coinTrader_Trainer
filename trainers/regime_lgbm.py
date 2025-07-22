import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict


def train_regime_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    use_gpu: bool = True,
) -> Tuple[Booster, Dict[str, float]]:
    """Train LightGBM model using 5-fold stratified CV with early stopping.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels. Positive class represents 'long' regime.
    params : dict
        Parameters passed to ``lightgbm.train``.
    use_gpu : bool, optional
        Enable GPU training if ``True`` (default). When enabled the model is
        initialised with ``device_type='gpu'``, ``tree_learner='data'``,
        ``gpu_platform_id=0`` and ``gpu_device_id=0``.

    Returns
    -------
    Booster
        Model fitted on the full dataset.
    dict
        Dictionary with averaged CV metrics: ``accuracy``, ``f1``,
        ``precision_long`` and ``recall_long``.
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    best_iterations = []

    gpu_defaults = {
        "device_type": "gpu",
        "tree_learner": "data",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid)

        train_params = dict(params)
        if use_gpu:
            for k, v in gpu_defaults.items():
                train_params.setdefault(k, v)

        booster = lgb.train(
            train_params,
            train_set,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(params.get("early_stopping_rounds", 50), verbose=False)
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

    final_num_boost_round = int(np.mean(best_iterations)) if best_iterations else params.get("num_boost_round", 100)
    final_set = lgb.Dataset(X, label=y)

    final_params = dict(params)
    final_params.pop("early_stopping_rounds", None)
    if use_gpu:
        for k, v in gpu_defaults.items():
            final_params.setdefault(k, v)

    final_model = lgb.train(
        final_params,
        final_set,
        num_boost_round=final_num_boost_round,
    )

    return final_model, metrics
