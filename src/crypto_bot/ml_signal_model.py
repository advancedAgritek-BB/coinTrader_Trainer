import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler


def train_from_csv(csv_path: str, use_gpu: bool = False) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a signal model from a CSV file.

    Parameters
    ----------
    csv_path: str
        Path to the CSV file containing market data.
    use_gpu: bool, optional
        Placeholder flag to enable GPU training in the future.

    Returns
    -------
    Tuple[LogisticRegression, StandardScaler]
        The trained estimator and the fitted scaler.
    """
    df = pd.read_csv(csv_path)

    # Derive binary labels: 1 if next close is higher, else 0
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['elasticnet'],
        'l1_ratio': [0.2, 0.5, 0.8],
        'solver': ['saga'],
    }

    base_model = LogisticRegression(max_iter=10000)
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_scaled, y)
    best_model = grid.best_estimator_

    f1_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='f1')
    y_pred_prob = cross_val_predict(best_model, X_scaled, y, cv=5, method='predict_proba')[:, 1]
    roc_auc = roc_auc_score(y, y_pred_prob)

    print(f"F1 scores: {f1_scores}")
    print(f"Mean F1 score: {f1_scores.mean():.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    joblib.dump({'model': best_model, 'scaler': scaler}, 'ml_signal_model.joblib')

    # Optional Supabase upload - uncomment and configure to use
    # from supabase import create_client
    # url = os.getenv('SUPABASE_URL')
    # key = os.getenv('SUPABASE_KEY')
    # client = create_client(url, key)
    # with open('ml_signal_model.joblib', 'rb') as f:
    #     client.storage.from_('models').upload('ml_signal_model.joblib', f)

    return best_model, scaler
