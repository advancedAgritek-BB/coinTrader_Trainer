from __future__ import annotations

import numpy as np
import pandas as pd

MEME_SYMBOL_DEFAULT = "SOL-MEME"  # global model tag for CT runtime

REQUIRED_BASE_COLS = {"ts", "price"}
OPTIONAL_COLS = {
    "close", "volume", "mktcap", "liq",
    "pair_created_at", "token_created_at",
    "holders", "unique_buyers_5m", "unique_sellers_5m",
    "txn_count_1m", "txn_count_5m", "is_honeypot", "rug_flag",
    "renounced", "has_mint_auth", "has_freeze_auth",
    "dev_hold", "dev_wallet_top10_pct", "dev_balance_delta_5m",
    "sentiment", "mentions_1m", "mentions_5m",
}

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)

def _to_epoch_s(ts):
    # Accept epoch seconds/ms or ISO str; return epoch seconds (float)
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return (s.view("int64") // 10**9).astype("float64")

def detect_meme_csv_columns(df: pd.DataFrame) -> bool:
    has_required = (set(df.columns) >= REQUIRED_BASE_COLS) or (
        {"ts", "close"} <= set(df.columns)
    )
    devish = any(
        c in df.columns
        for c in ("dev_hold", "renounced", "has_mint_auth", "has_freeze_auth")
    )
    social = any(c in df.columns for c in ("sentiment", "mentions_1m", "mentions_5m"))
    structure = any(
        c in df.columns for c in ("pair_created_at", "token_created_at", "holders")
    )
    return bool(has_required and (devish or social or structure))

def build_features_and_labels(
    raw: pd.DataFrame,
    horizon_rows: int = 15,      # ~15 minutes if 1m rows
    tp: float = 0.5,             # +50% take-profit threshold
    sl: float = 0.30,            # -30% stop threshold
    symbol_tag: str = MEME_SYMBOL_DEFAULT,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns X, y, meta where:
      y ∈ {+1 (snipe), 0 (flat), -1 (avoid)}
    """

    df = raw.copy()

    # Normalize price column name
    if "price" not in df and "close" in df:
        df["price"] = df["close"]

    # Time columns
    if df["ts"].dtype.kind in "OUSM":
        df["ts_s"] = _to_epoch_s(df["ts"])
    else:
        ts = df["ts"].astype("float64")
        # handle potential ms
        df["ts_s"] = np.where(ts > 10**12, ts / 1000.0, ts)

    # Age minutes since pair/token creation (when available)
    if "pair_created_at" in df:
        t0 = _to_epoch_s(df["pair_created_at"])
        df["age_min"] = (df["ts_s"] - t0) / 60.0
    elif "token_created_at" in df:
        t0 = _to_epoch_s(df["token_created_at"])
        df["age_min"] = (df["ts_s"] - t0) / 60.0
    else:
        df["age_min"] = 0.0

    # Basic market features
    df["ret_1"] = df["price"].pct_change(1).fillna(0.0)
    df["ret_5"] = df["price"].pct_change(5).fillna(0.0)
    df["vol_5"] = df["ret_1"].rolling(5).std().fillna(0.0)
    df["vol_15"] = df["ret_1"].rolling(15).std().fillna(0.0)
    df["liquidity"] = _col(df, "liq")
    df["mktcap"] = _col(df, "mktcap")
    df["volume"] = _col(df, "volume")

    # Orderflow / participation
    df["buyers_sellers_ratio_5m"] = (
        (_col(df, "unique_buyers_5m", 0.0) + 1.0) / (_col(df, "unique_sellers_5m", 0.0) + 1.0)
    )
    df["txn_rate_1m"] = _col(df, "txn_count_1m", 0.0)
    df["txn_rate_5m"] = _col(df, "txn_count_5m", 0.0)

    # Dev / safety signals
    df["dev_hold"] = _col(df, "dev_hold", 0.0).clip(0.0, 1.0)
    df["dev_top10_pct"] = _col(df, "dev_wallet_top10_pct", 0.0).clip(0.0, 1.0)
    df["dev_balance_delta_5m"] = _col(df, "dev_balance_delta_5m", 0.0)
    df["renounced"] = _col(df, "renounced", 0.0).astype(float)
    df["has_mint_auth"] = _col(df, "has_mint_auth", 0.0).astype(float)
    df["has_freeze_auth"] = _col(df, "has_freeze_auth", 0.0).astype(float)
    df["rug_or_honey"] = (
        _col(df, "is_honeypot", 0.0).astype(float)
        + _col(df, "rug_flag", 0.0).astype(float)
    ).clip(0.0, 1.0)

    # Sentiment & its velocity
    df["sentiment"] = _col(df, "sentiment", 0.0)
    df["sentiment_ma5"] = df["sentiment"].rolling(5).mean().fillna(0.0)
    df["sentiment_d1"] = df["sentiment"].diff().fillna(0.0)
    df["mentions_1m"] = _col(df, "mentions_1m", 0.0)
    df["mentions_5m"] = _col(df, "mentions_5m", 0.0)
    df["mentions_momentum"] = (df["mentions_5m"] + 1.0) / (df["mentions_1m"] + 1.0)

    # Holder concentration (proxy safety)
    df["holders"] = _col(df, "holders", 0.0)
    df["holder_conc_score"] = (df["dev_hold"] * 0.6 + df["dev_top10_pct"] * 0.4)

    # Forward-return labels over window of 'horizon_rows'
    # Forward max/min price within horizon
    fwd_max = df["price"].shift(-1).rolling(horizon_rows).max()
    fwd_min = df["price"].shift(-1).rolling(horizon_rows).min()
    ret_up = (fwd_max / df["price"]) - 1.0
    ret_dn = (fwd_min / df["price"]) - 1.0

    # Label: +1 snipe, -1 avoid, 0 flat
    label = np.where(
        (df["rug_or_honey"] > 0.5) | (ret_dn <= -sl), -1,
        np.where(ret_up >= tp, 1, 0)
    )

    # Drop tail rows with no forward window
    valid = ~(ret_up.isna() | ret_dn.isna())
    X = df.loc[valid, [
        "age_min","price","ret_1","ret_5","vol_5","vol_15",
        "liquidity","mktcap","volume",
        "buyers_sellers_ratio_5m","txn_rate_1m","txn_rate_5m",
        "dev_hold","dev_top10_pct","dev_balance_delta_5m",
        "renounced","has_mint_auth","has_freeze_auth","holder_conc_score",
        "sentiment","sentiment_ma5","sentiment_d1",
        "mentions_1m","mentions_5m","mentions_momentum",
        "rug_or_honey","holders"
    ]].fillna(0.0)

    y = pd.Series(label[valid], index=X.index, name="label").astype(int)

    feature_list = list(X.columns)
    meta = {
        "symbol": symbol_tag,
        "feature_list": feature_list,
        "horizon_rows": horizon_rows,
        "tp": tp,
        "sl": sl,
        "n_samples": int(valid.sum()),
        "target_names": { -1: "avoid", 0: "flat", 1: "snipe" },
    }
    return X, y, meta
