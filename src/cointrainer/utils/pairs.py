from __future__ import annotations
import re
from pathlib import Path

COMMON_QUOTES = [
    "USDT","USDC","USD","BTC","ETH","EUR","GBP","JPY","KRW",
    "BUSD","FDUSD","AUD","CAD","BRL","ARS","MXN","IDR","TRY","INR","NGN","ZAR"
]

def _strip_timeframe_tokens(stem: str) -> str:
    # remove things like _1m, -1m, _5, -5m, 1h, etc. at the end
    return re.sub(r"([_\-\.]?\d+[a-zA-Z]*)+$", "", stem)

def canonical_pair_from_filename(path: str | Path) -> str:
    """
    Return canonical pair (no separators, upper), e.g. 'BTCUSDT' from:
        BTC-USDT.csv, btcusdt-1m.csv, BTCUSD_1.csv
    """
    stem = Path(path).stem.upper()
    stem = _strip_timeframe_tokens(stem)
    stem = re.sub(r"[^A-Z0-9]", "", stem)
    return stem or "UNKN"

def slug_from_canonical(sym: str) -> str:
    """
    Insert '-' before a recognized quote to produce slugs like BTC-USDT.
    If no known quote is found, keep uppercase and add '-' in the middle at best-effort.
    """
    s = sym.upper().replace("-", "")
    for q in COMMON_QUOTES:
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            return f"{base}-{q}"
    # no known quote detected; try to split roughly in half
    if len(s) > 6:
        return f"{s[:-3]}-{s[-3:]}"
    return s

def canonical_from_slug(slug: str) -> str:
    """BTC-USDT -> BTCUSDT"""
    return slug.replace("-", "").upper()
