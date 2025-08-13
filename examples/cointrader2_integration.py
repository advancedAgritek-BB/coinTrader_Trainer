"""Minimal coinTrader2 runtime integration example."""

import pandas as pd

from crypto_bot.regime.api import predict


def main() -> None:
    """Build a mock feature row and print the prediction."""
    df = pd.DataFrame(
        [
            {
                "rsi_14": 50.0,
                "atr_14": 100.0,
                "ema_8": 20000.0,
                "ema_21": 20100.0,
            }
        ]
    )
    pred = predict(df)
    print("action:", pred.action, "score:", pred.score)


if __name__ == "__main__":
    main()
