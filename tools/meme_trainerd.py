from __future__ import annotations
import os, time, logging, traceback
from pathlib import Path
from trainers.meme_trainer import train_meme_regime

logging.basicConfig(
    level=os.getenv("MCT_LOGLEVEL", "INFO"),
    format="%(asctime)s | meme_trainerd | %(levelname)s | %(message)s"
)

def main():
    input_path = Path(os.getenv("MCT_INPUT", "solana_meme_logs.csv"))
    interval_sec = int(os.getenv("MCT_INTERVAL_SEC", "1200"))   # default 20m
    symbol = os.getenv("MCT_SYMBOL", "SOL-MEME")
    use_gpu = os.getenv("MCT_USE_GPU", "1") == "1"
    federated = os.getenv("MCT_FEDERATED", "0") == "1"
    publish = os.getenv("MCT_PUBLISH", "1") == "1"

    last_size = -1
    last_mtime = 0.0

    logging.info(f"Starting meme_trainerd watching {input_path} every {interval_sec}s (GPU={use_gpu}, FED={federated}, PUBLISH={publish})")

    while True:
        try:
            if input_path.exists():
                stat = input_path.stat()
                changed = (stat.st_size != last_size) or (stat.st_mtime != last_mtime)
                if changed:
                    logging.info("Detected new data; training…")
                    train_meme_regime(
                        input_csv=str(input_path),
                        symbol=symbol,
                        use_gpu=use_gpu,
                        federated=federated,
                        publish=publish
                    )
                    last_size, last_mtime = stat.st_size, stat.st_mtime
                else:
                    logging.debug("No change detected.")
            else:
                logging.warning(f"Input file not found: {input_path}")

        except Exception as e:
            logging.error("Training loop error:\n" + "".join(traceback.format_exception(e)))
            # keep running; back off a bit

        time.sleep(interval_sec)

if __name__ == "__main__":
    main()
