from __future__ import annotations

"""Train the regime model and upload it to Supabase."""

import os

from utils.upload import upload_to_supabase


def train_regime_model(model_path: str = "regime_model.pkl", dest: str | None = None) -> str:
    """Train a placeholder regime model and upload the artifact.

    Parameters
    ----------
    model_path : str, optional
        Local path where the trained model will be written. Defaults to
        ``"regime_model.pkl"``.
    dest : str, optional
        Destination path in Supabase storage. If omitted the file is uploaded
        to ``models/<basename>``.
    """
    # Placeholder for actual training logic. A real implementation would
    # produce a model object and serialise it to ``model_path``.
    with open(model_path, "wb") as fh:
        fh.write(b"regime model")

    remote = dest or f"models/{os.path.basename(model_path)}"
    upload_to_supabase(model_path, remote)
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    train_regime_model()
