from __future__ import annotations

"""Train the meta selector model and upload it to Supabase."""

import os

from utils.upload import upload_to_supabase


def train_meta_selector(model_path: str = "meta_selector.pkl", dest: str | None = None) -> str:
    """Train a placeholder meta selector and upload the artifact.

    Parameters
    ----------
    model_path : str, optional
        Local path where the trained model will be written. Defaults to
        ``"meta_selector.pkl"``.
    dest : str, optional
        Destination path in Supabase storage. If omitted the file is uploaded
        to ``models/<basename>``.
    """
    # Placeholder for actual training logic. A real implementation would
    # produce a model object and serialise it to ``model_path``.
    with open(model_path, "wb") as fh:
        fh.write(b"meta selector model")

    remote = dest or f"models/{os.path.basename(model_path)}"
    upload_to_supabase(model_path, remote)
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    train_meta_selector()
