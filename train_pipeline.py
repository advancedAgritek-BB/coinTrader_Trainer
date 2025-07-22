"""Utilities for environment setup tasks like building LightGBM."""
from __future__ import annotations

import glob
import os
import platform
import subprocess
from pathlib import Path

from supabase import create_client


def ensure_lightgbm_gpu(supabase_url: str, supabase_key: str, script_path: str | None = None) -> bool:
    """Ensure a GPU-enabled LightGBM build and upload wheels to Supabase.

    When running on Windows and LightGBM with GPU support is not available,
    ``build_lightgbm_gpu.ps1`` is invoked via ``subprocess`` to build the
    wheel. All wheels produced are uploaded to the ``wheels`` bucket in
    Supabase using ``create_client``.

    Parameters
    ----------
    supabase_url : str
        Supabase project URL.
    supabase_key : str
        Service role key or other API key with storage access.
    script_path : str, optional
        Path to the PowerShell build script. Defaults to ``build_lightgbm_gpu.ps1``
        located next to this module.

    Returns
    -------
    bool
        ``True`` if a build was performed and wheels uploaded, ``False`` if
        skipped because GPU-enabled LightGBM was already installed or the
        platform is not Windows.
    """

    if platform.system() != "Windows":
        return False

    try:
        import lightgbm as lgb

        lgb.train(
            {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0},
            lgb.Dataset([[1.0]], label=[0]),
            num_boost_round=1,
        )
        return False
    except Exception:
        pass

    script = Path(script_path or Path(__file__).with_name("build_lightgbm_gpu.ps1"))
    subprocess.run([
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ], check=True)

    wheel_dir = script.with_name("LightGBM").joinpath("python-package", "dist")
    wheels = glob.glob(str(wheel_dir / "*.whl"))

    sb = create_client(supabase_url, supabase_key)
    bucket = sb.storage.from_("wheels")
    for whl in wheels:
        with open(whl, "rb") as fh:
            bucket.upload(os.path.basename(whl), fh)
    return True
