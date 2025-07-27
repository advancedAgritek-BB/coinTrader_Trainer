from __future__ import annotations

import logging
import os
import platform
import re
import subprocess

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pyopencl as cl  # type: ignore
except Exception as exc:  # pragma: no cover - pyopencl may be absent
    cl = None  # type: ignore
    logger.warning("pyopencl not available: %s", exc)


def verify_opencl() -> bool:
    """Return ``True`` if an AMD GPU OpenCL device is available."""
    if cl is None:
        raise ValueError("pyopencl not installed")

    platforms = cl.get_platforms()
    amd_found = False
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception:
            continue
        for dev in devices:
            vendor = getattr(dev, "vendor", "").lower()
            if "advanced micro devices" in vendor or "amd" in vendor:
                amd_found = True
                break
        if amd_found:
            break

    if not amd_found:
        raise ValueError("No AMD OpenCL device found")

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on system
        raise RuntimeError("rocm-smi not found") from exc
    except Exception as exc:  # pragma: no cover - unexpected subprocess errors
        raise RuntimeError(f"failed to run rocm-smi: {exc}") from exc

    output = result.stdout + result.stderr
    match = re.search(r"GPU\[\d+\]\s*:\s*(.+)", output)
    if match:
        logger.info("AMD GPU detected via ROCm SMI: %s", match.group(1).strip())
        return True

    raise RuntimeError("Failed to parse rocm-smi output")


def has_rocm() -> bool:
    """Return ``True`` if an AMD ROCm device is available."""
    try:
        return verify_opencl()
    except Exception:
        pass

    if platform.system() == "Windows":
        if os.path.exists("C:\\Program Files\\AMD\\ROCm") or "ROCM_PATH" in os.environ:
            return True

    return False
