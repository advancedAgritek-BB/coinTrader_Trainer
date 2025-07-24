from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pyopencl as cl  # type: ignore
except Exception as exc:  # pragma: no cover - pyopencl may be absent
    cl = None  # type: ignore
    logger.warning("pyopencl not available: %s", exc)


def verify_opencl() -> bool:
    """Return True if an AMD GPU OpenCL device is available."""
    if cl is None:
        raise ValueError("pyopencl not installed")

    platforms = cl.get_platforms()
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception:
            continue
        for dev in devices:
            vendor = getattr(dev, "vendor", "").lower()
            if "advanced micro devices" in vendor or "amd" in vendor:
                return True
    raise ValueError("No AMD OpenCL device found")
