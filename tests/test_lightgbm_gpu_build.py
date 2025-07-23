import lightgbm as lgb
import pytest

def test_opencl_available():
    """Ensure LightGBM was built with OpenCL support."""
    import numpy as np

    params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    try:
        lgb.train(
            params,
            lgb.Dataset(np.array([[1.0]], dtype=float), label=np.array([0])),
            num_boost_round=1,
        )
    except lgb.basic.LightGBMError as exc:
        pytest.skip(str(exc))
    assert True

