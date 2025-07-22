import lightgbm as lgb
import numpy as np
import pytest

def test_opencl_available():
    params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    try:
        lgb.train(params, lgb.Dataset([[1.0]], label=[0]), num_boost_round=1)
    except Exception:
        pytest.skip("GPU support not available")
        lgb.train(
            params,
            lgb.Dataset(np.array([[1.0]], dtype=float), label=np.array([0])),
            num_boost_round=1,
        )
    except lgb.basic.LightGBMError as exc:
        pytest.skip(f"OpenCL not available: {exc}")

