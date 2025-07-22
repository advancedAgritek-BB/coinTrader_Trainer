import lightgbm as lgb

def test_opencl_available():
    params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    lgb.train(params, lgb.Dataset([[1.0]], label=[0]), num_boost_round=1)

