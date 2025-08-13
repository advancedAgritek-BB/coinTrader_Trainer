import types
import pandas as pd


def test_predict_fallback_on_inference_failure(monkeypatch):
    # Force registry to "work" but return a model that raises
    from cointrainer import registry as reg

    def fake_pointer(prefix: str):
        return {"feature_list": ["a","b","c","d","e"], "label_order": [-1,0,1]}
    def fake_latest(prefix: str, allow_fallback: bool = False):
        import pickle
        class BadModel:
            def predict_proba(self, X):  # always raises (shape or otherwise)
                raise RuntimeError("bad shape")
        return pickle.dumps(BadModel())
    monkeypatch.setattr(reg, "load_pointer", fake_pointer)
    monkeypatch.setattr(reg, "load_latest", fake_latest)

    from crypto_bot.regime.api import predict
    df = pd.DataFrame({"a":[1,2,3], "b":[2,3,4], "c":[3,4,5], "d":[4,5,6], "e":[5,6,7]})
    out = predict(df)
    assert out.action in {"long","flat","short"}
    assert out.meta and out.meta.get("source") == "fallback"

