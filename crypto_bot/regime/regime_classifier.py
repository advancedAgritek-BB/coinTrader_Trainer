import pickle
from utils.supabase_client import download_model, load_fallback_model
from utils.token_registry import schedule_retrain
from config import Config

FALLBACK_B64 = "your_base64_string_here"

def get_regime_model():
    try:
        data = download_model(Config.SUPABASE_BUCKET, 'regime_lgbm.pkl')
        return pickle.loads(data)
    except Exception:
        print("Fallback to embedded model")
        model = load_fallback_model(FALLBACK_B64)
        schedule_retrain('regime', 'immediate')
        return model
