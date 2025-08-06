import os
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    SUPABASE_BUCKET = 'models-bucket'
    DEFAULT_CSV_PATH = 'data/sample_market.csv'
    USE_GPU = torch.cuda.is_available()
    VOL_THRESHOLD = 0.02
