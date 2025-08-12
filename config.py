from typing import ClassVar

import torch
from dotenv import load_dotenv

load_dotenv()


class Config:
    SUPABASE_BUCKET: ClassVar[str] = 'models-bucket'
    DEFAULT_CSV_PATH: str = 'data/sample_market.csv'
    USE_GPU: bool = torch.cuda.is_available()
    VOL_THRESHOLD: float = 0.02
