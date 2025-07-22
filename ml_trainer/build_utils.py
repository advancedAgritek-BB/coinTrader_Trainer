import subprocess
from pathlib import Path
from registry import ModelRegistry

def build_and_upload_lightgbm_wheel(supabase_url: str, supabase_key: str) -> Path:
    """Build LightGBM GPU wheel and upload to Supabase Storage."""
    script = Path(__file__).with_name("build_lightgbm_gpu.ps1")
    subprocess.check_call(["powershell.exe", str(script)])
    wheel = next(Path("LightGBM/python-package/dist").glob("lightgbm-4.5.0-*.whl"))
    data = wheel.read_bytes()
    registry = ModelRegistry(supabase_url, supabase_key, bucket="wheels")
    registry.upload(data, name="lightgbm_gpu_wheel", metrics={})
    return wheel

