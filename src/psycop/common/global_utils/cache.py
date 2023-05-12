from pathlib import Path

from joblib import Memory  # pants: no-infer-dep
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, PSYCOP_PKG_ROOT

# If on Windows
if Path("E:/").exists():
    cache_dir = OVARTACI_SHARED_DIR / "cache"
else:
    cache_dir = PSYCOP_PKG_ROOT / "test_utils" / "test_outputs" / "sql_cache"

cache_dir.mkdir(parents=True, exist_ok=True)

mem = Memory(location=cache_dir, verbose=1)
