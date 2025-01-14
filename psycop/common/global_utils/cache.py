import datetime

from joblib import Memory
from psycop.automation.environment import on_ovartaci
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, PSYCOP_PKG_ROOT


def shared_cache() -> Memory:
    # If on Windows
    if on_ovartaci():
        cache_dir = OVARTACI_SHARED_DIR / "cache"
    else:
        cache_dir = PSYCOP_PKG_ROOT / ".cache"

    cache_dir.mkdir(parents=True, exist_ok=True)

    shared_cache = Memory(location=cache_dir, verbose=1)
    shared_cache.reduce_size(age_limit=datetime.timedelta(hours=4))
    return shared_cache
