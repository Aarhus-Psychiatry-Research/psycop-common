from joblib import Memory
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

mem = Memory(location=OVARTACI_SHARED_DIR / "cache", verbose=1)
