from pathlib import Path
from typing import Literal

from joblib import Memory
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, PSYCOP_PKG_ROOT

# If on Windows
if Path("E:/").exists():
    cache_dir = OVARTACI_SHARED_DIR / "cache"
else:
    cache_dir = PSYCOP_PKG_ROOT / "test_utils" / "test_outputs" / "sql_cache"

cache_dir.mkdir(parents=True, exist_ok=True)

mem = Memory(location=cache_dir, verbose=1)


def cast_str_to_split_name(
    split_str: Literal["train", "test", "val"],
) -> SplitName:
    match split_str:
        case "train":
            return SplitName.TRAIN
        case "val":
            return SplitName.VALIDATION
        case "test":
            return SplitName.TEST
