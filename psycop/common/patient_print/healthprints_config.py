from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

HEALTHPRINTS_DATASETS_DIR = OVARTACI_SHARED_DIR / "healthprint" / "datasets"
HEALTHPRINTS_DATASETS_DIR.mkdir(exist_ok=True, parents=True)
