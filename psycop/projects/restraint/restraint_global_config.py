# Run elements that are required before wandb init first,
# then run the rest in main so you can wrap it all in
# wandb_alert_on_exception, which will send a slack alert
# if you have wandb alerts set up in wandb
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

RESTRAINT_PROJECT_INFO = ProjectInfo(
    project_name="coercion", project_path=OVARTACI_SHARED_DIR / "coercion"
)
