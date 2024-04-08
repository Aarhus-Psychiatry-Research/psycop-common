import traceback
from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()  # noqa: ERA001
    
    cfg_file = Path(__file__).parent / "config" / "ddpm_test.cfg"
    cfg = Config().from_disk(cfg_file)

    for lr in [0.0001, 0.0005, 0.001]:
        print(f"LR: {lr}")
        cfg_copy = cfg.copy()        
        cfg_copy["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["synthcity"]["model_params"]["lr"] = lr

        cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_copy))
        cfg_schema.logger.log_config(cfg_copy)
        try:
            result = cfg_schema.trainer.train()
        except Exception as e:
            cfg_schema.logger.fail(traceback.format_exc())