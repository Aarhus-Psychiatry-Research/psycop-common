from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)

if __name__ == "__main__":
    populate_baseline_registry()
    cfg_file = Path(__file__).parent / "lookbehind_config.cfg"

    cfg = Config().from_disk(cfg_file)
    # change the regex blacklist here
    blacklist_dict = {
        "within_two_years": "pred_.+(yearly_interval|multiple_interval).*",
        "yearly_interval": "pred_.+(traditional|multiple_interval).*",
        "multiple_intervals": "pred_.+(traditional|yearly_interval).*",
    }

    for blacklist_name, blacklist_regex in blacklist_dict.items():
        cfg_copy = cfg.copy()
        cfg_copy["trainer"]["preprocessing_pipeline"]["*"]["regex_column_blacklist"][
            "*"
        ] += [blacklist_regex]

        cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_copy))
        result = cfg_schema.trainer.train()
        print(f"AUC: {result.metric.value}")
