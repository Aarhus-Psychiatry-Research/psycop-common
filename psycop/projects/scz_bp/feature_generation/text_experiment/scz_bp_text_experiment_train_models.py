from pathlib import Path
from typing import Sequence

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def regex_match_string_anywhere(input: str) -> str:
    return f".*{input}.*"


def all_predictors_regex(note_types: Sequence[str], model_names: Sequence[str]) -> set[str]:
    return {
        regex_match_string_anywhere(f"{note_type}_{model_name}")
        for note_type in note_types
        for model_name in model_names
    }


if __name__ == "__main__":
    populate_baseline_registry()
    cfg_file = Path(__file__).parent / "text_exp_config.cfg"

    cfg = Config().from_disk(cfg_file)
    # change the regex blacklist here
    blacklist_dict = {
        "within_two_years": "pred_.+(yearly_interval|multiple_interval).*",
        "yearly_interval": "pred_.+(traditional|multiple_interval).*",
        "multiple_intervals": "pred_.+(traditional|yearly_interval).*",
    }

    note_types = ["aktuelt_psykisk", "all_relevant"]
    model_names = [
        "dfm-encoder-large",
        #            "e5-large",
        "dfm-encoder-large-v1-finetuned",
        "tfidf-500",
        "tfidf-1000",
    ]

    regex_match_all_predictors = all_predictors_regex(
        note_types=note_types, model_names=model_names
    )

    for note_type in note_types:
        for model_name in model_names:
            cfg_copy = cfg.copy()
            current_predictor_regex = set(regex_match_string_anywhere(f"{note_type}_{model_name}"))
            blacklist_all_but_current_predictor = (
                regex_match_all_predictors - current_predictor_regex
            )

            cfg_copy["trainer"]["preprocessing_pipeline"]["*"]["regex_column_blacklist"][
                "*"
            ] += list(blacklist_all_but_current_predictor)

            cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_copy))
            result = cfg_schema.trainer.train()
            print(f"{model_name}. AUC: {result.metric.value}")
