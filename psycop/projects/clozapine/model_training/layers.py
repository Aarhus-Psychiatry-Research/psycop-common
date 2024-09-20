import copy
from collections.abc import Sequence
from functools import partial
from itertools import combinations
from pathlib import Path

from joblib import Parallel, delayed

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

LAYER_NAMES = [
    "demographics",
    "contacts",
    "medication-overall",
    "broset + suicide risk assessment",
    "diagnoses",
    "medication",
    "coercion",
    "lab tests",
    "layer_text",
]


def _make_layer_permuations() -> Sequence[Sequence[str]]:
    permutations = []
    for r in range(1, len(LAYER_NAMES) + 1):
        permutations.extend(combinations(LAYER_NAMES, r))

    return [list(permutation) for permutation in permutations]


def train_clozapine_layers(cfg: PsycopConfig):
    layer_permutations = _make_layer_permuations()
    cfg.mut("logger.*.mlflow.experiment_name", "clozapine/feature combinations")

    def _prepare_config_and_train(layer: Sequence[str], cfg: PsycopConfig) -> float:
        layer_cfg = copy.deepcopy(cfg)
        layer_cfg.add("logger.*.mlflow.run_name", f"{layer!s}")

        layer_cfg.mut(
            "trainer.preprocessing_pipeline.*.layer_selector.keep_matching",
            f".+_layer_({'|'.join(layer)}).+",
        )
        return train_baseline_model_from_cfg(layer_cfg)

    fit_fn = partial(_prepare_config_and_train, cfg=cfg)
    Parallel(n_jobs=10)(delayed(fit_fn)(layer=layer) for layer in layer_permutations)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    populate_baseline_registry()

    train_clozapine_layers(
        cfg=PsycopConfig().from_disk(Path(__file__).parent / "clozapine_baseline.cfg")
    )
