from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.test_utils.str_to_df import str_to_pl_df


from polars import LazyFrame


@BaselineRegistry.data.register("minimal_test_data")
class MinimalTestData(BaselineDataLoader):
    def __init__(self) -> None:
        pass

    def load(self) -> LazyFrame:
        data = str_to_pl_df(
            """ pred_time_uuid, pred_1, outcome,    outcome_val,    pred_age
                1,              1,      1,          1,              1
                2,              1,      1,          1,              99
                3,              1,      1,          1,              99
                4,              0,      0,          0,              99
                5,              0,      0,          0,              99
                6,              0,      0,          0,              99
                                        """,
        ).lazy()

        return data