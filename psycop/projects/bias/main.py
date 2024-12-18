from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.model_training.training_output.dataclasses import get_predictions_for_positive_rate
from psycop.projects.bias.load_predictions import load_scz_predictions
from dalex.fairness import GroupFairnessClassification
import polars as pl
from psycop.projects.restraint.evaluation.evaluation_utils import parse_dw_ek_borger_from_uuid
import numpy as np

if __name__ == "__main__":

    ect = pl.DataFrame(load_scz_predictions())
    sex_df = pl.from_pandas(sex_female())
    
    eval_df = (
        parse_dw_ek_borger_from_uuid(ect).join(sex_df, on="dw_ek_borger", how="left")
    ).to_pandas().replace({True: "female", False: "male"})

    y_hat = np.array(get_predictions_for_positive_rate(
                desired_positive_rate=0.02, y_hat_probs=eval_df.y_hat_prob
            )[0])
    
    t=GroupFairnessClassification(y=np.array(eval_df.y), y_hat=y_hat, protected=eval_df.sex_female, privileged="male", label="base")
    t.fairness_check()
    pass