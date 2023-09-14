import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.utils import auroc_by_group
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH


def test_patchwork_grid(subsampled_synth_eval_df: pd.DataFrame):
    input_df = subsampled_synth_eval_df.rename(
        columns={"pred": "y", "pred_prob": "y_hat_probs"},
    )

    df = auroc_by_group(
        df=input_df,
        groupby_col_name="is_female",
        confidence_interval=False,
    ).reset_index()

    from psycop.projects.forced_admission_inpatient.model_eval.config import FA_PN_THEME

    p = (
        pn.ggplot(df, pn.aes(x="is_female", y="auroc")) + pn.geom_bar(stat="identity")
    ) + FA_PN_THEME
    p.save(TEST_PLOT_PATH / "test_pn_theme.png", width=4, height=2.5, dpi=600)
