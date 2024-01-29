import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.utils import auroc_by_group
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH


def test_patchwork_grid(subsampled_synth_eval_df: pd.DataFrame):
    input_df = subsampled_synth_eval_df.rename(columns={"pred": "y", "pred_prob": "y_hat_probs"})

    df = auroc_by_group(
        df=input_df, groupby_col_name="is_female", confidence_interval=False
    ).reset_index()

    plots = [
        (pn.ggplot(df, pn.aes(x="is_female", y="auroc")) + pn.geom_bar(stat="identity"))
        for _ in range(6)
    ]

    patchwork = create_patchwork_grid(plots=plots, single_plot_dimensions=(3.5, 2.5), n_in_row=2)

    patchwork.savefig(TEST_PLOT_PATH / "patchwork.png")
