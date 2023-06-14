import pandas as pd
import plotnine as pn
from care_ml.model_evaluation.config import COLOURS, FIGURES_PATH, PN_THEME
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.data_loader.data_loader import DataLoader

# load train and test splits using config
cfg, _ = setup(
    config_file_name="default_config.yaml",
    application_config_dir_relative_path="../../../../../../care_ml/model_training/application/config/",
)

train_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="train")
test_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="val")

train_df["dataset"] = "train"
test_df["dataset"] = "test"

df = pd.concat([train_df, test_df])
df = df[df["outcome_coercion_bool_within_2_days"] == 1]

df["time_bin"] = pd.PeriodIndex(df["timestamp"], freq="Q").format()
df["n_in_bin"] = df.groupby("time_bin")["time_bin"].transform("count")

df_q = df[["time_bin", "n_in_bin"]].drop_duplicates(keep="first")

# create plot
p = (
    pn.ggplot(data=df_q, mapping=pn.aes(x="time_bin", y="n_in_bin"))
    + pn.geom_bar(stat="identity", fill=COLOURS["blue"], width=0.9)
    + pn.labs(
        x="Quarter",
        y="Count",
        title="Target days per quarter",
    )
    + pn.geom_text(
        mapping=pn.aes(label="n_in_bin"),
        size=8,
        va="bottom",
    )
    + PN_THEME
    + pn.theme(figure_size=(7, 4), axis_text_x=pn.element_text(rotation=90))
)


FIGURES_PATH.mkdir(parents=True, exist_ok=True)

save_path = FIGURES_PATH.parent.parent.parent / "coercion_target_days_by_time.png"

p.save(save_path, dpi=600)
