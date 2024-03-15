"""Avert your gaze!"""

import polars as pl
import polars.selectors as cs

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, TEXT_EMBEDDINGS_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.text_experiment.feature_gen.scz_bp_text_experiment_feature_spec import (
    SczBpTextExperimentFeatures,
)

if __name__ == "__main__":
    ONLY_KEYWORDS = True
    project_path = OVARTACI_SHARED_DIR / "scz_bp" / "text_exp" / "keywords"
    project_info = ProjectInfo(project_name="scz_bp", project_path=project_path)

    note_types = ["aktuelt_psykisk", "all_relevant"]
    model_names = ["dfm-encoder-large", "dfm-encoder-large-v1-finetuned", "tfidf-500", "tfidf-1000"]

    if not ONLY_KEYWORDS:
        for note_type in note_types:
            for model_name in model_names:
                feature_set_name = f"text_exp_730_{note_type}_{model_name}"

                save_path = project_path / "flattened_datasets" / feature_set_name
                if save_path.exists():
                    print(f"{feature_set_name} already featurized. Skipping...")
                    continue

                generate_feature_set(
                    project_info=project_info,
                    eligible_prediction_times_frame=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times,
                    feature_specs=SczBpTextExperimentFeatures().get_feature_specs(
                        note_type=note_type, model_name=model_name, lookbehind_days=[730]
                    ),
                    n_workers=1,
                    do_dataset_description=False,
                    feature_set_name=feature_set_name,
                )

    # pse keywords
    base_feature_set_name = "text_exp_730_pse_keyword"
    save_path = project_path / "flattened_datasets" / base_feature_set_name

    pred_times = SczBpCohort.get_filtered_prediction_times_bundle().prediction_times
    filename = "pse_keyword_counts_all_sfis.parquet"
    embedded_text_df = pl.read_parquet(TEXT_EMBEDDINGS_DIR / filename).drop("overskrift")

    cols = embedded_text_df.drop("dw_ek_borger", "timestamp").columns
    # down cast
    embedded_text_df = embedded_text_df.with_columns(
        pl.col("dw_ek_borger").cast(pl.String)
    ).with_columns(
        cs.by_dtype(pl.NUMERIC_DTYPES).cast(pl.Int8), pl.col("dw_ek_borger").cast(pl.Int64)
    )
    print(f"n cols: {cols}")

    start_col = 0
    step = 30
    for end_col in range(step, len(cols), step):
        feature_set_name = f"{base_feature_set_name}_chunk_{start_col}_{end_col}"
        save_path = project_path / "flattened_datasets" / feature_set_name
        if save_path.exists():
            print(f"{feature_set_name} already featurized. Skipping...")
            continue
        print(f"Generating pse keyword features for chunk {start_col} to {end_col}...")

        sub_df = embedded_text_df.select(*cols[start_col:end_col], "timestamp", "dw_ek_borger")

        keyword_specs = SczBpTextExperimentFeatures().get_keyword_specs(
            lookbehind_days=[730], df=sub_df
        )

        print("Generating pse keyword features...")

        generate_feature_set(
            project_info=project_info,
            eligible_prediction_times_frame=pred_times,
            feature_specs=keyword_specs,
            n_workers=None,
            do_dataset_description=False,
            feature_set_name=f"{feature_set_name}_chunk_{start_col}_{end_col}",
        )
        start_col += step

    feature_set_name = f"{base_feature_set_name}_metadata_and_outcome"
    save_path = project_path / "flattened_datasets" / feature_set_name
    if save_path.exists():
        print(f"{feature_set_name} already featurized. Skipping...")
        quit()
    print("Generating metadata and outcome pse keyword features")
    keyword_specs = [
        SczBpTextExperimentFeatures()._get_outcome_specs(),  # type: ignore[reportPrivateUsage]
        SczBpTextExperimentFeatures()._get_metadata_specs(),  # type: ignore[reportPrivateUsage]
        SczBpTextExperimentFeatures().get_age_spec(),
    ]
    keyword_specs = [feature for sublist in keyword_specs for feature in sublist]
    generate_feature_set(
        project_info=project_info,
        eligible_prediction_times_frame=pred_times,
        feature_specs=keyword_specs,
        n_workers=None,
        do_dataset_description=False,
        feature_set_name=f"{feature_set_name}",
    )
