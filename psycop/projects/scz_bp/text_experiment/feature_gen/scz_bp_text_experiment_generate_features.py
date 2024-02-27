from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set_tsflattener_v1,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.text_experiment.feature_gen.scz_bp_text_experiment_feature_spec import (
    SczBpTextExperimentFeatures,
)

if __name__ == "__main__":
    project_path = OVARTACI_SHARED_DIR / "scz_bp" / "text_exp"
    project_info = ProjectInfo(project_name="scz_bp", project_path=project_path)

    note_types = ["aktuelt_psykisk", "all_relevant"]
    model_names = ["dfm-encoder-large", "dfm-encoder-large-v1-finetuned", "tfidf-500", "tfidf-1000"]

    for note_type in note_types:
        for model_name in model_names:
            feature_set_name = f"text_exp_730_{note_type}_{model_name}"

            save_path = project_path / "flattened_datasets" / feature_set_name
            if save_path.exists():
                print(f"{feature_set_name} already featurized. Skipping...")
                continue

            generate_feature_set_tsflattener_v1(
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
    feature_set_name = "text_exp_730_pse_keyword"
    save_path = project_path / "flattened_datasets" / feature_set_name

    keyword_specs = [
        SczBpTextExperimentFeatures()._get_outcome_specs(),  # type: ignore[reportPrivateUsage]
        SczBpTextExperimentFeatures()._get_metadata_specs(),  # type: ignore[reportPrivateUsage]
        SczBpTextExperimentFeatures().get_keyword_specs(lookbehind_days=[730]),  # type: ignore[reportPrivateUsage]
    ]
    keyword_specs = [feature for sublist in keyword_specs for feature in sublist]

    print("Generating pse keyword features...")

    generate_feature_set(
        project_info=project_info,
        eligible_prediction_times_frame=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times,
        feature_specs=keyword_specs,
        n_workers=None,
        do_dataset_description=False,
        feature_set_name=feature_set_name,
    )
