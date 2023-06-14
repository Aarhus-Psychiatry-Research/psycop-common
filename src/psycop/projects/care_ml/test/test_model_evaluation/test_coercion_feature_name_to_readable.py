from care_ml.model_evaluation.utils.feature_name_to_readable import (
    feature_name_to_readable,
)


def test_feature_name_to_readable():
    feature_names = [
        "pred_broeset_violence_checklist_within_3_days_mean_fallback_nan",  # special character
        "pred_skema_1_within_1_days_bool_fallback_0",  # feature mapping
        "pred_paa_grund_af_farlighed_within_3_days_bool_fallback_0",  # dangerous feature mapping
        "pred_physical_visits_to_psychiatry_within_7_days_bool_fallback_0",  # visit mapping
        "pred_aktuelt_psykisk-besøg_CountVectorizer_within_7_days_concatenate_fallback_nan",  # CountVectorizer, feature mapping
        "pred_aktuelt_psykisk-pt fortæller_TfidfVectorizer_within_30_days_concatenate_fallback_nan",  # TfIdfVectorizer, feature mapping
        "pred_aktuelt_psykisk-ord_TfidfVectorizer_within_10_days_concatenate_fallback_nan",  # TfIdfVectorizer, no feature mapping
    ]

    expected_output = [
        "Brøset violence checklist 3-day mean",
        "Deprivation of freedom 1-day bool",
        "Deprivation of freedom due to danger 3-day bool",
        "Hospital contacts at the psychiatric unit 7-day bool",
        "BoW 'visit' 7-day concatenate",
        "Tf-Idf 'patient tells' 30-day concatenate",
        "Tf-Idf Ord 10-day concatenate",
    ]

    extracted_feature_names = []
    for feature_name in feature_names:
        extracted_feature_names.append(feature_name_to_readable(feature_name))

    assert extracted_feature_names == expected_output
