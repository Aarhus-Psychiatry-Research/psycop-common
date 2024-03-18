import re


def parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")

    feature_capitalised = feature_name[0].upper() + feature_name[1:]

    manual_overrides = {"Age_in_years": "Age (years)"}

    if feature_capitalised in manual_overrides:
        feature_capitalised = manual_overrides[feature_capitalised]
    return feature_capitalised


def parse_temporal_feature(full_string: str) -> str:
    """Takes a temporal feature name and returns a human readable version of it."""
    feature_name = re.findall(r"pred_(.*)?_within", full_string)[0]

    feature_name_mappings = {
        "sex_female": "sex female",
        "physical_visits_to_psychiatry": "physical visits to psychiatry",
        "second_gen_antipsychotics": "2. generation antipsychotics",
        "first_gen_antipsychotics": "1. generation antipsychotics",
        "schizophrenia": "Schizophrenia",
        "f2_disorders": "F2-disorders",
        "baelte": "mechanical restraint",
        "skema_1": "skema 1",
        "beroligende_medicin": "acute sedative",
        "broeset_violence_checklist": "brøset violence checklist",
        "cancelled_standard_lab_results": "cancelled standard lab results",
    }

    if feature_name in feature_name_mappings:
        feature_name = feature_name_mappings[feature_name]
    elif "_disorders" in feature_name:
        words = feature_name.split("_")
        words[0] = words[0].capitalize()
        feature_name = " ".join(word for word in words)

    lookbehind = re.findall(r"within_(.*)?_days", full_string)[0]

    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    output_string = f"{lookbehind}-day {resolve_multiple} {feature_name}"
    return output_string


def feature_name_to_readable(full_string: str) -> str:
    """Takes a feature name and returns a human readable version of it."""
    if "within" not in full_string:
        output_string = parse_static_feature(full_string)
    else:
        output_string = parse_temporal_feature(full_string)

    return output_string
