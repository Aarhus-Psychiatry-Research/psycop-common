import re


def parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")

    feature_capitalised = feature_name[0].upper() + feature_name[1:]

    manual_overrides = {
        "Age_in_years": "Age (years)",
        "Sex_female": "Sex (m/f)",
        "Adm_day_count": "Day in admission (count)",
    }

    if feature_capitalised in manual_overrides:
        feature_capitalised = manual_overrides[feature_capitalised]

    return feature_capitalised


def parse_temporal_feature(full_string: str) -> str:
    """Takes a temporal feature name and returns a human readable version of it."""
    feature_name = re.findall(r"pred_(.*)?_within", full_string)[0]

    feature_name_mappings = {
        "skema_1": "Deprivation of freedom",
        "paa_grund_af_farlighed": "Deprivation of freedom due to danger",
        "tvangstilbageholdelse": "Forced detention",
        "farlighed": "Dangerousness",
        "af_helbredsmaessige_grunde": "Deprivation of freedom due to health",
        "physical_visits": "Hospital contacts",
        "physical_visits_to_somatic": "Hospital contacts at the somatic unit",
        "physical_visits_to_psychiatry": "Hospital contacts at the psychiatric unit",
    }

    special_characters = {"oe": "ø", "ae": "æ", "aa": "å"}

    if feature_name in feature_name_mappings:
        feature_name = feature_name_mappings[feature_name]
    else:
        for key in special_characters:
            if key in feature_name:
                feature_name = feature_name.replace(key, special_characters[key])

        words = feature_name.split("_")
        words[0] = words[0].capitalize()
        feature_name = " ".join(word for word in words)

    lookbehind = re.findall(r"within_(.*)?_days", full_string)[0]

    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    output_string = f"{feature_name} {lookbehind}-day {resolve_multiple}"

    return output_string


def parse_text_feature(full_string: str, warning: bool = True) -> str:
    """Takes a text feature name and returns a human readable version of it."""
    if "Count" in full_string:
        feature_name = re.findall(
            r"pred_aktuelt_psykisk-(.*)?_CountVectorizer_within",
            full_string,
        )[0]
        vectorizer = "BoW"
    elif "Tfidf" in full_string:
        feature_name = re.findall(
            r"pred_aktuelt_psykisk-(.*)?_TfidfVectorizer_within",
            full_string,
        )[0]
        vectorizer = "Tf-Idf"
    else:
        if warning:
            raise ValueError(
                f"feature_name {full_string} does not include 'CountVectorizer' or 'TfidfVectorizer'. Text feature parsing will not work optimally.",
            )
        feature_name = re.findall(
            r"pred_aktuelt_psykisk-(.*)?_within",
            full_string,
        )[0]
        vectorizer = "Unknown vectorizer"

    feature_name_mappings = {
        "besøg": "'visit'",
        "sover": "'sleeps'",
        "tilstand": "'condition'",
        "spørgsmål": "'question(s)'",
        "medicin": "'medicine'",
        "derhjemme": "'at home'",
        "bedre": "'better'",
        "personale": "'staff'",
        "føler": "'feels'",
        "behandling": "'treatment'",
        "selvmordstanker": "'suicidal thoughts'",
        "beskriver": "'describes'",
        "lade": "'lets'",  # ?
        "afdelingen": "'the department",
        "planer": "'plans'",
        "pt fortæller": "'patient tells'",
        "forpint": "'tormented'",
        "selvmordsrisiko": "'suicide risk'",
        "venlig": "'friendly'",
        "hjælper": "'helps'",
        "forsøger": "'tries'",
        "pn": "'pn'/'pro necessitate'",
        "morgen": "'morning'",
        "lejlighed": "'appartment'/'occasion",
        "snakker": "'talks'",
        "døren": "'the door'",
        "aftenen": "'the evening'",
        "indlæggelsen": "'the admission'",
        "angiver": "'states'/'mentions'",  # ?
        "kendt": "'known'",
        "inden": "'before'",
        "personalet": "'the staff'",
        "udskrevet": "'disharged'",
        "pt": "'pt'/'patient'",
        "sidder": "'sits'",
        "sovet": "'slept'",
    }

    if feature_name in feature_name_mappings:
        feature_name = feature_name_mappings[feature_name]
    else:
        words = feature_name.split("_")
        words[0] = words[0].capitalize()
        feature_name = " ".join(word for word in words)
        print(
            f"Warning: feature_name {feature_name} is not in feature_mapping and will not be translated for plotting.",
        )

    lookbehind = re.findall(r"within_(.*)?_days", full_string)[0]

    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    output_string = f"{vectorizer} {feature_name} {lookbehind}-day {resolve_multiple}"

    return output_string


def feature_name_to_readable(full_string: str, warning: bool = False) -> str:
    """Takes a feature name and returns a human readable version of it."""
    if "within" not in full_string:
        output_string = parse_static_feature(full_string)
    elif "Vectorizer" in full_string:
        output_string = parse_text_feature(full_string, warning=warning)
    else:
        output_string = parse_temporal_feature(full_string)

    return output_string
