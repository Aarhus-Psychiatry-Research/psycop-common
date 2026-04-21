import pandas as pd


def parse_diagnosegruppestreng_to_diagnoses(
    code_str: str, diagnosis_type: str | None = None
) -> list[str]:
    if pd.isna(code_str):
        return []

    parts = str(code_str).split("#")

    codes = []

    for p in parts:
        if ":" not in p:
            continue

        diag_prefix, code = p.split(":", 1)

        diag_prefix = diag_prefix.upper().strip()
        code = code.strip().lower()

        # ✅ REMOVE leading "d" from dataset codes
        if code.startswith("d"):
            code = code[1:]

        if diagnosis_type is None or diag_prefix == diagnosis_type:
            codes.append(code)

    return codes
