"""Example for loading structured SFIs."""

import psycop.common.feature_generation.loaders.raw.load_structured_sfi as struct_sfi_loader

if __name__ == "__main__":
    df = struct_sfi_loader.suicide_risk_assessment(n_rows=1000)

    input_dict = [{"predictor_df": "selvmordsrisiko", "allowed_nan_value_prop": 0.01}]
