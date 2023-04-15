from psycop_feature_generation.text_models.preprocessing import text_preprocessing

if __name__ == "__main__":
    df = text_preprocessing(
        text_sfi_names="Aktuelt psykisk",
        include_sfi_name=True,
        n_rows=10,
        split_name=["train", "val"],
    )

    df
