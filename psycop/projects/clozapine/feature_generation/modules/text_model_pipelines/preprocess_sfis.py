"""Preprocess sfis"""

from psycop.common.feature_generation.text_models.preprocessing import (
    text_preprocessing_pipeline,
)


def main() -> str:
    return text_preprocessing_pipeline()


if __name__ == "__main__":
    main()
