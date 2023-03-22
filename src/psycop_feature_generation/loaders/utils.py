"""Example of how to get tfidf vocab."""


from psycop_feature_generation.featurizers import FEATURIZERS_PATH


def get_tfidf_vocab(
    n_features: int,
) -> list[str]:
    with open(
        FEATURIZERS_PATH / f"tfidf_{str(n_features)}.txt",
    ) as f:
        return f.read().splitlines()


TFIDF_VOCAB = {n: get_tfidf_vocab(n) for n in [100, 500, 1000]}
