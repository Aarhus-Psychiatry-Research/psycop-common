"""Example for loading text."""

import psycop_feature_generation.loaders.raw.load_text as t

if __name__ == "__main__":
    df = t.load_aktuel_psykisk(n_rows=1000)
