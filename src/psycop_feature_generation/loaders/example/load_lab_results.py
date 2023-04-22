"""Example of loading lab results."""

import psycop_feature_generation.loaders.raw.load_lab_results as lb

if __name__ == "__main__":
    df = lb.cancelled_standard_lab_results(n_rows=5000)
