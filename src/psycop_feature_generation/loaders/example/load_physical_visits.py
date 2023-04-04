"""Example loader for physical visits."""

import psycop_feature_generation.loaders.raw as r

if __name__ == "__main__":
    psych = r.load_visits.physical_visits_to_psychiatry(timestamps_only=True)
    somatic = r.load_visits.physical_visits_to_somatic(n_rows=1000)

    print(f"Max date is {psych['timestamp'].max()}")
    pass
