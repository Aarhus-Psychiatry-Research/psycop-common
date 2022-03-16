class GetData:
    def event_times():
        from pyprojroot import here
        import pandas as pd

        df = pd.read_csv(here("./csv/df_first_t2d_processed.csv"))
        return df

    def prediction_times(frac=1):
        from loaders import sql_load

        view = "[FOR_besoeg_fysiske_fremmoeder_inkl_2021_feb2022]"
        sql = "SELECT * FROM [fct]." + view

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df_subsampled = df.sample(frac=frac)

        prediction_times = df_subsampled[df_subsampled["besoeg"] == 1][
            ["dw_ek_borger", "datotid_start"]  # Keep only the relevant columns
        ]

        return prediction_times
