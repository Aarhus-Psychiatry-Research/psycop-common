from psycop.common.feature_generation.loaders.raw import sql_load


def load_heart_procedure_codes():
    df = sql_load(
        query="SELECT * FROM [fct].[FOR_hjerte_procedurekoder_inkl_2021_feb2022] WHERE timestamp is NOT NULL"
    )

    pass


if __name__ == "__main__":
    load_heart_procedure_codes()
