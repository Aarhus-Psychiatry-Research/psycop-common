import time
from load_data import LoadData
from timeseriesflattener.flattened_dataset import FlattenedDataset
from wasabi import msg

prediction_times = LoadData.physical_visits()
event_times = LoadData.event_times()
hba1c_vals = LoadData.hba1c_vals()

flattened_df = FlattenedDataset(prediction_times_df=prediction_times)

# Predictors
msg.info("Adding predictors")
start_time = time.time()
predictor_list = [
    {
        "predictor_df": "hba1c_vals",
        "lookbehind_days": 365,
        "resolve_multiple": "latest",
        "fallback": 35,
        "source_values_col_name": "val",
        "new_col_name": "hba1c",
    }
]

predictor_dfs = {"hba1c_vals": hba1c_vals}


flattened_df.add_predictors_from_list_of_argument_dictionaries(
    predictor_list=predictor_list, predictor_dfs=predictor_dfs
)

end_time = time.time()
msg.good(f"Finished adding predictors, took {end_time - start_time}")


# # Outcomes
# msg.info("Adding outcome")
# start_time = time.time()

# flattened_df.add_outcome(
#     outcome_df=event_times, lookahead_days=1827, resolve_multiple="max", fallback=0
# )

# end_time = time.time()
# msg.good(f"Finished adding outcome, took {end_time - start_time}")
