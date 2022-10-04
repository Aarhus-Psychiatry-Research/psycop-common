# import wandb
# from psycopt2d.utils import calculate_performance_metrics


# def test_log_performance_metrics(synth_data):
#     pass
# # wandb.init(project="test")
# perf = calculate_performance_metrics(
#     synth_data,
#     outcome_col_name="label",
#     prediction_probabilities_col_name="pred_prob",
#     id_col_name="dw_ek_borger",
# )
# # stupid test - is actested in psycop-ml-utils
# # mainly used to test wandb but doesn't work GH actions
# assert type(perf) == dict
# # wandb.log(perf)
# # wandb.finish()
