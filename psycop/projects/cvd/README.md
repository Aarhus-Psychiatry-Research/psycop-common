# CVD prediction
Project-specific code for predicting CVD

## Installation
All requirements are specified in the `*requirements.txt` files. To install the project, clone it locally and run:

`pip install invoke`
`inv install-requirements`

### 1. Cohort definition
First, the cohort is defined. 
```bash
cvd/  
├── feature_generation/ 
│   └── cohort_definition/
│   │   └── cvd_cohort_definition.py # defining the cohort
```

<br />


#### PREDICTION TIMESTAMPS
Prediction timestamp df is derived from CVDCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame and has the following format:

| dw_ek_borger | timestamp           |
|--------------|---------------------|
| 1            | yyyy-mm-dd 00:00:00 |


#### OUTCOME TIMESTAMPS
Outcome timestamps are derived from CVDCohortDefiner.get_outcome_timestamps() and the resulting df has the following format:

| dw_ek_borger | timestamp       | value        |
|--------------|---------------------|---------------------|
| 1            | yyyy-mm-dd 00:00:00 | 1|


### 2. Feature generation
Second, features are generated based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.

Relevant files in the psycop-common repository: 
```bash
cvd/  
├── feature_generation/ 
│   └── main.py # main driver for generating feature set - different feature layers are defined in the script
```

The resulting feature set can be found here (on Ovartaci): 
```bash
E:/shared_resources/cvd/
├── feature_set/ 
│   ├── flattened_datasets/ 
│   │   └── cvd_feature_set/
│   │   │   └── cvd_feature_set.parquet
```


### 3. Model training
Third, the model training procedure (hyperparameter tuning using cross-validation) is performed.

```bash
cvd/ 
├── model_training/
│   └── cvd_baseline.cfg # baseline configuration for model hyperparameter tuning
│   └── main.py # main script for performing all hyperparameter tuning experiments
```

The original hyperparameter tuning experiments are no longer available on MlFlow.

A replicated hyperparameter tuning run using predictor layer 1+2 is found on MlFlow as "CVD-hyperparam-tuning-layer-2-xgboost-disk-logged"

### 4. Model evaluation
Fourth, model evaluation is performed. The main models are retrained on the combined train+validation set and tested on the test set (geographic split):

```bash
cvd/
├── model_training/
│   └── eval_geographic_split.py # reconfigure, retrain and evaluate geographic stability (trained on east and west sites, evaluated on central sites)
│   └──eval_geographic_test_on_cfg_from_disk # script for retraining model on a cfg file that uses the hyperparameter settings as documented in the supplementary materials of the published article
```
#### Figures and eval_df

The following script is used to produce all figures and tables for the article:

```bash
cvd/
├── paper_outputs/
│   └── multi_run.py # script for producing comparative table
│   └── single_run.py # script for producing figures and tables for the main model (XGBoost model based only on layers 1 + 2)
```
The plots and tables from the paper are currently not available.


#### Figures and eval_df
The test runs, eval dfs and outputs for the best model from the replicated hyperparameter tuning run and the model replicated from the supplementary materials can be found here:



```bash
E:/shared_resources/scz_bp/
├── eval_runs/
│   └── CVD-hyperparam-tuning-layer-2-xgboost-disk-logged_best_run_evaluated_on_test/
│   │   └── outputs/ # all figures and tables
│   │   └── config.cfg # model config
│   │   └── eval_df.parquet
│   │   └── logs.log
│   │   └── sklearn_pipe.pkl # full model sklearn pipeline
│   └── CVD-replicate-model-in-paper_best_run_evaluated_on_test/
│   │   └── outputs/ # all figures and tables
│   │   └── config.cfg # model config
│   │   └── eval_df.parquet
│   │   └── logs.log
│   │   └── sklearn_pipe.pkl # full model sklearn pipeline
```

The evaluation dataframes are `polars.DataFrame` and have the following format:
| pred_time_uuid        | y | y_hat_prob |
|-----------------------|---|------------|
| 1-yyyy-mm-dd-00:00:00 | 0 | 0.001      |
