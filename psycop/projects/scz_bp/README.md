# Transition to schizophrenia/bipolar prediction
Project code for predicting diagnostic progression to schizophrenia or bipolar disorder via machine learning

## Running the pipeline

### 1. Cohort definition
First, the cohort is defined. 
```bash
scz_bp/  
├── feature_generation/ 
│   └── cohort_definition/
│   │   └── scz_bp_cohort_definition.py # defining the cohort
```

<br />


#### PREDICTION TIMESTAMPS
Prediction timestamp df is derived from SczBpCohort.get_filtered_prediction_times_bundle().prediction_times.frame and has the following format:

| dw_ek_borger | timestamp           |
|--------------|---------------------|
| 1            | yyyy-mm-dd 00:00:00 |


#### OUTCOME TIMESTAMPS
Outcome timestamps are derived from SczBpCohort.get_outcome_timestamps() and the resulting df has the following format:

| dw_ek_borger | timestamp       | value        |
|--------------|---------------------|---------------------|
| 1            | yyyy-mm-dd 00:00:00 | 1|


### 2. Feature generation
Second, features are generated based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.

Relevant files in the psycop-common repository: 
```bash
scz_bp/  
├── feature_generation/ 
│   └── main.py # main driver for generating feature set - different feature layers are defined in the script
```

The resulting feature set can be found here (on Ovartaci): 
```bash
E:/shared_resources/scz_bp/  
├── flattened_datasets/ 
│   └── full_feature_set_structured_tfidf_750_all_outcomes/
│   │   └── full_with_pred_adm_day_count.parquet
```

#### TEXT
The tfidf features in the feature set are derived from previously generated embedded text files.

The embedded text file can be found at:
```bash
E:/shared_resources/
├── text_embeddings/ 
│   └── text_embeddings_all_relevant_tfidf-1000.parquet
```

### 3. Model training
Third, the model training procedure (hyperparameter tuning using cross-validation) is performed.

```bash
scz_bp/ 
├── model_training/
│   └── scz_bp_baseline.cfg # baseline configuration for model hyperparameter tuning
│   └── hyperparam.py # main script for training models across three different feature set: only strcutured features, only text features, and both
```

The main hyperparameter tuning experiments are named 'scz_bp-trunc-and-hp-{feature_set_name}-xgboost-no-lookbehind-filter-best_run_evaluated_on_test' on MlFlow.

### 4. Model evaluation
Fourth, model evaluation is performed. The main models are retrained on the combined train+validation set and tested on the test set. Furthermore, models for evaluating temporal and geographic stability are trained and evaluated on the test set:

```bash
scz_bp/
├── model_training/
│   └── evaluate_models_on_test_set/
│   │   └── eval_random_split.py # reconfigure, retrain and evaluate main model
│   │   └── eval_geographic_split.py # reconfigure, retrain and evaluate geographic stability (trained on east and west sites, evaluated on central sites)
│   │   └── eval_temporal_splits.py # reconfigure, retrain and evaluate temporal stability
```

The model evaluation experiments are named e.g. 'scz_bp-trunc-and-hp-{feature_set_name}-xgboost-no-lookbehind-filter' on MlFlow.

#### Figures and eval_df

The following script is used to produce all figures and tables for the article:

```bash
scz_bp/
├── paper_outputs/
│   └── single_run.py
```

The figures and plots are saved in the same folder as the eval_df generated from model retraining and evaluation on the test set:

```bash
E:/shared_resources/scz_bp/
├── eval_runs/
│   └── scz_bp-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test/
│   │   └── figures/ # all figures and tables
│   │   └── config.cfg # model config
│   │   └── eval_df.parquet
│   │   └── logs.log
│   │   └── sklearn_pipe.pkl # full model sklearn pipeline
│   └── scz_bp-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test/
│   │   └── figures/ # all figures and tables
│   │   └── config.cfg # model config
│   │   └── eval_df.parquet
│   │   └── logs.log
│   │   └── sklearn_pipe.pkl # full model sklearn pipeline
│   └── * # similar structure for all the other experiments
│   └── figures/ # all figures for the temporal stability analyses
```

The evaluation dataframes are `polars.DataFrame` and have the following format:
| pred_time_uuid        | y | y_hat_prob |
|-----------------------|---|------------|
| 1-yyyy-mm-dd-00:00:00 | 0 | 0.001      |
