# Forced admission inpatient
Project-specific code for reproducing results from [Predicting involuntary admission following inpatient psychiatric treatment using machine learning trained on electronic health record data](https://doi.org/10.1017/S0033291724002642).

## Running the pipeline

### 1. Cohort definition
First, the cohort is defined. 
```bash
forced_admission_inpatient/  
├── cohort/
│   └── forced_admissions_inpatient_cohort_definition.py # defining the cohort
```


#### PREDICTION TIMESTAMPS
Prediction timestamps are derived from `ForcedAdmissionInpatientCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas()`* and has the following format:

| timestamp           | dw_ek_borger | age  |
|---------------------|--------------|------|
| yyyy-mm-dd 00:00:00 | 1            | 20.2 |

*The argument (`washout_on_prior_forced_admissions: bool`) to washout high-risk patients can be set in `get_filtered_prediction_times_bundle()`.

#### OUTCOME TIMESTAMPS
Outcome timestamps are derived from `ForcedAdmissionInpatientCohortDefiner.get_outcome_timestamps()` and has the following format:

| timestamp           | dw_ek_borger | value |
|---------------------|--------------|-------|
| yyyy-mm-dd 00:00:00 | 1            | 1     |


### 2. Feature generation
Second, features are generated based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.

Relevant files in the psycop-common repository: 
```bash
forced_admission_inpatient/  
├── feature_generation/ 
│   └── main.py # main driver for generating feature set
```

The resulting feature set can be found here (on Ovartaci): 
```bash
E:/shared_resources/forced_admission_inpatient/  
├── flattened_datasets/ 
│   └── full_feature_set_with_sentence_transformers_and_tfidf_750/ # main full feature set
│   │   └── test.parquet
│   │   └── train.parquet
│   │   └── val.parquet
│   └── * # subsets of the main full feature set also have their own folders (but can also be generated from the full feature set)
│   └── full_feature_set_with_sentence_transformers_and_tfidf_750_no_washout/ # feature set with high-risk patients for the secondary analysis
│   │   └── test.parquet
│   │   └── train.parquet
│   │   └── val.parquet
```

#### TEXT
The text features in the feature set are derived from previously generated embedded text files. Information on the location of the relevant files is described here. 
The preprocessed text can be found via the SQL name `"fct.psycop_train_val_test_all_sfis_preprocessed"`. The model is located at:
```bash
E:/shared_resources/
├── text_models/ 
│   └── tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.pkl
```
and the vocabulary at:
```bash
E:/shared_resources/
├── text_models/ 
│   └── vocabulary/
│   │   └── vocab_tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet
```

The resulting embedded text file can be found at:
```bash
E:/shared_resources/
├── text_embeddings/ 
│   └── text_train_val_test_tfidf_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet
```


### 3. Model training
Third, the model training procedure is performed based on experiment configurations. 

```bash
forced_admission_inpatient/ 
├── model_training/
│   └── train_models_in_parallel.py # main driver for model training
│   └── config/ 
│   │   └── * # contains subfolders with configuration files for model training
```

The models are saved here:
```bash
E:/shared_resources/forced_admission_inpatient/ 
├── models/
│   └── full_model_with_text_features_train_val/ # structured and text features
│   │   └── chuddahs-caterwauls/ # all runs from hyperparameter tuning (both logistic regression and XGBoost)
│   │   └── chuddahs-caterwauls-eval-on-test/ # best models (one logistic regression, one XGBoost) retrained on test set
│   │   |   └── abrasiometerintergradient-eval-on-test # best XGBoost
│   │   |   └── belgiumigdyr-eval-on-test # best logistic regression
│   └── full_model_without_text_features/ # only structured features
│   └── only_tfidf_model_train_val/ # only tfidf features
│   └── * # all other subsets and secondary analyses
```

### 4. Model evaluation
Fourth, model evaluation is performed.
```bash
forced_admission_inpatient/ 
├── model_eval/
│   └── selected_runs.py # script for retraining model and evaluating on test
│   └── config.py # configuring which hyperparameter tuning experiment to run on
│   └── aggregate_eval/
│   │   └── single_pipeline_full_eval.py # produces all* figures and tables for selected run, as determined by the config, (*except table one, calibration curve, and decision-curve-analysis)
```

#### CROSS-VALIDATION TABLE
```bash
forced_admission_inpatient/ 
├── model_eval/
│   └── model_description/
│   │   └── performance/
│   │   │   └── cross_validation_performance_table.py # generates table with cv performances for all models (different feature sets and secondary analyses)
```

#### TEMPORAL EVALUATION
To evaluate temporal stability, temporal cross-validation needs to be performed (adapted to model_training_v2 common code):
```bash
forced_admission_inpatient/ 
├── model_training/
│   └── inpatient_v2.cfg # model_training_v2-adapted config which can replicate the best XGBoost model from main model´
│   └── replace_sklearn_pipe.py # function for adapting the abovementioned config to the best logistic regression model
│   └── temporal.py # run temporal cross validation (logs models on mlflow - expriment name: "FA Inpatient - temporal cv") OBS - Mlflow logs are deleted and logging this way is outdated
```

To generate temporal stability plots:
```bash
forced_admission_inpatient/ 
├── model_eval/
│   └── model_permutation/
│   │   └── temporal_view.py
```

#### EVAL_DF
The resulting evaluation dataframe has the following format:
| ids | pred_time_uuids       | pred_timestamps     | outcome_timestamps  | y | age  | is_female | y_hat_prob |
|-----|-----------------------|---------------------|---------------------|---|------|-----------|------------|
| 1   | 1-yyyy-mm-dd-00:00:00 | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00 | 0 | 20.2 | 0         | 0.001      |



Eval dataframes can be found as .parquet files in the model folders. Example of the location of the eval dataframe for the main model:
```bash
E:/shared_resources/forced_admission_inpatient/ 
├── models/
│   └── full_model_with_text_features_train_val/ # structured and text features
│   │   └── chuddahs-caterwauls-eval-on-test/ 
│   │   │   └── abrasiometerintergradient-eval-on-test # best XGBoost
│   │   │   │   └── config.json
│   │   │   │   └── config.pkl
│   │   │   │   └── evaluation_dataset.parquet # here it is (eval_df)!!
│   │   │   │   └── pipe_metadata.pkl
│   │   │   │   └── pipe.pkl
│   │   │   └── belgiumigdyr-eval-on-test # best logistic regression
│   │   │   │   └── config.json
│   │   │   │   └── config.pkl
│   │   │   │   └── evaluation_dataset.parquet # here it is (eval_df)!!
│   │   │   │   └── pipe_metadata.pkl
│   │   │   │   └── pipe.pkl
```

In the code, the evaluation_dataset.parquet files are read in and converted to the class `EvalDataset` which is used for the plots and figures.  
