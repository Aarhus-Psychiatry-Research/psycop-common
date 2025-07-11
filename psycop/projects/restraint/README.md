# Restraint prediction
Project-specific code for predicting mechanical restraint and related coercive measures. 

## Running the pipeline

### 1. Cohort definition
First, the cohort is defined. 
```bash
restraint/  
├── feature_generation/ 
│   └── cohort/
│   │   └── restraint_cohort_definer.py # defining the cohort
```

OBS: The cohort was (and can be) generated using the file above. However, because this process takes a very long time for this specific project, a version of the cohort has also been saved and can be loaded through an SQL path. <br />


#### PREDICTION TIMESTAMPS
The SQL table with prediction timestamps is called `"fct.psycop_coercion_outcome_timestamps"` and has the following format:
| dw_ek_borger | timestamp_admission | timestamp_discharge | pred_adm_day_count | timestamp           |
|--------------|---------------------|---------------------|--------------------|---------------------|
| 1            | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00 | 2                  | yyyy-mm-dd 00:00:00 |


#### OUTCOME TIMESTAMPS
The SQL table with outcome timestamps is called `"fct.psycop_coercion_outcome_timestamps_2"` and has the following format:
| dw_ek_borger | datotid_start       | datotid_slut        | all_restraint       | mechanical_restraint | chemical_restraint  | manual_restraint    |
|--------------|---------------------|---------------------|---------------------|----------------------|---------------------|---------------------|
| 1            | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00  | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00 |


### 2. Feature generation
Second, features are generated based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.

Relevant files in the psycop-common repository: 
```bash
restraint/  
├── feature_generation/ 
│   └── modules/
│   │   └── loaders/
│   │   │   └── load_restraint_outcome_timestamps.py # load outcome timestamps (without generating features)
│   │   │   └── load_restraint_prediction_timestamps.py # load prediction timestamps (without generating features)
│   └── main.py # main driver for generating feature set
```

The resulting feature set can be found here (on Ovartaci): 
```bash
E:/shared_resources/restraint/  
├── flattened_datasets/ 
│   └── full_feature_set_structured_tfidf_750_all_outcomes/
│   │   └── full_with_pred_adm_day_count.parquet
```

#### TEXT
The text features in the feature set are derived from previously generated embedded text files. Information on the location of the relevant files are described here. 
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
Third, the model training procedure is performed based on experiment configurations. The following files handle model training for all models in the paper; the first two train models used in the main paper and the last three train models mentioned in the appendices. Each file is accompanied by a .cfg file (called within the script). 

```bash
restraint/ 
├── training/
│   └── restraint_mechanical_tuning.py # hyperparameter tuning for model with mechanical restraint as outcome - MAIN PAPER (experiment_name="restraint_mechanical_tuning_v2")
│   └── restraint_all_tuning.py # hyperparameter tuning for model with any restraint as outcome - MAIN PAPER (experiment_name="restraint_all_tuning_v2")
│   └── restraint_mechanical_tuning_minimal.py # hyperparameter tuning for parsimonious model (i.e., minimal feature set) - APPENDIX (experiment_name="restraint_mechanical_tuning_minimal_v2")
│   └── restraint_all_tuning_minimal.py # hyperparameter tuning for parsimonious model (i.e., minimal feature set) - APPENDIX (experiment_name="restraint_all_tuning_minimal_v2")
│   └── restraint_split_tuning.py # hyperparameter tuning for model trained on any restraint, validated on mechanical restraint in oof - APPENDIX (experiment_name="restraint_split_tuning_v2")
```


### 4. Model evaluation
Fourth, model evaluation is performed.
```bash
restraint/
├── evaluation/
│   └── retrain_models/
│   │   └── retrain_best_model_on_test.py # takes best run from both main models and trains model on train+val and evaluates on test set
│   │   └── retrain_best_model_on_test_minimal.py # takes best run from both parsimonious models and trains model on train+val and evaluates on test set
│   └── main.py # main driver for generating all figures and tables for the paper and appendix
```

#### EVAL_DF
The resulting evaluation dataframe is saved at XXXXXXXXXXXXXXXXXXXXXX and has the following format:
| pred_time_uuid        | y | y_hat_probs |
|-----------------------|---|-------------|
| yyyy-mm-dd-00:00:00-1 | 0 | 0.001       |

