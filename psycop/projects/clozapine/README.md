# Clozapine prediction
Project-specific code for predicting clozapine initiation in patients with schizophrenia and schizoaffective disorder

## Running the pipeline

### 1. Cohort definition
First, the cohort is defined. 
```bash
restraint/  
├── feature_generation/ 
│   └── cohort_definition/
│   │   └── clozapine_cohort_definer.py # defining the cohort
```

#### PREDICTION TIMESTAMPS
Prediction timestamps are derived from `ClozapineCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas()` and has the following format:
| timestamp           | dw_ek_borger |
|---------------------|--------------|
| yyyy-mm-dd 00:00:00 | 1            |


#### OUTCOME TIMESTAMPS
Outcome timestamps are derived from `ClozapineCohortDefiner.get_outcome_timestamps()` and has the following format:

| timestamp           | dw_ek_borger | value |
|---------------------|--------------|-------|
| yyyy-mm-dd 00:00:00 | 1            | 1     |

Outcome timestamps are both derived from manual text validation (2013.01.01-2016.09.30) and structured prescription data (2016.10.01-2024.06.01)


### 2. Feature generation
Second, features are generated based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.

Relevant files in the psycop-common repository: 
```bash
clozapine/  
├── feature_generation/ 
│   └── clozapine_generate_features.py # main driver for generating full feature set (structured + tf-idf features)
│   └── clozapine_generate_unique_antipsychotics_feature_set.py # main driver for generating unique antipsychotics feature set

```

The resulting feature set can be found here (on Ovartaci): 
```bash
#full feature set (only text-models are derived from this in the model_training-scripts)
E:/shared_resources/clozapine/  
├── feature_set/flattened_datasets/
│   └── clozapine_full_feature_set_with_tfidf_2025_random_split/
│   │   └── clozapine_full_feature_set_with_tfidf_2025_random_split.parquet

#unique count antipsychotics
E:/shared_resources/clozapine/  
├── feature_set/flattened_datasets/
│   └── clozapine_demo_unique_count_antipsych/
│   │   └── clozapine_demo_unique_count_antipsych.parquet

#full feature set where clozapine/leponex as text predictor is removed (sensitivity analysis)
E:/shared_resources/clozapine/  
├── feature_set/flattened_datasets/
│   └── removed_text_predictor_clozapin_leponex_clozapine_full_feature_set_with_tfidf_2025_random_split/
│   │   └── removed_text_predictor_clozapin_leponex_clozapine_full_feature_set_with_tfidf_2025_random_split.parquet

```


#### TEXT
The text features in the feature set created from this code:
```bash
#preprocessing
clozapine/  
├── text_models/ 
│   └── preprocessing.py 

#fit tfidf-model
clozapine/  
├── text_models/ 
│   └── fit_and_save_TFIDF_model.py

#encode text with tfidf-model
clozapine/  
├── text_models/ 
│   └── encode_text_as_tfidf_scores.py
```

and the files are located at:
```bash
#preprocessed text are located at the regional sql-server:
"psycop_clozapine_train_val_test_all_sfis_preprocessed_added_psyk_konf_2025_random_split"

#fitted_tfidf_model
E:/shared_resources/
├── text_models/
│   └── tfidf_psycop_clozapine_preprocessed_added_psyk_konf_added_2025_random_split_train_val_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750

E:/shared_resources/
├── text_models/ 
│   └── vocabulary/
│   │   └── vocab_tfidf_psycop_clozapine_preprocessed_added_psyk_konf_added_2025_random_split_train_val_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet
```

The resulting embedded text file can be found at:
```bash
E:/shared_resources/
├── clozapine/
├── text_embeddings/ 
│   └── clozapine_text_tfidf_train_val_test_2025_random_split_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet.parquet
```


### 3. Model training
Third, the model training procedure is performed based on experiment configurations. The following files handle model training for all models in the paper;

```bash
#logreg
clozapine/ 
├── model_training/
├── log_reg/
│   └── run_all_logreg.py
#xgboost
clozapine/ 
├── model_training/
├── xgboost/
│   └── run_all_xgboost.py 

#Sensitivity analysis with no leponex and clozapine text predictors
clozapine/ 
├── model_training/
├── no_clozapine_leponex_tfidf_predictor/
│   └── hyperparam.py 

```


### 4. Model evaluation
Fourth, model evaluation is performed.
```bash
clozapine/
├── model_training/
│   └── eval_on_test_set_random_split.py # takes best run from the defined trained model in the script
clozapine/
├── model_eval/
│    # In this folder, there are 3 folders: performance, table_one, feature_importance
```

#### EVAL_DF
The resulting evaluation dataframe is a `polars.DataFrame` and has the following format:
| pred_time_uuid        | y | y_hat_prob |
|-----------------------|---|------------|
| 1-yyyy-mm-dd-00:00:00 | 0 | 0.001      |


```bash
E:/shared_resources/clozapine/
├── eval_runs/
│    # For each feature set there is 2 XGBoost models  and 2 Logistic regression models with different lookaheads (365 day and 730 day lookahead)
E:/shared_resources/clozapine/
├── eval/
├── tables/
│    # All tables
E:/shared_resources/clozapine/
├── eval/
├── figures/
│    # All figures
E:/shared_resources/clozapine/
├── eval/
├── predictor_importance/
│    # All predictor importance tables

```

