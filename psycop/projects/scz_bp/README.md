# Transition to schizophrenia/bipolar prediction
Project code for predicting diagnostic progression to schizophrenia or bipolar disorder via machine learning

## Running the pipeline

### 1. Cohort definition
First, the cohort is defined. 
```bash
scz_bp/  
├── feature_generation/ 
│   └── eligible_prediction_times/
│   │   └── scz_bp_prediction_time_loader.py # defining the cohort
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

The main diver script for generating features is (many different feature set combinations are generated for the article):
```bash
scz_bp/  
├── feature_generation/ 
│   └── scz_bp_generate_features.py # main driver for generating feature set
```

Other files in the folder are used to define feature layers, generate metadata features for tables, etc. 

The resulting feature sets can be found here (on Ovartaci): 
```bash
E:/shared_resources/scz_bp/  
├── flattened_datasets/ 
│   └── l1_l4-lookbehind_183_365_730-all_relevant_tfidf_1000_lookbehind_730.parquet
```

#### TEXT
Text models are fitted and text features generated in the *text_experiment* folder.

The text model used to generate the tf-idf features is located here:
```bash
E:/shared_resources/
├── text_models/ 
│   └── tfidf_region_split_train_val_all_relevant_preprocessed_sfi_type__ngram_range_12_max_df_09_min_df_2_max_features_1000.pkl
```

### 3. Model training
Third, the model training procedure (hyperparameter tuning using cross-validation) is performed.

```bash
scz_bp/ 
├── model_training/
│   └── hparam_search_all_combinations_scz_bp.py # main script for hyperparameter tuning all model experiments
```

Hyperparameter configs are found in:

```bash
scz_bp/ 
├── model_training/
│   └── config/
│   │   └──hparam_tuning/
```

The main hyperparameter tuning experiments are no longer available on MLflow.

### 4. Model evaluation
Fourth, model evaluation is performed. The best performing predictor set on the training set was the predictor set only containing the 1000 TFIDF features. The congfig for the models on the test set can be found in:

```bash
scz_bp/
├── model_training/
│   └── config/
│   │   └── test_set/
```

#### Eval df

The eval_df generated from model retraining and evaluation on the test set can be found here:

```bash
E:/shared_resources/scz_bp/
├── testing/
│   └── config.cfg # this is the config file for the model run on only TFIDF predictors on the tests set
│   └── eval_df.parquet # this is the resulting eval_df
```

The evaluation dataframes are `polars.DataFrame` and have the following format:
| pred_time_uuid        | y | y_hat_prob |
|-----------------------|---|------------|
| 1-yyyy-mm-dd-00:00:00 | 0 | 0.001      |
