OBS: needs to be updated!!

# Temporal stability of type 2 diabetes prediction model
Project-specific code for investigating the temporal stability of a type 2 diabetes prediction model.

## Running the pipeline

### 1. Cohort definition
First, the cohort is defined. 
```bash
t2d_extended/
├── feature_generation/
│   └── cohort_definition/
│   │   └── t2d_cohort_definer.py # defining the cohort
```

#### PREDICTION TIMESTAMPS
Prediction timestamps are derived from `T2DCohortDefiner2025.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas()` and has the following format:

| timestamp           | dw_ek_borger |
|---------------------|--------------|
| yyyy-mm-dd 00:00:00 | 1            |


#### OUTCOME TIMESTAMPS
Outcome timestamps are derived from `T2DCohortDefiner2025.get_outcome_timestamps()` and has the following format:

| timestamp           | dw_ek_borger | value |
|---------------------|--------------|-------|
| yyyy-mm-dd 00:00:00 | 1            | 1     |

### 2. Feature generation
Second, features are generated based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.

Relevant files in the psycop-common repository: 
```bash
t2d_extended/  
├── feature_generation/ 
│   └── main.py # main driver for generating feature set
```

The resulting feature set can be found here (on Ovartaci): 
```bash
E:/shared_resources/t2d_extended/  
├── feature_set/flattened_datasets/
│   └── t2d_extended_feature_set_2025/
│   │   └── t2d_extended_feature_set_2025.parquet
```

### 3. Model training
Third, the model training procedure is performed based on experiment configurations. 

```bash
t2d_extended/ 
├── model_training/
│   └── temporal.py # main driver for model training
│   └── t2d_extended.cfg 
```

The model is saved here:

```bash
E:/shared_resources/t2d/
├── model_eval/
│   └── urosepsis-helicoid/ 
│   │   └── * # all runs from hyperparameter tuning (both logistic regression and XGBoost)
│   └── urosepsis-helicoid-eval-on-test/ 
│   │   └── nonviolentstigmaria-eval-on-test/ # best XGBoost models retrained on test set
```


### 4. Model evaluation
Fourth, model evaluation is performed.
```bash
t2d/ 
├── paper_outputs/
│   └── selected_runs.py # script for retraining model and evaluating on test
│   └── config.py # configuring which hyperparameter tuning experiment to run on
│   └── aggregate_eval/
│   │   └── single_pipeline_full_eval.py # produces performance and robustness figures for main paper
│   │   └── supplementary_full_eval.py # runs full evaluation for the supplementary materials (XGBoost)
```

TEMPORAL EVALUATION
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
| ids | pred_time_uuids       | pred_timestamps     | outcome_timestamps  | y | age  | is_female | y_hat_prob | eval_hba1c_within_9999_days_count_fallback_nan |
|-----|-----------------------|---------------------|---------------------|---|------|-----------|------------|------------|
| 1   | 1-yyyy-mm-dd-00:00:00 | yyyy-mm-dd 00:00:00 | yyyy-mm-dd 00:00:00 | 0 | 20.2 | 0         | 0.001      | 3      |



Eval dataframes can be found as .parquet files in the model folders. Example of the location of the eval dataframe for the main model:

```bash
E:/shared_resources/t2d/
├── model_eval/
│   └── urosepsis-helicoid-eval-on-test/ 
│   │   └── nonviolentstigmaria-eval-on-test/ # best XGBoost model retrained on test set
│   │   │   └── cfg.json
│   │   │   └── cfg.pkl
│   │   │   └── evaluation_dataset.parquet # here it is (eval_df)!!
│   │   │   └── pipe_metadata.pkl
│   │   │   └── pipe.pkl
```


In the code, the evaluation_dataset.parquet files are read in and converted to the `EvalDataset` class, which is used for generating plots and figures.  

