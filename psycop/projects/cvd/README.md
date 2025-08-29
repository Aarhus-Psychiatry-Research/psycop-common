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
├── feafture_generation/ 
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

The main hyperparameter tuning experiments are no longer available on MlFlow.


### 4. Model evaluation
Fourth, model evaluation is performed. The main models are retrained on the combined train+validation set and tested on the test set. Furthermore, models for evaluating geographic stability are trained and evaluated on the test set:

```bash
cvd/
├── model_training/
│   └── eval_random_split.py # reconfigure, retrain and evaluate main model
│   └── eval_geographic_split.py # reconfigure, retrain and evaluate geographic stability (trained on east and west sites, evaluated on central sites)
```

The test runs and resulting data frames are currently not available.

#### Figures and eval_df

The following script is used to produce all figures and tables for the article:

```bash
cvd/
├── paper_outputs/
│   └── multi_run.py # script for producing comparative table
│   └── single_run.py # script for producing figures and tables for the main model (XGBoost model based only on layers 1 + 2)
```
The plots and tables from the paper are currently not available.

