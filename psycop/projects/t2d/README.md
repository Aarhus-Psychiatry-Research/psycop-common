# Type 2 diabetes prediction
Project-specific code for reproducing results from [Development and validation of a machine learning model for prediction of type 2 diabetes in patients with mental illness](https://doi.org/10.1111/acps.13687). 

## Installation
All requirements are specified in the `pyproject.toml`. To install the project, clone it locally and run:

`pip install -r dev-requirements`

## Running the pipeline

### 1. Cohort definition
First, define the cohort. 
```bash
t2d/  
├── feature_generation/ 
│   └── cohort_definition/
│   │   └── t2d_cohort_definer.py # defining the cohort
```

### 2. Feature generation
Secondly, generate features based on the cohort definition. Please note that the feature generation pipeline is dependent on access the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without approriate access.

```bash
t2d/  
├── feature_generation/ 
│   └── main.py # main driver for generating feature set
```

### 3. Model training 
Third, perform model training procedure based on experiment configurations.
```bash
t2d/  
├── model_training/ 
│   └── config/ 
        └── default_config.yaml # config file for model training
│   └── train_models_in_parallel.py # main driver for model training
```

### 4. Model evaluation
Fourth, perform model evaluation using your choice of evaluation metric. 
```bash
t2d/  
├── paper_outputs/
│   └── config.py # config file for model evaluation
│   └── *
```