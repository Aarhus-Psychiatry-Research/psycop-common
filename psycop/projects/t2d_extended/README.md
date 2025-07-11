# Temporal stability of type 2 diabetes prediction model
Project-specific code for investigating the temporal stability of a type 2 diabetes prediction model.

## Running the pipeline

### 1. Cohort definition
First, define the cohort. 
```bash
t2d_extended/  
├── feature_generation/ 
│   └── cohort_definition/
│   │   └── t2d_cohort_definer.py # defining the cohort
```

### 2. Feature generation
Secondly, generate features based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.
```bash
t2d_extended/  
├── feature_generation/ 
│   └── main.py # main driver for generating feature set
```

### 3. Model training
Third, perform model training procedure based on experiment configurations.
```bash
t2d_extended/  
├── model_training/ 
│   └── t2d_extended.cfg # config file for model training
│   └── temporal.py # main driver for model training
```

### 4. Model evaluation
Fourth, perform model evaluation using your choice of evaluation metric. 
```bash
t2d_extended/  
├── model_evaluation/
│   └── config.py # config file for model evaluation
│   └── ............
```
