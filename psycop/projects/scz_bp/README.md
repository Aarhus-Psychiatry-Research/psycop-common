# Prediction of progression to schizophrenia or bipolar disorder
Project-specific code for reproducing the results of [Predicting diagnostic progression to schizophrenia or bipolar disorder via machine learning applied to electronic health record data](https://doi.org/10.1101/2024.07.02.24309828).

## Running the pipeline

### 1. Cohort definition
First, define the cohort.
```bash
scz_bp/  
├──  
│   └── 
│   │   └── 
```

### 2. Feature generation
Secondly, generate features based on the cohort definition. Please note that the feature generation pipeline is dependent on access to the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without appropriate access.
```bash
scz_bp/  
├── feature_generation/
│   └── 
```

### 3. Model training
Third, perform model training procedure based on experiment configurations.
```bash
scz_bp/  
├── model_training/ 
│   └── 
│   └── 
```

### 4. Model evaluation
Fourth, perform model evaluation using your choice of evaluation metric.
```bash
scz_bp/  
├── evaluation/
│   └── 
│   └── 
```
