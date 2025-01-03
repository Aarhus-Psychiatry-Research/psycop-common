import pandas as pd

from pathlib import Path

file_path = Path("E:/shared_resources/cvd/feature_set/flattened_datasets/cvd_feature_set/cvd_feature_set.parquet")

df = pd.read_parquet(file_path)

# Count the number of columns
num_columns = df.shape[1]  # Access the second element of the shape tuple
print(f"Number of columns: {num_columns}")

# Calculate proportion of missing values for columns starting with "pred"
missing_proportions = {
    col: df[col].isnull().mean() for col in df.columns if col.startswith("_pred")
}

# Create a new DataFrame with the results
result_df = pd.DataFrame(list(missing_proportions.items()), columns=["Column", "Missing_Proportion"])


#save dataset to overtaci
file_path = Path("E:/anddan/CVD/missing_proportions.xlsx")

result_df.to_excel(file_path, index=False)