import pandas as pd

# Load datasets
mat_df = pd.read_csv(r"student-mat.csv", sep=';')
por_df = pd.read_csv(r"student-por.csv", sep=';')

# Merge datasets on common keys
merge_keys = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
              'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']
merged_df = pd.merge(mat_df, por_df, on=merge_keys, suffixes=('_mat', '_por'))

# Drop unnecessary columns to prevent leakage
columns_to_exclude = ['G1_mat', 'G2_mat', 'G1_por', 'G2_por', 'G3_por']
data = merged_df.drop(columns=columns_to_exclude)

# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Save preprocessed data
data_encoded.to_csv("preprocessed_data.csv", index=False)
print("✅ Data preprocessing complete. Saved to 'preprocessed_data.csv'.")
