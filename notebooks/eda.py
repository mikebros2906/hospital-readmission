import pandas as pd
import numpy as np

df = pd.read_csv('../data/diabetic_data.csv')
print("Shape:", df.shape)

df.replace('?', np.nan, inplace=True)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing': missing, 'pct': missing_pct})

print(missing_df[missing_df['missing'] > 0].sort_values('pct', ascending=False))

# Drop high-missing and irrelevant columns
df.drop(columns=['weight', 'max_glu_serum', 'A1Cresult', 
                 'medical_specialty', 'payer_code',
                 'encounter_id', 'patient_nbr'], inplace=True)

# Fix target variable — binary: readmitted within 30 days yes/no
df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
df.drop(columns=['readmitted'], inplace=True)

# Class balance
print("\nTarget distribution:")
print(df['readmitted_30'].value_counts())
print("\nClass %:")
print(df['readmitted_30'].value_counts(normalize=True).round(3))

# Shape after cleaning
print("\nShape after dropping:", df.shape)

# Save cleaned base
df.to_csv('../data/diabetic_cleaned.csv', index=False)
print("\nSaved to data/diabetic_cleaned.csv")

# Check age brackets
print("\nAge distribution:")
print(df['age'].value_counts().sort_index())

# Check numeric columns
print("\nNumeric columns:")
print(df.select_dtypes(include='number').columns.tolist())

# Check categorical columns
print("\nCategorical columns:")
print(df.select_dtypes(include='object').columns.tolist())