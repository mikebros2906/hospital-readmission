import pandas as pd
import numpy as np

df = pd.read_csv('../data/diabetic_cleaned.csv')
print("Loaded shape:", df.shape)

age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,
           '[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,
           '[80-90)':85,'[90-100)':95}
df['age'] = df['age'].map(age_map)

df['gender'] = df['gender'].map({'Male':1,'Female':0,'Unknown/Invalid':np.nan})

df['race'] = df['race'].fillna('Unknown')
df = pd.get_dummies(df, columns=['race'], drop_first=True)

med_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',
            'glimepiride','acetohexamide','glipizide','glyburide',
            'tolbutamide','pioglitazone','rosiglitazone','acarbose',
            'miglitol','troglitazone','tolazamide','examide',
            'citoglipton','insulin','glyburide-metformin',
            'glipizide-metformin','glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone']

med_map = {'No':0,'Steady':1,'Up':2,'Down':3}
for col in med_cols:
    df[col] = df[col].map(med_map).fillna(0).astype(int)

df['change'] = df['change'].map({'No':0,'Ch':1})
df['diabetesMed'] = df['diabetesMed'].map({'No':0,'Yes':1})

def map_diag(code):
    try:
        code = str(code)
        if code.startswith('V') or code.startswith('E'):
            return 0
        c = float(code)
        if 390<=c<=459 or c==785: return 1
        elif 460<=c<=519 or c==786: return 2
        elif 520<=c<=579 or c==787: return 3
        elif c==250: return 4
        elif 800<=c<=999: return 5
        elif 710<=c<=739: return 6
        elif 580<=c<=629 or c==788: return 7
        elif 140<=c<=239: return 8
        else: return 0
    except:
        return 0

df['diag_1'] = df['diag_1'].apply(map_diag)
df['diag_2'] = df['diag_2'].apply(map_diag)
df['diag_3'] = df['diag_3'].apply(map_diag)

df.dropna(inplace=True)

print("Final shape:", df.shape)
print("Remaining nulls:", df.isnull().sum().sum())

df.to_csv('../data/diabetic_features.csv', index=False)
print("Saved to data/diabetic_features.csv")