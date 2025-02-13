import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('StudentPerformance.csv')
print(df.head())

missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['Reading_Score'].fillna(df['Reading_Score'].mean(), inplace=True)
df['Writing_Score'].fillna(df['Writing_Score'].mean(), inplace=True)
df['Placement_Score'].fillna(df['Placement_Score'].mean(), inplace=True)
missing_values_after_imputation = df.isnull().sum()
print("\nMissing Values:\n", missing_values_after_imputation)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']])
plt.title('Boxplot to Detect Outliers')
plt.show()

Q1 = df[['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']].quantile(0.25)
Q3 = df[['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']] < (Q1 - 1.5 * IQR)) |
            (df[['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']] > (Q3 + 1.5 * IQR)))
print("\nOutliers in the dataset:\n", outliers)

df['Log_Placement_Score'] = np.log(df['Placement_Score'] + 1)
print(df[['Placement_Score', 'Log_Placement_Score']].head())