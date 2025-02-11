import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('score.csv')
df.head()
# Check for missing values in the dataset
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)
df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['English_Score'].fillna(df['English_Score'].mean(), inplace=True)
df['Science_Score'].fillna(df['Science_Score'].mean(), inplace=True)
df['Attendance_Percentage'].fillna(df['Attendance_Percentage'].mean(), inplace=True)
missing_values_after_imputation = df.isnull().sum()
print("\nMissing Values:\n", missing_values_after_imputation)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Math_Score', 'English_Score', 'Science_Score', 'Attendance_Percentage']])
plt.title('Boxplot to Detect Outliers')
plt.show()
Q1 = df[['Math_Score', 'English_Score', 'Science_Score', 'Attendance_Percentage']].quantile(0.25)
Q3 = df[['Math_Score', 'English_Score', 'Science_Score', 'Attendance_Percentage']].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[['Math_Score', 'English_Score', 'Science_Score', 'Attendance_Percentage']] < (Q1 - 1.5 * IQR)) |
            (df[['Math_Score', 'English_Score', 'Science_Score', 'Attendance_Percentage']] > (Q3 + 1.5 * IQR)))
print("\nOutliers in the dataset:\n", outliers)
df['Log_Attendance_Percentage'] = np.log(df['Attendance_Percentage'] + 1)
print(df[['Attendance_Percentage', 'Log_Attendance_Percentage']].head())
