import pandas as pd

# Use read_excel to read an Excel file
dt = pd.read_csv("C:\\Users\\Welcome\\Desktop\\bigdata lab\\ODI_Match_Results.csv")
print(dt)
# for display 1st 5 records
print(dt.head())

# for checking rows and cols
print(dt.shape)
# for missing values in dataset
print(dt.isnull().sum())
# for statistics 
summary_statistics= dt.describe()
print(summary_statistics)
# dtypes
print(dt.dtypes)


dt['Margin'] = pd.to_numeric(dt['Margin'], errors='coerce')  
dt['BR'] = pd.to_numeric(dt['BR'], errors='coerce')

print(dt.dtypes)