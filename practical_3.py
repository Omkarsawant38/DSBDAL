import pandas as pd
import numpy as np
from sklearn import preprocessing

df=pd.read_csv('StudentsPerformanceMF.csv')
# print(df.describe())

numeric_columns = ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score', 'Age', 'Placement_Offer_Count']

# # Calculate mean
# mean_values = df[numeric_columns].mean()

# # Calculate mode (this returns the most frequent value for each column)
# mode_values = df[numeric_columns].mode().iloc[0]

# # Calculate median
# median_values = df[numeric_columns].median()

# # Calculate minimum
# min_values = df[numeric_columns].min()

# # Calculate maximum
# max_values = df[numeric_columns].max()

# # For categorical columns like 'Gender' and 'Club_Join_Date', we calculate mode only
# mode_gender = df['Gender'].mode()[0]
# mode_club_join_date = df['Club_Join_Date'].mode()[0]

# # Print the results
# print("Mean values:\n", mean_values)
# print("\nMode values (numeric):\n", mode_values)
# print("\nMedian values:\n", median_values)
# print("\nMinimum values:\n", min_values)
# print("\nMaximum values:\n", max_values)
# print("\nMode for Gender:", mode_gender)
# print("\nMode for Club Join Date:", mode_club_join_date)



# enc = preprocessing.OneHotEncoder()
# enc_df = pd.DataFrame(enc.fit_transform(df[['Gender']]).toarray())
# # print(enc_df)



# df_encode =df.join(enc_df)
# print(df_encode)

std_deviation = df[numeric_columns].std()

# Print the standard deviation values
print("Standard Deviation values:\n", std_deviation)

