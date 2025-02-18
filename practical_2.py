import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv('StudentPerformance2.csv')
# print(df)

# col = ['Math_Score', 'Reading_Score' , 'Writing_Score','Placement_Score']
# df.boxplot(column=col)

# # # Display the plot
# plt.show()

# fig, ax = plt.subplots(figsize = (18,10))
# ax.scatter(df['Placement_Score'], df['Placement_Offer_Count'])
# plt.show()
# print(np.where((df['Placement_Score']<50) & (df['Placement_Offer_Count']>1)))
# print(np.where((df['Placement_Score']>85) & (df['Placement_Offer_Count']<3)))
# (array([], dtype=int64),)
# (array([], dtype=int64),)

z = np.abs(stats.zscore(df['Math_Score']))
# print(z)
threshold = 0.18
sample_outliers = np.where(z <threshold)
sample_outliers

sorted_rscore= sorted(df['Reading_Score'])
# # # print(sorted_rscore)

q1 = np.percentile(sorted_rscore, 25)
q3 = np.percentile(sorted_rscore, 75)
# # print(q1,q3)

IQR = q3-q1
lwr_bound = q1-(1.5*IQR)
upr_bound = q3+(1.5*IQR)
print(lwr_bound, upr_bound)

r_outliers = []
for i in sorted_rscore:
  if (i<lwr_bound or i>upr_bound):
   r_outliers.append(i)
# print(r_outliers)

new_df=df
for i in sample_outliers:
  new_df.drop(i,inplace=True)
# print(new_df)

df_stud=df
ninetieth_percentile = np.percentile(df_stud['Math_Score'], 90)
b = np.where(df_stud['Math_Score']>ninetieth_percentile,
ninetieth_percentile, df_stud['Math_Score'])
# print("New array:",b)
df_stud.insert(1,"m score",b,True)
# print(df_stud)

median=np.median(sorted_rscore)
print(median)