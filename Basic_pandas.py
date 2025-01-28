import pandas as pd
import numpy as np

print(np.arange(0,20).reshape(5,4))

# create dataframe
df=pd.DataFrame(data=np.arange(0,20).reshape(5,4),index=["Row1","Row2",
                                                         "Row3","Row4","Row5"],
                                                         columns=["Column1","Column2",
                                                         "Column3","Column4"])
print(df)
# displays top 5 records
print(df.head)
# display bottom up 5 records
print(df.tail)
# display top 2 records
print(df.head(2))
# display bottom up 3 records
print(df.tail(3))
# type of df
print(type(df))
# column datatype
print(df.info())
# returns description of the data in the DataFrame
print(df.describe())
# indexing
# columnname,rowindex[Loc],rowindex columnindex number[.iloc]
print(df[['Column1','Column2','Column3']])
# multiple rows and column
print(type(df[['Column1','Column2','Column3']]))
# single row and col
print(type(df['Column1']))
# shows row
print(df.loc['Row3'])
print(df.loc[['Row3','Row4']])
# by using iloc
print(df.iloc[2:4,2:])
print(df.iloc[0:,::3])
# convert dataframe to arrays
print(df.iloc[0:,0:].values)
# null values
print(df.isnull().sum())
# 
print(df['Column3'].value_counts())
print(df['Column3'].unique())

print(df>3)
print(df[df['Column2']>3])