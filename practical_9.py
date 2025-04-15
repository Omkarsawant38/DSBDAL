import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df= sns.load_dataset('titanic')

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='sex', y='age', hue='survived', palette='Set1')
plt.title('Box Plot of Age Distribution by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()


 Removing Outliers

Calculate the IQR for age
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for age
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data to remove outliers
filtered_titanic = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

# Create the box plot without outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_titanic, x='sex', y='age', hue='survived', palette='Set1')

# Set plot title and labels
plt.title('Box Plot of Age Distribution by Gender and Survival Status (Outliers Removed)')
plt.xlabel('Gender')
plt.ylabel('Age')

# Show plot
plt.show()


# Count of survival status grouped by gender
survival_counts = df.groupby(['sex', 'survived']).size().unstack()

# Define colors for better visualization
colors = ['red', 'blue']  # Red = Not survived, Blue = Survived

# Create the pie charts
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, gender in enumerate(survival_counts.index):
    axes[i].pie(
        survival_counts.loc[gender], 
        labels=['Not Survived', 'Survived'], 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=140,
        wedgeprops={'edgecolor': 'black'}
    )
    axes[i].set_title(f'Survival Distribution for {gender.capitalize()}')

# Show plot
plt.tight_layout()
plt.show()


