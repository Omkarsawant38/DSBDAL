import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(1, 2, 100),
    'Category': np.random.choice(['Category 1', 'Category 2', 'Category 3'], 100)
})

# Assign x and y as variable names
x = 'A'
y = 'B'

# A. Distribution Plots
# a. Dist-Plot
plt.figure()
sns.histplot(data[x], kde=True)
plt.title(f'Dist-Plot of {x}')
plt.show()

# b. Joint Plot (removed 'ax' because jointplot is a figure-level function)
sns.jointplot(x=x, y=y, data=data, kind='scatter')
plt.suptitle('Joint Plot', y=1.02)
plt.show()

# d. Rug Plot
plt.figure()
sns.rugplot(data[x])
plt.title(f'Rug Plot of {x}')
plt.show()

# B. Categorical Plots
# a. Bar Plot
plt.figure()
sns.barplot(x='Category', y=x, data=data)
plt.title(f'Bar Plot of {x}')
plt.show()

# b. Count Plot
plt.figure()
sns.countplot(x='Category', data=data)
plt.title('Count Plot')
plt.show()

# c. Box Plot
plt.figure()
sns.boxplot(x='Category', y=x, data=data)
plt.title(f'Box Plot of {x}')
plt.show()

# d. Violin Plot
plt.figure()
sns.violinplot(x='Category', y=x, data=data)
plt.title(f'Violin Plot of {x}')
plt.show()

# C. Advanced Plots
# a. Strip Plot
plt.figure()
sns.stripplot(x='Category', y=x, data=data)
plt.title(f'Strip Plot of {x}')
plt.show()

# b. Swarm Plot
plt.figure()
sns.swarmplot(x='Category', y=x, data=data)
plt.title(f'Swarm Plot of {x}')
plt.show()

# D. Matrix Plots
# a. Heat Map (only using numeric columns)
plt.figure()
sns.heatmap(data[['A', 'B']].corr(), annot=True, cmap='coolwarm')
plt.title('Heat Map')
plt.show()

# E. Scatter Plot (In a subplot, if you want to add more plots to the grid)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # Create a subplot for scatter plot
ax.scatter(data[x], data[y])
ax.set_title(f'Scatter Plot of {x} vs {y}')
ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()

# Adjust the layout for the 3x3 grid of plots
plt.tight_layout()
plt.show()

