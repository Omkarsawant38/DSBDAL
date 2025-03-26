import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score, precision_score, recall_score, f1_score

data =  pd.read_csv('social_network_ads.csv')
print(data.head(5))

print(data.info())

print(data.describe())

print(data.isnull().sum())

print(data.shape)

x = data.iloc[:,2:4]
# print(x)
y = data.iloc[:,4]
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

lr = LogisticRegression(random_state = 0,solver = 'lbfgs')
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
       
print(x_test[:10])
print('-'*15)
print(pred[:10])

print('Expected Output:',pred[:10])
print('-'*15)
print('Predicted Output:\n',y_test[:10])

matrix = confusion_matrix(y_test,pred,labels = lr.classes_)
print(matrix)

tp, fn, fp, tn = confusion_matrix(y_test,pred,labels=[1,0]).reshape(-1)

conf_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=lr.classes_)
conf_matrix.plot(cmap=plt.cm.Blues)
plt.show()

print(classification_report(y_test,pred))

print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test,pred)))
print('Error Rate: ',(fp+fn)/(tp+tn+fn+fp))
print('Sensitivity (Recall or True positive rate) :',tp/(tp+fn))
print('Specificity (True negative rate) :',tn/(fp+tn))
print('Precision (Positive predictive value) :',tp/(tp+fp))
print('False Positive Rate :',fp/(tn+fp))


# Assuming 'Gender' is in column index 1 and 'Purchased' is in column index 4
data['Purchased'] = data['Purchased'].astype(str)  # Convert to string for better visualization

# Countplot for Male and Female who have purchased
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Purchased', data=data, palette='viridis')

plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of Males and Females Who Purchased')
plt.legend(title='Purchased', labels=['Not Purchased', 'Purchased'])
plt.show()
