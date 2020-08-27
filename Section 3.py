import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

iris = datasets.load_iris()
x = iris.data
y = iris.target

# PLot data points
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()

# Linear SVM
clf1 = SVC(kernel='linear')
clf1.fit(x, y)
y_pred = clf1.predict(x)
print(accuracy_score(y, y_pred))

# Polynomial SVM
clf2 = SVC(kernel='poly', degree=3, gamma='auto')
clf2.fit(x, y)
y_pred = clf2.predict(x)
print(accuracy_score(y, y_pred))

# Gaussian SVM
clf3 = SVC(kernel='rbf', gamma='auto')
clf3.fit(x, y)
y_pred = clf3.predict(x)
print(accuracy_score(y, y_pred))

# Sigmoid SVM
# Better for binary classification
clf4 = SVC(kernel='sigmoid', gamma='auto')
clf4.fit(x, y)
y_pred = clf4.predict(x)
print(accuracy_score(y, y_pred))

# To read your dataset from excel sheet
data = pd.read_excel('file1.xlsx')
df = pd.DataFrame(data)
y = df['type']
X = df.drop(['type'], axis=1)