from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel('Book2.xlsx')
df = pd.DataFrame(data)
Y_target = df['number of instances per class']
X_data = df.drop(['number of instances per class'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_target)
clf = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(precision_score(y_test, y_predict))