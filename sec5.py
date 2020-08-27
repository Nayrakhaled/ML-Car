from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#single nn
iris = datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y)
clf = MLPClassifier()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
print(accuracy_score(y_test,y_predict))