from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#multi nn
iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y)  #train size 1/3,2/3 by default
clf=MLPClassifier(hidden_layer_sizes=(10,5))
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
print(accuracy_score(y_test,y_predict))