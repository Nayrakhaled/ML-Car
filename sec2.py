#gaussian lw el training num ex temp
#barnoulli lw binary 0-1
#multinomial lw integer


from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
print(iris)
print(iris.data)
#print(iris.target)
#print(iris.feature_names)
gnb = GaussianNB()
gnb.fit(iris.data,iris.target)    #ta5od el data and el target
y_predict = gnb.predict(iris.data)#ta5od el new data wa t2oly hya el classfication(target)
print(accuracy_score(iris.target,y_predict))
