from sklearn.linear_model import LinearRegression
import numpy as np

#regretion
x=np.array([[1,1],[1,2],[2,3]])
y=np.dot(x,[1,2])+3
#print(y)
reg=LinearRegression()
reg.fit(x,y)
y_predict=reg.predict([[3,5]])
#print(y_predict)
#print(reg.coef_)
print(reg.intercept_)