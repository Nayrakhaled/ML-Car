from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

#cluser
x=np.array([[1,0],[1,2],[10,0],[10,2]])
c=KMeans(n_clusters=2)
c.fit(x)
y=c.labels_
y_predict=c.predict([[0,0],[12,3]])
print(y_predict)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()