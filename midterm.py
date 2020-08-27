import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

data=pd.read_excel('Book1.xlsx')
df=pd.DataFrame(data)
y = df['target']
x = df.drop(['target'], axis=1)
nighbor = KNeighborsClassifier(n_neighbors=3)
nighbor.fit(x,y)
y_predict = nighbor.predict(x)
print(precision_score(y, y_predict))
x_axis=df['gender']
y_axis=df['fbs']
plt.scatter(x_axis, y_axis, c=y)
plt.show()