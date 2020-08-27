from sklearn import datasets     #3shan ageb el set x w a3ml 3leha test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris=datasets.load_iris()   #bgeb el dataset mn iris
x=iris.data
#print(x)

y=iris.target

#clf=SVC(kernel='linear') #7adedt eny hsht3'l linear  mafesh gamma w degree hna
clf=SVC(kernel='poly',gamma='scale',degree=6)  #el gamma w el degree de 7agatt optional w el default bta3 el gamma =auto
#clf=SVC(kernel='rbf',gamma='scale')
#clf=SVC(kernel='sigmoid',gamma='scale')

clf.fit(x,y)                  #bta5od el data w el target

y_pred=clf.predict(x)       #ba test b dataset w ashof el natega
#print (y_pred)

#print(accuracy_score(y,y_pred))      #btshof el y ely ana mtwak3aha shabh el y elaslya wla l2
plt.scatter(x[:,0],x[:,1],c=y)    #btrsm el sata ka no2t msh k line w b3tlha kol sfof mn awel
# 3mod ka X_axis w kol elsfof 3la el3amod eltany
#c=y  y3ny b2olop en an ana 3ndy 3 class f hyb2a 3nddy 3 alwan

plt.show()