'''
x = input("enter the number")
if x == 3:
    print("ok")
print("end")
'''
import numpy as np
from scipy import sparse
import pandas as pd   #table && excel
import matplotlib.pyplot as plt
x = np.array([[1,2,3],[4,5,6]])
#print(x)

eye_matrix = np.eye(4)
#print(eye_matrix)

#sparse most element in matrix 0 && dense most element!=0
sparse_matrix = sparse.csr_matrix(eye_matrix)
#print(sparse_matrix)

data = {
    'Name':['Ahmed','Ali','Eman'],
    'Age':[24,10,30]
}
#print(data)

#print(pd.DataFrame(data))

#x = [1,2,3]
x = np.linspace(-10,10,100)  #mn -10 l 10 wa el3dd 100num
#y=[100,200,300]
y = np.sin(x)
plt.plot(x,y)
plt.show()