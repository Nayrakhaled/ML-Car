import numpy as np
import cv2
from matplotlib import pyplot as plt

#2

x = np.random.permutation(1000)
print(x)
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = a[2,:]

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = a.reshape(-1)

f = np.random.randn(5,1)
g = f[f>0]
print(g)
x = np.zeros(10)+0.5
print(x)
y = 0.5*np.ones(len(x))
print(y)
z = x + y
a = np.arange(1,100)
b = a[::-1]
print(b)

#3
y = np.array([1, 2, 3, 4, 5, 6])
z = y.reshape(3,2)
max = np.max(z)
row = z[0]
col = z[:,0]
print("max" ,max)
print("row",row)
print("col",col)

v = np.array([1, 8, 8, 2, 1, 3, 9, 8])
where = np.where(v==1,"x",v)
print("replace",where)

#4
a = np.random.randint(256, size=(100, 100))
print(a)
f = open("inputAPS0Q1.npy", "w")
np.savetxt("inputAPS0Q1.npy",a)
f.close()
plt.hist(a.ravel(),256,[0,256])
plt.show()
_ = plt.hist(a, bins=20)
plt.show()
