import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('1.jpg',-1)
px = img[100,100]    #cooredinate of pixels wa 7erga3 materix feha 3 R G B
print(px)

blue = img[100,100,0]    # hna el z=0 y3ny BGR wa 3l4an el index 0 1 2  wa blaly el blue hwa el 0
print(blue)

img[100,100] = [255,255,255]  # b3mal reset ll matrix bta3t el value wa 5ltha white
print(img[100,100])
print(img.shape)  # rl sora kam fy kam
print(img.size)   # 48la kam bl byte
print(img.dtype)  #data type of photo

one = img[280:333,330:390]#makan el -1 fy el photo
img[273:333,100:160] = one

b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))# merge el 3 channel m3 b3d
cv2.imshow('image',img)
cv2.waitKey(0)
#cv2.destroyAllWindow

BLUE = [0,0,0]
