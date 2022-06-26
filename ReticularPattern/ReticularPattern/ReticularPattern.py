

#!/usr/bin/env python
# 使用基于阈值的方法提取网状影
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from toolFunc import *

# 得到灰度图
img = cv.imread('Lung_1.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
# 阈值过滤，二值化
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2) #进行开操作
cv.imshow('opening',opening)
# 形状过滤，去除血管
opening = areaFilter(opening)

kernel2 = np.ones((9,9),np.uint8)

#膨胀， 连接成区域
sure_bg = cv.dilate(opening,kernel2,iterations=3)
cv.imshow('sure_bg',sure_bg)
kernel3 = np.ones((15,15),np.uint8)
opening2 = cv.morphologyEx(sure_bg,cv.MORPH_CLOSE,kernel3, iterations = 2)
cv.imshow('open_bg',opening2)

# 生成mask,和原图像blend
backtorgb = cv.cvtColor(opening2,cv.COLOR_GRAY2RGB)
ret, markers = cv.connectedComponents(opening2)
markers[opening2==0] = 0
backtorgb[markers !=0] = [255,70,10]
result = cv.addWeighted(backtorgb, 0.5, img, 1,0)
cv.imshow('result', result)

cv.waitKey(0)
cv.destroyAllWindows()
