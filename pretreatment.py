
from cProfile import label
from pickletools import uint8
import SimpleITK 
import skimage
import cv2
import numpy as np
import copy
from scipy.ndimage import binary_fill_holes



pic = cv2.imread("4.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("test",pic)
# ret, binary = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow("test",binary)
s = 10
structureElenent = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(s,s))
structureElenent2 = cv2.getStructuringElement(cv2.MORPH_RECT,(s,s))
# res = cv2.morphologyEx(binary,cv2.MORPH_OPEN,structureElenent2)
# res = cv2.morphologyEx(res,cv2.MORPH_OPEN,structureElenent)
# contours, hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(pic,contours,-1,(0,0,255),3)
# cv2.imshow("lk",pic)
# cv2.imshow("final",res)


# pic2 =  cv2.morphologyEx(pic,cv2.MORPH_OPEN,cv2.MORPH_ELLIPSE,(3,3))

# cv2.imshow("3333",binary)

#闭操作：去除无关背景信息
pic1_5 = cv2.morphologyEx(pic,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
cv2.imshow("pic2",pic1_5)
#腐蚀：最小值滤波，用于能够将一大坨不平均的如蜂窝等区域搞得黑一点
pic2 = cv2.erode(pic1_5,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=10)
#二值化
ret, binary = cv2.threshold(pic2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
ret, nouse = cv2.threshold(pic1_5, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# binary2 =  cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
binary2 = binary

# 通过闭操作，填一些半开放的孔洞(放到后面)
binary2 =  cv2.morphologyEx(binary2,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
cv2.imshow("ref",nouse)
cv2.imshow("4444",binary2)
# 用连通域方法去除背景，并且去除闭操作难以去除的闭合的孔洞
labeled_img, num = skimage.measure.label(binary2, background=0, return_num=True)
print(num)
max_label = 0
max_num = 0
for i in range(1, num+1):  #注意这里的范围，为了与连通域的数值相对应
    # 计算面积，保留最大面积对应的索引标签，然后返回二值化最大连通域
    if np.sum(labeled_img == i) > max_num:
        max_num = np.sum(labeled_img == i)
        max_label = i
showPic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

noBackPicMask = (labeled_img != max_label).astype(np.uint8) * binary2
cv2.imshow("nobackmask", noBackPicMask)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(noBackPicMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contoursList = np.ndarray((0,1,2), contours[0].dtype)
for contour in contours:
    contoursList = np.concatenate((contoursList, contour), axis = 0)
hull = cv2.convexHull(contoursList)
hullList = [hull]
print(hull.shape)
kmeansPts = []
for i in range(0,hull.shape[0]):
    kmeansPts.append(hull[i,0,0])

compactness, labels, centers = cv2.kmeans(np.array(kmeansPts).astype(np.float32), 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
for i in range(0,hull.shape[0]):
    color = (0,0,0)
    if(labels[i] == 0):
        color = (255,0,0)
    else:
        color = (0,0,255)
    cv2.circle(showPic, (hull[i,0,:]),1,color=color)

boundary1 = 0
boundary2 = 0
nowLabel = labels[0][0]
for i in range(labels.shape[0]):
    if labels[i][0] != nowLabel:
        boundary1 = i
        nowLabel = labels[i][0]
        break
for i in range(boundary1, labels.shape[0]):
    if labels[i][0] != nowLabel:
        boundary2 = i
        break
outerMostPts1 = []
outerMostPts2 = []
for i in range(boundary1, boundary2):
    outerMostPts1.append(hull[i,0,:])
for i in range(boundary2, labels.shape[0]):
    outerMostPts2.append(hull[i,0,:])
for i in range(0, boundary1):
    outerMostPts2.append(hull[i,0,:])

print(outerMostPts1)

cv2.imshow("circle",showPic)
cv2.waitKey(0)
cv2.destroyAllWindows()

convexMask = np.zeros(pic.shape,np.uint8)
cv2.drawContours(convexMask, hullList, 0, (255,0,0), -1)
convexMask = cv2.erode(convexMask,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=10)
cv2.imshow("nobackmask",255 - pic1_5 * convexMask)
twoLungPic = 255 - pic1_5 * convexMask
ret, twoLungBinary = cv2.threshold(twoLungPic, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("twoLungBinary",twoLungBinary)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Z = np.float32(twoLungPic.reshape((-1, 1)))

# compactness, labels, centers = cv2.kmeans(Z, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
# centers = np.uint8(centers)
# kmeansRes = centers[labels.flatten()]
# kmeansPic = kmeansRes.reshape(pic.shape)
# cv2.imshow("aaa", kmeansPic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



for i in range(1, num + 1):
    if(i == max_label):
        continue
    lcc = (labeled_img == i)
    lcc = lcc.astype(np.uint8)


# 填孔洞
    noback = binary2 * lcc
    # noback =  cv2.dilate(noback,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=100)
    for i in range(len(outerMostPts1) - 1):
        cv2.line(noback, outerMostPts1[i], outerMostPts1[i + 1], color = (255,255,255), thickness= 1)
    for i in range(len(outerMostPts2) - 1):
        cv2.line(noback, outerMostPts2[i], outerMostPts2[i + 1], color = (255,255,255), thickness= 1)
    res = binary_fill_holes(np.asarray(noback).astype(int)).astype(np.uint8)
    
    finalResMask = cv2.erode(res,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=10)
    contours, hierarchy = cv2.findContours(finalResMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    finalRes = finalResMask * pic
    cv2.imshow("mask",finalResMask * 255)
    cv2.imshow("noback",finalRes)
    convexMask = np.zeros(pic.shape,np.int8)
    cv2.drawContours(pic, contours, 0, (255,0,0), thickness = 5)
    cv2.imshow("sdsdsdsd", pic) 
    maskedPic = finalResMask * pic
    bluredPic = maskedPic
    Z = np.float32(bluredPic.reshape((-1, 1)))

    compactness, labels, centers = cv2.kmeans(Z, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    kmeansRes = centers[labels.flatten()]
    kmeansPic = kmeansRes.reshape(pic.shape)

    tmp =  cv2.morphologyEx(kmeansPic,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    tmp =  cv2.morphologyEx(tmp,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    
    cv2.imshow("kmeans",kmeansPic)
    cv2.imshow("xtx",tmp)
    cv2.imshow("resultd",finalResMask * pic)

    contours2, hierarchy = cv2.findContours(finalRes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    
    cv2.drawContours(showPic,contours2,-1,(0,0,255),3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
cv2.imshow("result",showPic)

cv2.waitKey(0)
cv2.destroyAllWindows()


