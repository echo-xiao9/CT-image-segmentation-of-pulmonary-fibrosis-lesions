from math import floor
import cv2 as cv
import numpy as np
import pickle
from pandas import array
import numpy as np
import skimage
from getGLCM import imgClassifier
from uniformClassifier import uniformClassifier

# k-means算法，将img按灰度分为k级，并输出图像和排序后的灰度值list
# input: 灰度图img, 阶数k
# output: 灰度图res, 灰度值列表center
def kMeans(img:np.array, k:int):
    # 将输入图片转化为一维数组
    data = img.reshape((-1, 1))
    data = np.float32(data)
    # 执行kmeans
    critera = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    retval, bestLabel, centers = cv.kmeans(data, k, None, criteria=critera, attempts=10, flags=flags)
    # 重建图片
    centers = np.uint8(centers)
    data = centers[bestLabel.flatten()]
    res = data.reshape((img.shape))
    # 处理灰度值列表
    center = centers.flatten()
    center.sort()
    return res, center

# 分窗口处理生成训练集，需要输入原图、前景mask、四类病灶mask、窗口size，输出为一串numpy数组（以list封装）和一个tag数组(以list封装)，为分出的图块和对应的标签
# input: 灰度图img, 前景蒙版foregroundMask, 病灶蒙版mask1-mask4, 块大小size
# output: 寻找到的块的列表res, 对应标签列表tag
def processTrainData(img:np.array, foregroundMask:np.array, mask1:np.array, mask2:np.array, mask3:np.array, mask4:np.array, size:int):
    compressed, center = kMeans(img, 16)
    hIndex = floor(img.shape[0] / size)
    wIndex = floor(img.shape[1] / size)

    maxFlag = 255 * size * size

    res = []
    tag = []
    for i in range(hIndex):
        for j in range(wIndex):
            subPic = compressed[i * size : (i+1) * size, j * size : (j+1) * size]

            subMask0 = foregroundMask[i * size : (i+1) * size, j * size : (j+1) * size]
            flag0 = np.sum(subMask0)

            subMask1 = mask1[i * size : (i+1) * size, j * size : (j+1) * size]
            flag1 = np.sum(subMask1)

            subMask2 = mask2[i * size : (i+1) * size, j * size : (j+1) * size]
            flag2 = np.sum(subMask2)

            subMask3 = mask3[i * size : (i+1) * size, j * size : (j+1) * size]
            flag3 = np.sum(subMask3)

            subMask4 = mask4[i * size : (i+1) * size, j * size : (j+1) * size]
            flag4 = np.sum(subMask4)

            if (flag0 == maxFlag):
                if (flag1 == maxFlag):
                    res.append(subPic)
                    tag.append(1)
                elif (flag2 == maxFlag):
                    res.append(subPic)
                    tag.append(2)
                elif (flag3 == maxFlag):
                    res.append(subPic)
                    tag.append(3)
                elif (flag4 == maxFlag):
                    res.append(subPic)
                    tag.append(4) 
                elif (flag1 + flag2 + flag3 + flag4 == 0):
                    res.append(subPic)
                    tag.append(0)

    return res, tag

# 分窗口处理图片，需要输入原图、前景mask、使用的块大小列表sizes，输出为包含五个mask的list，代表正常区域及四种病灶
# input: 灰度图img, 前景蒙版foregroundMask, 使用块大小列表sizes
# output: 依次包含正常区域、实变区域、磨玻璃影区域、蜂窝影区域、网状影区域蒙版的列表masks
def processImage(img:np.array, foregroundMask:np.array, sizes:list, classifier:uniformClassifier):
    compressed, center = kMeans(img, 16)
    shap = img.shape

    tmpMask0 = np.zeros(shap)
    tmpMask1 = np.zeros(shap)
    tmpMask2 = np.zeros(shap)
    tmpMask3 = np.zeros(shap)
    tmpMask4 = np.zeros(shap)
    tmpMasks = [tmpMask0, tmpMask1, tmpMask2, tmpMask3, tmpMask4]

    mask0 = np.zeros(shap, np.uint8)
    mask1 = np.zeros(shap, np.uint8)
    mask2 = np.zeros(shap, np.uint8)
    mask3 = np.zeros(shap, np.uint8)
    mask4 = np.zeros(shap, np.uint8)
    masks = [mask0, mask1, mask2, mask3, mask4]

    for size in sizes:
        hIndex = floor(shap[0] / size)
        wIndex = floor(shap[1] / size)
        maxFlag = 255 * size * size
        for i in range(hIndex):
            for j in range(wIndex):
                subImg = compressed[i * size : (i+1) * size, j * size : (j+1) * size]

                fSubMask = foregroundMask[i * size : (i+1) * size, j * size : (j+1) * size]
                flag = np.sum(fSubMask)
                if(flag < maxFlag):
                    continue

                # tag = classifier.classify(subImg, size)
                # tmpMasks[tag][i * size : (i+1) * size, j * size : (j+1) * size] += 1
                tags = classifier.calcProb(subImg, size)
                for tagIdx in range(5):
                    tmpMasks[tagIdx][i * size : (i+1) * size, j * size : (j+1) * size] += tags[tagIdx]

    for i in range(shap[0]):
        for j in range(shap[1]):
            maxIdx = -1
            maxVal = 0
            for k in range(5):
                if tmpMasks[k][i][j] > maxVal:
                    maxIdx = k
                    maxVal = tmpMasks[k][i][j]
            if(maxIdx == -1):
                continue
            masks[maxIdx][i][j] = 255

    struct = cv.getStructuringElement(cv.MORPH_ELLIPSE, (min(sizes), min(sizes)))
    for i in range(5):
        masks[i] = cv.morphologyEx(masks[i], cv.MORPH_OPEN, struct)
        masks[i] = cv.morphologyEx(masks[i], cv.MORPH_CLOSE, struct)

    return masks
