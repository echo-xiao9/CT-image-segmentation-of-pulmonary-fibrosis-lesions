import cv2 
import numpy as np
from skimage.io import imread
import skimage
from skimage.feature import greycomatrix, greycoprops
#归一化
from sklearn.preprocessing import Normalizer
import warnings


def checkProp(prop):
  if(prop in  {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}):
    return True;
  print("illegal prop value: " + prop)
  return False;


def glcmAll(s): # s为图像路径
    values_temp = []
   
    input = cv2.imread(s)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    # [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4] 一共计算了四个方向，你也可以选择一个方向
    # 统计得到glcm
    glcm = skimage.feature.greycomatrix(input, [8], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    # 循环计算表征纹理的参数 
    for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = greycoprops(glcm, prop)
        values_temp.append(temp[0])
    values_temp.append(np.average(np.average(s, axis=1)))
    
    return (values_temp)

def glcmAll2(input): # s为图像路径
    
    values_temp = []
    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    # [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4] 一共计算了四个方向，你也可以选择一个方向
    # 统计得到glcm
    glcm = skimage.feature.greycomatrix(input, [8], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    # 循环计算表征纹理的参数 
    list = ['contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM']
    for i in range(0, len(list)):
        temp = greycoprops(glcm, list[i])
        values_temp.append(temp[0])
    return values_temp

def glcm(s,prop):
    values_temp = []
    input = cv2.imread(s)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    # [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4] 一共计算了四个方向，你也可以选择一个方向
    # 统计得到glcm
    # glcm = skimage.feature.greycomatrix(input, [2, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    glcm = skimage.feature.greycomatrix(input, [8], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    # 循环计算表征纹理的参数 
    if(checkProp(prop)==False) :
      return
    list =  ['contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM']
    for i in range(len(list)):
        prop = list[i]
        temp = greycoprops(glcm, prop)
        values_temp.append(temp)
    # mean = np.average(np.average(s, axis=1))
    # values_temp.append(mean)
    return (values_temp)

# 归一化list
def normList(list):
  array = np.array(list)
  return Normalizer().fit_transform(array)

# feature = glcmAll('../SE362-Projects/Project1/case_1/outputOri/155.png')
# data, tag = getSubImage('../Data/trainData1',1, 11)
# image = data[0]
# print('image')
# feature = glcmAll2(image)
# print(normList(feature).flatten())

