

import cv2 
import numpy as np
import skimage
from skimage.feature import  greycoprops
from classifier import * 
from evaluating import *

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


def loadTrainData(size, case):
  if case == 0 :
    trainDataName = '../data/allCase'+'/TD_allCase'+'_size'+ str(size)+'_ALL.txt'
    tagName = '../data/allCase'+'/TAG_allCase_size'+ str(size)+'.txt'
  else:
    trainDataName = '../data/trainData'+str(case)+'/TD_case'+ str(case)+'_size'+ str(size)+'_ALL'
    tagName = '../data/trainData'+str(case)+'/TAG_case'+ str(case)+'_size'+ str(size)
  trainData = np.loadtxt(trainDataName, delimiter=',')
  trainTag = np.loadtxt(tagName, delimiter=',')
  return trainData, trainTag

def predictSingleCase(size, case):
  trainData, trainTag = loadTrainData(size, case)
  trainTag = trainTag.astype(np.int)
  classifierType = "random_forest"
  cla = Classifier(classifierType)
  # 初始化方法2：加载已有模型
  cla = loadClassifier("random_forest",size)
  predArr = cla.predict(trainData).astype(np.int)
  print("predic")
  print(predArr)
  print("actual")
  print(trainTag)
  return test2(predArr, trainTag, size)


def loadClassifier(method, size):
  cla = Classifier(method)
  claFileName=  '../data/allCase/CLA_'+ method +'_allCase'+'_size'+ str(size) +'_ALL.pickle'
  cla.load(claFileName)
  return cla


# 记录所有size的训练结果
def predictAllCase():
  with open('../data/trainTxt/predResult_allCase','w') as f:
    for size in range(11,25,2):
      PA, MPA, Recall, dice = predictSingleCase(size, 0)
      f.write(str(PA)+','+str(MPA)+','+str( Recall)+','+str( dice)+'\n')

def visualize():
  predictAllCase()

visualize()