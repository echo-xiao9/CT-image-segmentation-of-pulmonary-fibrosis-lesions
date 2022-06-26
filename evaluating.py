from typing import Type
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class evaluating:
    def __init__(self):
        self.ConfusionMatrix = np.zeros((5,5),np.int64)
        pass

    # 像素准确率（分割器）/准确率（分类器）
    def PA(self):
        sum = np.sum(self.ConfusionMatrix)
        print("sum type")
        if sum == 0:
            return 0
        acc = np.sum(np.diagonal(self.ConfusionMatrix))
        return acc/sum

    # 类别平均像素准确率（分割器）/类别平均准确率（分类器）/平均精准率
    def MPA(self):
        valid = 0
        sumPA = 0
        for i in range(5):
            sum = np.sum(self.ConfusionMatrix[:,i])
            if not sum == 0:
                valid = valid + 1
                sumPA = sumPA + self.ConfusionMatrix[i][i] / sum
        if valid == 0:
            return 0
        return sumPA / valid
    
    # 平均召回率
    def Recall(self):
        valid = 0
        sumRecall = 0
        for i in range(5):
            sum = np.sum(self.ConfusionMatrix[i,:])
            if not sum == 0:
                valid = valid + 1
                sumRecall = sumRecall + self.ConfusionMatrix[i][i] / sum
        if valid == 0:
            return 0
        return sumRecall / valid

    

    # dice指数（f1-score）
    def dice(self):
        valid = 0
        sumDice = 0
        for i in range(5):
            sum = np.sum(self.ConfusionMatrix[i,:]) + np.sum(self.ConfusionMatrix[:,i])
            if not sum == 0:
                valid = valid + 1
                sumDice = sumDice + 2 * self.ConfusionMatrix[i][i] / sum
        if valid == 0:
            return 0
        return sumDice / valid

    def ConfusionMatrix_Visualization(self):
        plt.figure(figsize=(15,5))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False 
        plt.subplot(1,3,1)
        plt.title("原始混淆矩阵")
        df=pd.DataFrame(self.ConfusionMatrix,index=["正常","实变影","磨玻璃影","蜂窝影","网状影"],columns=["正常","实变影","磨玻璃影","蜂窝影","网状影"])
        sns.heatmap(df,annot=True)
        plt.subplot(1,3,2)
        plt.title("按行归一化混淆矩阵")
        ConfusionMatrix_normalized_row = np.array(self.ConfusionMatrix, np.float64)
        for i in range(5):
            if np.sum(ConfusionMatrix_normalized_row[i,:]) != 0:
                ConfusionMatrix_normalized_row[i,:] = ConfusionMatrix_normalized_row[i,:] / np.sum(ConfusionMatrix_normalized_row[i,:])
        df=pd.DataFrame(ConfusionMatrix_normalized_row,index=["正常","实变影","磨玻璃影","蜂窝影","网状影"],columns=["正常","实变影","磨玻璃影","蜂窝影","网状影"])
        sns.heatmap(df,annot=True)
        plt.subplot(1,3,3)
        plt.title("按列归一化混淆矩阵")
        ConfusionMatrix_normalized_col = np.array(self.ConfusionMatrix, np.float64)
        for i in range(5):
            if np.sum(ConfusionMatrix_normalized_col[:,i]) != 0:
                # print(np.sum(ConfusionMatrix_normalized_col[:,i]))
                # print(ConfusionMatrix_normalized_col[:,i])
                col = ConfusionMatrix_normalized_col[:,i] / np.sum(ConfusionMatrix_normalized_col[:,i])
                # print(col)
                # print(type(col))
                ConfusionMatrix_normalized_col[:,i] = col
                # print(ConfusionMatrix_normalized_col)
        # print(ConfusionMatrix_normalized_col)
        df=pd.DataFrame(ConfusionMatrix_normalized_col,index=["正常","实变影","磨玻璃影","蜂窝影","网状影"],columns=["正常","实变影","磨玻璃影","蜂窝影","网状影"])
        sns.heatmap(df,annot=True)
        plt.show()


class evalutaing_Segmentation(evaluating):
    # 数据输入格式：识别结果 [正常，实变影，磨玻璃影，蜂窝影，网状影]，标准结果 [前景，实变影，磨玻璃影，蜂窝影，网状影]
    def inputData(self, result: list, reference: list):
        reference[0] = 255-((255-reference[0]) | reference[1] | reference[2] | reference[3] | reference[4])
        for i in range(5):
            for j in range(5):
                self.ConfusionMatrix[i][j] = self.ConfusionMatrix[i][j] + np.sum((reference[i] & result[j]) == 255)
    
    # 平均交并比
    def mIoU(self):
        valid = 0
        sumIoU = 0
        for i in range(5):
            sum = np.sum(self.ConfusionMatrix[i,:]) + np.sum(self.ConfusionMatrix[:,i]) - self.ConfusionMatrix[i][i]
            if not sum == 0:
                valid = valid + 1
                sumIoU = sumIoU + self.ConfusionMatrix[i][i] / sum
        if valid == 0:
            return 0
        return sumIoU / valid
 
class evaluating_Classification(evaluating):
    # 数据输入格式：分类结果（整数，0-正常，1-实变，2-磨玻璃影，3-蜂窝，4-网状），实际结果（整数，0-正常，1-实变，2-磨玻璃影，3-蜂窝，4-网状）
    def inputData(self, predict:np.array, exact:np.array):
        for i in range(predict.shape[0]):
            self.ConfusionMatrix[exact[i]][predict[i]] = self.ConfusionMatrix[exact[i]][predict[i]] + 1




def test():
    pica = cv2.imread('E:\\WorksAndStudying\\2022Spring\\CV\\evaluating\\a.jpg',cv2.IMREAD_GRAYSCALE)
    picb = cv2.imread('E:\\WorksAndStudying\\2022Spring\\CV\\evaluating\\b.jpg',cv2.IMREAD_GRAYSCALE)
    ret,a = cv2.threshold(pica,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret,b = cv2.threshold(picb,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(type(a))
    none = np.zeros(a.shape, np.int8)
    all = np.ones(a.shape,np.int8) * 255
    cv2.imshow("a",pica)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    e = evalutaing_Segmentation()
    for i in range(10):
        print(i)
        e.inputData([none, a, none, none, none],[all, b, none, none, none])
    print(e.ConfusionMatrix)
    print(e.dice())
    print(e.mIoU())
    print(e.MPA())
    print(e.PA())
    print(e.Recall())
    e.ConfusionMatrix_Visualization()



if __name__ == "__main__":
    test()