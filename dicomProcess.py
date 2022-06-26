
from cProfile import label
import pickle
import pydicom
import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import os
import json
from uniformClassifier import uniformClassifier
from window import processTrainData
from window import processImage
from evaluating import evalutaing_Segmentation

# 窗位
LEVEL = -650
# 窗宽
WINDOW = 1500

# 从dicom文件生成训练集
# input: 包含dicom文件的目录路径dicomPath, 前景标注文件路径foregroundPath, 病灶标注文件路径pathologyPath, 输出文件路径outputPath, 块大小size
# output: 包含已提取的块列表和对应标签列表的文件
def toTrainData(dicomPath:str, foregroundPath:str, pathologyPath:str, outputPath:str, size:int):
    # if not os.path.exists(outputPath):
    #     os.makedirs(outputPath)
    
    dicomFiles = glob.glob(dicomPath + '/*.dcm')

    dicomFiles.sort(key=lambda s : int(s.split('\\')[-1].split('.dcm')[0]), reverse=True)
    
    foregroundLabel = nib.load(foregroundPath).get_fdata()
    pathologyLabel = nib.load(pathologyPath).get_fdata()

    res = []
    tag = []

    for i in range(len(dicomFiles)):
        print(i)
        dicom = pydicom.read_file(dicomFiles[i])
        dicomImg = dicom.pixel_array.astype(np.int16)
        intercept = dicom.RescaleIntercept

        dicomImg += np.int16(intercept)

        minBound = LEVEL - WINDOW // 2
        maxBound = LEVEL + WINDOW // 2
        dicomImg = (dicomImg - minBound) / (maxBound - minBound) * 255
        dicomImg[dicomImg > 255] = 255.
        dicomImg[dicomImg < 0] = 0.

        greyImg = np.array(dicomImg, np.uint8)

        foregroundMask = np.array(foregroundLabel[:, :, i], np.uint8).T

        label = np.array(pathologyLabel[:, :, i], np.uint8).T
        shape = label.shape

        mask1 = np.zeros(shape, np.uint8)
        mask1[label == 1] = 255

        mask2 = np.zeros(shape, np.uint8)
        mask2[label == 2] = 255

        mask3 = np.zeros(shape, np.uint8)
        mask3[label == 3] = 255

        mask4 = np.zeros(shape, np.uint8)
        mask4[label == 4] = 255

        tmpRes, tmpTag = processTrainData(greyImg, foregroundMask, mask1, mask2, mask3, mask4, size)
        res.extend(tmpRes)
        tag.extend(tmpTag)

    final = []
    final.append(res)
    final.append(tag)

    with open(outputPath, 'wb') as f:
        pickle.dump(final, f)

# 从dicom文件提取病灶
# input: 包含dicom文件的目录路径dicomPath, 前景标注文件路径foregroundPath, 输出目录路径outputPath, 块大小列表sizes
# output: 已标注的label列表finalLabels, 完成标注的图片文件
def processDicom(dicomPath:str, foregroundPath:str, outputPath:str, sizes:list):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    dicomFiles = glob.glob(dicomPath + '/*.dcm')

    dicomFiles.sort(key=lambda s : int(s.split('\\')[-1].split('.dcm')[0]), reverse=True)
    
    foregroundLabel = nib.load(foregroundPath).get_fdata()

    classifier = uniformClassifier("models/")

    finalLabels = []

    for i in range(len(dicomFiles)):
        print(i)
        dicom = pydicom.read_file(dicomFiles[i])
        dicomImg = dicom.pixel_array.astype(np.int16)
        intercept = dicom.RescaleIntercept

        dicomImg += np.int16(intercept)

        minBound = LEVEL - WINDOW // 2
        maxBound = LEVEL + WINDOW // 2
        dicomImg = (dicomImg - minBound) / (maxBound - minBound) * 255
        dicomImg[dicomImg > 255] = 255.
        dicomImg[dicomImg < 0] = 0.

        greyImg = np.array(dicomImg, np.uint8)

        foregroundMask = np.array(foregroundLabel[:, :, i], np.uint8).T

        labels = processImage(greyImg, foregroundMask, sizes, classifier)
        label = np.zeros(greyImg.shape, np.uint8)
        for j in range(1, 5):
            label[labels[j] == 255] = j
        finalLabels.append(label)

        colorImg = cv.cvtColor(greyImg, cv.COLOR_GRAY2BGR)
        colorLabel = cv.cvtColor(label, cv.COLOR_GRAY2BGR)
        
        colorMap = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for j in range(1, 5):
            colorLabel[label == j] = colorMap[j-1]
        
        vis = cv.addWeighted(colorLabel, 0.8, colorImg, 1.0, 0)
        cv.imwrite(f'{outputPath}/{i + 1}.png', cv.vconcat([colorImg, vis]))
        print(f'{outputPath}/{i + 1}.png')

    return finalLabels

# 从dicom文件提取病灶并评估提取结果
# input: 包含dicom文件的目录路径dicomPath, 前景标注文件路径foregroundPath, 病灶标注文件路径pathologyPath, 输出目录路径outputPath, 块大小列表sizes
# output: 已标注的label列表finalLabels, 评估指标列表eval, 完成标注的图片文件
def processDicomWithEvaluating(dicomPath:str, foregroundPath:str, pathologyPath:str, outputPath:str, sizes:list):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    print(dicomPath)

    evaluating = evalutaing_Segmentation()

    dicomFiles = glob.glob(dicomPath + '/*.dcm')

    dicomFiles.sort(key=lambda s : int(s.split('\\')[-1].split('.dcm')[0]), reverse=True)
    
    foregroundLabel = nib.load(foregroundPath).get_fdata()
    pathologyLabel = nib.load(pathologyPath).get_fdata()

    classifier = uniformClassifier("models/")

    finalLabels = []

    for i in range(len(dicomFiles)):
        print(i)
        dicom = pydicom.read_file(dicomFiles[i])
        dicomImg = dicom.pixel_array.astype(np.int16)
        intercept = dicom.RescaleIntercept

        dicomImg += np.int16(intercept)

        minBound = LEVEL - WINDOW // 2
        maxBound = LEVEL + WINDOW // 2
        dicomImg = (dicomImg - minBound) / (maxBound - minBound) * 255
        dicomImg[dicomImg > 255] = 255.
        dicomImg[dicomImg < 0] = 0.

        greyImg = np.array(dicomImg, np.uint8)

        foregroundMask = np.array(foregroundLabel[:, :, i], np.uint8).T

        stdLabel = np.array(pathologyLabel[:, :, i], np.uint8).T
        shape = stdLabel.shape

        stdMask1 = np.zeros(shape, np.uint8)
        stdMask1[stdLabel == 1] = 255

        stdMask2 = np.zeros(shape, np.uint8)
        stdMask2[stdLabel == 2] = 255

        stdMask3 = np.zeros(shape, np.uint8)
        stdMask3[stdLabel == 3] = 255

        stdMask4 = np.zeros(shape, np.uint8)
        stdMask4[stdLabel == 4] = 255

        stdMasks = [foregroundMask, stdMask1, stdMask2, stdMask3, stdMask4]

        labels = processImage(greyImg, foregroundMask, sizes, classifier)
        label = np.zeros(greyImg.shape, np.uint8)
        for j in range(1, 5):
            label[labels[j] == 255] = j
        finalLabels.append(label)

        evaluating.inputData(labels, stdMasks)

        colorImg = cv.cvtColor(greyImg, cv.COLOR_GRAY2BGR)
        colorLabel = cv.cvtColor(label, cv.COLOR_GRAY2BGR)
        
        colorMap = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for j in range(1, 5):
            colorLabel[label == j] = colorMap[j-1]
        
        vis = cv.addWeighted(colorLabel, 0.8, colorImg, 1.0, 0)
        cv.imwrite(f'{outputPath}/{i + 1}.png', cv.vconcat([colorImg, vis]))
        print(f'{outputPath}/{i + 1}.png')

    eval = []

    print(evaluating.ConfusionMatrix)
    eval.append(evaluating.ConfusionMatrix)

    print(evaluating.dice())
    eval.append(evaluating.dice())

    print(evaluating.mIoU())
    eval.append(evaluating.mIoU())

    print(evaluating.MPA())
    eval.append(evaluating.MPA())

    print(evaluating.PA())
    eval.append(evaluating.PA())

    print(evaluating.Recall())
    eval.append(evaluating.Recall())

    evaluating.ConfusionMatrix_Visualization()

    print("\n\n")

    return finalLabels, eval