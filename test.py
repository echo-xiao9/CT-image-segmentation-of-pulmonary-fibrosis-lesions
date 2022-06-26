import cv2 as cv
import numpy as np
import pickle
from dicomProcess import processDicom
from numpy import *

# 测试各size的数据集中各tag占比

sizes = [11, 13, 15, 17, 19, 21, 23]

# 统计数据集中每种size内各种类型数据占比
def doneTestSizeSp():
    for size in sizes:
        BASE_PATH = "../SE362-Projects/Project1/"
        PKL_PATH = f"trainData/trainData_{size}.pkl"
        count = 0
        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        with open(BASE_PATH + PKL_PATH, 'rb') as f:
            data = pickle.load(f)
            count += len(data[1])
            for label in data[1]:
                if label == 0:
                    count0 += 1
                elif label == 1:
                    count1 += 1
                elif label == 2:
                    count2 += 1
                elif label == 3:
                    count3 += 1
                elif label == 4:
                    count4 += 1
    
        print(f'size: {size}')
        print(f'count: {count}')
        print(f'tag0: {count0 / count}')
        print(f'tag1: {count1 / count}')
        print(f'tag2: {count2 / count}')
        print(f'tag3: {count3 / count}')
        print(f'tag4: {count4 / count}') 
        print('\n')

# 统计后处理后的数据集内各种类型数据占比
def doneTestAll():
    count = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for size in sizes:
        BASE_PATH = "../SE362-Projects/Project1/"
        PKL_PATH = f"trainData/trainData_{size}.pkl"
        with open(BASE_PATH + PKL_PATH, 'rb') as f:
            data = pickle.load(f)
            count += len(data[1])
            for label in data[1]:
                if label == 0:
                    count0 += 1
                elif label == 1:
                    count1 += 1
                elif label == 2:
                    count2 += 1
                elif label == 3:
                    count3 += 1
                elif label == 4:
                    count4 += 1
    
    print(f'count: {count}')
    print(f'tag0: {count0 / count}')
    print(f'tag1: {count1 / count}')
    print(f'tag2: {count2 / count}')
    print(f'tag3: {count3 / count}')
    print(f'tag4: {count4 / count}') 
    print('\n')

# 统计后处理前数据集内各种类型数据占比
def rawTestAll():
    count = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    BASE_PATH = "../SE362-Projects/Project1/"
    for i in range(1, 11):
        for size in sizes:
            PKL_PATH = f"case_{i}/trainData{i}/trainData{i}_{size}.pkl"
            with open(BASE_PATH + PKL_PATH, 'rb') as f:
                data = pickle.load(f)
                count += len(data[1])
                for label in data[1]:
                    if label == 0:
                        count0 += 1
                    elif label == 1:
                        count1 += 1
                    elif label == 2:
                        count2 += 1
                    elif label == 3:
                        count3 += 1
                    elif label == 4:
                        count4 += 1
    print(f'count: {count}')
    print(f'tag0: {count0 / count}')
    print(f'tag1: {count1 / count}')
    print(f'tag2: {count2 / count}')
    print(f'tag3: {count3 / count}')
    print(f'tag4: {count4 / count}') 
    print('\n')

# 统计评价指标
def evalAll(case:list):
    BASE_PATH = "../SE362-Projects/Project1/"
    confMtxs = []
    dices = []
    mIoUs = []
    MPAs = []
    PAs = []
    Recalls = []
    for i in case:
        EVAL_PATH = f"case_{i}/evaluating.pkl"
        with open(BASE_PATH + EVAL_PATH, 'rb') as f:
            data = pickle.load(f)
            confMtxs.append(data[0])
            dices.append(data[1])
            mIoUs.append(data[2])
            MPAs.append(data[3])
            PAs.append(data[4])
            Recalls.append(data[5])
    print(f"avg dice: {mean(dices)}")
    print(f"avg mIoU: {mean(mIoUs)}")
    print(f"avg MPA: {mean(MPAs)}")
    print(f"avg PA: {mean(PAs)}")
    print(f"avg Recall: {mean(Recalls)}")  

def main():
    evalAll([1])

if __name__ == "__main__":
    main()