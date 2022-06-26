import random
import cv2 as cv
import numpy as np
import pickle

# 用于后处理训练数据，平衡各label占比（主要是正常的块）

sizes = [11, 13, 15, 17, 19, 21, 23]
prob = {
    11 : 4/84,
    13 : 4/84,
    15 : 4/84,
    17 : 3/84,
    19 : 3/84,
    21 : 3/84,
    23 : 3/84,
}

for size in sizes:
    imgs = []
    tags = []
    for i in range(1, 11):
        if i==3:
            continue
        BASE_PATH = "../SE362-Projects/Project1/"
        CASE_PATH = f"case_{i}/trainData{i}/"
        PKL_PATH = f"trainData{i}_{size}.pkl"
        print(f"case: {i}\t\tsize: {size}" )
        with open(BASE_PATH + CASE_PATH + PKL_PATH, 'rb') as f:
            data = pickle.load(f)
            for j in range(len(data[1])):
                if(data[1][j] == 0):
                    magic = random.random()
                    if magic > prob[size]:
                        continue
                imgs.append(data[0][j])
                tags.append(data[1][j])
    final = [imgs, tags]
    with open(BASE_PATH + f"trainData/trainData_{size}.pkl", 'wb') as f:
        pickle.dump(final, f)    

