
from getGLCM import *
import numpy as np

# 统一分类器，根据size使用相应的分类器对图片进行分类
class uniformClassifier:
    def __init__(self, modelPath:str) -> None:
        self.classifiers = {}
        self.sizes = [11, 13, 15, 17, 19, 21, 23]
        for size in self.sizes:
            model = modelPath + f"random_forest_{size}.pickle"
            newClassyfier = imgClassifier(model)
            self.classifiers[size] = newClassyfier

    # 分类
    def classify(self, img:np.array, size:int):
        return self.classifiers[size].classify([img])[0]

    # 计算各tag的概率
    def calcProb(self, img:np.array, size:int):
        return self.classifiers[size].calcProb([img])[0]