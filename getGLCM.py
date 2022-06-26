
import pickle
from math import log2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from classifier import Classifier

def getFeature(oriImg: np.ndarray):
    mask = np.linspace(0,255,16)
    img = np.uint8(np.digitize(oriImg, mask))
    glcm = graycomatrix(img, [2,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16)
    contrast = graycoprops(glcm,'contrast')
    homogeneity = graycoprops(glcm,'homogeneity')
    ASM = graycoprops(glcm,'ASM')
    correlation = graycoprops(glcm,'correlation')
    mean_gray = np.sum(oriImg) / (oriImg.shape[0] * oriImg.shape[1])
    entropy = []
    for i in range(4):
        glcm_t = glcm[:,:,0,i]
        entropy_t = 0
        for ele in glcm_t.flat:
            if ele != 0:
                entropy_t = entropy_t + ele * log2(ele)
        entropy.append(entropy_t)
    for i in range(4):
        glcm_t = glcm[:,:,1,i]
        entropy_t = 0
        for ele in glcm_t.flat:
            if ele != 0:
                entropy_t = entropy_t + ele * log2(ele)
        entropy.append(entropy_t)
    feature_vec = np.insert(np.array([contrast, homogeneity, ASM, correlation]).reshape((32)),0,entropy)
    return np.insert(feature_vec, 0, mean_gray)




def makeDataSet(imgs:list, labels:list):
    featuresList = []
    for i in range(len(imgs)):
        feature = getFeature(imgs[i])
        label = labels[i]
        featuresList.append(feature)
        print("finish deal image %d" % i) 
    featureArray = np.array(featuresList)
    labelArray = np.array(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(featureArray, labelArray, test_size=0.3, random_state=42)
    res = {'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
    with open('dataSet.pickle','wb') as f:
        pickle.dump(res, f, 2)




class imgClassifier:
    def __init__(self,model:str) -> None:
        self.classifier = Classifier('random_forest')
        self.classifier.load(model)

    def classify(self,imgs:list):
        featuresList = []
        for i in range(len(imgs)):
            feature = getFeature(imgs[i])
            featuresList.append(feature)
            # print("finish deal image %d" % i) 
        featureArray = np.array(featuresList)
        return self.classifier.predict(featureArray)

    def calcProb(self, imgs:list):
        featuresList = []
        for i in range(len(imgs)):
            feature = getFeature(imgs[i])
            featuresList.append(feature)
            # print("finish deal image %d" % i) 
        featureArray = np.array(featuresList)
        return self.classifier.predict_prob(featureArray)