import re
from matplotlib.pyplot import cla
from numpy import size
from GLCM import *
from toolFunc import *
from classifier import *
from evaluating import *

def getTrainData(size, case):
  # 从subImage获取训练数据
  data, tag = getSubImage('../Data',case, size)
  trainData = []
  trainTag = []

  # for i in range(len(tag)):
  for i in range(20):
  # for i in range(len(tag)):
    image = data[i]
    feature = glcmAll2(image)
    featureVec = normList(feature).flatten()
    trainData.append(featureVec)
    trainTag.append(tag[i])

  # turn list into array
  trainData = np.array(trainData)
  # all 表示6个特征
  trainDataName = '../data/trainData'+str(case)+'/TD_case'+ str(case)+'_size'+ str(size)+'_ALL'
  tagName = '../data/trainData'+str(case)+'/TAG_case'+ str(case)+'_size'+ str(size)
  np.savetxt(trainDataName, trainData, fmt="%.2f", delimiter=',') #保存为2位小数的浮点数，用逗号分隔
  np.savetxt(tagName, trainTag,fmt="%d", delimiter=',')
  return trainData, trainTag

# 从保存的文件中得到训练数据
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

def readArr(path):
  data = np.loadtxt(path)
  return data

# 分割训练集和测试集并训练
def segmentAndTrain(trainData, trainTag,classifier,case,size):
  X_train, X_test, Y_train, Y_test = train_test_split(trainData, trainTag, test_size=0.3, random_state=42)
  classifier.setTrainData(X_train, Y_train)
  classifier.train()
  trainResult, testResult = classifier.test(X_test, Y_test)
 
  if case==0:
    claFileName = '../data/allCase/CLA_'+ classifier.method +'_allCase'+'_size'+ str(size) +'_ALL.pickle'
  else:
    claFileName = '../data/trainData'+str(case)+'/CLA_'+ classifier.method +'_case'+ str(case)+'_size'+str(size) +'_ALL.pickle'

  classifier.save(claFileName)
  return trainResult, testResult


def loadClassifier(method, size, case):
  cla = Classifier(method)
  if case == 0 :
    claFileName=  '../data/allCase/CLA_'+ method +'_allCase'+'_size'+ str(size) +'_ALL.pickle'
  else :
    claFileName = '../data/trainData'+str(case)+'/CLA_'+ cla.method +'_case'+ str(case)+'_size'+str(size) +'_ALL.pickle'
  cla.load(claFileName)
  return cla

# 合并训练数据
def combineFile():
  # Open file3 in write mode
    sizeList = [11,13,15,17,19,21,23]
    for size in sizeList:
      with open('../data/allCase/TD_allCase_size'+str(size)+'_ALL.txt', 'w') as outTD:
        for case in range(1,11):
          trainDataName = '../data/trainData'+str(case)+'/TD_case'+ str(case)+'_size'+ str(size)+'_ALL'
          with open(trainDataName) as infile:
            outTD.write(infile.read())
          outTD.write("\n")

    for size in sizeList:
      with open('../data/allCase/TAG_allCase_size'+str(size)+'.txt', 'w') as outTAG:
        for case in range(1,11):
          trainDataName = '../data/trainData'+str(case)+'/TAG_case'+ str(case)+'_size'+ str(size)
          with open(trainDataName) as infile:
            outTAG.write(infile.read())
          outTAG.write("\n")




# 用所有数据（特定size）训练得到的分类器来预测某个case的样本
def predictWithSingle():
  trainData, trainTag = getTrainData(11,1)
  classifierType = "random_forest"
  cla = Classifier(classifierType)
  # 初始化方法2：加载已有模型
  cla = loadClassifier("random_forest",11,0)
  predArr = cla.predict(trainData)

  # test2(predArr,trainTag)
  print("predic")
  print(predArr)
  print("actual")
  print(trainTag)



# 得到所有 size的 合并case训练数据
def trainDataAll():
  sizeList = [11,13,15,17,19,21,23]
  for case in range(1,11):
   for size in sizeList:
     getTrainData(size,case)
 
# 尝试训练所有的分类器
def classifierAll():
  sizeList = [11,13,15,17,19,21,23]
  classifierList =["svm","knn","random_forest","mlp","naive_bayes"]
  sizePrecise =[0,0,0,0,0,0,0]
  
  for className in classifierList:
    fileName = '../Data/trainTxt/'+className+'_trainResult.txt'
    with open(fileName, 'w') as f:
      f.write(className+'\n')
      print(className)
      totalTrain=0
      totalTest=0
      for size in sizeList:
        
        trainData, trainTag = loadTrainData(size,0)
        classifier = Classifier(className)
        trainResult, testResult = segmentAndTrain(trainData, trainTag, classifier,0,size)
        totalTrain += trainResult
        totalTest += testResult
        results =  '%.2f'%(trainResult)+','+'%.2f'%(testResult)+'\n'
        f.write(results)
      average = '%.2f'%(totalTrain/len(sizeList))+','+ '%.2f'%(totalTest/len(sizeList))
      f.write(average)
    f.close()

# 训练 random_forest 在所有size上的模型
def classifierAll2():
  sizeList = [11,13,15,17,19,21,23]

  for case in range(1,2):
   for size in sizeList:
     trainData, trainTag = loadTrainData(size,case)
     classifier = Classifier("random_forest")
     segmentAndTrain(trainData, trainTag, classifier,case,size)

# 训练单个size的特定分类器
def trainSingleClassifier(trainData, trainTag,method, size):
  classifier = Classifier(method)
  return segmentAndTrain(trainData, trainTag,classifier,0,size)

# 用特定的classifer 进行预测， 调用evalutating 进行可视化
def claPredict(classifier, testData, testTag):
  predArr = classifier.predict(testData).astype(np.int)
  testTag = testTag.astype(np.int)
  return test2(predArr, testTag)

# 预测size=11 的结果，也可以更改size
def predictAll():
  f = open("CLA_random_forest.txt", "w")
  for claCase in range(1,11):
    cla = loadClassifier("random_forest",11,claCase)
    f.write("classifier random_forest from case"+ str(claCase)+" preditc \n")
    for case in range(1,11):
      testData, testTag = loadTrainData(11,case)
      PA,MPA,Recall, dice = claPredict(cla, testData, testTag)
      f.write("case"+str(case)+"PA:"+str(PA)+" MPA:"+str(MPA)+" Recall:"+Recall+" dice:"+dice+"\n")
      # f.write("case"+str(case)+"PA:"+str(PA))
  f.close()


# 整合得到所有训练数据流程（random_forest）
def getTrainDataAll():
  trainDataAll()
  combineFile()
  fileName = '../Data/trainTxt/'+"random_forest"+'_trainResult.txt'
  with open(fileName, 'w') as f:
    for size in range(11,25,2):
      trainData, trainTag = loadTrainData(size,0)
      trainResult, testResult = trainSingleClassifier(trainData, trainTag,"random_forest",size)
      f.write('%.2f'%trainResult+","+'%.2f'%(testResult)+"\n")
  f.close()
