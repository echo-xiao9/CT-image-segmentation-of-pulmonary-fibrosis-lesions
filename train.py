import time
from getGLCM import getFeature,makeDataSet
import pickle
import classifier
import evaluating
size = 11
with open('./trainData/trainData_%d.pkl'%(size),'rb') as f:
    obj = pickle.load(f)
makeDataSet(obj[0], obj[1], "./dataset/dataset_%d"%(size))


f =  open("./dataset/dataset_%d"%(size), 'rb') 
data = pickle.load(f)
c = classifier.Classifier("random_forest")
c.setTrainData(data['X_train'], data['Y_train'])
c.train()
c.save("./models/mlp_%d.pickle"%(size))
c.load("./models/random_forest_%d.pickle"%(size))
c.test(data['X_test'], data['Y_test'])
e = evaluating.evaluating_Classification()
y_predict = c.predict(data['X_test'])
e.inputData(y_predict,data['Y_test'])
e.ConfusionMatrix_Visualization()
print("F1-score: %f"%(e.dice()))
print("类别平均召回率: %f"%(e.Recall()))
print("类别平均精准率: %f"%(e.MPA()))
print("准确率: %f"%(e.Recall()))

