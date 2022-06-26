from tabnanny import verbose
from sklearn.neural_network import MLPClassifier
from sklearn import naive_bayes
from sklearn import datasets
from sklearn import svm
from sklearn import multiclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

class Classifier:
    def __init__(self, method:str = "mlp") -> None:
        if method == "mlp" or method == "naive_bayes" or method == "svm" or method == "knn" or method == "random_forest":
            self.method = method
        else:
            raise Exception("Unsupported machine learning method.")
    def train(self):
        if self.method == "mlp":
            self.classifier = MLPClassifier(hidden_layer_sizes=(20,10,5),activation="relu",verbose=True, learning_rate='adaptive',max_iter=1000)
        elif self.method == "svm":
            self.classifier = multiclass.OneVsOneClassifier(svm.SVC(verbose=True))
        elif self.method == "naive_bayes":
            self.classifier = naive_bayes.GaussianNB()
        elif self.method == "knn":
            self.classifier = KNeighborsClassifier(10)
        elif self.method == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators= 20, verbose= True)
        self.classifier.fit(self.data,self.target)
        
    # input: 2d-array, 每行是一个待分类向量
    # output: 1d-array, 结果的下标对应输入的行标
    def setTrainData(self, data, target):
        self.data = data 
        self.target = target

    def test(self, testData, testTarget):
        print("训练集:", self.classifier.score(self.data, self.target))
        print("测试集:", self.classifier.score(testData, testTarget))  
        return  self.classifier.score(self.data, self.target), self.classifier.score(testData, testTarget)
    
    # 输出预测结果
    # input: 2d-array, 每行是一个待分类向量
    # output: 1d-array, 结果的下标对应输入的行标
    def predict(self, data):
        return self.classifier.predict(data)

    # 输出各个分类结果的预测概率
    # input: 2d-array, 每行是一个待分类向量
    # output: 2d-array, 每一行是对应行号的输入向量的分类结果，其中的每一个数字代表对应tag的概率
    def predict_prob(self,data):
        if self.method == "svm":
            res = self.classifier.predict(data)
            max = self.classifier.classes_.shape[0]
            print(self.classifier.classes_)
            prob = np.zeros((res.shape[0], max))
            for i in range(res.shape[0]):
                prob[i][res[i]] = 1
            return prob
        else:
            return self.classifier.predict_proba(data)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load(self,filename:str):
        with open(filename, 'rb') as f:
            self.classifier = pickle.load(f)



def main():
    # Usage
    # 依据希望使用的分类器定义类, 类别包含“knn”（K-近邻算法）、“svm”（基于支持向量机二分类器的OneVsOne多分类）、“mlp”（多层感知机）、“naive_bayes”（朴素贝叶斯分类器）、“random_forest”（随机森林分类器）
    a = Classifier("random_forest")
    iris = datasets.load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    # 初始化方法1：输入训练集
    a.setTrainData(X_train, Y_train)
    # 初始化方法1：开始训练
    a.train()
    # 测试
    a.test(X_test,Y_test)
    # 模型使用
    print(a.predict([[6.2,	3.4	,5.4,	2.3], [5.4, 3. , 4.5, 1.5]]))
    print(a.predict_prob([[6.2,	3.4	,5.4,	2.3], [5.4, 3. , 4.5, 1.5]]))
    # 保存模型
    a.save("./tmp.pickle")
    b = Classifier("random_forest")
    # 初始化方法2：加载已有模型
    b.load("./tmp.pickle")
    print("predict")
    print(b.predict([[6.2,	3.4	,5.4,	2.3], [5.4, 3. , 4.5, 1.5]]))
    print("predict_prob")
    print(b.predict_prob([[6.2,	3.4	,5.4,	2.3], [5.4, 3. , 4.5, 1.5]]))

if __name__ == "__main__":
    main()