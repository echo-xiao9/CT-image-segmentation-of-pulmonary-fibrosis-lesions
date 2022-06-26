# CT image segmentation of pulmonary fibrosis lesions

## Documents

详细算法说明请见以下文档，在release中也可以找到。

[结题报告+康艺潇+518431910002.pdf](https://github.com/echo-xiao9/CT-image-segmentation-of-pulmonary-fibrosis-lesions/files/8986719/%2B.%2B518431910002.pdf)

[可视计算中期报告_康艺潇_518431910002.pdf](https://github.com/echo-xiao9/CT-image-segmentation-of-pulmonary-fibrosis-lesions/files/8986721/_._518431910002.pdf)

[中期汇报.pptx](https://github.com/echo-xiao9/CT-image-segmentation-of-pulmonary-fibrosis-lesions/files/8986722/default.pptx)

[期末答辩.pptx](https://github.com/echo-xiao9/CT-image-segmentation-of-pulmonary-fibrosis-lesions/files/8986723/default.pptx)

### 以下文件是我们的项目文件，需要与SE362-Projects置于同一父目录下使用，结果将输出到相应目录下

* processData.py: 处理入口，按输入的list处理相应的case，生成输出图片、标注文件和评估数据
* dicomProcess.py: 包含处理dicom文件所用的方法，包括处理dicom文件生成数据集、处理dicom文件生成标注和结果图片、处理dicom文件生成标注和结果文件和评估数据
* window.py: 包含用于分窗口处理图像的方法，包括分窗口处理图像生成数据集、分窗口处理图像生成mask
* uniformClassifier.py: 对机器学习模型的统一封装，根据size调用相应模型
* subProcess.py: 对数据集进行后处理以平衡正常标签的含量
* test.py: 一些统计用方法，包括对数据集内标签比例的统计、对评估数据的统计等
* classifier.py: 定义了机器学习的分类器，在类的初始化函数中指明要用的分类器的类型，setTrainData()可以输入数据，然后调用其train（）函数对分类器进行训练。在使用时，使用predict()函数或者predict_prob()函数依据已有的特征向量预测目标类别或者各个类别的概率。并且save()和load()函数可以进行模型的存取。
* getGLCM.py: 定义了灰度共生矩阵特征提取和特征向量的制作。调用getFeature()函数可以将输入的图片的特征值提取出来转换为向量。makeDataSet()函数用来制备数据集。imgClassifier类定义了一个图像分类的接口，输入训练好的分类器模型路径，然后输入图片，就可以给出其分类结果。
evaluating.py文件定义了分割算法和分类算法的评价指标，在预测结果出来之后将预测的结果和正确的结果用inputData()方法输入评估器，调用评估器的相关函数便可以输出其各类评价指标。
* train.py: 定义了依据分割好的数据集进行模型训练的流程。
* Reconstruction3D.py: 结果的三维重建.
### 以下是一些重要的中间结果，虽然并没有在项目中实际部署，但是对于我们的最终结果具有重要意义
* GLCM_prev.py、predict_prev.py、train_prev.py、toolFunc_prev.py，旧版代码，在这版代码的实验中我们发现了我们数据集和模型存在问题，后来对其进行了重构。这一版本代码是对我们最终结果有重要意义的尝试，故同时提交。
* pretreatment.py，用于图片格式的数据预处理，提取肺实质前景，因为项目后期作业提供的数据集中已经完成了这部分工作，故没有在项目中应用。
* ReticularPattern/，该文件夹下面是前期使用传统方法提取网状影的算法，因为只有网状影部分在前期工作中取得了较好效果，故同时提交这部分代码。# CT-image-segmentation-of-pulmonary-fibrosis-lesions



Grade: 93/100
