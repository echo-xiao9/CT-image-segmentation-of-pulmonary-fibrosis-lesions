# CT image segmentation of pulmonary fibrosis lesions 肺部CT纤维化病灶图像分割

## 项目背景
间质性肺病（Interstitial Lung Disease, ILD）是一类以弥漫性肺实质、肺泡炎症和间质纤维化为病理指征的病理实体总称。其中，肺纤维化（Pulmonary Fibrosis）的典型病理可大致分为实变、蜂窝、网状、磨玻璃影四类。利用图像处理技术对肺部CT影像进行病灶自动化识别，能够有效减少影像科医生的工作量，并为其提供辅助诊断依据。

## 项目要求
1. 查阅相关文献，整理肺部CT影像病灶分割方法及分割结果评价指标。
2. 病灶区域像素分类：给定未知类别的病灶区域，设计并实现算法，对该区域内的病变组织进行像素级分类。
3. 病灶区域分割：给定肺实质前景区域，设计并实现算法，识别并分割各类病灶组织。
（你可以将2、3看作两个独立的问题分别求解，也可以在3中使用2中的算法）
4. 分割结果评价：采用量化指标对2、3的处理结果进行评价。

## 提交材料
1. 算法源代码及说明文档；
2. 可视化结果：四类病灶分类、分割结果（NIFTI）、肺部CT的三维重建；
3. 总结报告：总结实验流程，包括详细的算法描述、中间过程和最终效果的展示，以及对不同效果的实验分析。请注明参考文献。
## 参考示例
红色：实变；绿色：磨玻璃影；黄色：蜂窝影；蓝色：网状影


<img width="272" alt="截屏2022-06-27 上午9 23 34" src="https://user-images.githubusercontent.com/60086218/175843480-a6e08738-4983-48b8-8417-e061592211b0.png">


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
