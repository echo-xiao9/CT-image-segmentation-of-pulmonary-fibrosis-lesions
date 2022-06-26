
import nibabel as nib
import numpy as np

# 参数为2维数组的list，原文件路径和新文件路径
def generateNIFTIFile(arrayList:list, ORIGIN_FILE_PATH, PATH_TO_SAVE):
    #读取原文件
    originImage = nib.load(ORIGIN_FILE_PATH)

    #获取原文件的数据
    header = originImage.header
    affine = originImage.affine 
    #originImageData = originImage .get_fdata()
    #print(header)
    #print(np.shape(originImageData))

    # 根据list生成3维数组
    newImageData = np.zeros(shape=(512,512,len(arrayList)))
    for index in range(len(arrayList)):
        array2D = arrayList[index]
        for w in range(0,512):
            for h in range(0,512):
                newImageData[w,h,index] = array2D[h,w]

    #生成新文件,使用原文件的header
    nifti_file = nib.Nifti1Image(newImageData, affine, header)
    nib.save(nifti_file, PATH_TO_SAVE)
    
    return


# 测试
# aList = []
# for i in range(0,234):
#     newArray = np.ones(shape=(512,512))
#     aList.append(newArray)
# ORIGIN_FILE_PATH = 'd:\SJTU\ComputerVision\SE362-Projects\Project1\case_1\\pathology.nii.gz'
# PATH_TO_SAVE = 'd:\SJTU\ComputerVision\SE362-Projects\Project1\case_1\\newPathology2.nii.gz'
# generateNIFTIFile(aList, ORIGIN_FILE_PATH, PATH_TO_SAVE)

