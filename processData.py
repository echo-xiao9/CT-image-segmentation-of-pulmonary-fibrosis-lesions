from dicomProcess import *
from GenerateNIFTI import generateNIFTIFile

# 处理文件，生成标注
# index为待处理的case number
index = [1]
for i in index:
    BASE_PATH = f'../SE362-Projects/Project1/case_{i}/'
    DICOM_PATH = 'DICOM'
    FOREGROUND_PATH = 'foreground.nii.gz'
    PATHOLOGY_PATH = 'pathology.nii.gz'
    NEW_PATHOLOGY_PATH = 'newPathology.nii.gz'
    NEW_PATHOLOGY_PATH_R = 'newPathologyR.nii.gz'
    OUTPUT_PATH = f'outputData{i}'

    sizes = [17, 19, 21]

    labels, eval = processDicomWithEvaluating(BASE_PATH + DICOM_PATH, BASE_PATH + FOREGROUND_PATH, BASE_PATH + PATHOLOGY_PATH, BASE_PATH + OUTPUT_PATH, sizes)
    with open(BASE_PATH + "tmpNII.pkl", "wb") as f:
        pickle.dump(labels, f)
    with open(BASE_PATH + "evaluating.pkl", "wb") as f:
        pickle.dump(eval, f)
    generateNIFTIFile(labels, BASE_PATH + PATHOLOGY_PATH, BASE_PATH + NEW_PATHOLOGY_PATH)