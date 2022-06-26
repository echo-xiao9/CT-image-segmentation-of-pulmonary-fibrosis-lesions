import vtk

#病灶nii文件、肺实质前景nii文件，dicom文件夹的路径，如果不需要显示则设置为None
PATHOLOGY_PATH = 'd:\SJTU\ComputerVision\SE362-Projects\Project1\case_1\\newPathology.nii.gz'
#PATHOLOGY_PATH = None
FOREGROUND_PATH = 'd:\SJTU\ComputerVision\SE362-Projects\Project1\case_1\\foreground.nii.gz'
#FOREGROUND_PATH = None
DICOM_PATH = 'd:\SJTU\ComputerVision\SE362-Projects\Project1\case_1\\DICOM'
#DICOM_PATH = None

#设置身体、骨头、病灶、肺实质的 不透明度 （越小越透明）
BODY_OPACITY = 0    #身体
BONE_OPACITY = 0.5    #骨头
PATHOLOGY_OPACITY = 0.9   #病灶
FOREGROUND_OPACITY = 0.01  #肺实质


# 参数为dicom文件夹目录，vtkRenderer,身体，骨头的不透明度, 返回vtkVolume
def DICOMReconstruction(directory_path, renderer, body_opacity, bone_opacity):
    # 读取文件
    dicomReader = vtk.vtkDICOMImageReader()
    dicomReader.SetDataSpacing(1.0, 1.0, 1.0)
    dicomReader.SetDirectoryName(directory_path)
    #dicomReader.SetMemoryRowOrderToFileNative()
    dicomReader.SetDataByteOrderToLittleEndian()
    dicomReader.Update()

    # 创建volumeMapper
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputConnection(dicomReader.GetOutputPort())
    mapper.SetBlendModeToComposite()

    # 设置颜色
    volumeColor = vtk.vtkColorTransferFunction()
    volumeColor.AddRGBPoint(-500, 0.0, 0.0, 0.0)
    volumeColor.AddRGBPoint(0,    0.2, 0.2, 0.2)
    volumeColor.AddRGBPoint(400, 1.0, 1.0, 1.0)

    # 设置透明度
    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    volumeScalarOpacity.AddPoint(-501, 0)
    volumeScalarOpacity.AddPoint(-500, body_opacity)
    volumeScalarOpacity.AddPoint(200, body_opacity)
    volumeScalarOpacity.AddPoint(400, bone_opacity)
    #volumeScalarOpacity.AddPoint(1000, bone_opacity)

    # 设置参数
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(volumeColor)
    volumeProperty.SetScalarOpacity(volumeScalarOpacity)
    #volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.2)
    volumeProperty.SetDiffuse(0.6)
    volumeProperty.SetSpecular(0.4)
    volumeProperty.SetSpecularPower(10.0)

    # 渲染结果
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volumeProperty)

    return volume

# 参数为nifti文件目录，vtkRenderer,病灶的不透明度, 返回vtkVolume
def NIFTIReconstruction(file_path, renderer, opacity):
    # 读取文件
    niftiReader = vtk.vtkNIFTIImageReader()
    niftiReader.SetFileName(file_path)
    niftiReader.Update()
    #niftiReader.SetDataSpacing(0.1, 0.1, 0.1)

    # 创建mapper
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(niftiReader.GetOutput())
    #volumeMapper.SetBlendModeToComposite()

    # 设置透明度
    popacity = vtk.vtkPiecewiseFunction()
    popacity.AddPoint(0, 0.0)
    popacity.AddPoint(1, opacity)
    popacity.AddPoint(2, opacity)
    popacity.AddPoint(3, opacity)
    popacity.AddPoint(4, opacity)
    popacity.AddPoint(255, opacity)
    #popacity.AddPoint(8000, 0.0)

    # 设置颜色
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0, 0, 0)
    color.AddRGBPoint(1,  1, 0, 0)
    color.AddRGBPoint(2, 0, 1, 0)
    color.AddRGBPoint(3, 0, 0, 1)
    color.AddRGBPoint(4, 1, 1, 0)
    color.AddRGBPoint(255, 0.5, 0.5, 0.5)

    # 设置参数
    property = vtk.vtkVolumeProperty()
    property.SetColor(color)
    property.SetScalarOpacity(popacity)
    property.ShadeOn()
    #property.SetInterpolationTypeToLinear()
    property.SetShade(0, 1)
    property.SetDiffuse(0.5)
    property.SetAmbient(0.4)
    property.SetSpecular(0.2)
    property.SetSpecularPower(10.0)
    property.SetComponentWeight(0, 1)
    property.SetDisableGradientOpacity(1)
    property.DisableGradientOpacityOn()
    property.SetScalarOpacityUnitDistance(1)

    # 渲染
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(property)
    
    return volume

# 修改niftiVolume的transform使之与dicomVolume匹配
def AdjustTransform(niftiVolume, dicomVolume):
    #比例
        transf = vtk.vtkTransform()
        transf.PostMultiply()
        transf.Concatenate(niftiVolume.GetMatrix())
        c1 = niftiVolume.GetCenter()
        c2 = dicomVolume.GetCenter()
        factor = c2[0] / c1[0]
        transf.Scale(factor,factor,factor)
        niftiVolume.SetUserTransform(transf)
        #旋转
        transf = vtk.vtkTransform()
        transf.PostMultiply()
        transf.Concatenate(niftiVolume.GetMatrix())
        transf.RotateX(180)
        niftiVolume.SetUserTransform(transf)
        #位移
        transf = vtk.vtkTransform()
        transf.PostMultiply()
        transf.Concatenate(niftiVolume.GetMatrix())
        c1 = niftiVolume.GetCenter()
        c2 = dicomVolume.GetCenter()
        x = c2[0] - c1[0]
        y = c2[1] - c1[1]
        z = c2[2] - c1[2]
        transf.Translate(x, y, z)
        transf.Translate(0, 0, 10.6)
        niftiVolume.SetUserTransform(transf)
        return

# 参数为dicom文件夹目录, nifti文件目录，身体，骨头，病灶不透明度
def Reconstruction3D(dicom_path, pathology_path, foreground_path, body_opacity, bone_opacity, pathology_opacity, foreground_opacity):
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(512, 512)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # 绘制NIFTI 
    if pathology_path:
        pathologyVolume = NIFTIReconstruction(pathology_path, ren, pathology_opacity) #foreground
        ren.AddViewProp(pathologyVolume)

    if foreground_path:
        foregroundVolume = NIFTIReconstruction(foreground_path, ren, foreground_opacity) #foreground
        ren.AddViewProp(foregroundVolume)

    # 绘制DICOM
    if dicom_path:
        dicomVolume = DICOMReconstruction(dicom_path, ren, body_opacity, bone_opacity)
        ren.AddViewProp(dicomVolume)

    #同时显示dicom和nifti时，修正大小，旋转，位移
    if pathology_path and dicom_path:
        AdjustTransform(pathologyVolume, dicomVolume)
    if foreground_path and dicom_path:
        AdjustTransform(foregroundVolume, dicomVolume)
    
    # 创建camera
    camera = ren.GetActiveCamera()
    if pathology_path:
        c = pathologyVolume.GetCenter() 
    elif dicom_path:
        c = dicomVolume.GetCenter() 
    elif foreground_path:
        c = foregroundVolume.GetCenter() 
    else:
        c= [0,0,0]
    #print(c)
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.SetPosition(c[0], c[1] + 500, c[2] + 400)
    camera.SetViewUp(0, 0, -1)


    # 渲染
    iren.Initialize()
    renWin.Render()
    iren.Start()


# 参数为dicom文件夹目录, 病灶nifti文件目录，肺实质nifti文件目录，身体，骨头，病灶，肺实质不透明度
Reconstruction3D(DICOM_PATH, PATHOLOGY_PATH, FOREGROUND_PATH, BODY_OPACITY, BONE_OPACITY, PATHOLOGY_OPACITY, FOREGROUND_OPACITY)


