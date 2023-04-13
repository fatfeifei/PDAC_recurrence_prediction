import numpy as np
import os
import SimpleITK as sitk
import glob


# 调整窗宽、窗位
def window(volume):
    """Normalize the volume"""
    min = -75
    max = 175
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


# 对医疗图像进行重采样
def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)



main_path = r"C:\Users\admin\Desktop\portal"
all_files=[]
for root,dirnames,filenames in os.walk(main_path):
    for files in filenames:
        file_path = os.path.join(root,files)
        if 'pv' in file_path:
            all_files.append(file_path)

for file in all_files:
    file_path = file[:-9]
    img = sitk.ReadImage(file)

    # 统一图像的spacing
    img = resample_image(img, out_spacing=[1.0, 1.0, 3.0])

    img_arr = sitk.GetArrayFromImage(img)

    size = img.GetSize()
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    pixelType = sitk.sitkUInt8

    # 调整窗宽、窗位
    new_img_arr = window(img_arr)

    image_new = sitk.Image(size, pixelType)

    image_new = sitk.GetImageFromArray(new_img_arr)
    image_new.SetDirection(direction)
    image_new.SetSpacing(spacing)
    image_new.SetOrigin(origin)

    #print(file)
    sitk.WriteImage(image_new, file_path + 'PP.nii.gz')

