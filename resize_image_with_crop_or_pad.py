import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from scipy.ndimage import zoom
import glob
import matplotlib.pyplot as plt




def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0)+1
    return (zstart, ystart, xstart), (zstop, ystop, xstop)



def resize_image_with_crop_or_pad(image, img_size=(16, 512, 512), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)



main_path = r"C:\Users\admin\Desktop\pancreas data"
save_path = r"C:\Users\admin\Desktop\pancreas"

all_ct_files=[]
all_seg_files=[]
for root,dirnames,filenames in os.walk(main_path):
    for files in filenames:
        file_path = os.path.join(root,files)
        if 'process' in file_path:
            all_ct_files.append(file_path)
        if 'seg' in file_path:
            all_seg_files.append(file_path)


all_data = []
for ct_path, seg_path in zip(all_ct_files, all_seg_files):
    print(ct_path)
    print(seg_path)
    mask_path = seg_path[:-10]
    ct = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg)
    image_array = np.multiply(ct_array, seg_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(seg_array, use2D=False)
    roi_images = image_array[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_images2 = np.array(roi_images, dtype=np.float64)
    roi_images3 = resize_image_with_crop_or_pad(roi_images2, img_size=(50, 50, 50))
    all_data.append(roi_images3)


#保存数据
all_data = np.array(all_data)
all_data = all_data.transpose(0,2,3,1)
np.save(os.path.join(save_path,"all_data_pad_50_50_50.npy"), all_data)








