from PIL import Image
import nibabel as nib
import numpy as np
import os
import Augmentation

'''
T1-横切片
reg_T1 - 纵切片 240*240 * 48  值（-11-700）都有
T1_mask - 一个模板 256 * 256 * 48 灰度值（0-1）
reg_IR - 纵切片 灰度值(2047-1260)， 需要除以10才能正常显示 240 * 240 * 48
IR - 纵切片 240 * 240 * 48 (1200-2400)
FLAIR - 纵切片 240*240*48 灰度值(0-800) 
'''
sourcepath = "/Volumes/PowerExtension/training/4"
subdirs = ["1", "4", "5", "7", "14", "070", "148"]
subfolder = ["orig", "pre"]
filenames = ["FLAIR.nii.gz","reg_IR.nii.gz" ,"IR.nii.gz"]
segname = "segm.nii.gz"

filepath = os.path.join(sourcepath, segname)

img = nib.load(filepath)
array_data = img.get_fdata()
print(array_data.shape)
print(type(array_data))

slice = [array_data[:,:,i]*30 for i in range(48)]
a = np.array(slice)
bundle = np.ndarray(shape=a.shape, buffer=a)
transformed_image = Augmentation.elastic_transformations(alpha=1000, sigma=60)(bundle)

for i in range(48):

    image = Image.fromarray(slice[i])
    image.show(title="No." + str(i) + " Slice")
    #reshape(1,image.size[0],image.size(1))
    t_image = Image.fromarray(transformed_image[i])
    t_image.show(title="No." + str(i) + " T_Slice")
    dist = np.linalg.norm(slice[i] - transformed_image[i])
    print(i,'-',image.size,"-", dist)




#header = img.header
#print(header)
#print(header.get_data_shape())

#array_data = np.arange(48, dtype=np.uint16)
#img_data = nib.Nifti1Image(array_data)
#print(type(img_data))

