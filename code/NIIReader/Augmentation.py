from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import os

sourcepath = './../dataset/orig'
outputpath = './../dataset/processed'
subdirs = ["1", "4", "5", "7", "14", "070", "148"]
subfolder = ["orig"]
filenames = ["FLAIR.nii.gz","reg_IR.nii.gz" ,"IR.nii.gz"]
segname = "segm.nii.gz"

def getPaths():

    image_paths = []
    label_paths = []
    for i in subdirs:
        p_path = os.path.join(sourcepath, i)
        label_path_i = os.path.join(p_path, segname)
        label_paths.append(label_path_i)
        r_path = os.path.join(p_path, subfolder[0])
        image_path = []
        for j in filenames:
            image_path.append(os.path.join(r_path, j))
        image_paths.append(image_path)

    return image_paths, label_paths

def rotatePair(image, label, anger):

    rot_image = image
    rot_label = label
    for i in range(1, anger + 1):
        rot_image = np.rot90(rot_image)
    #t_image = Image.fromarray(rot_image)
    #t_image.show(title="No." + str(i) + " T_Slice")
    for i in range(1, anger + 1):
        rot_label = np.rot90(rot_label)
    #t_image = Image.fromarray(rot_label*30)
    #t_image.show(title="No." + str(i) + " T_Slice")
    return [rot_image, rot_label]

def rollPair(image, label, shift):

    roll_image = np.roll(image, -8)
    roll_label = np.roll(label, -8)
    return [roll_image, roll_label]
    #t_image = Image.fromarray(roll_image)
    #t_image.show(title="No." + " T_Slice")

    #t_image = Image.fromarray(roll_label * 30)
    #t_image.show(title="No." + " T_Slice")



def dataAugmentation(image_paths, label_paths):

    for i in range(len(label_paths)):

        images_x = []
        orig_images_x = []
        shrink = [1., 0.1, 0.1, 1.]
        weights = [0.5, 0.25, 0.25]
        weights2 = [0.4, 0.3, 0.3]
        shifts = [5]

        for seq_path in image_paths[i]:
            # seq = nib.load(seq_path)
            # seq_array_data = seq.get_fdata()
            seq = sitk.ReadImage(seq_path)
            seq_array_data = sitk.GetArrayFromImage(seq)
            seq_array_data = np.transpose(seq_array_data,(1, 2, 0))
            # TODO: Resize image data from 240*240*48 to 256*256*48
            resize_seq_array_data = np.zeros((256,256,48))
            for j in range(48):
                resize_seq_array_data[:, :, j].fill(seq_array_data[0][0][j])
                resize_seq_array_data[8:248,8:248, j] = seq_array_data[:, :, j]
            slice = [resize_seq_array_data[:, :, s] for s in range(48)]
            a = np.array(slice)
            # bundle = np.ndarray(shape=a.shape, buffer=a)
            images_x.append(a)
            # images_x.append(bundle)
            orig_images_x.append(slice)

        # seg = nib.load(label_paths[i])
        # seg_array_data = seg.get_fdata()
        seg = sitk.ReadImage(label_paths[i])
        seg_array_data = sitk.GetArrayFromImage(seg)
        seg_array_data = np.transpose(seg_array_data, (1, 2, 0))
        # TODO: Resize seg data from 240*240*48 to 256*256*48
        resize_seg_array_data = np.zeros((256,256,48))
        for j in range(48):
            resize_seg_array_data[:, :, j].fill(seg_array_data[0][0][j])
            resize_seg_array_data[8:248,8:248, j] = seg_array_data[:, :, j]
        slice = [resize_seg_array_data[:, :, s] for s in range(48)]
        a = np.array(slice)
        # bundle = np.ndarray(shape=a.shape, buffer=a)
        images_x.append(a)
        # images_x.append(bundle)
        orig_images_x.append(slice)

        anx = []
        anx.append(orig_images_x)

        anx.append(elastic_transformations_bundle(1000, 40)(images_x))
        anx.append(elastic_transformations_bundle(1000, 50)(images_x))
        anx.append(elastic_transformations_bundle(1000, 60)(images_x))
        anx.append(elastic_transformations_bundle(1000, 80)(images_x))


        # yuep: define anx names to write as file names
        anx_names = ['orig', 't1', 't2', 't3', 't4']

        for bundle_x in anx:
            for j in range(len(shrink)):
                for k in range(48):
                    bundle_x[j][k] *= shrink[j]

        output = []
        for b_idx, bundle_x in enumerate(anx):
            for j in range(48):
                weighted_image = bundle_x[0][j] * weights[0]  + bundle_x[1][j] * weights[1] + bundle_x[2][j] * weights[2]
                label = bundle_x[3][j]
                pair = [weighted_image, label]
                pair = np.array(pair)

                # yuep: save one image as a file
                path = os.path.join(outputpath, subdirs[i] + '_' +  (anx_names[b_idx] + '_w1_' + str(j)))
                np.save(file = path, arr = pair)

                weighted2_image = bundle_x[0][j] * weights2[0] + bundle_x[1][j] * weights2[1] + bundle_x[2][j] * weights2[2]
                pair = [weighted2_image, label]
                pair = np.array(pair)

                # yuep: save one image as a file
                path = os.path.join(outputpath, subdirs[i] + '_' + (anx_names[b_idx] + '_w2_' + str(j)))
                np.save(file = path, arr = pair)


                rotated_pair = np.array(rotatePair(weighted_image, label, 2))

                # yuep: save one image as a file
                path = os.path.join(outputpath, subdirs[i] + '_' + (anx_names[b_idx] + '_w1r_' + str(j)))
                np.save(file = path, arr = rotated_pair)


                rotated_pair = np.array(rotatePair(weighted2_image, label, 2))

                # yuep: save one image as a file
                path = os.path.join(outputpath, subdirs[i] + '_' + (anx_names[b_idx] + '_w2r_' + str(j)))
                np.save(file = path, arr = rotated_pair)






# Elastic transform

def elastic_transformations(alpha, sigma, rng=np.random.RandomState(42),
                            interpolation_order=1):
    def _elastic_transform_2D(images):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images[0].shape
        # Make random fields
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set

        transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                              for image in images]
        return transformed_images
    return _elastic_transform_2D

def elastic_transformations_bundle(alpha, sigma, rng=np.random.RandomState(42),
                            interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(images_x):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images_x[0][0].shape
        # Make random fields
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set
        transformed_images_x = []
        for images in images_x:

            transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                                  order=interpolation_order).reshape(image_shape)
                                  for image in images]
            transformed_images_x.append(transformed_images)
        return transformed_images_x
    return _elastic_transform_2D
