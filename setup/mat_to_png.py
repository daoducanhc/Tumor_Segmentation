import os
import numpy as np
import matplotlib.image
from tqdm import tqdm
import h5py

def get_image_data(filename, path):
    path = os.path.join(path, filename+'.mat')
    file = h5py.File(path, 'r')
    data = dict()
    data['image'] = np.array(file.get('cjdata/image'))
    data['mask'] = np.array(file.get('cjdata/tumorMask'))
    return data

def save_image_data(filename, path, data):
    path_image = os.path.join(path, filename + '.png')
    path_mask = os.path.join(path, filename + '_mask.png')
    matplotlib.image.imsave(path_image, data['image'], cmap='gray', format='png')
    matplotlib.image.imsave(path_mask, data['mask'], cmap='gray', format='png')


root = 'C:/Users/Admin/Desktop/Downloads/brain_tumor/data/'
read_path = os.path.join(root, 'mat_dataset')
save_path = os.path.join(root, 'png_dataset')

# for filename in tqdm(range(1, 3064+1)):
#     filename = str(filename)
#     data = get_image_data(filename, read_path)
#     save_image_data(str(int(filename)-1), save_path, data)
