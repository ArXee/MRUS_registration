import os
import numpy as np
import torch
from skimage.transform import resize
import nibabel as nib
from torch.utils.data import Dataset
def resize_3d_image(image, shape):
    resized_image = resize(image, output_shape=shape)
    if np.amax(resized_image) == np.amin(resized_image):
        normalised_image = resized_image
    else:
        normalised_image = (resized_image - np.amin(resized_image)) / (np.amax(resized_image) - np.amin(resized_image))
    return normalised_image


class MedicalImageDataset(Dataset):
    def __init__(self, f_path, moving_image_shape, fixed_image_shape, with_label_inputs=True):
        self.moving_image_shape = moving_image_shape
        self.fixed_image_shape = fixed_image_shape
        self.with_label_inputs = with_label_inputs

        self.moving_images_path = os.path.join(f_path, 'us_images')
        self.fixed_images_path = os.path.join(f_path, 'mr_images')
        self.all_names = os.listdir(self.fixed_images_path)

        if with_label_inputs:
            self.moving_labels_path = os.path.join(f_path, 'us_labels')
            self.fixed_labels_path = os.path.join(f_path, 'mr_labels')

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        f_name = self.all_names[idx]

        moving_image = nib.load(os.path.join(self.moving_images_path, f_name)).get_fdata()
        fixed_image = nib.load(os.path.join(self.fixed_images_path, f_name)).get_fdata()

        moving_image = resize_3d_image(moving_image, self.moving_image_shape)
        fixed_image = resize_3d_image(fixed_image, self.fixed_image_shape)



        if self.with_label_inputs:
            moving_label = nib.load(os.path.join(self.moving_labels_path, f_name)).get_fdata()
            fixed_label = nib.load(os.path.join(self.fixed_labels_path, f_name)).get_fdata()

            label_to_select = np.random.randint(6)  # pick one label randomly for training
            moving_label = resize_3d_image(moving_label[:, :, :, label_to_select], self.moving_image_shape)
            fixed_label = resize_3d_image(fixed_label[:, :, :, label_to_select], self.fixed_image_shape)
            inputs = (torch.from_numpy(moving_image).float(),
                      torch.from_numpy(fixed_image).float(),
                      torch.from_numpy(moving_label).float(),
                      torch.from_numpy(fixed_label).float())
            outputs = (torch.from_numpy(fixed_image).float(),
                       torch.from_numpy(fixed_label).float())

        else:
            zero_phis = np.zeros(self.moving_image_shape[:-1] + (3,))
            outputs = (torch.from_numpy(fixed_image).float().unsqueeze(0),
                       torch.from_numpy(zero_phis).float().unsqueeze(0))

        return inputs, outputs


