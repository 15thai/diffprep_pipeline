import numpy as np
import os
import nibabel as nib
from dipy.denoise.GibbRemoval import  gibbremoval
from dipy.denoise.randomlpca_denoise import randomlpca_denoise
from dipy.denoise.fsl_bet import fsl_bet_mask
from dipy.align.reslice import reslice
from dipy.denoise.ChangeFOV import changeFoV

def get_affine_image_data(image_fn):
    image = nib.load(image_fn)
    return image.affine, image.get_data()


def get_GibbRemoved_data(input_arr):
    output = gibbremoval(input_arr)
    return output

def change_field_of_view(input_image_nib, fov, com = False):
    output, output_affine = changeFoV(input_image_nib, fov, com)
    return output, output_affine

def get_b0_image(image_data, b0_id):
    if image_data.ndim  == 3:
        return image_data

    if image_data.ndim > 3:
        return image_data[...,b0_id]
    else:
        return 0


def denoising_image (image_data_4D):
    if image_data_4D.ndim != 4:
        return 0
    data_denoise, out_noise, sigma_noise = randomlpca_denoise(image_data_4D)
    return data_denoise, out_noise, sigma_noise


def creating_mask(image_filename, b0_masked_filename, isBinary = True):
    fsl_bet_mask(input_image_fn= image_filename,
                 output_image_fn = b0_masked_filename,
                 binary_mask= isBinary)


def dmc_make_target( b0_image_fn, b0_bi_mask_data):
    fct = 0.9
    b0_image = nib.load(b0_image_fn)
    current_resolution = np.array( b0_image.header.get_zooms()[:3])
    new_resolution = current_resolution /fct
    transform_b0_data, transform_b0_affine = reslice(b0_image.get_data(),
                                                b0_image.affine,
                                                current_resolution,
                                                new_resolution,
                                                order = 3)
    transform_mask_data, transform_mask_affine = reslice(b0_bi_mask_data,
                                                     b0_image.affine,
                                                     current_resolution,
                                                     new_resolution,
                                                     order=3)

    transform_mask_data[(transform_mask_data < 0.5)] = 0
    transform_mask_data[(transform_mask_data >= 0.5)] = 1

    x, y, z = np.where(transform_mask_data == 1)

    minx = min(x)
    miny = min(y)
    minz = min(z)
    maxx = max(x)
    maxy = max(y)
    maxz = max(z)
    if minx == transform_mask_data.shape[0] + 5:
        minx = 0
    if miny == transform_mask_data.shape[1] + 5:
        miny = 0
    if minz == transform_mask_data.shape[2] + 5:
        minz = 0
    if maxx == -1:
        maxx = transform_mask_data.shape[0] - 1
    if maxy == -1:
        maxy = transform_mask_data.shape[1] - 1
    if maxz == -1:
        maxz = transform_mask_data.shape[2] - 1

    minx = max(minx - 2, 0)
    maxx = min(maxx + 2, transform_mask_data.shape[0] - 1)
    miny = max(miny - 2, 0)
    maxy = min(maxy + 2, transform_mask_data.shape[1] - 1)
    minz = max(minz - 2, 0)
    maxz = min(maxz + 2, transform_mask_data.shape[2] - 1)

    start = np.array([minx, miny, minz])
    sz = np.array([maxx - minx + 1, maxy - miny + 1, maxz - minz + 1])

    if np.sum(sz) < 0.3 * np.sum(transform_mask_data.shape):
        start = np.zeros(3)
        sz = transform_mask_data.shape
    DMC_image = np.zeros(transform_mask_data.shape)
    DMC_image[start[0]: start[0] + sz[0],
              start[1]: start[1] + sz[1],
              start[2]: start[2] + sz[2]] = transform_b0_data[start[0]: start[0] + sz[0],
                                                              start[1]: start[1] + sz[1],
                                                              start[2]: start[2] + sz[2]]

    msk_cnt = len(b0_bi_mask_data[b0_bi_mask_data != 0])
    npixel = b0_bi_mask_data.size

    # Set the equal to 1 - Keep the orginal image
    if msk_cnt < 0.02 * npixel:
        b0_bi_mask_data[:] = 1
    return DMC_image, b0_bi_mask_data


def choose_range(b0_image_data, curr_vol, b0_mask_image):
    fixed_signal = np.sort(b0_image_data[b0_mask_image != 0])
    moving_signal = np.sort(curr_vol[b0_mask_image != 0])
    koeff = 0.005
    nb = fixed_signal.shape[0]
    ind = int((nb-1) - koeff * nb)
    lim_arr = np.zeros(4)
    lim_arr[0] = 0.1
    lim_arr[1] = fixed_signal[ind]
    lim_arr[2] = 0.1
    lim_arr[3] = moving_signal[ind]
    return lim_arr

def set_images_in_scale(lim_arr, fixed_image, moving_image):
    if lim_arr is None:
        fixed_min = fixed_image.min()
        fixed_max = fixed_image.max()
        moving_min = moving_image.min()
        moving_max = moving_image.max()
    else:
        fixed_min = lim_arr[0]
        fixed_max = lim_arr[1]
        moving_min = lim_arr[2]
        moving_max = lim_arr[3]
    # Setting image boundaries
    # where fixed_image > fixed_max , fixed_image =fixed_max
    fixed_image[fixed_image > fixed_max] = fixed_max
    fixed_image[fixed_image < fixed_min] = fixed_min
    moving_image[moving_image > moving_max] = moving_max
    moving_image[moving_image < moving_min] = moving_min
    return fixed_image, moving_image

def get_gradients_params (resolution, sizes, default = 21):
    sz = sizes
    grad_params = np.zeros(default)
    grad_params[:3] = resolution[:3] * 1.25
    grad_params[3:6] = 0.05
    grad_params[6:9] = resolution[2] * 1.5 / (sz[:3] / 2 * resolution[:3]) * 2
    grad_params[9] = 0.5 * resolution[2] * 10 / (sz[0] / 2 * resolution[0]) / (sz[1] / 2 * resolution[1])
    grad_params[10] = 0.5 * resolution[2] * 10 / (sz[0] / 2 * resolution[0]) / (sz[2] / 2 * resolution[2])
    grad_params[11] = 0.5 * resolution[2] * 10 / (sz[1] / 2 * resolution[0]) / (sz[2] / 2 * resolution[2])

    grad_params[12] = resolution[2] * 5. / (sz[0] / 2. * resolution[0]) / (sz[0] / 2. * resolution[0])
    grad_params[13] = resolution[2] * 8. / (sz[0] / 2. * resolution[0]) / (sz[0] / 2. * resolution[0]) / 2.

    grad_params[14] = 5. * resolution[2] * 4 / (sz[0] / 2. * resolution[0]) / (sz[1] / 2. * resolution[1]) / (
                sz[2] / 2. * resolution[2])

    grad_params[15] = 5. * resolution[2] * 1 / (sz[0] / 2. * resolution[0]) / (sz[0] / 2. * resolution[0]) / (
                sz[0] / 2. * resolution[0])
    grad_params[16] = 5. * resolution[2] * 1. / (sz[1] / 2. * resolution[1]) / (sz[1] / 2. * resolution[1]) / (
                sz[1] / 2. * resolution[1])
    grad_params[17] = 5. * resolution[2] * 1. / (sz[0] / 2. * resolution[0]) / (sz[2] / 2. * resolution[2]) / (
                sz[2] / 2. * resolution[2])
    grad_params[18] = 5. * resolution[2] * 1. / (sz[1] / 2. * resolution[1]) / (sz[2] / 2. * resolution[2]) / (
                sz[2] / 2. * resolution[2])
    grad_params[19] = 5. * resolution[2] * 1. / (sz[0] / 2. * resolution[0]) / (sz[0] / 2. * resolution[0]) / (
                sz[0] / 2. * resolution[0])
    grad_params[20] = 5. * resolution[2] * 1. / (sz[0] / 2. * resolution[0]) / (sz[0] / 2. * resolution[0]) / (
                sz[0] / 2. * resolution[0])
    grad_params[:] = grad_params[:] / 1.5
    return grad_params



def decimalToBinary(n):
    a = []
    if n> 1:
        b = decimalToBinary(n//2)
        a.append(b)

    return a
