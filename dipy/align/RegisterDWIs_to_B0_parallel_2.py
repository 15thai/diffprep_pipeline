# This version is using multiprocessor !
# Compare which one is faster.

import os
import numpy as np
import nibabel as nib
import multiprocessing, ctypes
from contextlib import closing
import pymp
import time

from dipy.align.sub_processes import choose_range
image_b0= "/home/anhpeo/Desktop/dipy_data_test/register_test/test_B/AP_b2500_b0.nii"

image_main = "/home/anhpeo/Desktop/dipy_data_test/register_test/AP_b2500.nii"
image_mask = "/home/anhpeo/Desktop/dipy_data_test/register_test/test_B/AP_b2500_b0_fsl_mask_mask.nii"

def init (shared_input_,
          share_output_,
          b0_arr_,
          mask_arr_,
          image_shape_) :
    global shared_input
    global shared_output
    global b0_arr
    global mask_arr
    global im_shape
    global ma_shape
    shared_input = shared_input_
    shared_output = share_output_
    b0_arr = b0_arr_
    mask_arr = mask_arr_
    im_shape = image_shape_
    ma_shape = mask_arr_.shape

def wrapper(vol_id):
    inp = np.frombuffer(shared_input)
    sh_input = inp.reshape(im_shape)

    out = np.frombuffer(shared_output)
    sh_out = out.reshape(ma_shape)
    sh_out[:,vol_id] = choose_range(b0_arr,
                                    sh_input[:,:,:,vol_id],
                                    mask_arr)



b0_image = nib.load(image_b0)
main_image = nib.load(image_main)
mask_image = nib.load(image_mask)

print(b0_image.shape, main_image.shape, mask_image.shape)
threads_to_use = multiprocessing.cpu_count()

b0_arr = b0_image.get_data()
mask_arr = mask_image.get_data()

imageSize= 1
for i in main_image.shape:
    print(i)
    imageSize = imageSize*i

# input array
mp_arr = multiprocessing.RawArray(ctypes.c_double, imageSize)
shared_arr = np.frombuffer(mp_arr)
shared_input = shared_arr.reshape(main_image.shape)
shared_input[:] = main_image.get_data()

mp_arr2 = multiprocessing.RawArray(ctypes.c_double, 4*main_image.shape[-1])
shared_arr2 = np.frombuffer(mp_arr2)
shared_output = shared_arr2.reshape((4,main_image.shape[-1]))

with closing(multiprocessing.Pool(threads_to_use,
                                  initializer= init,
                                  initargs=(shared_arr,
                                            shared_arr2,
                                            b0_arr,
                                            mask_arr,
                                            main_image.shape))) as p:
    p.map_async(wrapper, [vol_id for vol_id in range(0,main_image.shape[-1])])

p.join()

print(shared_output)
print(shared_output.shape)


### Synchronous
lim_arr = np.zeros((4, main_image.shape[-1]))
for vol_id in range(main_image.shape[-1]):
    lim_arr[:,vol_id] = choose_range(b0_arr,
                                     main_image.dataobj[...,vol_id],
                                     mask_arr)

print(lim_arr)
print(lim_arr.shape)