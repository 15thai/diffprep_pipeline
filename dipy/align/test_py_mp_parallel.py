import pymp
from dipy.align.sub_processes import  choose_range
import nibabel as nib
import numpy as np
import time

b0_target_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0.nii"
moving_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc.nii"
b0_target_mask = nib.load("/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0_mask_mask.nii")
target_mask_arr = b0_target_mask.get_data()

b0_image = nib.load(b0_target_image)
b0_arr = b0_image.get_data()

moving_image = nib.load(moving_image)

ex_array = pymp.shared.array((moving_image.shape), dtype=np.float32)
ex_array[:] = moving_image.get_data()
print('1')
start_time = time.time()

lim_arr = pymp.shared.array((4, moving_image.shape[-1]), dtype= np.float32)
with pymp.Parallel() as p:
    for index in p.range(0, moving_image.shape[-1]):
        p.print(index)
        curr_vol = ex_array[:,:,:,index]
        lim_arr[:,index]  = choose_range(b0_arr, curr_vol, target_mask_arr)
        # p.print(a)
        # The parallel print function takes care of asynchronous output.

print("Time cost {}", time.time() - start_time)
print(lim_arr)
print(lim_arr.shape)
image_arr = moving_image.get_data()
start_time = time.time()

for index in range(0, moving_image.shape[-1]):
    lim_arr = choose_range(b0_arr, image_arr[:,:,:,index], target_mask_arr)
print("Time cost {}", time.time() - start_time)
