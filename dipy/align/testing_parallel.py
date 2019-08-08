import nibabel as nib
import time, pymp
import numpy as np
import multiprocessing, ctypes
from dipy.align.sub_processes import choose_range
from contextlib import closing

def init (shared_image_,
          shared_output_,
          shared_b0_,
          shared_mask_,
          shape_image_,
          shape_out_):

    global shared_image, shared_b0, shared_mask
    global shape_image, shape_out, shared_output

    shared_image = shared_image_
    shared_b0 = shared_b0_
    shared_mask = shared_mask_
    shape_image = shape_image_
    shape_out= shape_out_
    shared_output = shared_output_



def wrapper(vol_id):
    inp = np.frombuffer(shared_image)
    shr_image = inp.reshape(shape_image)

    out = np.frombuffer(shared_output)
    shr_output = out.reshape(shape_out)

    shr_output[:,vol_id] = choose_range(shared_b0,
                                        shr_image[:,:,:,vol_id],
                                        shared_mask)


def test1(whole_image, b0_image, mask_image):

    whole_image = nib.load(whole_image)
    b0_image = nib.load(b0_image)
    mask_image = nib.load(mask_image)

    whole_data = whole_image.get_data()

    start_time = time.time()
    mp_arr = multiprocessing.RawArray(ctypes.c_double, whole_data.size)
    shared_arr = np.frombuffer(mp_arr)
    shared_input = shared_arr.reshape(whole_data.shape)
    shared_input[:] = whole_data[:]

    mp_arr2 = multiprocessing.RawArray(ctypes.c_double, 4* whole_data.shape[-1])
    shared_arr2 = np.frombuffer(mp_arr2)
    shared_output = shared_arr2.reshape((4, whole_data.shape[-1]))

    threads_to_use = multiprocessing.cpu_count()

    with closing(multiprocessing.Pool(threads_to_use,
                                      initializer= init,
                                      initargs= (shared_arr,
                                                 shared_arr2,
                                                 b0_image.get_data(),
                                                 mask_image.get_data(),
                                                 whole_data.shape,
                                                 [4,whole_data.shape[-1] ]))) as p:
        p.map_async(wrapper,[slices for slices in range(whole_data.shape[-1])])
    p.join()

    print("consuming time %s",time.time() -start_time)
    print(shared_output)


def test2(whole_image, b0_image, mask_image):


    whole_image = nib.load(whole_image)
    b0_image = nib.load(b0_image)
    mask_image = nib.load(mask_image)

    whole_data = whole_image.get_data()
    start_time = time.time()

    lim_arr = np.zeros((4, whole_data.shape[-1]))
    for i in range(whole_data.shape[-1]):
        lim_arr[:,i] = choose_range(b0_image.get_data(),
                                    whole_data[:,:,:,i],
                                    mask_image.get_data())
    print("consuming time %s",time.time() -start_time)
    print(lim_arr)


def test3(whole_image, b0_image, mask_image):
    whole_image = nib.load(whole_image)
    b0_image = nib.load(b0_image)
    mask_image = nib.load(mask_image)

    whole_data = whole_image.get_data()
    # start_time = time.time()

    b0_arr = b0_image.get_data()
    mask_arr = mask_image.get_data()
    shared_data = pymp.shared.array(whole_image.shape, dtype= np.float32)
    shared_data[:] =whole_data[:]
    shared_out = pymp.shared.array((4, whole_image.shape[-1]), dtype= np.float32)

    start_time = time.time()
    with pymp.Parallel() as p:
        for index in p.range(0,whole_image.shape[-1]):
            curr_vol = shared_data[:,:,:,index]
            shared_out[:,index] = choose_range(b0_arr,
                                               curr_vol,
                                               mask_arr)
    print("consuming time %s",time.time() -start_time)
    print(shared_out)

whole_image = "/home/anhpeo/Desktop/dipy_data_test/register_test/AP_b2500.nii"
b0_image = "/home/anhpeo/Desktop/dipy_data_test/register_test/test_B/AP_b2500_b0.nii"
mask_image = "/home/anhpeo/Desktop/dipy_data_test/register_test/test_B/AP_b2500_b0_fsl_mask_mask.nii"
test1(whole_image, b0_image, mask_image)
test2(whole_image, b0_image, mask_image)
test3(whole_image, b0_image, mask_image)