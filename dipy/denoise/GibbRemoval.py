import nibabel as nib
import time
import numpy as np
import multiprocessing
import ctypes, argparse
from contextlib import closing
import argparse
from dipy.denoise.unring import  unring_2d


# ### To run UnRing from the command line ###
# parser = argparse.ArgumentParser(description= "python UnRing indir outdir")
# parser.add_argument('indir', type = str, help = '/path/to/input/file')
# parser.add_argument('outdir', type = str, help = '/path/to/output/file')
# args = parser.parse_args()
# #######################################################################

def unring_wrapper(slices):

    inp = np.frombuffer(shared_input)
    sh_input = inp.reshape(arr_shape)

    out = np.frombuffer(shared_output)
    sh_out = out.reshape(arr_shape)

    if len(arr_shape) == 3:
        slice_data = sh_input[:, :, slices]
        result_slice = unring_2d(slice_data, nsh, minW, maxW)
        sh_out[:, :, slices] = result_slice
    else:
        for k in range(arr_shape[3]):
            slice_data = sh_input[:, :, slices, k]
            result_slice = unring_2d(slice_data, nsh, minW, maxW)
            sh_out[:, :, slices, k] = result_slice


def init(shared_input_, shared_output_, arr_shape_, params_):
    # initialization of the global shared arrays
    global shared_input, shared_output, arr_shape, nsh, minW, maxW
    shared_input = shared_input_
    shared_output = shared_output_
    arr_shape = arr_shape_
    nsh = params_[0]
    minW = params_[1]
    maxW = params_[2]


def gibbremoval(arr, nsh=25, minW=1, maxW=5, out_dtype=None, num_threads=None):
    r"""Gibbs ringing correction for 4D DWI datasets.
    Parameters
    ----------
    arr : 3D, and 4D array
        Array of data to be corrected. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    nsh : int, optional
        Number of shifted images on one side. Default: 25. The total number of
        shifted images will be 2*nsh+1
    minW : int, optional
        Minimum neighborhood distance. Default:1
    maxW : int, optional
        Maximum neighborhood distance. Default:5
    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.
    num_threads : int, optional
         The number of threads that the algorithm can create. Default: Use all cores.

    Returns
    -------
    corrected_arr : 4D array
        This is the corrected array of the same size as that of the input data,
        clipped to non-negative values

    References
    ----------
    .. [Kellner2015] Kellner E., Bibek D., Valerij K. G., Reisert M.(2015)
                  Gibbs-ringing artifact removal based on local subvoxel-shifts.
                  Magnetic resonance in Medicine 76(5), p1574-1581.
                  https://doi.org/10.1002/mrm.26054
    Example:
    ----------
    from dipy.denoise import GibbRemoval
    from nibabel import nib

    image = nib.load("Gibb_image.nii")
    data = image.get_data()

    unring_image = GibbRemoval.gibbremoval(data)
    output_image = nib.Nifti1Image(unring_image, image.affine)
    nib.save(output_image, 'GibbRemoved_image.nii')
    """
    start_time = time.time()
    # We perform the computations in float64. However we output
    # with the original data_type
    if out_dtype is None:
        out_dtype = arr.dtype

    # if  arr.ndim == 3:
        # print('Converting input array from 3D to 4D...')
        # arr = arr.reshape([arr.shape[0], arr.shape[1], arr.shape[2]])

    if arr.ndim == 2:
        return unring_2d(arr, nsh, minW, maxW)

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = multiprocessing.cpu_count()


    # Creating input and output shared arrays for multi-process processing:

    # input array
    mp_arr = multiprocessing.RawArray(ctypes.c_double, arr.size)
    shared_arr = np.frombuffer(mp_arr)
    shared_input = shared_arr.reshape(arr.shape)
    shared_input[:] = arr[:]
    # output array
    mp_arr2 = multiprocessing.RawArray(ctypes.c_double, arr.size)
    shared_arr2 = np.frombuffer(mp_arr2)
    shared_output = shared_arr2.reshape(arr.shape)
    # parameters
    params = [nsh, minW, maxW]

    # multi-processing
    with closing(multiprocessing.Pool(threads_to_use, initializer=init,
                                      initargs=(shared_arr, shared_arr2, arr.shape, params))) as p:
        p.map_async(unring_wrapper, [slices for slices in range(0, arr.shape[2])])
    p.join()

    print("Gibbs ringing correction took --- %s seconds ---" % (time.time() - start_time))

    return shared_output.astype(out_dtype)


if __name__ == "__main__":
    image = nib.load(args.indir)
    affine = image.affine
    arr = image.get_data()
    unrang_arr = gibbremoval(arr)
    image_out = nib.Nifti1Image(unrang_arr, affine)
    nib.save(image_out, args.outdir)
    print("Done")