from dipy.align.imwarp import get_direction_and_spacings
from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration, transform_centers_of_mass
import time
import numpy as np
import multiprocessing
import ctypes, argparse
from contextlib import closing
import nibabel as nib
from dipy.align import sub_processes


def get_angles_list (step = 45):
    angles_list = []
    for x in range(-180, 180, step):
        for y in range(-180, 180, step):
            for z in range(-180, 180, step):
                mx = x / 180 * np.pi
                my = y / 180 * np.pi
                mz = z / 180 * np.pi
                angles_list.append([mx, my, mz])
    return angles_list

def register_images (target_arr, target_affine,
                     moving_arr, moving_affine,
                     phase,
                     lim_arr=None,
                     registration_type='quadratic',
                     initialize=False,
                     optimizer_setting=False):
    fixed_image, moving_image = sub_processes.set_images_in_scale(lim_arr,
                                                                  target_arr,
                                                                  moving_arr)

    sz = np.array(fixed_image.shape)
    moving_sz = np.array(moving_image.shape)

    dim = len(fixed_image.shape)

    orig_fixed_grid2world = target_affine
    orig_moving_grid2world = moving_affine

    mid_index = (sz - 1) / 2.

    # Physical Coordinate of the center index = 0,0,0
    new_orig_temp = - orig_fixed_grid2world[0:3, 0:3].dot(mid_index)
    fixed_grid2world = orig_fixed_grid2world.copy()
    fixed_grid2world[0:3, 3] = new_orig_temp

    new_orig_temp = - orig_moving_grid2world[0:3, 0:3].dot(mid_index)
    moving_grid2world = orig_moving_grid2world.copy()
    moving_grid2world[0:3, 3] = new_orig_temp

    _, resolution = get_direction_and_spacings(fixed_grid2world, dim)

    transformRegister = QuadraticRegistration(phase, registration_type=registration_type)
    OptimizerFlags = transformRegister.optimization_flags
    initializeTransform = QuadraticMap(phase)

    if initialize and (np.sum(OptimizerFlags[0:3]) != 0):
        initializeTransform = transform_centers_of_mass(fixed_image, moving_image, phase,
                                                        static_grid2world=fixed_grid2world,
                                                        moving_grid2world=moving_grid2world)

    grad_scale = sub_processes.get_gradients_params(resolution, sz)

    if initialize and ~optimizer_setting:
        print("initializing")
        grad_scale2 = grad_scale.copy()
        grad_scale2[3:6] = grad_scale2[3:6] * 2

        transformRegister.factors = [3]
        transformRegister.sigmas = [0]
        transformRegister.levels = 1

        flag2 = np.zeros(21)
        flag2[0:6] = transformRegister.optimization_flags[:6]
        transformRegister.set_optimizationflags(flag2)

        try:
            initializeTransform = transformRegister.optimize(fixed_image, moving_image, phase=phase,
                                                             static_grid2world=fixed_grid2world,
                                                             moving_grid2world=moving_grid2world,
                                                             grad_params=grad_scale2)
        except:
            ValueError("ERROR")

    if optimizer_setting:
        print("start optimizer")
        angles_list = get_angles_list()

        flags2 = np.zeros(21)
        flags2[:6] = 1
        best_reg_val = 1.
        best_params = np.zeros(21)
        transformRegister = QuadraticRegistration(phase)
        transformRegister.set_optimizationflags(flags2)
        transformRegister.grad_params = grad_scale
        transformRegister.factors = [3]
        transformRegister.sigmas = [0]
        transformRegister.levels = 1

        for i in range(len(angles_list)):
            dummy_transformMap = QuadraticMap(phase)
            dummy_transformMap.QuadraticParams[3:6] = angles_list[i]

            transformRegister.initial_QuadraticParams = dummy_transformMap.QuadraticParams
            dummy_transformMap = transformRegister.optimize(fixed_image, moving_image, phase,
                                                            static_grid2world=fixed_grid2world,
                                                            moving_grid2world=moving_grid2world)
            reg_val = transformRegister.current_level_cost

            if reg_val < best_reg_val:
                print(reg_val, best_reg_val)
                best_reg_val = reg_val
                best_params = dummy_transformMap.QuadraticParams

        finalparams = best_params

    else:
        print("not using optimizer")
        initializeTransform.get_QuadraticParams()
        transformRegister = QuadraticRegistration(phase)

        transformRegister.initial_QuadraticParams = initializeTransform.get_QuadraticParams()
        finalTransform = transformRegister.optimize(fixed_image, moving_image,
                                                    phase=phase,
                                                    static_grid2world=fixed_grid2world,
                                                    moving_grid2world=moving_grid2world,
                                                    grad_params=grad_scale)
        finalparams = finalTransform.get_QuadraticParams()

    finalTransform = QuadraticMap(phase, finalparams, fixed_image.shape, fixed_grid2world,
                                  moving_image.shape, moving_grid2world)

    return finalTransform

def wrapper(vol):
    inp = np.frombuffer(shared_input)
    sh_input = inp.reshape(arr_shape)

    out = np.frombuffer(shared_output)
    sh_out = out.reshape(arr_shape)


    moving_arr = sh_input[:, :,:, vol]
    result_slice = register_images(sh_out,
                                   target_affine,
                                   moving_arr,
                                   moving_affine,
                                   phase)
    sh_input[:, :, :, vol] = result_slice


def init(shared_input_,
         shared_output_,
         arr_shape_,
         params_):
    # initialization of the global shared arrays
    global shared_input
    global shared_output
    global arr_shape
    global target_affine
    global moving_affine
    global phase
    shared_input = shared_input_
    shared_output = shared_output_
    arr_shape = arr_shape_
    target_affine = params_[0]
    moving_affine = params_[1]
    phase = params_[2]

def run_pararllel(image_name,
                  b0_target_name,
                  b0_id = 0,
                  phase = 'vertical',
                  num_threads = None):
    start_time = time.time()

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = multiprocessing.cpu_count()

    image = nib.load(image_name)
    image_size = 1
    for i in image.shape:
        image_size *=i

    target_image = nib.load(b0_target_name)
    target_arr = target_image.get_data()

    if len(image.shape) > 3:
        if image.shape[:-1] != target_arr.shape:
            raise IOError ("Two images are not in the same dimensions")
    else:
        if image.shape != target_arr.shape:
            raise IOError ("Two images are not in the same dimensions")

    mp_arr = multiprocessing.RawArray(ctypes.c_double, image_size)
    shared_arr = np.frombuffer(mp_arr)
    shared_input = shared_arr.reshape(image.shape)
    shared_input[:] = image.get_data()[:]
    # output array
    mp_arr2 = multiprocessing.RawArray(ctypes.c_double, target_arr.size)
    shared_arr2 = np.frombuffer(mp_arr2)
    shared_output = shared_arr2.reshape(target_arr.shape)
    shared_output[:] = target_arr[:]
    # parameters
    params = [image.affine, target_image.affine, phase]

    # multi-processing
    with closing(multiprocessing.Pool(threads_to_use, initializer=init,
                                      initargs=(shared_arr, shared_arr2, params))) as p:
        p.map_async(wrapper, [vol for vol in range(0, image.shape[-1]) if vol != b0_id])
    p.join()

    print("Gibbs multi correction took --- %s seconds ---" % (time.time() - start_time))

    return shared_input, image.affine



def test ():
    b0_target_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0.nii"
    moving_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc.nii"
    output, affine = run_pararllel(moving_image, b0_target_image)
    out_test = nib.Nifti1Image(output, affine )
    nib.save(out_test, moving_image.split('.nii') + '_out_Test.nii')


test()