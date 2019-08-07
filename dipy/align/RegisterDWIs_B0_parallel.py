from dipy.align.imwarp import get_direction_and_spacings
from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration, transform_centers_of_mass
import time
import numpy as np
import pymp
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
    image_transform = finalTransform.transform(image = moving_arr,QuadraticParams=finalparams)
    return finalparams,image_transform




def test ():
    b0_target_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0.nii"
    moving_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc.nii"
    mask_target_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0_mask_mask.nii"

    b0_target = nib.load(b0_target_image)
    moving_image = nib.load(moving_image)
    mask_target = nib.load(mask_target_image)
    phase = 'vertical'
    moving_image_shr = pymp.shared.array((moving_image.shape), dtype = np.float32)
    moving_image_shr[:] = moving_image.get_data()

    b0_arr = b0_target.get_data()
    mask_arr = mask_target.get_data()
    lim_arr = pymp.shared.array((4, moving_image.shape[-1]), dtype=np.float32)

    transformation = pymp.shared.array((moving_image.shape[-1],21), dtype=np.float64)
    start_time = time.time()

    with pymp.Parallel() as p:
        for index in p.range(1, moving_image.shape[-1]):
            curr_vol = moving_image_shr[:, :, :, index]
            lim_arr[:, index] = sub_processes.choose_range(b0_arr,
                                                           curr_vol,
                                                           mask_arr)

    with pymp.Parallel() as p:
        for index in p.range(1, moving_image.shape[-1]):
            curr_vol = moving_image_shr[:, :, :, index]

            transformation[index,:],moving_image_shr[:,:,:,index] =  register_images(b0_arr, b0_target.affine,
                                 curr_vol, moving_image.affine,
                                 phase,
                                lim_arr=lim_arr[:,index],
                                registration_type='quadratic',
                                initialize=False,
                                optimizer_setting=False)

    print("Time cost {}", time.time() - start_time)

    np.savetxt('transformations_test.txt', transformation)
    image_out = nib.Nifti1Image(moving_image_shr, moving_image.affine)
    nib.save(image_out, 'Image_test.nii')

test()