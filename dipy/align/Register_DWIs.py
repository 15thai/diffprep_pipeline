# Authors: Okan, Anh Thai, Britney
import numpy as np
from dipy.align.imwarp import get_direction_and_spacings
import nibabel as nib
from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration, transform_centers_of_mass

# To Register DWIs images into B0 image, from Anh_RegisterDWItoB0_2.py in the lab
def set_image_in_scale(lim_arr, fixed_image, moving_image):
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

def register_dwi_to_b0 (fixed_im_nib, moving_im_nib, phase,
                        lim_arr = None, registration_type = 'quadratic',
                        initialize = False, optimizer_setting = False):
    fixed_image = fixed_im_nib.get_data().copy()
    moving_image = moving_im_nib.get_data().copy()

    fixed_image, moving_image = set_image_in_scale(lim_arr, fixed_image, moving_image)

    sz = np.array(fixed_image.shape)
    moving_sz = np.array(moving_image.shape)

    dim = len(fixed_image.shape)

    orig_fixed_grid2world = fixed_im_nib.affine
    orig_moving_grid2world = moving_im_nib.affine

    mid_index = (sz -1) /2.

    # Physical Coordinate of the center index = 0,0,0
    new_orig_temp = - orig_fixed_grid2world[0:3,0:3].dot(mid_index)
    fixed_grid2world = orig_fixed_grid2world.copy()
    fixed_grid2world[0:3,3] = new_orig_temp

    new_orig_temp = - orig_moving_grid2world[0:3,0:3].dot(mid_index)
    moving_grid2world = orig_moving_grid2world.copy()
    moving_grid2world[0:3,3] = new_orig_temp

    _, resolution = get_direction_and_spacings(fixed_grid2world, dim)

    transformRegister = QuadraticRegistration(phase, registration_type = registration_type)
    OptimizerFlags = transformRegister.optimization_flags
    initializeTransform = QuadraticMap(phase)

    if initialize and (np.sum(OptimizerFlags[0:3]) != 0 ):
        initializeTransform = transform_centers_of_mass(fixed_image, moving_image, phase,
                                                          static_grid2world= fixed_grid2world,
                                                          moving_grid2world = moving_grid2world)


    grad_scale = get_gradients_params(resolution, sz)

    if initialize and ~optimizer_setting:
        print("initializing")
        grad_scale2 = grad_scale.copy()
        grad_scale2[3:6] = grad_scale2[3:6]*2

        transformRegister.factors = [3]
        transformRegister.sigmas  = [0]
        transformRegister.levels  = 1

        flag2 = np.zeros(21)
        flag2[0:6] = transformRegister.optimization_flags[:6]
        transformRegister.set_optimizationflags(flag2)

        try:
            initializeTransform = transformRegister.optimize(fixed_image, moving_image, phase = phase,
                                                             static_grid2world=fixed_grid2world,
                                                             moving_grid2world = moving_grid2world, grad_params= grad_scale2)
        except:
            ValueError("ERROR")


    if optimizer_setting:
        print("optimizing")

        angles_list =[]
        for x in range(-180, 180, 45):
            for y in range(-180, 180, 45):
                for z in range(-180, 180, 45):
                    mx = x/180 *np.pi
                    my = y/180 *np.pi
                    mz = z/180 *np.pi
                    angles_list.append([mx,my,mz])

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
            dummy_transformMap = transformRegister.optimize(fixed_image, moving_image, phase, static_grid2world= fixed_grid2world,
                                                            moving_grid2world = moving_grid2world)
            reg_val = transformRegister.current_level_cost

            if reg_val < best_reg_val:
                print(reg_val, best_reg_val)
                best_reg_val = reg_val
                best_params = dummy_transformMap.QuadraticParams
                best_map = dummy_transformMap

        # finalTransform = best_map(phase, QuadraticParams= best_params)
        finalTransform = best_map.copy()

    else:
        print("not optimizing")
        initializeTransform.get_QuadraticParams()
        transformRegister = QuadraticRegistration(phase)

        # transformRegister = QuadraticRegistration(phase,initial_QuadraticParams= initializeTransform.get_QuadraticParams(),
        #                                            gradients_params = grad_scale )
        transformRegister.initial_QuadraticParams = initializeTransform.get_QuadraticParams()
        finalTransform = transformRegister.optimize(fixed_image, moving_image,
                                            phase=phase,
                                            static_grid2world=fixed_grid2world,
                                            moving_grid2world=moving_grid2world,
                                            grad_params=grad_scale)
        finalTransform = QuadraticMap(phase, finalTransform.get_QuadraticParams(), fixed_image.shape, fixed_grid2world,
                                        moving_image.shape, moving_grid2world)
    final_image = finalTransform.transform(moving_image)

    return finalTransform, final_image


def register_to_b0 (fixed_im_nib,
                    moving_data,
                    moving_affine,
                    phase,
                    lim_arr = None,
                    registration_type = 'quadratic',
                    initialize = False,
                    optimizer_setting = False):
    fixed_image = fixed_im_nib.get_data().copy()
    moving_image = moving_data

    fixed_image, moving_image = set_image_in_scale(lim_arr, fixed_image, moving_image)

    sz = np.array(fixed_image.shape)
    moving_sz = np.array(moving_image.shape)

    dim = len(fixed_image.shape)

    orig_fixed_grid2world = fixed_im_nib.affine
    orig_moving_grid2world = moving_affine

    mid_index = (sz -1) /2.

    # Physical Coordinate of the center index = 0,0,0
    new_orig_temp = - orig_fixed_grid2world[0:3,0:3].dot(mid_index)
    fixed_grid2world = orig_fixed_grid2world.copy()
    fixed_grid2world[0:3,3] = new_orig_temp

    new_orig_temp = - orig_moving_grid2world[0:3,0:3].dot(mid_index)
    moving_grid2world = orig_moving_grid2world.copy()
    moving_grid2world[0:3,3] = new_orig_temp

    _, resolution = get_direction_and_spacings(fixed_grid2world, dim)

    transformRegister = QuadraticRegistration(phase, registration_type = registration_type)
    OptimizerFlags = transformRegister.optimization_flags
    initializeTransform = QuadraticMap(phase)

    if initialize and (np.sum(OptimizerFlags[0:3]) != 0 ):
        initializeTransform = transform_centers_of_mass(fixed_image, moving_image, phase,
                                                          static_grid2world= fixed_grid2world,
                                                          moving_grid2world = moving_grid2world)


    grad_scale = get_gradients_params(resolution, sz)

    if initialize and ~optimizer_setting:
        print("initializing")
        grad_scale2 = grad_scale.copy()
        grad_scale2[3:6] = grad_scale2[3:6]*2

        transformRegister.factors = [3]
        transformRegister.sigmas  = [0]
        transformRegister.levels  = 1

        flag2 = np.zeros(21)
        flag2[0:6] = transformRegister.optimization_flags[:6]
        transformRegister.set_optimizationflags(flag2)

        try:
            initializeTransform = transformRegister.optimize(fixed_image, moving_image, phase = phase,
                                                             static_grid2world=fixed_grid2world,
                                                             moving_grid2world = moving_grid2world, grad_params= grad_scale2)
        except:
            ValueError("ERROR")


    if optimizer_setting:
        print("optimizing")
        angles_list =[]
        for x in range(-180, 180, 45):
            for y in range(-180, 180, 45):
                for z in range(-180, 180, 45):
                    mx = x/180 *np.pi
                    my = y/180 *np.pi
                    mz = z/180 *np.pi
                    angles_list.append([mx,my,mz])

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
            dummy_transformMap = transformRegister.optimize(fixed_image, moving_image, phase, static_grid2world= fixed_grid2world,
                                                            moving_grid2world = moving_grid2world)
            reg_val = transformRegister.current_level_cost

            if reg_val < best_reg_val:
                print(reg_val, best_reg_val)
                best_reg_val = reg_val
                best_params = dummy_transformMap.QuadraticParams

        # finalTransform = best_map(phase, QuadraticParams= best_params)
        finalparams = best_params

    else:
        print("not optimizing")
        initializeTransform.get_QuadraticParams()
        transformRegister = QuadraticRegistration(phase)

        # transformRegister = QuadraticRegistration(phase,initial_QuadraticParams= initializeTransform.get_QuadraticParams(),
        #                                            gradients_params = grad_scale )
        transformRegister.initial_QuadraticParams = initializeTransform.get_QuadraticParams()
        finalTransform = transformRegister.optimize(fixed_image, moving_image,
                                            phase=phase,
                                            static_grid2world=fixed_grid2world,
                                            moving_grid2world=moving_grid2world,
                                            grad_params=grad_scale)
        finalparams = finalTransform.get_QuadraticParams()

    finalTransform = QuadraticMap(phase, finalparams, fixed_image.shape, fixed_grid2world,
                                        moving_image.shape, moving_grid2world)
    final_image = finalTransform.transform(moving_image)

    return finalTransform, final_image


