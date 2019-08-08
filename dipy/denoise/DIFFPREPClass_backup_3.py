import numpy as np
import nibabel as nib
from dipy.denoise.randomlpca_denoise import randomlpca_denoise
from dipy.denoise.GibbRemoval import gibbremoval
from dipy.denoise.ChangeFOV import changeFoV
from dipy.align.reslice import reslice
from dipy.denoise.fsl_bet import fsl_bet_mask
from dipy.align.Register_DWIs import set_image_in_scale, get_gradients_params, get_direction_and_spacings

from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration, transform_centers_of_mass


# DIFFPREP -i path_to_list_input_file --do_QC 0 -e off --res 1 1 1 -s acpc_align.nii


class diffprep(object):
    def __init__(self, input_image_path,
                 encoding_phase,
                 output_image_path = None,
                 mask_image_path = None,
                 bvals = None,
                 bvecs = None,
                 Eddy = False,
                 fov = None,
                 activeGibbRemoving = False,
                 activeDenoising = False,
                 center_of_mass = False,
                 new_resolution = None, interp = 1):

        # Inter from 0-5, "Nearest", "Lanczos", "Bilinear", "Bicubic", "Cubic"
        self.input_image_fn = input_image_path
        self.encoding_phase = encoding_phase
        if output_image_path is None:
            output_image_path = input_image_path.split('.')[0] + "_DIFFPREP_proc.nii"

        self.output_image_fn = output_image_path
        self.bvals = bvals          # Temporary - not Used
        self.bvecs = bvecs          # Temporary - not Used


        if fov:
            if len(fov) >3 :
                print("ERROR, too many size")
            fov = np.array(fov)
            self.fov = fov

        self.input_image = nib.load(input_image_path)
        self.input_data = self.input_image.get_data()
        self.input_affine = self.input_image.affine
        self.input_resolution = np.array(self.input_image.header.get_zooms()[:3])
        self.nVols = self.input_data.shape[-1]
        self.input_size = self.input_data.size
        self.input_shape = self.input_data.shape

        # Activate Set-up
        self.activeGibbRemoving = activeGibbRemoving
        self.activeDenoising = activeDenoising
        self.center_of_mass = center_of_mass
        self.Eddy = Eddy

        if self.input_data.ndim  <4:
            print("Data is in 3D, not performing DWI-Denoising")
            self.activeDenoising = False

        self.output_data = self.input_data
        self.interp = interp
        self.b0_id = 0
        self.b0 = self.input_data[:,:,:,self.b0_id]

        if new_resolution is not None:
            try:
                self.new_resolution = np.array(new_resolution)

            except:
                print("Can't Convert to Numpy Array for resolution")
        if mask_image_path is not None:
            self.mask_image_path = mask_image_path
            self.mask_image = nib.load(mask_image_path)
            self.mask_data = self.mask_image.get_data()


#------------DIFFPREP - run all----------------------------#
    def execute(self):
        # Do QC
        self.output_affine = self.input_affine.copy()
        # Change FOV
        if self.fov is not None:
            print("----------Changing field of view ----------------")
            output_arr, output_affine = self.change_field_of_view(self.input_image, self.fov, self.center_of_mass)
            self.input_data = output_arr
            self.output_affine = output_affine

        # Denoising
        if self.activeDenoising:
            print("----------Denoising the DWIs ----------------")
            output_arr, output_noise_arr, sigma_arr = self.get_Denoising_data(self.input_data)
            self.input_data = output_arr

        # Gibb Removing
        if self.activeGibbRemoving:
            print("----------Gibb Removal ----------------------")
            output_arr = self.get_GibbRemoved_data(self.input_data)
            self.input_data = output_arr

     #   # Upsampling
     #   if self.new_resolution is not None:
     #       print("----------Upsampling data-------------------")

            #if self.new_resolution != self.input_resolution:
           #     output_arr = self.upsampling_data(self.input_data, self.input_affine, self.input_resolution, self.new_resolution, self.interp)
          #      self.input_data = output_arr

        # Creating Mask
        if self.mask_image_path is None:
            self.create_mask()
            self.mask_affine, self.mask_data = self.get_affine_image_data(self.mask_image_path)

        if self.Eddy:
            print("Perform Registration......")

            b0_img_target= self.dmc_make_target(self.b0,self.mask_data)

        self.output_data = self.input_data
        self.save_output_tofile(self.output_data,self.output_affine,self.input_image_fn.split('.nii')[0] + "_DIFFPREP_proc.nii"  )
        return self.output_data



########### GETTING SUB_FUNCTIONS ####################
    def get_affine_image_data(self, image_fn):
        image = nib.load(image_fn)
        return image.affine, image.get_data()

    def get_field_of_view (self):
        return self.fov

    def get_output_data(self):
        return self.output_data

    def get_GibbRemoved_data(self, input_arr):
        output = gibbremoval(input_arr)
        return output

    def get_b0_image(self, b0_image_fn = None):
        if b0_image_fn is None:
            b0_data = self.b0
            b0_image = nib.Nifti1Image(b0_data, self.input_affine)
            self.b0_image_fn = self.input_image_fn.split('.')[0] + '_b0.nii'
            nib.save(b0_image, self.b0_image_fn)
        else:
            b0_image = nib.load(b0_image_fn)
        return b0_image

    def get_Denoising_data(self, input_arr, save_to_file = False):
        output, output_noise, sigma = randomlpca_denoise(input_arr)
        self.output_sigma_noise = sigma
        self.save_output_tofile(output_noise, self.output_image_fn.split('.nii')[0] + "_noise.nii")
        return output, output_noise, sigma


    def get_sigma_noise(self):
        if not self.activeDenoising:
            print("None data exists")
            return 0
        return self.output_sigma_noise


    def change_field_of_view(self, input_image_nib, fov, com = False):
        output, output_affine = changeFoV(input_image_nib, fov, com)
        return output, output_affine

    def upsampling_data(self, input_image_arr,
                        input_affine,
                        input_resolution,
                        output_resolution,
                        order = 1 ):
        new_image, new_affine = reslice(input_image_arr,
                            input_affine,
                            input_resolution,
                            output_resolution,
                            order= order)
        return new_image, new_affine



    def create_mask(self, b0_image_fn= None, mask_fn = None, mask = True):
        if b0_image_fn:
            self.b0_image_fn = b0_image_fn

        if mask_fn is None:
            mask_fn = self.input_image_fn.split('.')[0] + '_mask.nii'
        fsl_bet_mask(self.b0_image_fn, mask_fn, mask)
        self.mask_image_fn = mask_fn
        self.mask_mask_fn = mask_fn.split('.nii')[0] + '_mask.nii'
        # get_mask_info
        mask_image = nib.load(mask_fn)
        mask_data = mask_image.get_data()

        mask_mask_image = nib.load(self.mask_mask_fn)
        mask_mask_data = mask_mask_image.get_data()
        mask_affine = mask_image.affine

        return mask_data, mask_mask_data,  mask_affine


    def dmc_make_target(self, b0_image, b0_image_mask):
        # Import b0_image_data, and b0_image_mask produced by fsl_bet
        fct = 0.9
        new_resolution = self.input_resolution / fct
        transformed_b0, transformed_affine_b0 = self.upsampling_data(b0_image, self.input_affine, self.input_resolution, new_resolution)
        transformed_mask, transformed_affine_mask = self.upsampling_data(b0_image_mask,self.input_affine, self.input_resolution, self.new_resolution)

        # minx = transformed_mask.shape[0] + 5
        # maxx = -1
        # miny = transformed_mask.shape[1] + 5
        # maxy = -1
        # minz = transformed_mask.shape[2] + 5
        # maxz = -1

        transformed_mask[(transformed_mask<0.5)]=0
        transformed_mask[(transformed_mask >= 0.5)] = 1
        x,y,z = np.where(transformed_mask==1)
        [minx, miny, minz] = min(x,y,z)
        [maxx, maxy, maxz] = max(x,y,z)

        print( [minx, miny, minz])
        print([maxx, maxy, maxz] )
        if minx == transformed_mask.shape[0] + 5:
            minx = 0
        if miny == transformed_mask.shape[1] + 5:
            miny = 0
        if minz == transformed_mask.shape[2] + 5:
            minz = 0

        minx = max(minx-2, 0)
        maxx = min(maxx + 2, transformed_mask.shape[0] - 1)
        miny = max(miny-2, 0)
        maxy = min(maxy + 2, transformed_mask.shape[1] - 1)
        minz = max(minz-2, 0)
        maxz = min(maxz + 2, transformed_mask.shape[2] - 1)

        if (maxx-minx+1) +(maxy-miny+1) + (maxz-minz+1) < 0.3*np.sum(transformed_mask.shape):
            minx=0
            maxx=transformed_mask.shape[0]-1
            miny = 0
            maxy = transformed_mask.shape[1] - 1
            minz = 0
            maxz = transformed_mask.shape[2] - 1

        new_transformed_mask = transformed_mask[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        final_image =transformed_b0[minx:maxx + 1, miny:maxy + 1, minz:maxz + 1]

        msk_cnt = len(b0_image_mask[b0_image_mask != 0])
        npixel = b0_image_mask.size

        # Set the equal to 1 - Keep the orginal image
        if msk_cnt < 0.02 * npixel:
            b0_image_mask[:] = 1
            final_image = b0_image_mask
        return final_image


    def choose_range(self, b0_image, curr_vol, b0_mask_image):
        fixed_signal = np.sort(b0_image[b0_mask_image != 0])
        moving_signal = np.sort(curr_vol[b0_mask_image != 0])
        koeff = 0.005
        nb = fixed_signal.shape[0]
        ind = (nb-1) - koeff * nb

        lim_arr = np.zeros(4)
        lim_arr[0] = 0.1
        lim_arr[1] = fixed_signal[ind]
        lim_arr[2] = 0.1
        lim_arr[3] = moving_signal[ind]
        return lim_arr


    def save_output_tofile(self,
                           output_data,
                           output_affine= None,
                           output_name = None):
        if output_name is None:
            output_name = self.output_image_fn
        if output_affine is None:
            output_affine = self.input_affine

        self.output_image = nib.Nifti1Image(output_data, output_affine)
        nib.save(self.output_image, output_name)


    # This is for motion & eddy distortion
    """step 1"""
    def multi_sub_(self, curr_vol, b0_target, b0_target_bin, correction_mode = False):
        signal_ranges = self.choose_range(b0_target, curr_vol, b0_target_bin)
        signal_ranges[2:3] = 0
        if not correction_mode:
            id_trans = QuadraticMap(self.encoding_phase)
            # id_trans.Setting_Identity()
            curr_trans = id_trans.get_QuadraticParams()

        else:
            curr_trans = self.RegisterDWI_to_B0(b0_target, curr_vol,self.encoding_phase, signal_ranges, optimizer_setting = True)
        return curr_trans


    def EDDY_Current_Distortion(self, b0_target, b0_target_bin, num_threads = None):
        import multiprocessing, ctypes, time
        from functools import partial

        if num_threads is not None:
            threads_to_use = num_threads
        else:
            threads_to_use = multiprocessing.cpu_count()
        parse_function = partial(self.multi_sub_,
                                 target_vol = b0_target,
                                 b0_target_bin = b0_target_bin,
                                 correction_mode = False)

        with multiprocessing.Pool(threads_to_use) as p:
            transform_vols = p.map(parse_function, [self.input_data[...,i] for i in range(self.nVols)])
        return transform_vols


    def RegisterDWI_to_B0(self, target_vol, curr_vol,
                           phase_encoding,
                           signal_range = None,
                           register_type = 'linear',
                           optimizer_setting = False,
                           optimization_flags =  None,
                           initialize = True,
                           ):
        fixed_image, moving_image = set_image_in_scale(signal_range, target_vol, curr_vol)

        sz = np.array(fixed_image.shape)
        moving_sz = np.array(moving_image.shape)

        dim = len(fixed_image.shape)

        orig_fixed_grid2world = self.input_affine
        orig_moving_grid2world = self.input_affine

        mid_index = (sz - 1) / 2.

        # Physical Coordinate of the center index = 0,0,0
        new_orig_temp = - orig_fixed_grid2world[0:3, 0:3].dot(mid_index)
        fixed_grid2world = orig_fixed_grid2world.copy()
        fixed_grid2world[0:3, 3] = new_orig_temp

        new_orig_temp = - orig_moving_grid2world[0:3, 0:3].dot(mid_index)
        moving_grid2world = orig_moving_grid2world.copy()
        moving_grid2world[0:3, 3] = new_orig_temp

        _, resolution = get_direction_and_spacings(fixed_grid2world, dim)
        phase = phase_encoding

        transformRegister = QuadraticRegistration(phase_encoding, registration_type= register_type)
        OptimizerFlags = transformRegister.optimization_flags
        initializeTransform = QuadraticMap(phase)

        if initialize and (np.sum(OptimizerFlags[0:3]) != 0):
            initializeTransform = transform_centers_of_mass(fixed_image, moving_image, phase,
                                                            static_grid2world=fixed_grid2world,
                                                            moving_grid2world=moving_grid2world)

        grad_scale = get_gradients_params(resolution, sz)

        if initialize and ~optimizer_setting:
            print("initializing and not optimizing")
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
            print("optimizing")

            angles_list = []
            for x in range(-180, 180, 45):
                for y in range(-180, 180, 45):
                    for z in range(-180, 180, 45):
                        mx = x / 180 * np.pi
                        my = y / 180 * np.pi
                        mz = z / 180 * np.pi
                        angles_list.append([mx, my, mz])

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
                    best_map = dummy_transformMap
                    finalTransform = best_map


        else:
            print("not optimizing part")
            initializeTransform.get_QuadraticParams()
            # transformRegister = QuadraticRegistration(phase,initial_QuadraticParams= initializeTransform.get_QuadraticParams(),
            #                                            gradients_params = grad_scale )
            transformRegister.initial_QuadraticParams = initializeTransform.get_QuadraticParams()
            finalTransform = transformRegister.optimize(fixed_image, moving_image,
                                                        phase=phase,
                                                        static_grid2world=fixed_grid2world,
                                                        moving_grid2world=moving_grid2world, grad_params=grad_scale)
            finalTransform = QuadraticMap(finalTransform.get_Parameters(), "horizontal", fixed_image.shape,
                                          fixed_grid2world,
                                          moving_image.shape, moving_grid2world)

        final_image = finalTransform.transform(moving_image)

        # return finalTransform, final_image
        return finalTransform.get_QuadraticParams()



