import numpy as np
import nibabel as nib
import os
from dipy.align import sub_processes
from dipy.denoise.ChangeFOV import changeFoV
from dipy.align.reslice import reslice
from dipy.align.Register_DWIs import register_to_b0
class diffprep(object):
    def __init__(self,
                 input_image_fn,
                 phase_encoding='vertical',
                 b0_id=0,
                 output_image_path=None,
                 b0_mask_fn=None,
                 b0_image_fn = None,
                 fov=None,
                 activeChangeFOV = False,
                 activeGibbRemoving = False,
                 activeDenoising = False,
                 activeUpsample = False,
                 activeEddy=False,
                 center_of_mass = False,
                 new_resolution = None, interp = 3):
        """Constructor"""
        # Inter from 0-5, "Nearest", "Lanczos", "Bilinear", "Bicubic", "Cubic"
        self.input_image_fn = input_image_fn
        self.input_image = nib.load(input_image_fn)
        self.input_affine = self.input_image.affine

        input_resolution = self.input_image.header.get_zooms()[:3]
        self.input_resolution = np.array(input_resolution)
        self.phase_encoding = phase_encoding
        self.CreateMask = True
        if new_resolution:
            self.new_resolution = new_resolution
        # ------------------------------------------------------------------
        self.b0_mask_fn = b0_mask_fn
        if self.b0_mask_fn:
            self.CreateMask = False

        self.b0_image_fn = b0_image_fn

        if output_image_path is None:
            output_image_path = input_image_fn.split('.')[0] + \
                                "_DIFFPREP_PROC.nii"
        self.output_image_fn = output_image_path
        self.b0_id = b0_id

        if fov:
            if len(fov) >3 :
                print("ERROR, too many size")
            fov = np.array(fov)
            self.fov = fov
        else:
            self.fov = self.input_resolution*self.input_image.shape[:3]

        # Activate Set-up
        self.activeChangeFOV = activeChangeFOV
        self.activeGibbRemoving = activeGibbRemoving
        self.activeDenoising = activeDenoising
        if activeUpsample and (len(new_resolution) != 0):
            self.activeUpsampling = activeUpsample
        else:
            self.activeUpsampling = False

        self.activeUpsampling = activeUpsample

        self.center_of_mass = center_of_mass
        self.activeEddy = activeEddy

        folder = os.path.dirname(self.input_image_fn)
        self.temp_wd = os.path.join(folder, 'process')
        if not os.path.exists(self.temp_wd):
            os.makedirs(self.temp_wd)


########### GETTING SUB_FUNCTIONS ####################
    def execute(self,activeRegister = False):
        self.current_data = self.input_image.get_data()
        self.current_affine = self.input_image.affine
        if self.activeChangeFOV and (len(self.fov) != 0):
            print("----> Changing Field Of View")
            self.current_data, self.current_affine = changeFoV(self.input_image,
                                                               self.fov,
                                                               self.center_of_mass)
        if self.activeDenoising:
            print("----> Random Matrix Noise Reduction")
            self.current_data, output_noise, sigma_noise = sub_processes.denoising_image(self.current_data)

        if self.activeGibbRemoving:
            print("----> Gibb Removing")
            self.current_data = sub_processes.gibbremoval(self.current_data)

        if self.activeUpsampling and self.new_resolution:
            print("----> Upsampling")
            self.current_data, self.current_affine = reslice(self.current_data,
                                                             self.current_affine,
                                                             self.input_resolution,
                                                             self.new_resolution)

        if (self.activeUpsampling or
            self.activeGibbRemoving or
            self.activeDenoising or
            self.activeChangeFOV):

            # save to temporary_folder before registration
            temp_file = os.path.join(self.temp_wd, 'temp.nii')
            self.save_output_tofile(self.current_data,
                                    self.current_affine,temp_file)
            print("----> Saving to temporary corrected file for registration")
            print("----> saving ", temp_file)


        # save b0_temp after correction
        temp_file = os.path.join(self.temp_wd, 'temp_b0.nii')
        self.save_output_tofile(self.current_data[...,self.b0_id],
                                self.current_affine, temp_file)
        print("----> Saving to temporary file for registration")
        print("----> saving ", temp_file)


        # creating mask base on b0_temp
        temp_mask = temp_file.split('.nii')[0] + "_mask.nii"

        if np.any(self.current_data[..., self.b0_id]) and (self.CreateMask):
            sub_processes.creating_mask(temp_file, temp_mask)
        else:
            if self.b0_mask_fn:
                temp_mask = self.b0_mask_fn.split('.nii')
            else:
                print("Invalid Image")

        if self.activeEddy:
            temp_mask_mask = temp_mask.split('.nii')[0] + "_mask.nii"
            temp_mask_mask_image = nib.load(temp_mask_mask)
            dmc_image, _ = sub_processes.dmc_make_target(temp_file,temp_mask_mask_image.get_data())

            temp_file = os.path.join(self.temp_wd, 'b0_DMC_target.nii')
            self.save_output_tofile(dmc_image,
                                    self.current_affine, temp_file)
            print("----> Saving to temporary file for registration")
            print("----> saving ", temp_file)

            b0_target = nib.load(temp_file)
            if activeRegister:
                # import multiprocessing
                # from functools import partial
                # threads_to_use = multiprocessing.cpu_count()
                #
                # parse_function = partial(self.multi_sub_,
                #                          b0_target=b0_target,
                #                          b0_target_bin=temp_mask_mask_image.get_data(),
                #                          correction_mode=False)
                #
                # with multiprocessing.Pool(threads_to_use) as p:
                #     transform_vols = p.map(parse_function, [self.input_image.dataobj[..., i]
                #                                             for i in range(self.input_image.shape[-1])])
                # np.savetxt(os.path.join(self.temp_wd,'log_1.txt'), transform_vols)
                # return transform_vols
                print("Nah Y_Y")

    def multi_sub_(self, curr_vol, b0_target, b0_target_bin, correction_mode = False):
        signal_ranges = sub_processes.choose_range(b0_target, curr_vol, b0_target_bin)
        signal_ranges[2:3] = 0
        curr_trans = register_to_b0(b0_target,
                                    curr_vol,
                                    self.phase_encoding,
                                    signal_ranges,
                                    optimizer_setting = True)
        return curr_trans

    def save_output_tofile(self,
                           output_data,
                           output_affine,
                           output_name):
        output_image = nib.Nifti1Image(output_data, output_affine)
        nib.save(output_image, output_name)
        return output_name







