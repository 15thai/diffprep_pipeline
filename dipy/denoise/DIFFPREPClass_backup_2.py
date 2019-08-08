import numpy as np
import nibabel as nib
from dipy.denoise.randomlpca_denoise import randomlpca_denoise
from dipy.denoise.GibbRemoval import gibbremoval
from dipy.denoise.ChangeFOV import changeFoV
from dipy.align.reslice import reslice
from dipy.denoise.fsl_bet import fsl_bet_mask
from dipy.align.Register_DWIs import register_dwi_to_b0


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
        self.b0 = self.input_data[:,:,:,0]


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

        # Upsampling
        if self.new_resolution is not None:
            print("---------- Upsampling data-------------------")

            if self.new_resolution != self.input_resolution:
                output_arr = self.upsampling_data(self.input_data, self.input_affine, self.input_resolution, self.new_resolution, self.interp)
                self.input_data = output_arr

        # Creating Mask
        if self.mask_image_path is None:
            self.create_mask()
            self.mask_affine, self.mask_data = self.get_image_data(self.mask_image_path)

        if self.Eddy:
            print("Perform Registration......")

        self.output_data = self.input_data
        self.save_output_tofile(self.output_data,self.output_affine,self.input_image_fn.split('.nii')[0] + "_DIFFPREP_proc.nii"  )
        return self.output_data



########### GETTING SUB_FUNCTIONS ####################
    def get_image_data(self, image_fn):
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

    # TODO: EDDY CURRENT DISTORTION
    def EDDY_Current_Distortion(self):
        register_dwi_to_b0()
        pass


    def create_mask(self, b0_image_fn= None):
        if b0_image_fn is None:
            b0_image_data = self.b0
            b0_image = nib.Nifti1Image(b0_image_data, self.input_affine)
            b0_image_fn = self.input_image_fn.split('.nii')[0] + '_b0.nii'
            self.b0_image_fn = b0_image_fn
            nib.save(b0_image, self.b0_image_fn)
            print('---save mask in {}'.format(self.b0_image_fn))
        self.create_mask_fslbet(b0_image_fn)


    def create_mask_fslbet(self, b0_image_fn, b0_image_mask_name=None):

        if b0_image_mask_name is None:
            b0_image_mask_name = self.input_image_fn.split('.')[0] \
                                 + '_mask' + self.input_image_fn.split('.')[-1]
        fsl_bet_mask(b0_image_fn, b0_image_mask_name)
        self.mask_image_path = b0_image_mask_name
        return b0_image_mask_name

    def dmc_make_target(self, b0_image, b0_image_mask):
        fct = 0.9
        new_resolution = self.input_resolution * fct
        transformed_b0, transformed_affine_b0 = self.upsampling_data(self.b0, self.input_affine, self.input_resolution, new_resolution)
        transformed_mask, transformed_affine_mask = self.upsampling_data(self.mask_data, self.mask_affine)
        minx = transformed_mask.shape[0] + 5
        maxx = -1
        miny = transformed_mask.shape[1] + 5
        maxy = -1
        minz = transformed_mask.shape[2] + 5
        maxz = -1






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
