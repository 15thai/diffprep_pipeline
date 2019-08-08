import numpy as np
import nibabel as nib
from dipy.denoise.randomlpca_denoise import randomlpca_denoise
from dipy.denoise.GibbRemoval import gibbremoval
from dipy.denoise.ChangeFOV import changeFoV
from dipy.align.reslice import reslice
from dipy.denoise.fsl_bet import fsl_bet_mask


# DIFFPREP -i path_to_list_input_file --do_QV 0 -e off --res 1 1 1 -s acpc_align.nii


class diffprep(object):
    def __init__(self, input_image_path, output_image_path = None,
                 bvals = None, bvecs = None, Eddy = False,
                 fov = None, activeGibbRemoving = False, activeDenoising = False, center_of_mass = False,
                 new_resolution = None, interp = 1):

        # Inter from 0-5, "Nearest", "Lanczos", "Bilinear", "Bicubic", "Cubic"
        self.input_image_fn = input_image_path

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
        self.input_resolution = self.input_image.header.get_zooms()[:3]

        self.input_size = self.input_data.size
        self.input_shape = self.input_data.shape

        # Activate Set-up
        self.activeGibbRemoving = activeGibbRemoving
        self.activeDenoising = activeDenoising
        self.center_of_mass = center_of_mass
        self.Eddy = Eddy

        if self.input_data.ndim  <4:
            self.activeDenoising = False

        self.output_data = self.input_data
        self.interp = interp

        if new_resolution is not None:
            try:
                self.new_resolution = np.array(new_resolution)

            except:
                print("Can't Convert to Numpy Array for resolution")


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
            print("---------- Upsampling data--------------------")

            if self.new_resolution != self.input_resolution:
                output_arr = self.upsampling_data(self.input_data, self.input_affine, self.input_resolution, self.new_resolution, self.interp)
                self.input_data = output_arr

        if self.Eddy:
            print("Perform Registration......")

        self.output_data = self.input_data
        self.save_output_tofile(self.output_data,self.output_affine,self.input_image_fn.split('.nii')[0] + "_DIFFPREP_proc.nii"  )
        return self.output_data


    def get_field_of_view (self):
        return self.fov

    def get_output_data(self):
        return self.output_data

    def get_GibbRemoved_data(self, input_arr):
        output = gibbremoval(input_arr)
        return output

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

    def upsampling_data(self, input_image_arr, input_affine, input_resolution, output_resolution, order = 1 ):
        new_image = reslice(input_image_arr, input_affine,
                                input_resolution, output_resolution, order= order)
        return new_image

    def save_output_tofile(self, output_data, output_affine= None,  output_name = None):
        if output_name is None:
            output_name = self.output_image_fn
        if output_affine is None:
            output_affine = self.input_affine

        self.output_image = nib.Nifti1Image(output_data, output_affine)
        nib.save(self.output_image, output_name)


# def main():
#     DIFFPREP = diffprep("/home/anhpeo/Desktop/dipy_data_test/unring_test/T2W.nii", activeGibbRemoving = True)
#     DIFFPREP.save_output_tofile(DIFFPREP.output_data)
#
#     # DIFFPREP = diffprep("/home/anhpeo/Desktop/dipy_data_test/denoise_test/ABCD-DTI_run-20161212123806_DR.nii", activeDenoising= True)
#     # DIFFPREP.save_output_tofile(DIFFPREP.get_output_data())
#     # DIFFPREP.get_output_noise(save_to_file=True)
#
#     # DIFFPREP = diffprep("/home/anhpeo/Desktop/dipy_data_test/unring_test/T2W.nii", fov = [200, 200, 150])
#     # output_, affine_ = DIFFPREP.change_field_of_view(com = True)
#     # DIFFPREP.save_output_tofile(output_, affine_, DIFFPREP.output_image_fn.split('.nii') [0]+ '_newFov_com.nii')
#
#     # DIFFPREP = diffprep("/home/anhpeo/Desktop/dipy_data_test/fov_unring_test/T2W.nii", fov = [200, 200, 150], center_of_mass=True)
#
#
# main()