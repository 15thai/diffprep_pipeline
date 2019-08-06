
import dipy.denoise.DIFFPREPClass as difp
import nibabel as nib
from dipy.align.Register_DWIs import register_dwi_to_b0
input_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_DIFFPREP_proc_Denoised.nii"
mask_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_mask_mask.nii"

#
fixed_image ="/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_T0.nii"
moving_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_T21.nii"
phase = 'horizontal'
ouput_image_name = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_T0_T21.nii"

def registering_images(fixed_image_name, moving_image_name, phase, ouput_image_name):
    fixed_image = nib.load(fixed_image_name)
    moving_image = nib.load(moving_image_name)
    transformation_, image_out = register_dwi_to_b0(fixed_image, moving_image, phase = phase,
                                                    initialize= True, registration_type='quadratic',
                                                    optimizer_setting=False)

    image_out = nib.Nifti1Image(image_out, fixed_image.affine)
    nib.save(image_out, ouput_image_name)




registering_images(fixed_image_name=fixed_image,
                   moving_image_name= moving_image,
                   phase = phase,
                   ouput_image_name = ouput_image_name)