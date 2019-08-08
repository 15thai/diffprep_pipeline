import numpy as np
import nibabel as nib
from dipy.align.Register_DWIs import  register_dwi_to_b0

def registering_dwis():
    fixed_image ="/qmi_home/anht/Desktop/DIFFPREP_test_data/b0.nii"
    moving_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/dwi.nii"

    fixed_image = nib.load(fixed_image)
    moving_image = nib.load(moving_image)
    transformation_, image_out = register_dwi_to_b0(fixed_image, moving_image, phase = 'horizontal',
                                                    initialize= True, registration_type='quadratic',
                                                    optimizer_setting=True)

    image_out = nib.Nifti1Image(image_out, fixed_image.affine)
    nib.save(image_out, "/qmi_home/anht/Desktop/DIFFPREP_test_data/test_out_linear_2_anh.nii")
