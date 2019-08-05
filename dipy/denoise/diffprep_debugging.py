import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import DIFFPREPClass as difp



input_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test_subj1_anh/subj1_raw/subj1.nii"

d = difp.diffprep(input_fn, phase_encoding= 'vertical',
                  mask_image_path= '/qmi_home/anht/Desktop/DIFFPREP_test_data/test_subj1_anh/subj1_raw/mask.nii.gz',
                  activeDenoising=False,
                  activeGibbRemoving= False,
                  Eddy = True)
# d.create_mask()
image = d.input_data
d.execute()

final_image = d.dmc_make_target(d.b0, d.mask_mask_data)
print('pause here')