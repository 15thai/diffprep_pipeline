import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import DIFFPREPClass as difp



input_fn = "/home/anhpeo/Desktop/dipy_data_test/register_test/test_diffprep_work/AP_b2500_for_test.nii"

d = difp.diffprep(input_fn, encoding_phase= 'vertical')
d.create_mask()
image = d.input_data

print('pause here')