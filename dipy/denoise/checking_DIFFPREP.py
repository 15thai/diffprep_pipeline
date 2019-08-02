import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import DIFFPREPClass as difp


input_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test1/AP_b2500.nii"
d = difp.diffprep(input_fn)
image_resolution = np.array(d.input_resolution)
fct = 0.9
# print(np.array(image_resolution))
new_resolution = image_resolution * fct
d = difp.diffprep(input_fn, new_resolution=new_resolution)
out = d.execute()
print(out.shape)


