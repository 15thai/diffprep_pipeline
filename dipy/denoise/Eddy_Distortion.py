import numpy as np
import multiprocessing, ctypes, time
from contextlib import closing
from functools import partial
import os
from dipy.align.Register_DWIs import register_dwi_to_b0
from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration
import nibabel as nib
from dipy.denoise.DIFFPREPClass import diffprep


threads_to_use = multiprocessing.cpu_count()
print("Number of threads:", threads_to_use)

vol = 0
input_fn = "/home/anhpeo/Desktop/dipy_data_test/register_test/test_diffprep_work/AP_b2500_for_test.nii"
correction_mode = False
DiffPrep = diffprep(input_fn, encoding_phase='vertical')
# Create b0, b0_masked, mask_mask
b0_image_target, mask_mask_data, mask_affine= DiffPrep.create_mask()

signal_ranges = np.zeros(DiffPrep.nVols-1, 4)
QuadraticTransform = []
for i in range(DiffPrep.nVols -1 ):
    if i == DiffPrep.b0_id:
        continue
    curr_vol = DiffPrep.input_data[:,:,:,i]
    signal_ranges[i,:] = DiffPrep.choose_range(DiffPrep.b0, curr_vol, b0_image_target)
    signal_ranges[:,2:3] = 0
    if not correction_mode:
        quadMap = QuadraticMap(phase = 'vertical')
        QuadraticTransform.append(quadMap.get_QuadraticParams())


class RegisterDWIs_B0():
    def __init__(self, dwi_image_fn,
                 b0_image_fn = None,
                 phase_encoding = 'veritcal'):

        # Get the DWI
        self.dwi_image = nib.load(dwi_image_fn)
        self.dwi_data = self.dwi_image.get_data()
        self.dwi_affine = self.dwi_image.affine

        # Get B0_image
        self.b0_id = 0
        if b0_image_fn is None:
            i_s =  "There is no b0_image as input "
            i_s += "so we assume that b0_image is at"
            i_s += " {}".format(self.b0_id)
            print(i_s)
            self.b0_data = self.dwi_data[:,:,:,self.b0_id]
            self.b0_image = nib.Nifti1Image(self.b0_data, self.dwi_affine)
        else:
            self.b0_image = nib.load(b0_image_fn)
            self.b0_data = self.b0_image.get_data()





# if correction_mode:
#     """
#     OkanQuadraticTransformType::Pointer
#     curr_trans = RegisterDWIToB0(b0_img_target, curr_vol, list.GetPhaseEncodingDirection(),
#                                  this->mecc_settings->getDWIOptimizer(), this->mecc_settings, mstr, true, signal_ranges);"""


    #---------------- RAW images--------------------------
# new_func = partial(sub.get_patches_image_and_mask_with_tensors_fn,
#                    tensor_ids = default.tensors,
#                    norm = True)
# with multiprocessing.Pool(threads_to_use) as p:
#     raw_data = p.map(new_func, list_of_images)
# p.close()
# raw_image = [raw_[0] for raw_ in raw_data]
# raw_label = [raw_l[1] for raw_l in raw_data]
# raw_image = np.asarray(raw_image)
# raw_label = np.asarray(raw_label)
# raw_image = raw_image.reshape(-1, default.patch_shape[0], default.patch_shape[1])
# raw_label = raw_label.reshape(-1, default.patch_shape[0], default.patch_shape[1])
#
# sub.save_to_hdf5(os.path.join(default.save_folder,
#                               default.image_hdf5),
#                               raw_image, 'images')
#
# sub.save_to_hdf5(os.path.join(default.save_folder,
#                               default.label_hdf5), raw_label, 'labels')
