
import dipy.denoise.DIFFPREPClass as difp
input_fn = "/home/anhpeo/Desktop/dipy_data_test/register_test/test_diffprep_work/AP_b2500_for_test.nii"
# mask_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_mask_mask.nii"



test = difp.diffprep(input_image_fn= input_fn,
                     phase_encoding= 'horizontal')
test.activeEddy = True
test.execute()
print('AAAa')

