
import dipy.denoise.DIFFPREPClass as difp
input_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_DIFFPREP_proc_Denoised.nii"
mask_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc_mask_mask.nii"



test = difp.diffprep(input_image_fn= input_fn,
                     phase_encoding= 'horizontal',
                     activeDenoising=False,
                     activeGibbRemoving=False,
                     activeChangeFOV=False,
                     activeEddy= True)
# test.activeEddy = True
test.execute()
print('AAAa')

