
from dipy.denoise.DIFFPREPClass import diffprep
input_filename = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc.nii"
d = diffprep(input_filename,
             phase_encoding= 'horizontal',
             b0_id = 0,
             center_of_mass= True,
             activeDenoising=True,
             activeGibbRemoving=True,
             Eddy = True)
d.execute()

