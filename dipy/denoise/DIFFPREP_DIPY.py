import os
import numpy as np
import argparse
import nibabel as nib
# from dipy.denoise.randomlpca_denoise import randomlpca_denoise

"""Follow the DIFFPREP in C++"""
parser = argparse.ArgumentParser(prog= 'DIFFPREP',description= "The main DIFFPREP program. It takes in "
                                              "                           dwi: path to input image,"
                                              "                           output_dwi: path to output image,"
                                              "                           bvecs: path to bvecs txt file,"
                                              "                           bvals: path to bvals txt file,"
                                              " and a registration settings file "
                                              "and optionally a structural and reorientation image and performs motion,"
                                              " eddy-currents, susceptibility distortion correction and reorients the DWIs onto "
                                              "the desired space with Bmatrix reorientation.")


# parser.add_argument('--input_list',dest = 'input_list', type = str, help = '/path/to/input/listfile')
parser.add_argument('--input', type = str, help = '/path/to/input/nii file ')
parser.add_argument('--output', type = str, help = '/path/to/output/output_nii file')
parser.add_argument('--bvecs', type = str, help = '/path/to/input/bvecs file')
parser.add_argument('--bvals', type = str, help = '/path/to/input/bvals file')
parser.add_argument('--phase', type = int, choices=[0,1,2], default=0, help = 'value of phase 0: rowwise, 1: colwise, 2: slice-wise')

parser.add_argument('--structural', type = str, default = None, help = '/path/to/input/strutuctural file, optional but strongly suggested')
parser.add_argument('--reorientation',type = str, default = None, help = '/path/to/reoriented image file')

group = parser.add_mutually_exclusive_group()
group.add_argument('--new_resolution', nargs = 3, help = 'array input of new resolution for each dimension')
group.add_argument('--up_factor', type = int, help = 'array input of new resolution for each dimension')

parser.add_argument('--method',type = int,default=3, help = 'interpolation methods 0: nearest, 1: bilinear, 3: cubic')
parser.add_argument('--reg_setting',type = str, help = '/path/to/registration file')
parser.add_argument('--voxelwise_bmatrices', type = str, help = '/path/to/bmatrix_txt file')
parser.add_argument('--gib_ringing', type = bool,choices= [0,1], default = 0, help = 'active gib_ringing : 1, inactive gib_ringing : 0 ')
parser.add_argument('--denoising',type = bool, choices= [0,1], default = 0, help = 'active denoising : 1, inactive denoising: 0')
parser.add_argument('--center_of_mass',type = bool, choices= [0,1], help = 'using center_of_mass:1 , otherwise : 0')
parser.add_argument('--Fov',nargs = 3, help = 'input new field of view, [fovx fovy fovz]')

args = parser.parse_args()

if __name__ == "__main__":
    from DIFFPREPClass import diffprep
    # DiffPrep = diffprep(input_image_path= args.input,
    #                     output_image_path= args.output,
    #                     phase_encoding= args.phase,
    #                     activeGibbRemoving= args.gib_ringing,
    #                     activeDenoising= args.denoising,
    #                     center_of_mass= args.center_of_mass)

