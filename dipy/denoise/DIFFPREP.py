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
parser.add_argument('--dwi', type = str, help = '/path/to/input/nii file ')
parser.add_argument('--output', type = str, help = '/path/to/output/output_nii file')
parser.add_argument('--bvecs', type = str, help = '/path/to/input/bvecs file')
parser.add_argument('--bvals', type = str, help = '/path/to/input/bvals file')
parser.add_argument('--phase', type = int, choices=[0,1,2], default=0, help = 'value of phase 0: rowwise, 1: colwise, 2: slice-wise')

parser.add_argument('--structural', type = str, help = '/path/to/input/strutuctural file, optional but strongly suggested')
parser.add_argument('--reorientation',type = str, help = '/path/to/reoriented image file')

# They can not exist together (either new_resolution or up_factor)
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


def main():
    """Uncomment below when finish with bvals and bvecs"""
    # if (not os.path.exists(args.dwi)) or (not os.path.exists(args.bvecs)) or (not os.path.exists(args.bvals)):
    #     return print("Error, No file exists")
    """DIFFPREP Process"""
    # reading in image
    if (not os.path.exists(args.dwi)):
        return print("ERROR no {} exists".format(args.dwi))

    dwi_image = nib.load(args.dwi)
    dwi_affine = dwi_image.affine
    arr = dwi_image.get_data()

    # reading in structural image
    if args.structural:
        if os.path.exists(args.structural):
            structural_image = nib.load(args.structural)
        else:
            raise IOError("No file exists")

    # reading in structural image
    if args.reorientation:
        if os.path.exists(args.reorientation):
            reorientation_image = nib.load(args.reorientation)
        else:
            raise IOError("No file exists")

    # Performing Noise reduction
    # if args.denoising:
    #     arr = randomlpca_denoise(arr)

    # Performing Gibb artifacts removal
    if args.gib_ringing:
        print("---Performing Gibb Ringing Removal step---")
        from GibbRemoval import gibbremoval
        arr = gibbremoval(arr)

    # Performing adjust field of view removal
    if args.Fov:
        print("---Performing adjust field of view step---")

        from ChangeFOV3D import changefov , changefov_with_com
        if not args.center_of_mass:
            arr, dwi_affine = changefov(args.dwi,args.Fov[0], args.Fov[1], args.Fov[2])
        else:
            arr, dwi_affine = changefov_with_com(args.dwi, args.Fov[0], args.Fov[1], args.Fov[2])

    # Resample image according to new_resolution
    if args.new_resolution:
        print("---Performing resampling image")
        from dipy.align.reslice import reslice
        zooms = dwi_image.header.get_zooms()[:3]
        new_zooms = np.array(args.new_resolution)
        arr, dwi_affine = reslice(arr, dwi_affine, zooms, new_zooms)

    # if args.up_factor:
    #     from dipy.denoise. import resample_3D_parallel
    #     arr = resample_3D_parallel(arr, up_factors = args.up_factors, methods = args.method)

    # Saving to output file
    output_image = nib.Nifti1Image(arr, dwi_affine)
    nib.save(output_image, args.output_dwi)


if __name__ == "__main__":
    main()





