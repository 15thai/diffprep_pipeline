import sys
import nibabel as nib
import os
from scipy.ndimage import center_of_mass
from dipy.align.imwarp import get_direction_and_spacings
import numpy as np


def _get_centerOfGravityPt(arr):
    com = center_of_mass(arr)
    com = np.array(com)
    com = np.round(com).astype(int)
    return com

def _transformIndextoPhysicalPoint_with_affine(affine, pts):
    physical_pts = affine[:3,:3].dot(pts) + affine [:3,3]
    return physical_pts

def _transformPhysicalPointToIndex(point, original, spacing):
    index = (point - original) // spacing
    return index

def _transformIndextoPhysicalPoint(index, spacing, direction, origin):
    pt = index * (spacing.dot(direction)) + origin
    return pt


# This version is the updating Version, which will NOT use the SIMPLE ITK
def changeFOVimage(image, new_FOV):
    arr = image.get_data()
    affine = image.affine
    n = 3
    if isinstance(new_FOV, list):
        try:
            new_FOV = np.array(new_FOV)
        except:
            raise IOError ("list of fields of view")

    direction, spacing = get_direction_and_spacings(affine, arr.ndim)
    print(direction, spacing)
    arr_size = arr.size
    arr_shape = np.array(arr.shape)
    old_fov = np.abs(spacing.dot(direction)*arr_shape)
    origin = affine[:3,3]

    print("Input Image has:"
          "FOVs: {} "
          "Our New FOVs: {}".format(old_fov, new_FOV))

    doit = False
    for f, fov in enumerate(old_fov):
        if (abs(fov - new_FOV[f]) >= 2 * spacing[f]):
            doit = True
        else:
            doit = False
            break

    if not doit:
        print("Can't Change FOV - Return original image")
        return image, affine

    total_add_3D, total_remove_3D = np.zeros(n).astype(int), np.zeros(n).astype(int)

    new_origin_index = np.zeros(n).astype(int)
    new_size = np.zeros(n).astype(int)
    start_from = np.zeros(n).astype(int)
    start_from_new, from_old_size = np.zeros(n).astype(int), np.zeros(n).astype(int)

    for d, fov in enumerate(old_fov[:n]):
        print(d)
        if new_FOV[d] > fov:
            # print("Dimension {} process add".format(d))
            total_add_3D[d] = (np.ceil((new_FOV[d] - fov) / spacing[d]))
            if ((total_add_3D[d] % 2) == 1):
                total_add_3D[d] += 1
            new_origin_index[d] = - (total_add_3D[d] / 2)
            new_size[d] = (arr_shape[d] + total_add_3D[d])
            start_from[d] = 0
            start_from_new[d] = total_add_3D[d] / 2
            from_old_size[d] = arr_shape[d]

        else:
            # print("Dimension {} process remove".format(d))
            total_remove_3D[d] = (np.floor((fov - new_FOV[d]) / spacing[d]))
            if ((total_remove_3D[d] % 2) == 1):
                total_remove_3D[d] -= 1
            new_origin_index[d] = (total_remove_3D[d] / 2)
            new_size[d] = (arr_shape[d] - total_remove_3D[d])
            start_from[d] = total_remove_3D[d] / 2
            start_from_new[d] = 0
            from_old_size[d] = arr_shape[d] - total_remove_3D[d]

    # new_origin = origin + spacing * new_origin_index
    # new_origin = _transformIndextoPhysicalPoint(new_origin_index, spacing, direction, origin)
    new_origin = _transformIndextoPhysicalPoint_with_affine(affine, new_origin_index)
    new_affine = affine.copy()
    new_affine[0:3,3] = new_origin
    if arr.ndim > 3:

        new_image = arr[start_from[0]: start_from[0] + from_old_size[0],
                  start_from[1]: start_from[1] + from_old_size[1],
                  start_from[2]: start_from[2] + from_old_size[2],:]
    else:
        new_image = arr[start_from[0]: start_from[0] + from_old_size[0],
                  start_from[1]: start_from[1] + from_old_size[1],
                  start_from[2]: start_from[2] + from_old_size[2]]
    return new_image, new_affine

def crop4D(image, new_FOV, b0_image):
    arr = image.get_data()
    affine = image.affine
    n = arr.ndim

    if isinstance(new_FOV, list):
        try:
            new_FOV = np.array(new_FOV)
        except:
            raise IOError("list of fields of view")

    direction, spacing = get_direction_and_spacings(affine, arr.ndim)
    # print(direction, spacing)
    arr_size = arr.size
    arr_shape = np.array(arr.shape)
    old_fov = np.abs(spacing.dot(direction) * arr_shape)
    origin = affine[:3, 3]
    print("Input Image has:"
          "FOVs: {} "
          "Our New FOVs: {}".format(old_fov, new_FOV))

    doit = False
    for f, fov in enumerate(old_fov):
        if (abs(fov - new_FOV[f]) >= 2 * spacing[f]):
            doit = True

    if not doit:
        return image.get_data(), image.affine
    # get center of mass index and point
    center_index = _get_centerOfGravityPt(b0_image)
    center_pt = _transformIndextoPhysicalPoint( center_index,spacing, direction, origin)

    scl = direction[:3, :3] * np.diag(spacing)



    new_size = np.ceil(new_FOV / spacing)
    new_center_index = (new_size - 1) // 2
    new_origin = np.zeros(3)

    # Find new_origin
    for r in range(3):
        sm = np.sum(scl[r, :] * new_center_index)
        new_origin[r] = center_pt[r] - sm

    spc4 = np.zeros(4)
    spc4[:-1] = spacing
    spc4[-1] = 1

    dir4 = np.eye(4, 4)
    dir4[:3, :3] = direction[:3, :3]

    new_origin4 = np.zeros(4)
    new_origin4[:3] = new_origin

    destination_index = np.zeros(3).astype(int)
    source_start = np.zeros(3).astype(int)
    source_size = np.zeros(3).astype(int)

    for d in range(3):
        if (new_center_index[d] < center_index[d]):
            destination_index[d] = 0
            source_start[d] = center_index[d] - new_center_index[d]

            if (source_start[d] + new_size[d] > arr.shape[d]):
                source_size[d] = arr.shape[d] - source_start[d] - 1

            else:
                source_size[d] = new_size[d]

        else:
            destination_index[d] = new_center_index[d] - center_index[d]
            source_start[d] = 0

            if (arr.shape[d] > new_size[d]):
                source_size[d] = new_size[d]
            else:
                source_size[d] = arr.shape[d]
    if arr.ndim >3:
        new_image = arr[source_start[0]: source_start[0] + source_size[0],
                source_start[1]: source_start[1] + source_size[1],
                source_start[2]: source_start[2] + source_size[2],:]

    else:
        new_image = arr[source_start[0]: source_start[0] + source_size[0],
                    source_start[1]: source_start[1] + source_size[1],
                    source_start[2]: source_start[2] + source_size[2]]
    new_affine = affine
    new_affine[0:3,3] = new_origin
    return new_image, new_affine


# main function for changing fov
def changeFoV(image_nib, new_FOV, com = False):

    if isinstance(new_FOV, list):
        try:
            new_FOV = np.array(new_FOV)
        except:
            raise IOError("list of fields of view")

    image_shape = np.array(image_nib.shape)
    direction, spacing = get_direction_and_spacings(image_nib.affine, image_nib.ndim)
    old_fov = image_shape * spacing

    print("Input Image has:"
          "FOVs: {} "
          "Our New FOVs: {}".format(old_fov, new_FOV))

    doit = False
    for f, fov in enumerate(old_fov):
        if (abs(fov - new_FOV[f]) >= 2 * spacing[f]):
            doit = True
        else:
            doit = False
            break

    if not doit:
        return image_nib.get_data(), image_nib.affine


    arr = image_nib.get_data()
    # Assume B_0 image is always the first vol
    if image_nib.ndim > 3:
        b0_image = arr[:,:,:,0]
    else:
        b0_image = arr

    nVols = image_shape[-1]

    if com:
        new_image, new_affine = crop4D(image_nib, new_FOV, b0_image)
    else:
        new_image, new_affine = changeFOVimage(image_nib, new_FOV)

    return new_image, new_affine