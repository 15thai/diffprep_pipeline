from copy import deepcopy
import enum
from itertools import product
import logging
import os
import time
import six

import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import (Tractogram,
                                            PerArraySequenceDict,
                                            PerArrayDict)
import numpy as np

from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines
from dipy.io.dpy import Dpy


class Space(enum.Enum):
    """ Enum to simplify future change to convention """
    VOX = 'vox'
    VOXMM = 'voxmm'
    RASMM = 'rasmm'


class StatefulTractogram(object):
    """ Class for stateful representation of collections of streamlines
    Object designed to be identical no matter the file format
    (trk, tck, fib, dpy). Facilitate transformation between space and
    data manipulation for each streamline / point.
    """

    def __init__(self, streamlines, reference, space,
                 shifted_origin=False,
                 data_per_point=None, data_per_streamline=None):
        """ Create a strict, state-aware, robust tractogram

        Parameters
        ----------
        streamlines : list or ArraySequence
            Streamlines of the tractogram
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header or trk.header (dict)
            Reference that provides the spatial attribute.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        space : string
            Current space in which the streamlines are (vox, voxmm or rasmm)
            Typically after tracking the space is VOX, after nibabel loading
            the space is RASMM
        shifted_origin : bool
            Information on the position of the origin,
            False is Trackvis standard, default (corner of the voxel)
            True is NIFTI standard (center of the voxel)
        data_per_point : dict
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
            Y_i being the number of points on streamlines #i
        data_per_streamline : dict
            Dictionary in which each key has X items
            X being the number of streamlines

        Notes
        -----
        Very important to respect the convention, verify that streamlines
        match the reference and are effectively in the right space.

        Any change to the number of streamlines, data_per_points or
        data_per_streamlines requires particular verification.

        In a case of manipulation not allowed by this object, use Nibabel
        directly and be careful.
        """
        if data_per_point is None:
            data_per_point = {}

        if data_per_streamline is None:
            data_per_streamline = {}

        self._tractogram = Tractogram(streamlines,
                                      data_per_point=data_per_point,
                                      data_per_streamline=data_per_streamline)

        space_attribute = get_reference_info(reference)
        if space_attribute is None:
            raise TypeError('Reference MUST be one of the following:\n' +
                            'Nifti or Trk filename, Nifti1Image or TrkFile, ' +
                            'Nifti1Header or trk.header (dict)')

        (self._affine, self._dimensions,
         self._voxel_sizes, self._voxel_order) = space_attribute
        self._inv_affine = np.linalg.inv(self._affine)

        if space not in Space:
            raise ValueError('Space MUST be from Space enum, e.g Space.VOX')
        self._space = space

        if not isinstance(shifted_origin, bool):
            raise TypeError('shifted_origin MUST be a boolean')
        self._shifted_origin = shifted_origin
        logging.debug(self)

    def __str__(self):
        """ Generate the string for printing """
        text = 'Affine: \n{}'.format(
            np.array2string(self._affine,
                            formatter={'float_kind': lambda x: "%.6f" % x}))
        text += '\ndimensions: {}'.format(
            np.array2string(self._dimensions))
        text += '\nvoxel_sizes: {}'.format(
            np.array2string(self._voxel_sizes,
                            formatter={'float_kind': lambda x: "%.2f" % x}))
        text += '\nvoxel_order: {}'.format(self._voxel_order)

        text += '\nstreamline_count: {}'.format(self._get_streamline_count())
        text += '\npoint_count: {}'.format(self._get_point_count())
        text += '\ndata_per_streamline keys: {}'.format(
            self.get_data_per_point().keys())
        text += '\ndata_per_points keys: {}'.format(
            self.get_data_per_streamline().keys())

        return text

    def __len__(self):
        """ Define the length of the object """
        return self._get_streamline_count()

    def get_space_attribute(self):
        """ Getter for spatial attribute """
        return self._affine, self._dimensions, self._voxel_sizes, \
            self._voxel_order

    def get_current_space(self):
        """ Getter for the current space """
        return self._space

    def get_current_shift(self):
        """ Getter for shift """
        return self._shifted_origin

    def get_streamlines(self):
        """ Partially safe getter for streamlines """
        return self._tractogram.streamlines

    def get_streamlines_copy(self):
        """ Safe getter for streamlines (for slicing) """
        return deepcopy(list(self._tractogram.streamlines))

    def set_streamlines(self, streamlines):
        """ Modify streamlines. Creating a new object would be less risky.

        Parameters
        ----------
        streamlines : list or ArraySequence (list and deepcopy recommanded)
            Streamlines of the tractogram
        """
        self._tractogram._streamlines = ArraySequence(streamlines)
        self.set_data_per_point(self.get_data_per_point())
        self.set_data_per_streamline(self.get_data_per_streamline())
        logging.warning('Streamlines has been modified')

    def get_data_per_point(self):
        """ Getter for data_per_point """
        return self._tractogram.data_per_point

    def set_data_per_point(self, data):
        """ Modify point data . Creating a new object would be less risky.

        Parameters
        ----------
        data : dict
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
            Y_i being the number of points on streamlines #i
        """
        self._tractogram.data_per_point = data
        logging.warning('Data_per_point has been modified')

    def get_data_per_streamline(self):
        """ Getter for data_per_streamline """
        return self._tractogram.data_per_streamline

    def set_data_per_streamline(self, data):
        """ Modify point data . Creating a new object would be less risky.

        Parameters
        ----------
        data : dict
            Dictionary in which each key has X items, each items has Y_i items
            X being the number of streamlines
        """
        self._tractogram.data_per_streamline = data
        logging.warning('Data_per_streamline has been modified')

    def to_vox(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOXMM:
            self._voxmm_to_vox()
        elif self._space == Space.RASMM:
            self._rasmm_to_vox()

    def to_voxmm(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOX:
            self._vox_to_voxmm()
        elif self._space == Space.RASMM:
            self._rasmm_to_voxmm()

    def to_rasmm(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOX:
            self._vox_to_rasmm()
        elif self._space == Space.VOXMM:
            self._voxmm_to_rasmm()

    def to_center(self):
        """ Safe function to shift streamlines so the center of voxel is
        the origin """
        if self._shifted_origin:
            self._shift_voxel_origin()

    def to_corner(self):
        """ Safe function to shift streamlines so the corner of voxel is
        the origin """
        if not self._shifted_origin:
            self._shift_voxel_origin()

    def compute_bounding_box(self):
        """ Compute the bounding box of the streamlines in their current state

        Returns
        -------
        output : ndarray
            8 corners of the XYZ aligned box, all zeros if no streamlines
        """
        if self._tractogram.streamlines.data.size > 0:
            bbox_min = np.min(self._tractogram.streamlines.data, axis=0)
            bbox_max = np.max(self._tractogram.streamlines.data, axis=0)

            return np.asarray(list(product(*zip(bbox_min, bbox_max))))

        return np.zeros((8, 3))

    def is_bbox_in_vox_valid(self):
        """ Verify that the bounding box is valid in voxel space
        Will transform the streamlines for OBB, slow for big tractogram

        Returns
        -------
        output : bool
            Are the streamlines within the volume of the associated reference
        """
        old_space = deepcopy(self.get_current_space())
        old_shift = deepcopy(self.get_current_shift())

        # Do to rotation, equivalent of a OBB must be done
        self.to_vox()
        self.to_corner()
        bbox_corners = deepcopy(self.compute_bounding_box())

        is_valid = True
        if np.any(bbox_corners < 0):
            logging.error('Voxel space values lower than 0.0')
            logging.debug(bbox_corners)
            is_valid = False

        if np.any(bbox_corners[:, 0] > self._dimensions[0]) or \
                np.any(bbox_corners[:, 1] > self._dimensions[1]) or \
                np.any(bbox_corners[:, 2] > self._dimensions[2]):
            logging.error('Voxel space values higher than dimensions')
            logging.debug(bbox_corners)
            is_valid = False

        if old_space == Space.RASMM:
            self.to_rasmm()
        elif old_space == Space.VOXMM:
            self.to_voxmm()

        if not old_shift:
            self.to_center()

        return is_valid

    def _get_streamline_count(self):
        """ Safe getter for the number of streamlines """
        return len(self._tractogram.streamlines._offsets)

    def _get_point_count(self):
        """ Safe getter for the number of streamlines """
        return len(self._tractogram.streamlines.data)

    def _vox_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOX:
            if self._tractogram.streamlines.data.size > 0:
                self._tractogram.streamlines._data *= np.asarray(
                    self._voxel_sizes)
                self._space = Space.VOXMM
                logging.info('Moved streamlines from vox to voxmm')
        else:
            logging.warning('Wrong initial space for this function')
            return

    def _voxmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOXMM:
            if self._tractogram.streamlines.data.size > 0:
                self._tractogram.streamlines._data /= np.asarray(
                    self._voxel_sizes)
                self._space = Space.VOX
                logging.info('Moved streamlines from voxmm to vox')
        else:
            logging.warning('Wrong initial space for this function')
            return

    def _vox_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOX:
            if self._tractogram.streamlines.data.size > 0:
                self._tractogram.apply_affine(self._affine)
                self._space = Space.RASMM
                logging.info('Moved streamlines from vox to rasmm')
        else:
            logging.warning('Wrong initial space for this function')
            return

    def _rasmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.RASMM:
            if self._tractogram.streamlines.data.size > 0:
                self._tractogram.apply_affine(self._inv_affine)
                self._space = Space.VOX
                logging.info('Moved streamlines from rasmm to vox')
        else:
            logging.warning('Wrong initial space for this function')
            return

    def _voxmm_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.VOXMM:
            if self._tractogram.streamlines.data.size > 0:
                self._tractogram.streamlines._data /= np.asarray(
                    self._voxel_sizes)
                self._tractogram.apply_affine(self._affine)
                self._space = Space.RASMM
                logging.info('Moved streamlines from voxmm to rasmm')
        else:
            logging.warning('Wrong initial space for this function')
            return

    def _rasmm_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        if self._space == Space.RASMM:
            if self._tractogram.streamlines.data.size > 0:
                self._tractogram.apply_affine(self._inv_affine)
                self._tractogram.streamlines._data *= np.asarray(
                    self._voxel_sizes)
                self._space = Space.VOXMM
                logging.info('Moved streamlines from rasmm to voxmm')
        else:
            logging.warning('Wrong initial space for this function')
            return

    def _shift_voxel_origin(self):
        """ Unsafe function to switch the origin from center to corner
        and vice versa """
        shift = np.asarray([0.5, 0.5, 0.5])
        if self._space == Space.VOXMM:
            shift = shift * self._voxel_sizes
        elif self._space == Space.RASMM:
            tmp_affine = np.eye(4)
            tmp_affine[0:3, 0:3] = self._affine[0:3, 0:3]
            shift = apply_affine(tmp_affine, shift)
        if self._shifted_origin:
            shift *= -1

        self._tractogram.streamlines._data += shift
        if not self._shifted_origin:
            logging.info('Origin moved to the center of voxel')
        else:
            logging.info('Origin moved to the corner of voxel')

        self._shifted_origin = not self._shifted_origin


def save_tractogram(sft, filename, bbox_valid_check=True):
    """ Save the stateful tractogram in any format (trk, tck, fib, dpy)

    Parameters
    ----------
    sft : StatefulTractogram
        The stateful tractogram to save
    filename : string
        Filename with valid extension

    Returns
    -------
    output : bool
        Did the saving work properly
    """

    _, extension = os.path.splitext(filename)
    if extension not in ['.trk', '.tck', '.vtk', '.fib', '.dpy']:
        TypeError('Output filename is not one of the supported format')

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot ' +
                         'save a valid file if some coordinates are invalid')

    old_space = deepcopy(sft.get_current_space())
    old_shift = deepcopy(sft.get_current_shift())

    sft.to_rasmm()
    sft.to_center()

    timer = time.time()
    if extension in ['.trk', '.tck']:
        tractogram_type = detect_format(filename)
        header = create_tractogram_header(tractogram_type,
                                          *sft.get_space_attribute())
        new_tractogram = Tractogram(sft.get_streamlines(),
                                    affine_to_rasmm=np.eye(4))

        if extension == '.trk':
            new_tractogram.data_per_point = sft.get_data_per_point()
            new_tractogram.data_per_streamline = sft.get_data_per_streamline()

        fileobj = tractogram_type(new_tractogram, header=header)
        nib.streamlines.save(fileobj, filename)

    elif extension in ['.vtk', '.fib']:
        save_vtk_streamlines(sft.get_streamlines(), filename, binary=True)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='w')
        dpy_obj.write_tracks(sft.get_streamlines())
        dpy_obj.close()

    logging.debug('Save %s with %s streamlines in %s seconds',
                  filename, len(sft), round(time.time() - timer, 3))

    if old_space == Space.VOX:
        sft.to_vox()
    elif old_space == Space.VOXMM:
        sft.to_voxmm()

    if old_shift:
        sft.to_corner()

    return True


def load_tractogram(filename, reference, to_space=Space.RASMM,
                    shifted_origin=False, bbox_valid_check=True,
                    trk_header_check=True):
    """ Load the stateful tractogram from any format (trk, tck, fib, dpy)

    Parameters
    ----------
    filename : string
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict)
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        streamlines generation
    space : string
        Space in which the streamlines will be transformed after loading
        (vox, voxmm or rasmm)
    shifted_origin : bool
        Information on the position of the origin,
        False is Trackvis standard, default (center of the voxel)
        True is NIFTI standard (corner of the voxel)

    Returns
    -------
    output : StatefulTractogram
        The tractogram to load (must have been saved properly)
    """
    _, extension = os.path.splitext(filename)
    if extension not in ['.trk', '.tck', '.vtk', '.fib', '.dpy']:
        logging.error('Output filename is not one of the supported format')
        return False

    if to_space not in Space:
        logging.error('Space MUST be one of the 3 choices (Enum)')
        return False

    if trk_header_check and extension == '.trk':
        if not is_header_compatible(filename, reference):
            logging.error('Trk file header does not match the provided ' +
                          'reference')
            return False

    timer = time.time()
    data_per_point = None
    data_per_streamline = None
    if extension in ['.trk', '.tck']:
        tractogram_obj = nib.streamlines.load(filename).tractogram
        streamlines = tractogram_obj.streamlines
        if extension == '.trk':
            data_per_point = tractogram_obj.data_per_point
            data_per_streamline = tractogram_obj.data_per_streamline

    elif extension in ['.vtk', '.fib']:
        streamlines = load_vtk_streamlines(filename)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='r')
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()
    logging.debug('Load %s with %s streamlines in %s seconds',
                  filename, len(streamlines), round(time.time() - timer, 3))

    sft = StatefulTractogram(streamlines, reference, Space.RASMM,
                             shifted_origin=shifted_origin,
                             data_per_point=data_per_point,
                             data_per_streamline=data_per_streamline)

    if to_space == Space.VOX:
        sft.to_vox()
    elif to_space == Space.VOXMM:
        sft.to_voxmm()

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot ' +
                         'load a valid file if some coordinates are invalid')

    return sft


def get_reference_info(reference):
    """ Will compare the spatial attribute of 2 references

    Parameters
    ----------
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict)
        Reference that provides the spatial attribute.

    Returns
    -------
    output : tuple
        - affine ndarray (4,4), np.float32, tranformation of VOX to RASMM
        - dimensions list (3), int, volume shape for each axis
        - voxel_sizes  list (3), float, size of voxel for each axis
        - voxel_order, string, Typically 'RAS' or 'LPS'
    """

    is_nifti = False
    is_trk = False
    if isinstance(reference, six.string_types):
        try:
            header = nib.load(reference).header
            is_nifti = True
        except nib.filebasedimages.ImageFileError:
            pass
        try:
            header = nib.streamlines.load(reference, lazy_load=True).header
            is_trk = True
        except ValueError:
            pass
    elif isinstance(reference, nib.nifti1.Nifti1Image):
        header = reference.header
        is_nifti = True
    elif isinstance(reference, nib.streamlines.trk.TrkFile):
        header = reference.header
        is_trk = True
    elif isinstance(reference, nib.nifti1.Nifti1Header):
        header = reference
        is_nifti = True
    elif isinstance(reference, dict) and 'magic_number' in reference:
        header = reference
        is_trk = True

    if is_nifti:
        affine = np.eye(4).astype(np.float32)
        affine[0, 0:4] = header['srow_x']
        affine[1, 0:4] = header['srow_y']
        affine[2, 0:4] = header['srow_z']
        dimensions = header['dim'][1:4]
        voxel_sizes = header['pixdim'][1:4]
        voxel_order = ''.join(nib.aff2axcodes(affine))
    elif is_trk:
        affine = header['voxel_to_rasmm']
        dimensions = header['dimensions']
        voxel_sizes = header['voxel_sizes']
        voxel_order = header['voxel_order']
    else:
        raise TypeError('Input reference is not one of the supported format')

    if isinstance(voxel_order, np.bytes_):
        voxel_order = voxel_order.decode('utf-8')

    return affine, dimensions, voxel_sizes, voxel_order


def is_header_compatible(reference_1, reference_2):
    """ Will compare the spatial attribute of 2 references

    Parameters
    ----------
    reference_1 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.
    reference_2 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.

    Returns
    -------
    output : bool
        Does all the spatial attribute match
    """

    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = get_reference_info(
        reference_1)
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = get_reference_info(
        reference_2)

    identical_header = True
    if not np.allclose(affine_1, affine_2):
        logging.error('Affine not equal')
        identical_header = False

    if not np.array_equal(dimensions_1, dimensions_2):
        logging.error('Dimensions not equal')
        identical_header = False

    if not np.allclose(voxel_sizes_1, voxel_sizes_2):
        logging.error('Voxel_size not equal')
        identical_header = False

    if voxel_order_1 != voxel_order_2:
        logging.error('Voxel_order not equal')
        identical_header = False

    return identical_header


def create_tractogram_header(tractogram_type, affine, dimensions, voxel_sizes,
                             voxel_order):
    """ Write a standard trk/tck header from spatial attribute """
    if isinstance(tractogram_type, six.string_types):
        tractogram_type = detect_format(tractogram_type)

    new_header = tractogram_type.create_empty_header()
    new_header[nib.streamlines.Field.VOXEL_SIZES] = tuple(voxel_sizes)
    new_header[nib.streamlines.Field.DIMENSIONS] = tuple(dimensions)
    new_header[nib.streamlines.Field.VOXEL_TO_RASMM] = affine
    new_header[nib.streamlines.Field.VOXEL_ORDER] = voxel_order

    return new_header


def create_nifti_header(affine, dimensions, voxel_sizes):
    """ Write a standard nifti header from spatial attribute """
    new_header = nib.Nifti1Header()
    new_header['srow_x'] = affine[0, 0:4]
    new_header['srow_y'] = affine[1, 0:4]
    new_header['srow_z'] = affine[2, 0:4]
    new_header['dim'][1:4] = dimensions
    new_header['pixdim'][1:4] = voxel_sizes

    return new_header


def _is_data_per_point_valid(streamlines, data):
    """ Verify that the number of item in data is X and that each of these items
        has Y_i items.

        X being the number of streamlines
        Y_i being the number of points on streamlines #i

    Parameters
    ----------
    streamlines : list or ArraySequence
        Streamlines of the tractogram
    data : dict
        Contains the organized point's metadata (hopefully)
    Returns
    -------
    output : bool
        Does all the streamlines and metadata attribute match
    """
    if not isinstance(data, (dict, PerArraySequenceDict)):
        logging.error('data_per_point MUST be a dictionary')
        return False
    elif data == {}:
        return True

    total_point = 0
    total_streamline = 0
    for i in streamlines:
        total_streamline += 1
        total_point += len(i)

    for key in data.keys():
        total_point_entries = 0
        if not len(data[key]) == total_streamline:
            logging.error('Missing entry for streamlines points data (1)')
            return False

        for values in data[key]:
            total_point_entries += len(values)

        if total_point_entries != total_point:
            logging.error('Missing entry for streamlines points data (2)')
            return False

    return True


def _is_data_per_streamline_valid(streamlines, data):
    """ Verify that the number of item in data is X

        X being the number of streamlines

    Parameters
    ----------
    streamlines : list or ArraySequence
        Streamlines of the tractogram
    data : dict
        Contains the organized streamline's metadata (hopefully)
    Returns
    -------
    output : bool
        Does all the streamlines and metadata attribute match
    """
    if not isinstance(data, (dict, PerArrayDict)):
        logging.error('data_per_point MUST be a dictionary')
        return False
    elif data == {}:
        return True

    total_streamline = 0
    for _ in streamlines:
        total_streamline += 1

    for key in data.keys():
        if not len(data[key]) == total_streamline:
            logging.error('Missing entry for streamlines data (3)')
            return False

    return True
