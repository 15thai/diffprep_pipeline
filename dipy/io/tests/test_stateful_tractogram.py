import json
import os

from nibabel.tmpdirs import InTemporaryDirectory
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose

from dipy.data import fetch_gold_standard_io
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram

from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury')

filepath_dix = {}
files, folder = fetch_gold_standard_io()
for filename in files:
    filepath_dix[filename] = os.path.join(folder, filename)

with open(filepath_dix['points_data.json']) as json_file:
    points_data = dict(json.load(json_file))

with open(filepath_dix['streamlines_data.json']) as json_file:
    streamlines_data = dict(json.load(json_file))


# UNIT TESTS
def trk_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def tck_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


@npt.dec.skipif(not have_fury)
def fib_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def dpy_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def trk_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def tck_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


@npt.dec.skipif(not have_fury)
def fib_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def dpy_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def trk_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def tck_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


@npt.dec.skipif(not have_fury)
def fib_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def dpy_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.data, atol=1e-3, rtol=1e-6)


def switch_voxel_sizes_from_rasmm():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    sft_switch = StatefulTractogram(sft.streamlines,
                                    filepath_dix['gs_3mm.nii'],
                                    Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])

    sft_switch.to_rasmm()
    assert_allclose(tmp_points_rasmm,
                    sft_switch.streamlines.data, atol=1e-3, rtol=1e-6)

    sft_switch.to_voxmm()
    assert_allclose(tmp_points_voxmm,
                    sft_switch.streamlines.data, atol=1e-3, rtol=1e-6)


def switch_voxel_sizes_from_voxmm():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    sft_switch = StatefulTractogram(sft.streamlines,
                                    filepath_dix['gs_3mm.nii'],
                                    Space.VOXMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])

    sft_switch.to_rasmm()
    assert_allclose(tmp_points_rasmm,
                    sft_switch.streamlines.data, atol=1e-3, rtol=1e-6)

    sft_switch.to_voxmm()
    assert_allclose(tmp_points_voxmm,
                    sft_switch.streamlines.data, atol=1e-3, rtol=1e-6)


def trk_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.trk')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.trk', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.data,
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.trk')


def tck_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.tck')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.tck', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.data,
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.tck')


@npt.dec.skipif(not have_fury)
def fib_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.fib')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.fib', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.data,
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.fib')


def dpy_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.dpy')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.dpy', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.data,
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.dpy')


def iterative_to_vox_transformation():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    for _ in range(1000):
        sft.to_vox()
        sft.to_rasmm()
        assert_allclose(tmp_points_rasmm,
                        sft.streamlines.data, atol=1e-3, rtol=1e-6)


def iterative_to_voxmm_transformation():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    for _ in range(1000):
        sft.to_voxmm()
        sft.to_rasmm()
        assert_allclose(tmp_points_rasmm,
                        sft.streamlines.data, atol=1e-3, rtol=1e-6)


def shift_corner_from_rasmm():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_1.to_corner()
    bbox_1 = sft_1.compute_bounding_box()

    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.RASMM)
    sft_2.to_corner()
    sft_2.to_vox()
    bbox_2 = sft_2.compute_bounding_box()

    assert_allclose(bbox_1, bbox_2, atol=1e-3, rtol=1e-6)


def shift_corner_from_voxmm():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_1.to_corner()
    bbox_1 = sft_1.compute_bounding_box()

    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOXMM)
    sft_2.to_corner()
    sft_2.to_vox()
    bbox_2 = sft_2.compute_bounding_box()

    assert_allclose(bbox_1, bbox_2, atol=1e-3, rtol=1e-6)


def iterative_shift_corner():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()

    for _ in range(1000):
        sft._shift_voxel_origin()

    assert_allclose(sft.get_streamlines_copy(),
                    tmp_streamlines, atol=1e-3, rtol=1e-6)


def replace_streamlines():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()[::-1]

    try:
        sft.streamlines = tmp_streamlines
        return True
    except (TypeError, ValueError):
        return False


def subsample_streamlines():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()[0:8]

    try:
        sft.streamlines = tmp_streamlines
        return False
    except (TypeError, ValueError):
        return True


def reassign_both_data_sep_to_empty():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)

    try:
        sft.data_per_point = {}
        sft.data_per_streamline = {}
    except (TypeError, ValueError):
        return False

    return sft.data_per_point == {} and \
        sft.data_per_streamline == {}


def reassign_both_data_sep():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)

    try:
        sft.data_per_point = points_data
        sft.data_per_streamline = streamlines_data
    except (TypeError, ValueError):
        return False

    return True


def bounding_bbox_valid(shift):
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          shifted_origin=shift, bbox_valid_check=False)

    return sft.is_bbox_in_vox_valid()


def remove_invalid_streamlines(resize):
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    if resize:
        sft._dimensions[2] = 5

    sft.remove_invalid_streamlines()
    return len(sft)


def random_point_color():
    np.random.seed(0)
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])

    random_colors = np.random.randint(0, 255, (13, 8, 3))
    coloring_dict = {}
    coloring_dict['colors'] = random_colors

    try:
        sft.data_per_point = coloring_dict
        with InTemporaryDirectory():
            save_tractogram(sft, 'random_points_color.trk')
        return True
    except (TypeError, ValueError):
        return False


def random_point_gray():
    np.random.seed(0)
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])

    random_colors = np.random.randint(0, 255, (13, 8, 1))
    coloring_dict = {}
    coloring_dict['color_x'] = random_colors
    coloring_dict['color_y'] = random_colors
    coloring_dict['color_z'] = random_colors

    try:
        sft.data_per_point = coloring_dict
        with InTemporaryDirectory():
            save_tractogram(sft, 'random_points_gray.trk')
        return True
    except ValueError:
        return False


def random_streamline_color():
    np.random.seed(0)
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])

    uniform_colors_x = np.random.randint(0, 255, (13, 1))
    uniform_colors_y = np.random.randint(0, 255, (13, 1))
    uniform_colors_z = np.random.randint(0, 255, (13, 1))
    uniform_colors_x = np.expand_dims(
        np.repeat(uniform_colors_x, 8, axis=1), axis=-1)
    uniform_colors_y = np.expand_dims(
        np.repeat(uniform_colors_y, 8, axis=1), axis=-1)
    uniform_colors_z = np.expand_dims(
        np.repeat(uniform_colors_z, 8, axis=1), axis=-1)

    coloring_dict = {}
    coloring_dict['color_x'] = uniform_colors_x
    coloring_dict['color_y'] = uniform_colors_y
    coloring_dict['color_z'] = uniform_colors_z

    try:
        sft.data_per_point = coloring_dict
        with InTemporaryDirectory():
            save_tractogram(sft, 'random_streamlines_color.trk')
        return True
    except (TypeError, ValueError):
        return False


def out_of_grid(value):
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])
    sft.to_vox()
    tmp_streamlines = list(sft.get_streamlines_copy())
    tmp_streamlines[0] += value

    try:
        sft.streamlines = tmp_streamlines
        return sft.is_bbox_in_vox_valid()
    except (TypeError, ValueError):
        return True


def test_iterative_transformation():
    iterative_to_vox_transformation()
    iterative_to_voxmm_transformation()


def test_iterative_saving_loading():
    trk_iterative_saving_loading()
    tck_iterative_saving_loading()
    fib_iterative_saving_loading()
    dpy_iterative_saving_loading()


def test_equal_in_vox_space():
    trk_equal_in_vox_space()
    tck_equal_in_vox_space()
    fib_equal_in_vox_space()
    dpy_equal_in_vox_space()


def test_equal_in_rasmm_space():
    trk_equal_in_rasmm_space()
    tck_equal_in_rasmm_space()
    fib_equal_in_rasmm_space()
    dpy_equal_in_rasmm_space()


def test_equal_in_voxmm_space():
    trk_equal_in_voxmm_space()
    tck_equal_in_voxmm_space()
    fib_equal_in_voxmm_space()
    dpy_equal_in_voxmm_space()


def test_switch_reference():
    switch_voxel_sizes_from_rasmm()
    switch_voxel_sizes_from_voxmm()


def test_shifting_corner():
    shift_corner_from_rasmm()
    shift_corner_from_voxmm()
    iterative_shift_corner()


def test_replace_streamlines():
    # First two is expected to fail
    if not subsample_streamlines():
        raise AssertionError()
    if not replace_streamlines():
        raise AssertionError()
    if not reassign_both_data_sep():
        raise AssertionError()
    if not reassign_both_data_sep_to_empty():
        raise AssertionError()


def test_bounding_box():
    # First is expected to fail
    if not bounding_bbox_valid(False):
        raise AssertionError()
    if bounding_bbox_valid(True):
        raise AssertionError()
    # Last two are expected to fail
    if out_of_grid(100):
        raise AssertionError()
    if out_of_grid(-100):
        raise AssertionError()


def test_invalid_streamlines():
    if not remove_invalid_streamlines(True) == 5:
        raise AssertionError()
    if not remove_invalid_streamlines(False) == 13:
        raise AssertionError()


def test_trk_coloring():
    if not random_streamline_color():
        raise AssertionError()
    if not random_point_gray():
        raise AssertionError()
    if not random_point_color():
        raise AssertionError()


if __name__ == '__main__':
    npt.run_module_suite()
