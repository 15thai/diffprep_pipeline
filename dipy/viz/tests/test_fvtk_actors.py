import os
import numpy as np

from dipy.viz import actor, window

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory
from dipy.tracking.streamline import center_streamlines, transform_streamlines
from dipy.align.tests.test_streamlinear import fornix_streamlines


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_slicer():

    renderer = window.renderer()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine)
    window.add(renderer, slicer)
    # window.show(renderer)

    # copy pixels in numpy array directly
    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)

    # The slicer can cut directly a smaller part of the image
    slicer.SetDisplayExtent(10, 30, 10, 30, 35, 35)
    slicer.Update()
    renderer.ResetCamera()

    window.add(renderer, slicer)

    # save pixels in png file not a numpy array
    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'slice.png')
        # window.show(renderer)
        arr = window.snapshot(renderer, fname)
        report = window.analyze_snapshot(fname, find_objects=True)
        npt.assert_equal(report.objects, 1)

    npt.assert_raises(ValueError, actor.slicer, np.ones(10))

    renderer.clear()

    rgb = np.zeros((30, 30, 30, 3))
    rgb[..., 0] = 1.
    rgb_actor = actor.slicer(rgb)

    renderer.add(rgb_actor)

    renderer.reset_camera()
    renderer.reset_clipping_range()

    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [True])


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_streamtube_and_line_actors():

    renderer = window.renderer()

    line1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    line2 = line1 + np.array([0.5, 0., 0.])

    lines = [line1, line2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    c = actor.line(lines, colors, linewidth=3)
    window.add(renderer, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    arr = window.snapshot(renderer)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_bundle_maps():

    renderer = window.renderer()
    bundle = fornix_streamlines()
    bundle, shift = center_streamlines(bundle)

    mat = np.array([[1, 0, 0, 100],
                    [0, 1, 0, 100],
                    [0, 0, 1, 100],
                    [0, 0, 0, 1.]])

    bundle = transform_streamlines(bundle, mat)

    # metric = np.random.rand(*(200, 200, 200))
    metric = 100 * np.ones((200, 200, 200))

    # add lower values
    metric[100, :, :] = 100 * 0.5

    # create a nice orange-red colormap
    lut = actor.colormap_lookup_table(scale_range=(0., 100.),
                                      hue_range=(0., 0.1),
                                      saturation_range=(1, 1),
                                      value_range=(1., 1))

    line = actor.line(bundle, metric, linewidth=0.1, lookup_colormap=lut)
    window.add(renderer, line)
    window.add(renderer, actor.scalar_bar(lut, ' '))

    report = window.analyze_renderer(renderer)

    npt.assert_almost_equal(report.actors, 1)
    # window.show(renderer)

    renderer.clear()

    nb_points = np.sum([len(b) for b in bundle])
    values = 100 * np.random.rand(nb_points)
    # values[:nb_points/2] = 0

    line = actor.streamtube(bundle, values, linewidth=0.1, lookup_colormap=lut)
    renderer.add(line)
    # window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')

    renderer.clear()

    colors = np.random.rand(nb_points, 3)
    # values[:nb_points/2] = 0

    line = actor.line(bundle, colors, linewidth=2)
    renderer.add(line)
    # window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')
    # window.show(renderer)

    arr = window.snapshot(renderer)
    report2 = window.analyze_snapshot(arr)
    npt.assert_equal(report2.objects, 1)


if __name__ == "__main__":

    npt.run_module_suite()
