# from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.viz.colormap import line_colors
from dipy.viz.utils import numpy_to_vtk_points, numpy_to_vtk_colors
from dipy.viz.utils import set_input, trilinear_interp
from dipy.core.ndindex import ndindex

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

if have_vtk:

    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()


def slice(data, affine=None, opacity=1., lookup_colormap=None):
    """ Cuts 3D volumes into images

    Parameters
    ----------
    data : array, shape (X, Y, Z)
        A 3D volume as a numpy array.
    affine : array, shape (3, 3)
        Grid to space (usually RAS 1mm) transformation matrix
    opacity : float
        Opacity of 0 means completely transparent and 1 completely visible.
    lookup_colormap : vtkLookupTable
        If None (default) then a grayscale map is created.

    Returns
    -------
    image_actor : ImageActor
        An object that is capable of displaying different parts of the volume
        as slices. The key method of this object is ``display_extent`` where
        one can input grid coordinates and display the slice in space (or grid)
        coordinates as calculated by the affine parameter.

    """

    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    vol = vol.astype('uint8')

    im = vtk.vtkImageData()
    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()
    I, J, K = vol.shape[:3]
    im.SetDimensions(I, J, K)
    voxsz = (1., 1., 1)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    # copy data
    for index in ndindex(vol.shape):
        i, j, k = index
        im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    if affine is None:
        affine = np.eye(4)

    # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy((
        affine[0][0], affine[0][1], affine[0][2], affine[0][3],
        affine[1][0], affine[1][1], affine[1][2], affine[1][3],
        affine[2][0], affine[2][1], affine[2][2], affine[2][3],
        affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = vtk.vtkImageReslice()
    image_resliced.SetInputData(im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()
    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    if lookup_colormap is None:
        # Create a black/white lookup table.
        lut = colormap_lookup_table((0, 255), (0, 0), (0, 0), (0, 1))
    else:
        lut = lookup_colormap

    x1, x2, y1, y2, z1, z2 = im.GetExtent()

    plane_colors = vtk.vtkImageMapToColors()
    plane_colors.SetLookupTable(lut)
    plane_colors.SetInputConnection(image_resliced.GetOutputPort())
    plane_colors.Update()

    class ImageActor(vtk.vtkImageActor):

        def input_connection(self, output_port):
            self.GetMapper().SetInputConnection(output_port)

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            self.SetDisplayExtent(x1, x2, y1, y2, z1, z2)
            self.Update()

        def opacity(self, value):
            self.GetProperty().SetOpacity(value)

    image_actor = ImageActor()
    image_actor.input_connection(plane_colors.GetOutputPort())
    image_actor.display_extent(x1, x2, y1, y2, z2/2, z2/2)
    image_actor.opacity(opacity)

    return image_actor


def streamtube(lines, colors=None, opacity=1, linewidth=0.01, tube_sides=9,
               lod=True, lod_points=10 ** 4, lod_points_size=3,
               spline_subdiv=None, lookup_colormap=None):
    """ Uses streamtubes to visualize polylines

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), tuple (3,), array (K,), array (X, Y, Z)

    opacity : float
    linewidth : float
    tube_sides : int
    lod : bool
        use vtkLODActor rather than vtkActor
    lod_points : int
        number of points to be used when LOD is in effect
    lod_points_size : int
        size of points when lod is in effect
    spline_subdiv : int
        number of splines subdivision to smooth streamtubes
    lookup_colormap : bool
        add a default lookup table to the colormap

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors=np.random.rand(2, 3)
    >>> c=fvtk.streamtube(lines, colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)

    Notes
    -----
    Streamtubes can be heavy on GPU when loading many streamlines and
    therefore, you may experience slow rendering time depending on system GPU.
    A solution to this problem is to reduce the number of points in each
    streamline. In Dipy we provide an algorithm that will reduce the number of
    points on the straighter parts of the streamline but keep more points on
    the curvier parts. This can be used in the following way

    from dipy.tracking.distances import approx_polygon_track
    lines = [approx_polygon_track(line, 0.2) for line in lines]

    Alternatively we suggest using the ``line`` actor which is much more
    efficient.

    See Also
    --------
    dipy.viz.fvtk.line
    """
    # Poly data with lines and colors
    poly_data, is_colormap = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # Set Normals
    poly_normals = set_input(vtk.vtkPolyDataNormals(), next_input)
    poly_normals.ComputeCellNormalsOn()
    poly_normals.ComputePointNormalsOn()
    poly_normals.ConsistencyOn()
    poly_normals.AutoOrientNormalsOn()
    poly_normals.Update()
    next_input = poly_normals.GetOutputPort()

    # Spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0):
        spline_filter = set_input(vtk.vtkSplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    # Add thickness to the resulting lines
    tube_filter = set_input(vtk.vtkTubeFilter(), next_input)
    tube_filter.SetNumberOfSides(tube_sides)
    tube_filter.SetRadius(linewidth)
    # tube_filter.SetVaryRadiusToVaryRadiusByScalar()
    tube_filter.CappingOn()
    tube_filter.Update()
    next_input = tube_filter.GetOutputPort()

    # Poly mapper
    poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("Colors")
    poly_mapper.GlobalImmediateModeRenderingOn()
    # poly_mapper.SetColorModeToMapScalars()
    poly_mapper.Update()

    # Color Scale with a lookup table
    if is_colormap:
        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()
        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = vtk.vtkActor()

    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetAmbient(0.1)  # .3
    actor.GetProperty().SetDiffuse(0.15)  # .3
    actor.GetProperty().SetSpecular(0.05)  # .3
    actor.GetProperty().SetSpecularPower(6)
    # actor.GetProperty().SetInterpolationToGouraud()
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetOpacity(opacity)

    return actor


def line(lines, colors=None, opacity=1, linewidth=1,
         spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3,
         lookup_colormap=None):
    """ Create an actor for one or more lines.

    Parameters
    ------------
    lines :  list of arrays

    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g.

    opacity : float, optional

    linewidth : float, optional
        Line thickness.
    spline_subdiv : int, optional
        number of splines subdivision to smooth streamtubes
    lookup_colormap : bool, optional
        add a default lookup table to the colormap

    Returns
    ----------
    v : vtkActor object
        Line.

    Examples
    ----------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10,3), np.random.rand(20,3)]
    >>> colors=np.random.rand(2,3)
    >>> c=fvtk.line(lines, colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)
    """
    # Poly data with lines and colors
    poly_data, is_colormap = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # use spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0):
        spline_filter = set_input(vtk.vtkSplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("Colors")
    # poly_mapper.SetColorModeToMapScalars()
    poly_mapper.Update()

    # Color Scale with a lookup table
    if is_colormap:

        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()

        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = vtk.vtkActor()

    actor = vtk.vtkActor()
    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def lines_to_vtk_polydata(lines, colors=None):
    """ Create a vtkPolyData with lines and colors

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), list of arrays, tuple (3,), array (K,), None
        If None then a standard orientation colormap is used for every line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, ) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    Returns
    -------
    poly_data :  vtkPolyData
    is_colormap : bool, true if the input color array was a colormap
    """

    # Get the 3d points_array
    points_array = np.vstack(lines)

    nb_lines = len(lines)
    nb_points = len(points_array)

    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    points_per_line = np.zeros([nb_lines], np.int64)
    current_position = 0
    for i in lines_range:
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position

    lines_array = np.array(lines_array)

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.GetData().DeepCopy(numpy_support.numpy_to_vtk(lines_array))
    vtk_lines.SetNumberOfCells(len(lines))

    is_colormap = False
    # Get colors_array (reformat to have colors for each points)
    #           - if/else tested and work in normal simple case
    if colors is None:  # set automatic rgb colors
        cols_arr = line_colors(lines)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
    else:
        cols_arr = np.asarray(colors)
        if cols_arr.dtype == np.object:  # colors is a list of colors
            vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
        else:
            if len(cols_arr) == nb_points:
                vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

            elif cols_arr.ndim == 1:  # the same colors for all points
                vtk_colors = numpy_to_vtk_colors(
                    np.tile(255 * cols_arr, (nb_points, 1)))

            elif cols_arr.ndim == 2:  # map color to each line
                colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
            else:  # colormap
                #  get colors for each vertex
                cols_arr = trilinear_interp(cols_arr, points_array)
                vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

    vtk_colors.SetName("Colors")

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    poly_data.GetPointData().SetScalars(vtk_colors)
    return poly_data, is_colormap


def colormap_lookup_table(scale_range=(0, 1), hue_range=(0.8, 0),
                          saturation_range=(1, 1), value_range=(0.8, 0.8)):
    """ Lookup table for the colormap

    Parameters
    ----------
    scale_range : tuple
        It can be anything e.g. (0, 1) or (0, 255). Usually it is the mininum
        and maximum value of your data.
    hue_range : tuple of floats
        HSV values (min 0 and max 1)
    saturation_range : tuple of floats
        HSV values (min 0 and max 1)
    value_range : tuple of floats
        HSV value (min 0 and max 1)

    Returns
    -------
    lookup_table : vtkLookupTable

    """
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetRange(scale_range)
    lookup_table.SetTableRange(scale_range)

    lookup_table.SetHueRange(hue_range)
    lookup_table.SetSaturationRange(saturation_range)
    lookup_table.SetValueRange(value_range)

    lookup_table.Build()
    return lookup_table


def scalar_bar(lookup_table, title=" "):
    """ Default Scalar bar actor for the colormap
    """
    lookup_table_copy = vtk.vtkLookupTable()
    # Deepcopy the lookup_table because sometimes vtkPolyDataMapper deletes it
    lookup_table_copy.DeepCopy(lookup_table)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetTitle(title)
    scalar_bar.SetLookupTable(lookup_table_copy)
    scalar_bar.SetNumberOfLabels(6)

    return scalar_bar
