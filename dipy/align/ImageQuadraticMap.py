# Author Okan, Anh Thai, Britney
# from Anh_ImageQuadraticMap_2.py in lab 
import numpy as np
import numpy.linalg as npl
import scipy.ndimage as ndimage
import math
from dipy.align.imwarp import get_direction_and_spacings
from dipy.align.parzenhist import (ParzenJointHistogram,
                                   compute_parzen_mi)
from dipy.align.scalespace import  IsotropicScaleSpace
from dipy.align.quadratictransform import quadratic_transform
from dipy.align import VerbosityLevels
from dipy.align.DIFFPREPOptimizer import DIFFPREPOptimizer

_interp_options = ['nearest', 'linear', 'quadratic']
_NQUADPARAMS = 21

class QuadraticInversionError(Exception):
    pass
class QuadraticInvalidValuesError(Exception):
    pass
####################################################
class QuadraticMap (object):

    def __init__(self, phase, QuadraticParams = None,
                 domain_grid_shape = None, domain_grid2world = None,
                 codomain_grid_shape = None, codomain_grid2world = None, registration_type = None):

        # Return self.phase and self.phase_id
        self.set_Phase(phase)

        # Setting Quadratic Parameters with length = 21
        if QuadraticParams is None:
            QuadraticParams = np.zeros(_NQUADPARAMS)
            QuadraticParams[6 + self.phase_id] = 1
        self.set_QuadraticParams(QuadraticParams)

        self.ComputeMatrix()

        # Setting if Quadratic Transform is activate - Do CUBIC INTERPOLATION
        if np.sum(self.QuadraticParams[14:21]) != 0:
            self.do_cubic = 1
        else:
            self.do_cubic = 0

        self.domain_shape = domain_grid_shape
        self.domain_grid2world = domain_grid2world
        self.codomain_shape = codomain_grid_shape
        self.codomain_grid2world = codomain_grid2world

    def set_Phase(self, phase):
        """ Sets the phase encoding direction of dMRI acquisition
        """
        if phase is None:
            raise ValueError("Phase information should be entered as vertical or horizontal or slice")
        if phase not in ['vertical', 'horizontal', 'slice']:
            raise ValueError("Phase information should be one of \'vertical\' or \'horizontal\' or \'slice\'")

        self.phase = phase
        if phase == "vertical":
            self.phase_id = 1
        elif phase == "horizontal":
            self.phase_id = 0
        else:
            self.phase_id = 2

    def ComputeMatrix(self, Angles = None):
        if Angles is None:
            m_AngleX = self.QuadraticParams[3]
            m_AngleY = self.QuadraticParams[4]
            m_AngleZ = self.QuadraticParams[5]
        else:
            [m_AngleX, m_AngleY, m_AngleZ] = Angles

        cx = math.cos(m_AngleX)
        sx = math.sin(m_AngleX)
        cy = math.cos(m_AngleY)
        sy = math.sin(m_AngleY)
        cz = math.cos(m_AngleZ)
        sz = math.sin(m_AngleZ)

        RotationX = np.zeros((3, 3))
        RotationX[0, 0] = 1
        RotationX[1, 1] = cx
        RotationX[1, 2] = -sx
        RotationX[2, 1] = sx
        RotationX[2, 2] = cx

        RotationY = np.zeros((3, 3))
        RotationY[0, 0] = cy
        RotationY[0, 2] = sy
        RotationY[1, 1] = 1
        RotationY[2, 0] = -sy
        RotationY[2, 2] = cy

        RotationZ = np.zeros((3, 3))
        RotationZ[0, 0] = cz
        RotationZ[0, 1] = -sz
        RotationZ[1, 0] = sz
        RotationZ[1, 1] = cz
        RotationZ[2, 2] = 1
        self.Matrix = RotationZ.dot(RotationY.dot(RotationX))

    def set_QuadraticParams(self, QuadraticParams):
        try:
            self.QuadraticParams = np.array(QuadraticParams)
        except:
            raise TypeError('Input must be type ndarray, or be convertible to one.')

        if self.QuadraticParams.shape[0] != _NQUADPARAMS:
            raise QuadraticInversionError('Incorrect Number of Parameters')

        if not np.all(np.isfinite(QuadraticParams)):
            raise QuadraticInvalidValuesError('Quadratic transform contains invalid elements')

    def get_QuadraticParams(self):
        return self.QuadraticParams.copy()

    def transform(self, image, QuadraticParams = None,
                  image_grid2world = None, sampling_grid_shape = None, sampling_grid2world = None, interpolation_method = 'quadratic'):
        if sampling_grid_shape is None:
            sampling_grid_shape = self.domain_shape
        dim = len(sampling_grid_shape)
        shape = np.array(sampling_grid_shape, dtype=np.int32)

        if sampling_grid2world is None:
            sampling_grid2world = self.domain_grid2world
            if sampling_grid2world is None:
                sampling_grid2world = np.eye(dim + 1)

        if image_grid2world is None:
            image_grid2world = self.codomain_grid2world
            if image_grid2world is None:
                image_grid2world = np.eye(dim + 1)

        if QuadraticParams is None:
            quad = self.QuadraticParams
        else:
            quad = QuadraticParams

        image_transformed = quadratic_transform(image, shape, quad, self.Matrix, sampling_grid2world,
                                               image_grid2world, self.phase_id, self.do_cubic, interpolation_method)
        return np.array(image_transformed)

class MutualInformationMetric(object):
    def __init__(self, phase, nbins = 100, starting_QuadraticParams = None):
        self.phase = phase
        self.histogram = ParzenJointHistogram(nbins)
        self.metric_val = None
        self.metric_grad = None

        self.starting_QuadraticParams = starting_QuadraticParams
        self.QuadraticMap_ = QuadraticMap(phase)

    def setup(self, transform, static, moving, static_grid2world=None,
              moving_grid2world=None, starting_QuadraticParams=None):

        self.dim = len(static.shape)
        if moving_grid2world is None:
            moving_grid2world = np.eye(self.dim + 1)
        if static_grid2world is None:
            static_grid2world = np.eye(self.dim + 1)
        self.transform = transform
        self.static = np.array(static).astype(np.float64)
        self.moving = np.array(moving).astype(np.float64)
        self.static_grid2world = static_grid2world
        self.static_world2grid = npl.inv(static_grid2world)
        self.moving_grid2world = moving_grid2world
        self.moving_world2grid = npl.inv(moving_grid2world)
        self.static_direction, self.static_spacing = \
            get_direction_and_spacings(static_grid2world, self.dim)
        self.moving_direction, self.moving_spacing = \
            get_direction_and_spacings(moving_grid2world, self.dim)

        self.QuadraticMap_ = QuadraticMap(self.phase, QuadraticParams= starting_QuadraticParams,
                                         domain_grid_shape=  static.shape, domain_grid2world= static_grid2world,
                                         codomain_grid_shape= moving.shape, codomain_grid2world= moving_grid2world)
        self.histogram.setup(self.static, self.moving)

    def _update_histogram(self):
        static_values = self.static
        moving_values = self.QuadraticMap_.transform(self.moving)
        self.histogram.update_pdfs_dense(static_values, moving_values)
        return static_values, moving_values

    def _update_mutual_information(self, QuadraticParams):
        self.QuadraticMap_.set_QuadraticParams(QuadraticParams)
        static_values, moving_values = self._update_histogram()

        H = self.histogram  # Shortcut to `self.histogram`
        grad = None
        # Call the cythonized MI computation with self.histogram fields
        self.metric_val = compute_parzen_mi(H.joint, H.joint_grad, H.smarginal, H.mmarginal, grad)

    def distance(self, QuadraticParams):
        try:
            self._update_mutual_information(QuadraticParams)

        except (QuadraticInversionError, QuadraticInvalidValuesError):
            return np.inf
        return -1 *self.metric_val

class QuadraticRegistration(object):
    def __init__(self, phase, initial_QuadraticParams = None,
                 gradients_params = None,
                 metric = None, levels = None,
                 sigmas = None, factors = None, registration_type=None,
                 options=None,
                 verbosity=VerbosityLevels.STATUS):
        if phase not in ['vertical', 'horizontal', 'slice']:
            raise ValueError("Phase information should be one of \'vertical\' or \'horizontal\' or \'slice\'")
        self.phase = phase
        if phase == "vertical":
            self.phase_id = 1
        elif phase == "horizontal":
            self.phase_id = 0
        else:
            self.phase_id = 2

        self.initial_QuadraticParams = initial_QuadraticParams
        self.grad_params = gradients_params
        self.metric = metric
        if metric is None:
            self.metric = MutualInformationMetric(phase = self.phase, starting_QuadraticParams = initial_QuadraticParams)

        if levels is None:
            levels = 3
        self.levels = levels
        if self.levels == 0:
            raise ValueError('The iterations sequence cannot be empty')
        self.options = options

        if registration_type is None:
            registration_type = 'quadratic'

        self.optimization_flags = np.zeros(_NQUADPARAMS)
        if registration_type == 'translation':
            self.optimization_flags[0:3] = 1
        elif registration_type == 'rotation':
            self.optimization_flags[3:6] = 1
        elif registration_type == 'rigid':
            self.optimization_flags[0:6] = 1
        elif registration_type == 'eddy_only':
            self.optimization_flags[6:14] = 1
        elif registration_type == 'quadratic':
            self.optimization_flags[0:14] = 1
        elif registration_type == 'cubic':
            self.optimization_flags[:] = 1
        else:
            raise ValueError('Invalid registration type')

        if factors is None:
            factors = [4, 2, 1]
        if sigmas is None:
            sigmas = [1., 0.25, 0]
        self.factors = factors
        self.sigmas = sigmas
        self.verbosity = verbosity

    def set_optimizationflags (self, new_optimizationflags):
        self.optimization_flags = new_optimizationflags

    def optimize(self, static, moving, phase=None,
                 static_grid2world=None, moving_grid2world=None,
                 grad_params=None):

        if grad_params is not None:
            self.grad_params = grad_params
        self.dim = 3
        if phase is not None:
            self.phase = phase
        # Extract information from Affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_grid2world, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_grid2world, self.dim)

        # This functions are already remodified - No Normalization
        self.moving_ss = IsotropicScaleSpace(moving, self.factors,
                                             self.sigmas,
                                             moving_grid2world,
                                             moving_spacing, False)

        self.static_ss = IsotropicScaleSpace(static, self.factors,
                                             self.sigmas,
                                             static_grid2world,
                                             static_spacing, False)

        # Multi-resolution iterations
        original_static_shape = self.static_ss.get_image(0).shape
        original_static_grid2world = self.static_ss.get_affine(0)
        original_moving_shape = self.moving_ss.get_image(0).shape
        original_moving_grid2world = self.moving_ss.get_affine(0)

        Quadratic_map = transform_centers_of_mass(static,moving,self.phase,
                                                  static_grid2world,
                                                  moving_grid2world)

        if self.initial_QuadraticParams is not None:
            QuadraticParams = Quadratic_map.get_QuadraticParams()
            QuadraticParams[3:6] = self.initial_QuadraticParams[3:6]
            Quadratic_map.set_QuadraticParams(QuadraticParams)

        for level in range(self.levels -1 , -1, -1):
            self.current_level = level
            if self.verbosity >= VerbosityLevels.STATUS:
                print('Optimizing level %d' % level)

            # Resample the smooth static image to the shape of this level
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_grid2world = self.static_ss.get_affine(level)
            smooth_static = self.static_ss.get_image(level)

            id_Quadratic_map = QuadraticMap( phase, None,
                                            current_static_shape,
                                            current_static_grid2world,
                                            original_static_shape,
                                            original_static_grid2world)

            current_static = id_Quadratic_map.transform(smooth_static)

            # The moving image is full resolution
            current_moving_grid2world = original_moving_grid2world
            current_moving = self.moving_ss.get_image(level)

            # Prepare the metric for iterations at this resolution
            self.metric.setup(Quadratic_map, current_static, current_moving,
                              current_static_grid2world,
                              current_moving_grid2world)

            opt = DIFFPREPOptimizer(self.metric, Quadratic_map.get_QuadraticParams(),
                                                            self.optimization_flags, self.grad_params)
            params = opt.xopt
            self.current_level_cost = opt.CurrentCost

            Quadratic_map.set_QuadraticParams(params)
        return Quadratic_map

def transform_centers_of_mass(static, moving, phase, static_grid2world = None, moving_grid2world = None):

    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = ndimage.measurements.center_of_mass(np.array(static))
    c_static = static_grid2world.dot(c_static + (1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_grid2world.dot(c_moving + (1,))

    params = np.zeros(_NQUADPARAMS)
    if phase == "vertical":
        params[7] = 1
    elif phase == "horizontal":
        params[6] = 1
    else:
        params[8] = 1

    params[0:3] = (c_moving - c_static)[:dim]
    Quadratic_map = QuadraticMap( phase,params,
                                 static.shape, static_grid2world,
                                 moving.shape, moving_grid2world)
    return Quadratic_map
