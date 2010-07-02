""" Testing DTI

"""

import numpy as np

import dipy.core.dti as dti
#for reading in nifti test data
import nibabel as nib
from dipy.io.bvectxt import read_bvec_file

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_almost_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

import os

@parametric
def test_tensor_scalar_attributes():
    """
    Tests that the tensor class scalar attributes (FA, ADC, etc...) are
    calculating properly.

    """
    ### DEFINING ANALYTICAL VALUES ###
    evals = np.array([2., 1., 0.])
    a = 1. / np.sqrt(2)
    #evec[:,j] is pair with eval[j]
    evecs = np.array([[a, 0, -a], [a, 0, a], [0, 1., 0]]) 
    D = np.array([[1., 1., 0], [1., 1., 0], [0, 0, 1.]])
    FA = np.sqrt(1./2*(1+4+1)/(1+4+0)) # 0.7745966692414834
    MD = 1.
    ADC = 1.

    ### CALCULATE ESTIMATE VALUES ###
    dummy_data = np.zeros((1,10)) #single voxel
    dummy_gtab = np.zeros((10,3))
    dummy_bval = np.zeros((10,))
    tensor = dti.Tensor(dummy_data,dummy_bval,dummy_gtab)
    tensor.evals = evals.reshape((-1,)+evals.shape)
    tensor.evecs = evecs.reshape((-1,)+evecs.shape)
    
    ### TESTS ###
    yield assert_almost_equal(np.abs(np.dot(evecs[:, 2], 
                tensor[0].evecs[:, 2].T)), 1., 
                msg = "Calculation of third eigenvector is not right")
    yield assert_array_almost_equal(D, tensor[0].D, err_msg = "Recovery of self diffusion tensor from eig not adaquate")
    yield assert_almost_equal(ADC, tensor[0].adc(), msg = "Calculation of ADC of self diffusion tensor is not adequate")
    yield assert_almost_equal(FA, tensor[0].fa(), msg = "Calculation of FA of self diffusion tensor is not adequate")
    yield assert_almost_equal(MD, tensor[0].md(), msg = "Calculation of MD of self diffusion tensor is not adequate")

    
    #yield assert_equal(m_list.shape, n_list.shape)
    #yield assert_equal(m_list.ndim, 2)
    #yield assert_equal(m_list.shape, (45,1))
    #yield assert_true(np.all(np.abs(m_list) <= n_list))
    #yield assert_array_equal(n_list % 2, 0)
    #yield assert_raises(ValueError, qball.sph_harm_ind_list, 1)

@parametric
def test_WLS_fit():
    """
    Tests the WLS fitting function to see if it returns the correct
    eigenvalues and eigenvectors.

    Uses data/55dir_grad.bvec as the gradient table and 3by3by56.nii 
    as the data.

    """
    
    ### Defining Test Voxel (avoid nibabel dependency) ###

    #Recall: D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_0)] and D ~ 10^-4 mm^2 /s 
    D = np.array([1., 1., 1., 1., 0., 0., np.log(1000) * 10.**4]) * 10.**-4
    evals = np.array([2., 1., 0.]) * 10.**-4
    tensor = np.empty((3,3))
    tensor[0, 0] = D[0]
    tensor[1, 1] = D[1]
    tensor[2, 2] = D[2]
    tensor[0, 1] = tensor[1, 0] = D[3]
    tensor[0, 2] = tensor[2, 0] = D[4]
    tensor[1, 2] = tensor[2, 1] = D[5]
    #Design Matrix
    gtab, bval = read_bvec_file(os.path.join(os.path.dirname(__file__),
                    'data','55dir_grad.bvec'))
    X = dti.design_matrix(gtab, bval)
    #Signals
    Y = np.exp(np.dot(X,D))
    Y.shape = (-1,) + Y.shape
    
    ### Testing WLS Fit on Single Voxel ###
    #Estimate tensor from test signals
    tensor_est = dti.Tensor(Y,bval,gtab.T)
    yield assert_array_almost_equal(tensor_est[0].evals, evals)
    yield assert_array_almost_equal(tensor_est[0].D, tensor,err_msg= "Calculation of tensor from Y does not compare to analytical solution")

