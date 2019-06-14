import numpy as np
from dipy.reconst.opt_msd import MultiShellResponse
from dipy.reconst.csdeconv import auto_response
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.sims.voxel import (single_tensor)
from dipy.reconst import shm
from dipy.data import default_sphere
from dipy.core.gradients import GradientTable
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import dipy.reconst.dti as dti
from dipy.reconst.opt_msd import MultiShellDeconvModel
from dipy.viz import window, actor
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

# static file paths for experiments
fbvals = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.bvals'
fbvecs = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.bvecs'
fdwi = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.nii.gz'
ft1 = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fcf.neuro-anat-t1w.acpc_aligned/t1.nii.gz'

t1, t1_affine = load_nifti(ft1)

dwi, dwi_affine = load_nifti(fdwi)
b0_mask, mask = median_otsu(dwi)
print("Data Loaded!")

"""
Now we will define the other two parameters for the segmentation algorithm.
We will segment three classes, namely corticospinal fluid (CSF), white matter
(WM) and gray matter (GM).
"""
nclass = 3
"""
Then, the smoothness factor of the segmentation. Good performance is achieved
with values between 0 and 0.5.
"""
beta = 0.1

# denoising
sigma = estimate_sigma(t1, True, N=6)

t1[mask == 0] = 0

t1_den = nlmeans(t1, sigma=sigma)

# save_nifti('t1_masked.nii.gz', t1, t1_affine)
# save_nifti('t1_class_masked.nii.gz', PVE, t1_affine)

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# fitting the model with DTI
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(dwi, mask)

# save_nifti('t1_denoised.nii.gz', t1_den, t1_affine)

# getting the mean diffusivities and FAs from DTI
FA = tenfit.fa
MD = tenfit.md

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1_den, nclass,
                                                              beta)

csf = PVE[..., 0]
cgm = PVE[..., 1]

save_nifti('pve.nii.gz', PVE, dwi_affine)

indices_csf = np.where(((FA < 0.2) & (csf > 0.95)))
indices_cgm = np.where(((FA < 0.2) & (cgm > 0.95)))

selected_csf = np.zeros(FA.shape, dtype='bool')
selected_cgm = np.zeros(FA.shape, dtype='bool')

selected_csf[indices_csf] = True
selected_cgm[indices_cgm] = True

csf_md = np.mean(tenfit.md[selected_csf])
cgm_md = np.mean(tenfit.md[selected_cgm])

save_nifti('md.nii.gz', MD, dwi_affine)
save_nifti('fa.nii.gz', FA, dwi_affine)

# generating the autoresponse
dwi[mask == 0] = 0
response, ratio = auto_response(gtab, dwi, roi_radius=10, fa_thr=0.7)
evals_d = response[0]


def sim_response(sh_order=8, bvals=bvals, evals=evals_d, csf_md=csf_md,
                 gm_md=cgm_md):
    bvals = np.array(bvals, copy=True)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    big_sphere = default_sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    A = shm.real_sph_harm(0, 0, 0, 0)

    response = np.empty([len(bvals), len(n) + 2])
    for i, bvalue in enumerate(bvals):
        gtab = GradientTable(big_sphere.vertices * bvalue)
        wm_response = single_tensor(gtab, 1., evals, evecs, snr=None)
        response[i, 2:] = np.linalg.lstsq(B, wm_response)[0]

        response[i, 0] = np.exp(-bvalue * csf_md) / A
        response[i, 1] = np.exp(-bvalue * gm_md) / A

    return MultiShellResponse(response, sh_order, bvals)


response_msd = sim_response(sh_order=8, bvals=bvals, evals=evals_d,
                            csf_md=csf_md, gm_md=cgm_md)

msd_model = MultiShellDeconvModel(gtab, response_msd)

data = dwi[:, :, 68: 68 + 1]
mask_tmp = mask[:, :, 68: 68 + 1]

msd_fit = msd_model.fit(data, mask_tmp)
msd_odf = msd_fit.odf(sphere)
fodf_spheres = actor.odf_slicer(msd_odf, mask=mask_tmp,
                                sphere=sphere, scale=0.001, norm=False,
                                colormap='plasma')
interactive = True
ren = window.Renderer()
ren.add(fodf_spheres)

if interactive:
    window.show(ren)
