"""
=============================================================
An introduction to the Deterministic Maximum Direction Getter
=============================================================

Deterministic maximum direction getter is the deterministic version of the
probabilistic direction getter. It can be used with the same local models
and has the same parameters. Deterministic maximum fiber tracking follows
the trajectory of the most probable pathway within the tracking constraint
(e.g. max angle). In other words, it follows the direction with the highest
probability from a distribution, as opposed to the probabilistic direction
getter which draws the direction from the distribution. Therefore, the maximum
deterministic direction getter is equivalent to the probabilistic direction
getter returning always the maximum value of the distribution.

Deterministic maximum fiber tracking is an alternative to EuDX deterministic
tractography and unlike EuDX does not follow the peaks of the local models but
uses the entire orientation distributions.

This example is an extension of the :ref:`example_tracking_probabilistic`
example. We begin by loading the data, fitting a Constrained Spherical
Deconvolution (CSD) reconstruction model for the tractography and fitting
the constant solid angle (CSA) reconstruction model to define the tracking
mask (tissue classifier).
"""

# Enables/disables interactive visualization
interactive = False

from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.viz import window, actor, colormap, has_fury

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)

"""
The Fiber Orientation Distribution (FOD) of the CSD model estimates the
distribution of small fiber bundles within each voxel. This distribution
can be used for deterministic fiber tracking. As for probabilistic tracking,
there are many ways to provide those distributions to the deterministic maximum
direction getter. Here, the spherical harmonic representation of the FOD
is used.
"""

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_trk
from dipy.tracking.streamline import Streamlines

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=30.,
                                                             sphere=default_sphere)
streamline_generator = LocalTracking(detmax_dg, classifier, seeds, affine,
                                     step_size=.5)
streamlines = Streamlines(streamline_generator)
save_trk("'tractogram_deterministic_dg.trk", streamlines, affine,
         labels.shape)

if has_fury:
    r = window.Renderer()
    r.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(r, out_path='tractogram_deterministic_dg.png',
                  size=(800, 800))
    if interactive:
        window.show(r)

"""
.. figure:: tractogram_deterministic_dg.png
   :align: center

   **Corpus Callosum using deterministic maximum direction getter**
"""
"""
.. include:: ../links_names.inc

"""
