"""
====================================================
Bootstrap and Closest Peak Direction Getters Example
====================================================

This example shows how choices in direction-getter impact fiber
tracking results by demonstrating the bootstrap direction getter (a type of
probabilistic tracking, as described in Berman et al. (2008) [Berman2008]_ a
nd the closest peak direction getter (a type of deterministic tracking).
(Amirbekian, PhD thesis, 2016)

This example is an extension of the :ref:`example_tracking_introduction_eudx`
example. Let's start by loading the necessary modules for executing this
tutorial.
"""

# Enables/disables interactive visualization
interactive = False

from dipy.data import read_stanford_labels
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.viz import window, actor, colormap, has_fury

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine

seed_mask = (labels == 2)
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

"""
Next, we fit the CSD model.
"""

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

"""
we use the CSA fit to calculate GFA, which will serve as our tissue
classifier.
"""

from dipy.reconst.shm import CsaOdfModel
csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)

"""
Next, we need to set up our two direction getters
"""

"""
Example #1: Bootstrap direction getter with CSD Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from dipy.direction import BootDirectionGetter
from dipy.tracking.streamline import Streamlines
from dipy.data import small_sphere

boot_dg_csd = BootDirectionGetter.from_data(data, csd_model, max_angle=30.,
                                            sphere=small_sphere)
boot_streamline_generator = LocalTracking(boot_dg_csd, classifier, seeds,
                                          affine, step_size=.5)
streamlines = Streamlines(boot_streamline_generator)

save_trk("tractogram_bootstrap_dg.trk", streamlines, affine, labels.shape)

if has_fury:
    r = window.Renderer()
    r.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(r, out_path='tractogram_bootstrap_dg.png', size=(800, 800))
    if interactive:
        window.show(r)

"""
.. figure:: tractogram_bootstrap_dg.png
   :align: center

   **Corpus Callosum Bootstrap Probabilistic Direction Getter**

We have created a bootstrapped probabilistic set of streamlines. If you repeat
the fiber tracking (keeping all inputs the same) you will NOT get exactly the
same set of streamlines.
"""

"""
Example #2: Closest peak direction getter with CSD Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from dipy.direction import ClosestPeakDirectionGetter

pmf = csd_fit.odf(small_sphere).clip(min=0)
peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                              sphere=small_sphere)
peak_streamline_generator = LocalTracking(peak_dg, classifier, seeds, affine,
                                          step_size=.5)
streamlines = Streamlines(peak_streamline_generator)

save_trk("closest_peak_dg_CSD.trk", streamlines, affine, labels.shape)

if has_fury:
    r = window.Renderer()
    r.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(r, out_path='tractogram_closest_peak_dg.png',
                  size=(800, 800))
    if interactive:
        window.show(r)

"""
.. figure:: tractogram_closest_peak_dg.png
   :align: center

   **Corpus Callosum Closest Peak Deterministic Direction Getter**

We have created a set of streamlines using the closest peak direction getter,
which is a type of deterministic tracking. If you repeat the fiber tracking
(keeping all inputs the same) you will get exactly the same set of streamlines.
"""


"""
References
----------
.. [Berman2008] Berman, J. et al., Probabilistic streamline q-ball
tractography using the residual bootstrap, NeuroImage, vol 39, no 1, 2008

.. include:: ../links_names.inc

"""
