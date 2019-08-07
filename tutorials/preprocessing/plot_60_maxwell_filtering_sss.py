"""
.. _tut-artifact-sss:

Signal-space separation and Maxwell filtering
=============================================

This tutorial covers reducing environmental noise and compensating for head
movement with SSS and Maxwell filtering.

.. contents:: Page contents
   :local:
   :depth: 2

As usual we'll start by importing the modules we need, loading some
:ref:`example data <sample-dataset>`, and cropping it to save on memory:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()

###############################################################################
# Signal-space separation (SSS) [1]_ [2]_ is a technique based on the physics
# of electromagnetic fields. SSS separates the measured signal into components
# attributable to sources *inside* the measurement volume of the sensor array
# (the *internal components*), and components attributable to sources *outside*
# the measurement volume (the *external components*). The internal and external
# components are linearly independent, so it is possible to simply discard the
# external components to reduce environmental noise. *Maxwell filtering* is a
# related procedure that omits the higher-order components of the internal
# subspace, which are dominated by sensor noise. Typically, Maxwell filtering
# and SSS are performed together (in MNE-Python they are implemented together
# in a single function).
#
# Like :ref:`SSP <tut-artifact-ssp>`, SSS is a form of projection. Whereas SSP
# empirically determines a noise subspace based on data (empty-room recordings,
# EOG or ECG activity, etc) and projects the measurements onto a subspace
# orthogonal to the noise, SSS mathematically constructs the external and
# internal subspaces from `spherical harmonics`_ and reconstructs the sensor
# signals using only the internal subspace.
#
# .. warning::
#
#     Maxwell filtering was originally developed for Elekta Neuromag® systems,
#     and should be considered *experimental* for non-Neuromag data. See the
#     Notes section of the :func:`~mne.preprocessing.maxwell_filter` docstring
#     for details.
#
#
# Using SSS and Maxwell filtering in MNE-Python
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For optimal use of SSS with data from Elekta Neuromag® systems, you should
# provide the path to the fine calibration file (which encodes site-specific
# information about sensor orientation and calibration) as well as a crosstalk
# compensation file (which reduces interference between Elekta's co-located
# magnetometer and paired gradiometer sensor units.

fine_cal_file = os.path.join(sample_data_folder, 'SSS', 'sss_cal_mgh.dat')
crosstalk_file = os.path.join(sample_data_folder, 'SSS', 'ct_sparse_mgh.fif')

###############################################################################
# Before we perform SSS we'll set a couple additional bad channels — ``MEG
# 2313`` has some DC jumps and ``MEG 1032`` has some large-ish low-frequency
# drifts. After that, performing SSS and Maxwell filtering is done with a
# single call to :func:`~mne.preprocessing.maxwell_filter`, with the crosstalk
# and fine calibration filenames provided (if available):

raw.info['bads'].extend(['MEG 1032', 'MEG 2313'])
raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file,
                                           calibration=fine_cal_file)

###############################################################################
#  .. warning::
#
#      Automatic bad channel detection is not currently implemented. It is
#      critical to mark bad channels in ``raw.info['bads']`` *before* calling
#      :func:`~mne.preprocessing.maxwell_filter` in order to prevent bad
#      channel noise from spreading.
#
# To see the effect, we can plot the data before and after SSS / Maxwell
# filtering.

raw.pick(['meg']).plot(duration=2, butterfly=True)
raw_sss.pick(['meg']).plot(duration=2, butterfly=True)

###############################################################################
# Notice that channels marked as "bad" have been effectively repaired by SSS,
# eliminating the need to perform :ref:`interpolation <tut-bad-channels>`.
# The heartbeat artifact has also been substantially reduced.
#
# The :func:`~mne.preprocessing.maxwell_filter` function has parameters
# ``int_order`` and ``ext_order`` for setting the order of the spherical
# harmonic expansion of the interior and exterior components; the default
# values are appropriate for most use cases. Additional parameters include
# ``coord_frame`` and ``origin`` for controlling the coordinate frame ("head"
# or "meg") and the origin of the sphere; the defaults are appropriate for most
# studies that include digitization of the scalp surface / electrodes. See the
# documentation of :func:`~mne.preprocessing.maxwell_filter` for details.
#
#
# Spatiotemporal SSS (tSSS)
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An assumption of SSS is that the measurement volume (the spherical shell
# where the sensors are physically located) is free of electromagnetic sources.
# The thickness of this source-free measurement shell should be 4-8 cm for SSS
# to perform optimally. In practice, there may be sources falling within that
# measurement volume; these can often be mitigated by using Spatiotemporal
# Signal Space Separation (tSSS) [2]_. tSSS works by looking for temporal
# correlation between components of the internal and external subspaces, and
# projecting out any components that are common to the internal and external
# subspaces. The projection is done in an analogous way to
# :ref:`SSP <tut-artifact-ssp>`, except that the noise vector is computed
# across time points instead of across sensors.
#
# To use tSSS in MNE-Python, pass a time (in seconds) to the parameter
# ``st_duration`` of :func:`~mne.preprocessing.maxwell_filter`. This will
# determine the "chunk duration" over which to compute the temporal projection.
# The chunk duration effectively acts as a high-pass filter with a cutoff
# frequency of :math:`\frac{1}{\mathtt{st\_duration}}~\mathrm{Hz}`; this
# effective high-pass has some important consequences:
#
# - In general, larger values of ``st_duration`` are better (provided that your
#   computer has sufficient memory) because larger values of ``st_duration``
#   will have a smaller effect on the signal.
#
# - It is best to choose a chunk duration that evenly divides your data length,
#   so that the effective high-pass filtering is the same for all spans of your
#   data.
#
# Additional parameters affecting tSSS include ``st_correlation`` (to set the
# correlation value above which correlated internal and external components
# will be projected out) and ``st_only`` (to apply only the temporal projection
# without also performing SSS and Maxwell filtering). See the docstring of
# :func:`~mne.preprocessing.maxwell_filter` for details.
#
#
# Movement compensation
# ^^^^^^^^^^^^^^^^^^^^^
#
# If you have information about subject head position relative to the sensors
# (i.e., continuous head position indicator coils, or :term:`cHPI <hpi>`), SSS
# can take that into account when projecting sensor data onto the internal
# subspace. Head position data is loaded with the
# :func:`~mne.chpi.read_head_pos` function. The :ref:`example data
# <sample-dataset>` doesn't include cHPI, so here we'll load a :file:`.pos`
# file used for testing, just to demonstrate:

head_pos_file = os.path.join(mne.datasets.testing.data_path(), 'SSS',
                             'test_move_anon_raw.pos')
head_pos = mne.chpi.read_head_pos(head_pos_file)
mne.viz.plot_head_positions(head_pos, mode='traces')

###############################################################################
# The cHPI data file could also be passed as the ``head_pos`` parameter of
# :func:`~mne.preprocessing.maxwell_filter`. Not only would this account for
# movement within a given recording session, but also would effectively
# normalize head position across different measurement sessions and subjects.
# See :ref:`here <example-movement-comp>` for an extended example of applying
# movement compensation during Maxwell filtering / SSS.
#
#
# Caveats to using SSS / Maxwell filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 1. There are patents related to the Maxwell filtering algorithm, which may
#    legally preclude using it in commercial applications. More details are
#    provided in the documentation of
#    :func:`~mne.preprocessing.maxwell_filter`.
#
# 2. SSS works best when both magnetometers and gradiometers are present, and
#    is most effective when gradiometers are planar (due to the need for very
#    accurate sensor geometry and fine calibration information). Thus its
#    performance is dependent on the MEG system used to collect the data.
#
#
# References
# ^^^^^^^^^^
#
# .. [1] Taulu S and Kajola M. (2005). Presentation of electromagnetic
#        multichannel data: The signal space separation method. *J Appl Phys*
#        97, 124905 1-10. https://doi.org/10.1063/1.1935742
#
# .. [2] Taulu S and Simola J. (2006). Spatiotemporal signal space separation
#        method for rejecting nearby interference in MEG measurements. *Phys
#        Med Biol* 51, 1759-1768. https://doi.org/10.1088/0031-9155/51/7/008
#
#
# .. LINKS
#
# .. _spherical harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics
