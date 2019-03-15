# -*- coding: utf-8 -*-
"""
.. _sss-tutorial:

Signal-space separation and Maxwell filtering
=============================================

.. include:: ../../tutorial_links.inc

This tutorial describes how to use signal-space separation (SSS), and the
related operation called Maxwell filtering, to both reduce environmental noise
and compensate for subject movement in MEG data. As usual we'll start by
importing the modules we need, and loading some example data:
"""

import os
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

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
# Like :ref:`SSP <ssp-tutorial>`, SSS is a form of projection. Whereas SSP
# empirically determines a noise subspace based on data (empty-room recordings,
# EOG or ECG activity, etc) and projects the measurements onto a subspace
# orthogonal to the noise, SSS mathematically constructs the external and
# internal subspaces from `spherical harmonics`_ and reconstructs the sensor
# signals using only the internal subspace.
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
# and fine calibration filenames provided:

raw.info['bads'].extend(['MEG 1032', 'MEG 2313'])
raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file,
                                           calibration=fine_cal_file)

###############################################################################
# To see the effect, we can plot the data before and after SSS / Maxwell
# filtering.

start, stop = raw.time_as_index([0, 2])
ylabels = dict(mag='Magnetometers (fT)', grad='Gradiometers (fT/m)')
raw_objs = dict(RAW=raw, SSS=raw_sss)

fig, axs = plt.subplots(2, 2, sharex=True, sharey='row')
for column, (title, raw_obj) in enumerate(raw_objs.items()):
    for row, (sensor_type, ylabel) in enumerate(ylabels.items()):
        data, times = raw_obj.get_data(picks=sensor_type, start=start,
                                       stop=stop, return_times=True)
        data /= 1e-15  # tesla to femtotesla
        axs[row, column].plot(times, data.T, color='k', linewidth=0.2,
                              alpha=0.3)
        # label axes
        axs[row, 0].set_ylabel(ylabel)
    axs[0, column].set_title(f'{title} DATA')
    axs[1, column].set_xlabel('time (s)')
# zoom in on gradiometers
axs[1, 0].set_ylim(-2e5, 2e5)
fig.tight_layout()

###############################################################################
# Notice that bad channels have been effectively repaired by SSS, eliminating
# the need to perform :ref:`interpolation <interpolating-bads-tutorial>`.
#
# Interactive plots are possible with MNE-Python's built-in plotting methods,
# so you can easily scroll through the rest of the datafile or look at
# individual channels:

kwargs = dict(duration=2, color='#00000033', bad_color='r')
raw.pick_types().plot(**kwargs)
raw_sss.pick_types().plot(**kwargs)

###############################################################################
# Movement compensation
# ^^^^^^^^^^^^^^^^^^^^^
#
# If you have information about subject head position relative to the sensors
# (i.e., continuous head position indicator coils, or "cHPI") SSS can take that
# into account when projecting sensor data onto the internal subspace. Head
# position data is loaded with the :func:`~mne.chpi.read_head_pos` function:
#
# .. code-block: python3
#
#     head_pos_file = mne.chpi.read_head_pos('path_to_chpi_file.pos')
#
# The cHPI data file is then passed as the ``head_pos`` parameter of
# :func:`~mne.preprocessing.maxwell_filter`:
#
# .. code-block: python3
#
#     raw_sss = mne.preprocessing.maxwell_filter(raw, head_pos=head_pos_file,
#                                                cross_talk=crosstalk_file,
#                                                calibration=fine_cal_file)
#
# Not only does this account for movement within a given recording session, but
# also effectively normalizes head position across different measurement
# sessions and subjects. See :ref:`here <movement-compensation-example>` for an
# extended example of applying movement compensation during Maxwell filtering /
# SSS.
#
#
# Caveats to using SSS / Maxwell filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 1. There are patents related to the Maxwell filtering algorithm, which may
#    legally preclude using it in commercial applications. More details are
#    provided in the documetation of :func:`~mne.preprocessing.maxwell_filter`.
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
#        multichannel data:The signal space separation method. *J Appl Phys*
#        97, 124905 1-10. doi:10.1063/1.1935742
#
# .. [2] Taulu S and Simola J. (2006). Spatiotemporal signal space separation
#        method for rejecting nearby interference in MEG measurements. *Phys
#        Med Biol* 51, 1759-1768. doi:10.1088/0031-9155/51/7/008
