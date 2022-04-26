# -*- coding: utf-8 -*-
"""
.. _ex-eeg-bridging:

===============================================
Identify EEG Electrodes Bridged by too much Gel
===============================================

Research-grade EEG often uses a gel based system, and when too much gel is
applied the gel conducting signal from the scalp to the electrode for one
electrode connects with the gel conducting signal from another electrode
"bridging" the two signals. This is undesirable as the signals from the two
(or more) electrodes are not as independent as they would otherwise be;
spatial smearing is caused wherein the signals are more similar to each other
been developed to detect electrode bridging :footcite:`TenkeKayser2001`, which
was previously implemented in EEGLAB :footcite:`DelormeMakeig2004`.
Unfortunately, there is not a lot to be done about electrode brigding once
the data has been collected as far as preprocessing. Therefore, the
recommendation is to check for electrode bridging early in data collection
and address the problem. Or, if the data has already been collected, quantify
the extent of the bridging so as not to introduce bias into the data from
this effect and exclude subjects with bridging that might effect the outcome
of a study. Preventing electrode bridging is ideal but awareness of the
problem at least will mitigate its potential as a confound to a study.
This tutorial follows
https://psychophysiology.cpmc.columbia.edu/software/eBridge/tutorial.html.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pyplot as plt

import mne

print(__doc__)

# %%
# Let's look at the histograms of electrical distances for the EEGBCI dataset.
# As we can see, for subjects 6, 7 and 8 (and to a lesser extent 4), there is
# a different shape of the distribution of electrical distances than for the
# other subjects. These subjects' distributions have a peak around 0
# :math:`{\\mu}`V:sup:`2` distance and a trough around 5
# :math:`{\\mu}`V:sup:`2` which is indicative of electrode bridging.
#  The rest of the subjects' distributions increase monotonically,
# indicating normal spatial separation of sources.

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title('Electrical Distance Distribution for EEGBCI Subjects')
ax.set_ylabel('Count')
ax.set_xlabel(r'Electrical Distance ($\mu$$V^2$)')
for sub in range(1, 11):
    print(f'Computing electrode bridges for subject {sub}')
    raw = mne.io.read_raw(mne.datasets.eegbci.load_data(
        subject=sub, runs=(1,))[0], preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)  # set channel names
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, verbose=False)

    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(raw)
    # ed_matrix is upper triangular so exclude bottom half of NaNs
    hist, edges = np.histogram(ed_matrix[~np.isnan(ed_matrix)].flatten(),
                               bins=np.linspace(0, 100, 51))
    ax.plot((edges[1:] + edges[:-1]) / 2, hist,
            label=f'Sub {sub} #={len(bridged_idx)}')

ax.legend(loc=(1.04, 0))
fig.subplots_adjust(right=0.725, bottom=0.15)

# %%
# Let's look at one subject with bridged electrodes and plot what's
# going on to try and understand electrode bridging better.

raw = mne.io.read_raw(mne.datasets.eegbci.load_data(
    subject=6, runs=(1,))[0], preload=True)
mne.datasets.eegbci.standardize(raw)  # set channel names
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(raw)
mne.viz.plot_bridged_electrodes(
    raw.info, bridged_idx, ed_matrix, title='Bridged Electrodes',
    topomap_args=dict(names=raw.ch_names, show_names=True))

# %%
# Electrode bridging is often brought about by inserting more gel in order
# to bring impendances down. Thus it can be helpful to compare bridging
# to impedances in the quest to be an ideal EEG technician! Low
# impedances lead to less noisy data and EEG without bridging is more
# spatially precise. Impedances are recommended to be stored in an EEG dataset
# in the :ref:`electrodes-tsv` file within th eBrain Imaging Data Structure
# (BIDS). Since the impedances are not stored for this dataset, we will fake
# them to demonstrate how they would be plotted.

np.random.seed(11)  # seed for reproducibility
impedances = np.random.random((len(raw.ch_names,))) * 10
fig, ax = plt.subplots(figsize=(5, 5))
im, cn = mne.viz.plot_topomap(impedances, raw.info, axes=ax)
ax.set_title('Electrode Impendances Audio/Visual Task')
cax = fig.colorbar(im, ax=ax)
cax.set_label(r'Impedance (k$\Omega$)')


# %%
# References
# ----------
# .. footbibliography::
#
#
# .. _electrodes-tsv: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html#electrodes-description-_electrodestsv  # noqa E501
