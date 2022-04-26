# -*- coding: utf-8 -*-
"""
.. _ex-eeg-briding:

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
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

# %%
# Load sample subject data
meg_path = data_path / 'MEG' / 'sample'
raw = mne.io.read_raw_fif(meg_path / 'sample_audvis_raw.fif')
raw = raw.pick_types(eeg=True, exclude=raw.info['bads']).load_data()

# %%
# Compute the bridged channels and plot them on a topomap.

bridged_idx, scores = mne.preprocessing.compute_bridged_electrodes(raw)
mne.viz.plot_bridged_electrodes(bridged_idx, scores, raw.info,
                                title='Bridged Electrodes')

# %%
# Electrode bridging is often brought about by inserting more gel in order
# to bring impendances down. Thus it can be helpful to compare bridging
# to impedances in the quest to be an ideal EEG technician! Low
# impedances lead to less noisy data and EEG without bridging is more
# spatially precise. Impedances can be stored with an EEG dataset
# like in the :ref:`electrodes-tsv` Brain Imaging Data Structure (BIDS)
# file. Since the impedances are not stored for this dataset, we will fake
# them to demonstrate how they would be plotted.

np.random.seed(11)  # seed for reproducibility
impedances = np.random.random((len(raw.ch_names,)))
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
