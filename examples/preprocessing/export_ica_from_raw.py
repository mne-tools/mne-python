"""
==================================
Compute ICA components on Raw data
==================================

ICA is used to decompose raw data in 25 sources.
The sources are then exportes as raw object which then
can be used for conducting analyses in the ICA space.
It also can be used to store the sources and browse them
using mne_browse_raw.


"""
print __doc__

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
#
# License: BSD (3-clause)

import os
import mne
from mne.fiff import Raw
from mne.artifacts.ica import ICA
from mne.layouts import make_grid_layout
from mne.datasets import sample

###############################################################################
# Setup

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                            stim=False, exclude=raw.info['bads'])


###############################################################################
# Perform ICA

# setup ica seed
# Sign and order of components is non deterministic.
# setting the random state to 0 helps stabilizing the solution.
ica = ICA(noise_cov=None, n_components=25, random_state=0)

# 1 minute exposure should be sufficient for artifact detection.
# However, rejection performance may significantly improve when using
# the entire data range
start, stop = raw.time_as_index([100, 160])

# decompose sources for raw data
ica.decompose_raw(raw, start=start, stop=stop, picks=picks)
print ica

###############################################################################
# export ICA as raw for subsequent analyses in ICA space

ica_raw = ica.export_sources(raw, start=start, stop=stop)

print ica_raw.ch_names

ica_lout = make_grid_layout(ica_raw.info)

# uncomment the following two line to save sources and layut
ica_raw.save('ica_raw.fif')
ica_lout.save(os.path.join(os.environ['HOME'], '.mne/lout/ica.lout'))
