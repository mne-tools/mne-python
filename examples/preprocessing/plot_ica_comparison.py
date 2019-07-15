"""
===========================================
Compare the different ICA algorithms in MNE
===========================================

Different ICA algorithms are fit to raw MEG data, and the corresponding maps
are displayed.

"""
# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: BSD (3-clause)

from time import time

import mne
from mne.preprocessing import ICA
from mne.datasets import sample


print(__doc__)

###############################################################################
# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)

picks = mne.pick_types(raw.info)
reject = dict(mag=5e-12, grad=4000e-13)
raw.filter(1, 30, fir_design='firwin')


###############################################################################
# Define a function that runs ICA on the raw MEG data and plots the components


def run_ica(method, fit_params=None):
    ica = ICA(n_components=20, method=method, fit_params=fit_params,
              random_state=0)
    t0 = time()
    ica.fit(raw, picks=picks, reject=reject)
    fit_time = time() - t0
    title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
    ica.plot_components(title=title)

###############################################################################
# FastICA
run_ica('fastica')

###############################################################################
# Picard
run_ica('picard')

###############################################################################
# Infomax
run_ica('infomax')

###############################################################################
# Extended Infomax
run_ica('infomax', fit_params=dict(extended=True))
