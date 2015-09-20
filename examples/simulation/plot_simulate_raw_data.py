"""
==============================
Generate simulated raw data
==============================

"""
# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#         Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from mne import (read_proj, read_forward_solution, read_cov,
                 pick_types_forward)
from mne.io import Raw
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw

print(__doc__)

###############################################################################
# Load real data as templates
data_path = sample.data_path()

raw = Raw(data_path + '/MEG/sample/sample_audvis_raw.fif')
raw.crop(0., 60.)  # only 60s of data is enough
proj = read_proj(data_path + '/MEG/sample/sample_audvis_ecg_proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])

cov = read_cov(cov_fname)

bem_fname = (data_path +
             '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')

# Generate times series for 2 dipoles

rng = np.random.RandomState(42)


def data_fun(times):
    """Function to generate random sine source time courses around 10Hz"""
    return 1e-9 * np.sin(2. * np.pi * (10. + 2. * rng.randn(1)) * times)

stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=raw.times,
                          random_state=42)

trans = fwd['mri_head_t']
src = fwd['src']
raw_sim = simulate_raw(info=raw.info, stc=stc, trans=trans, src=src,
                       bem=bem_fname, times=raw.times)

raw_sim.plot()
