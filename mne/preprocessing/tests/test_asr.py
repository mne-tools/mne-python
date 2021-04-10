# -*- coding: utf-8 -*-
"""Test the compute_current_source_density function.

For each supported file format, implement a test.
"""
# Authors: Dirk Guetlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

import pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.preprocessing.asr import (asr_calibrate, clean_windows, asr_process, yulewalk_filter,yulewalk, ASR)
from mne.io import read_raw_eeglab
from scipy.io import loadmat


data_path = op.join(testing.data_path(download=False), 'EEGLAB')
eeg_fname = op.join(data_path, 'test_raw.set')

raw = read_raw_eeglab(eeg_fname)

#clean, sample_mask = clean_windows(raw.get_data(), raw.info["sfreq"])

eeglab_euclid = read_raw_eeglab("mne/preprocessing/tests/data/asr_euclid_std20.set")
asr_data = loadmat("mne/preprocessing/tests/data/asr_results.mat")["asr_results"]
clean_asr_data = loadmat("mne/preprocessing/tests/data/matlab_asr_data.mat")["data"][0][0][0]
sfreq = raw.info["sfreq"]
n_chs = len(raw.ch_names)

# set the parameters
data = raw.get_data()  * 1e6
sfreq = raw.info["sfreq"]
cutoff = 2.5
blocksize = 10
window_len = 0.5
window_overlap = 0.66
max_dropout_fraction = 0.1
min_clean_fraction = 0.25

# process params
lookahead = window_len/2
stepsize = 32
maxdims = 0.66

X_filt, iir = yulewalk_filter(data, sfreq=sfreq, zi=np.ones([len(data), 8]))

#M, T = asr_calibrate(data,sfreq,cutoff,blocksize,window_len,window_overlap,max_dropout_fraction,min_clean_fraction)
#clean = asr_process(data, sfreq, M, T, window_len, lookahead, stepsize, maxdims, Zi=iir)
#print(np.corrcoef(clean.reshape(-1), clean_asr_data.reshape(-1)))
#print(np.mean([np.corrcoef(i, j)[0, 1] for (i, j) in zip(clean, clean_asr_data)]))


# corrcoef must be: 0.95763382
# avg corrcoef over channels must be: 0.9562659317460613
raw.load_data()
asr = ASR(sfreq=raw.info["sfreq"], cutoff=cutoff, blocksize=blocksize, win_len=window_len, win_overlap=window_overlap,
          max_dropout_fraction=max_dropout_fraction, min_clean_fraction=min_clean_fraction, ab=None)
asr.fit(raw.get_data())
clean = asr.transform(raw.get_data())
print(np.corrcoef(clean.reshape(-1), clean_asr_data.reshape(-1)))
print(np.mean([np.corrcoef(i, j)[0, 1] for (i, j) in zip(clean, clean_asr_data)]))


from mne.preprocessing.asr_utils import nonlinear_eigenspace
nonlinear_eigenspace(data[:5, 1, :5], k = 2)



"""
@pytest.fixture(scope='function', params=[testing._pytest_param()])
def evoked_csd_sphere():
    ""Get the MATLAB EEG data.""
    data = loadmat(eeg_fname)['data']
    coords = loadmat(coords_fname)['coords'] * 1e-3
    csd = loadmat(csd_fname)['csd']
    sphere = np.array((0, 0, 0, 0.08500060886258405))  # meters
    sfreq = 256  # sampling rate
    # swap coordinates' shape
    pos = np.rollaxis(coords, 1)
    # swap coordinates' positions
    pos[:, [0]], pos[:, [1]] = pos[:, [1]], pos[:, [0]]
    # invert first coordinate
    pos[:, [0]] *= -1
    dists = np.linalg.norm(pos, axis=-1)
    assert_allclose(dists, sphere[-1], rtol=1e-2)  # close to spherical, meters
    # assign channel names to coordinates
    ch_names = [str(ii) for ii in range(len(pos))]
    dig_ch_pos = dict(zip(ch_names, pos))
    montage = make_dig_montage(ch_pos=dig_ch_pos, coord_frame='head')
    # create info
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    # make Evoked object
    evoked = EvokedArray(data=data, info=info, tmin=-1)
    evoked.set_montage(montage)
    return evoked, csd, sphere
    
    
from scipy import linalg
def polyeig(*A):
    '''
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x=0 

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X,e = polyeig(A0,A1,..,Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X,e = polyeig(K,C,M)

    '''
    if len(A)<=0:
        raise Exception('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise Exception('All matrices must have the same shapes')

    n = A[0].shape[0]
    l = len(A)-1
    # Assemble matrices for generalized problem
    C = np.block([
        [np.zeros((n*(l-1),n)), np.eye(n*(l-1))],
        [-np.column_stack( A[0:-1])]
        ])
    D = np.block([
        [np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
        [np.zeros((n, n*(l-1))), A[-1]          ]
        ])
    # Solve generalized eigenvalue problem
    e, X = linalg.eig(C, D)
    if np.all(np.isreal(e)):
        e=np.real(e)
    X=X[:n,:]

    # Sort eigenvalues/vectors
    #I = np.argsort(e)
    #X = X[:,I]
    #e = e[I]

    # Scaling each mode by max
    X /= np.tile(np.max(np.abs(X),axis=0), (n,1))

    return X, e
"""

# run_tests_if_main()
