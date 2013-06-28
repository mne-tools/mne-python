"""
====================================================
Dynamic imaging of coherent sources (DICS) pSPM maps
====================================================

Work in progress.

"""

# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import matplotlib.pylab as pl
from matplotlib import mlab
from scipy import linalg

import nitime.algorithms as tsa

import mne

from mne.fiff import Raw
from mne.fiff.constants import FIFF
from mne.fiff.pick import pick_channels_forward
from mne.minimum_norm.inverse import _get_vertno
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read raw data
raw = Raw(raw_fname)

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read noise covariance
noise_cov = mne.read_cov(fname_cov)
noise_cov = mne.cov.regularize(noise_cov, raw.info,
                               mag=0.05, grad=0.05, eeg=0.1, proj=True)

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Pick times relative to the onset of the MEG measurement.
start, stop = raw.time_as_index([100, 115], use_first_samp=False)

# Export to nitime using a copy of the data
raw_ts = raw.to_nitime(start=start, stop=stop, picks=picks, copy=True)

# Prepare specification of cross spectral density computation
csd_method = {}
csd_method['Fs'] = raw_ts.sampling_rate

# Using Welch's method as in the original DICS publication
csd_method['this_method'] = 'welch'

# Time window length of 1s and overlap of 0.5s
csd_method['NFFT'] = int(np.round(raw_ts.sampling_rate/2))
csd_method['n_overlap'] = int(np.round(raw_ts.sampling_rate/4))

# Calculate cross spectral density using three different methods for
# comparison.
# Comparing the three methods using PSD or CSD.
channel_1 = 121
channel_2 = channel_1  # Set to something else to obtain CSD

# Using nitime
frequencies, csds = tsa.get_spectra(raw_ts.data, method=csd_method)
pl.figure()

# Making the CSD matrix symmetrical
for i in range(csds.shape[2]):
    csds[:, :, i] = csds[:, :, i] + csds[:, :, i].conj().T -\
                    np.diag(csds[:, :, i].diagonal())

# Converting to dB
csd = 10 * np.log10(abs(csds[channel_1, channel_2, :]))
pl.plot(frequencies, csd)
pl.ylabel('Power or Cross Spectral Density (dB)')
pl.xlabel('Frequency (Hz)')
pl.title('Nitime version')

# Using matplotlib.pylab's csd function for comparison
# Note the matplotlib version only computes the psd/csd for a pair of channels,
# whereas the nitime version calculates the entire csd matrix for each
# frequency (actually using matplotlib.mlab.csd when Welch's method is
# selected)
pl.figure()
csd_pl, frequencies_pl = pl.csd(raw_ts.data[channel_2, :],
                                raw_ts.data[channel_1, :], csd_method['NFFT'],
                                csd_method['Fs'],
                                noverlap=csd_method['n_overlap'])
pl.title('Matplotlib version for comparison')

# Averaging CSD across frequency range of interest
freq_range = [8, 12]
freq_ids = [i for i in frequencies if i >= freq_range[0] and i < freq_range[1]]
csd = np.mean(csds[:, :, freq_ids], 2)


# What follows is mostly beamforming code that could be refactored into a
# separate function that would be reused between DICS and LCMV

# Setting parameters that would've been set by calling _apply_lcmv
reg = 0.1
label = None

# TODO: DICS, in the original 2001 paper, used a free orientation beamformer,
# however selection of the max-power orientation was also employed depending on
# whether a dominant component was present
pick_ori = None

is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

if pick_ori in ['normal', 'max-power'] and not is_free_ori:
    raise ValueError('Normal or max-power orientation can only be picked '
                     'when a forward operator with free orientation is '
                     'used.')
if pick_ori == 'normal' and not forward['surf_ori']:
    raise ValueError('Normal orientation can only be picked when a '
                     'forward operator oriented in surface coordinates is '
                     'used.')
if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
    raise ValueError('Normal orientation can only be picked when a '
                     'forward operator with a surface-based source space '
                     'is used.')

ch_names = [raw.info['ch_names'][k] for k in picks]

# Restrict forward solution to selected channels
forward = pick_channels_forward(forward, include=ch_names)

# Get gain matrix (forward operator)
if label is not None:
    vertno, src_sel = label_src_vertno_sel(label, forward['src'])

    if is_free_ori:
        src_sel = 3 * src_sel
        src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
        src_sel = src_sel.ravel()

    G = forward['sol']['data'][:, src_sel]
else:
    vertno = _get_vertno(forward['src'])
    G = forward['sol']['data']

# TODO: I don't know what to do about SSPs and whitening, SSP should probably
# be applied to data before calculating CSD

# Handle SSPs
#proj, ncomp, _ = make_projector(info['projs'], ch_names)
#G = np.dot(proj, G)

# Handle whitening + data covariance
#whitener, _ = compute_whitener(noise_cov, info, picks)

# whiten the leadfield
#G = np.dot(whitener, G)

# Apply SSPs + whitener to data covariance
#data_cov = pick_channels_cov(data_cov, include=ch_names)
#Cm = data_cov['data']
#Cm = np.dot(proj, np.dot(Cm, proj.T))
#Cm = np.dot(whitener, np.dot(Cm, whitener.T))

Cm = csd

# Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
Cm_inv = linalg.pinv(Cm, reg)

# Compute spatial filters
W = np.dot(G.T, Cm_inv)
n_orient = 3 if is_free_ori else 1
n_sources = G.shape[1] // n_orient
source_power = np.zeros(n_sources)
for k in range(n_sources):
    Wk = W[n_orient * k: n_orient * k + n_orient]
    Gk = G[:, n_orient * k: n_orient * k + n_orient]
    Ck = np.dot(Wk, Gk)

    # Find source orientation maximizing output source power
    # TODO: max-power is not used in this example, however DICS does employ
    # orientation picking when one eigen value is much larger than the other
    if pick_ori == 'max-power':
        eig_vals, eig_vecs = linalg.eigh(Ck)

        # Choosing the eigenvector associated with the middle eigenvalue.
        # The middle and not the minimal eigenvalue is used because MEG is
        # insensitive to one (radial) of the three dipole orientations and
        # therefore the smallest eigenvalue reflects mostly noise.
        for i in range(3):
            if i != eig_vals.argmax() and i != eig_vals.argmin():
                idx_middle = i

        # TODO: The eigenvector associated with the smallest eigenvalue
        # should probably be used when using combined EEG and MEG data
        max_ori = eig_vecs[:, idx_middle]

        Wk[:] = np.dot(max_ori, Wk)
        Ck = np.dot(max_ori, np.dot(Ck, max_ori))
        is_free_ori = False

    if is_free_ori:
        # Free source orientation
        Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
    else:
        # Fixed source orientation
        Wk /= Ck

    # TODO: Vectorize outside of the loop?
    source_power[k] = np.real_if_close(np.dot(Wk, np.dot(csd,
                                                         Wk.conj().T)).trace())

# Preparing noise normalization
# TODO: Noise normalization in DICS takes into account noise CSD, but maybe
# this isn't necessary if whitening is applied to data before calculating
noise_norm = np.sum((W * W.conj()), axis=1)
noise_norm = np.real_if_close(noise_norm)
if is_free_ori:
    noise_norm = np.sum(np.reshape(noise_norm, (-1, 3)), axis=1)
noise_norm = np.sqrt(noise_norm)

# Applying noise normalization
source_power /= noise_norm
