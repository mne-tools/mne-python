"""
==================================
Basic analysis of an SSVEP/vSSR dataset
==================================

Example script to compute frequency spectrum and extract snr of a target frequency

We use a simple example dataset with frequency tagged visual stimulation (a.k.a.
steady state visually evoked potentials, SSVEP, or visual steady-state responses, 
vSSR):

N=2 participants observed checkerboards patterns inverting with a constant frequency
of either 12Hz of 15Hz. 10 trials of 30s length each. 32ch wet EEG was recorded.

Data format: BrainVision .eeg/.vhdr/.vmrk files organized according to BIDS standard.

Data can be downloaded at https://osf.io/7ne6y/
"""  # noqa: E501
# Authors: Dominik Welke <dominik.welke@ae.mpg.de>
#          Evgenii Kalenkovich <e.kalenkovich@gmail.com>
#
# License: BSD (3-clause)

import warnings
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import make_bids_basename, read_raw_bids
from scipy.stats import ttest_rel, ttest_ind

###############################################################################
# Load raw data
# -------------
event_id = {
    '12hz': 10001,
    '15hz': 10002
}

bids_root = "./data/"
bids_filename = make_bids_basename(
    subject='02',
    session='01', task='ssvep'
) + '_eeg.vhdr'


# read_raw_bids issues warnings about missing electrodes.tsv and coordsystem.json.
# These warning prevent successful building of the tutorial.
# As a quick workaround, we just suppress the warnings here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = read_raw_bids(bids_filename, bids_root, verbose=False)
raw.load_data()

###############################################################################
# Minimal preprocessing
# ---------------------
#
# Due to a generally high SNR in SSVEP/vSSR, typical preprocessing steps
# are considered optional. this doesnt mean, that a proper cleaning would not
# increase your signal quality!
#
# Raw data comes with FCz recording reference, so we will apply common-average
# rereferencing.

###############################################################################
# Set montage
# ^^^^^^^^^^^

montage_style = 'easycap-M1'
montage = mne.channels.make_standard_montage(
    montage_style,
    head_size=0.095)  # head_size parameter default = 0.095
raw.set_montage(montage)

###############################################################################
# Set common average reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

raw.set_eeg_reference('average', projection=False)

###############################################################################
# Apply notch filtering
# ^^^^^^^^^^^^^^^^^^^^^
notch = np.arange(raw.info['line_freq'], raw.info['lowpass'] / 2,
                  raw.info['line_freq'])
raw.notch_filter(notch, filter_length='auto', phase='zero')

###############################################################################
# Apply linear filtering
# ^^^^^^^^^^^^^^^^^^^^^^

hp = .1
lp = 250.
raw.filter(hp, lp, fir_design='firwin')


###############################################################################
# Frequency analysis
# ------------------
# We use Welch's method for frequency decomposition, since it is really fast.
# You can compare it with, e.g., multitaper to get an impression of the
# influence on SNR. All the other methods implemented in MNE can be used as
# well.

###############################################################################
# Construct epochs
# ^^^^^^^^^^^^^^^^

events, _ = mne.events_from_annotations(raw)
raw.info["events"] = events
tmin, tmax = -1., 30.  # in s
baseline = None
epochs = mne.Epochs(raw, events=events, event_id=event_id['12hz'], tmin=tmin,
                    tmax=tmax, baseline=baseline)

###############################################################################
# Calculate power spectral density
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tmin = 0.
tmax = 30.
fmin = 1.
fmax = 90.
sf = epochs.info['sfreq']

psds, freqs = mne.time_frequency.psd_welch(
    epochs,
    n_fft=int(sf * (tmax - tmin)), n_overlap=int(sf * .5), n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax)


###############################################################################
# Extract SSVEP/vSSR
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The function below calculates the ratio of power in the target frequency bin
# to average power in a set of neighbor (noise) bins. The composition of noise
# bins can be tweaked by two parameters:
#
# * how many noise bins do you want?
# * do you want to skip n bins directly next to the target bin?

###############################################################################
# SNR calculation function
# ^^^^^^^^^^^^^^^^^^^^^^^^
def snr_spectrum(psd, noise_n_neighborfreqs=1, noise_skip_neighborfreqs=1):
    """
    Parameters
    ----------
    psd - np.array
        containing psd values as spit out by mne functions. must be 2d or 3d
        with frequencies in the last dimension
    noise_n_neighborfreqs - int
        number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighborfreqs - int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr - np.array
        array containing snr for all epochs, channels, frequency bins.
        NaN for frequencies on the edge, that do not have enoug neighbors on
        one side to calculate snr
    """

    # prep not epoched / single channel data
    is_2d = True if (len(psd.shape) == 2) else False
    if is_2d:
        psd = psd.reshape((1, psd.shape[0], psds.shape[1]))

    # SNR loop
    snr = np.empty(psd.shape)
    for i_freq in range(psd.shape[2]):

        # skip freqs on the edges (without noise neighbors)

        start_freq_i = noise_n_neighborfreqs + noise_skip_neighborfreqs
        stop_freq_i = (psd.shape[2] - noise_n_neighborfreqs
                       - noise_skip_neighborfreqs)
        if not (stop_freq_i > i_freq >= start_freq_i):
            snr[:, :, i_freq] = np.nan
            continue

        # extract signal level
        signal = psd[:, :, i_freq]

        # ... and average noise level
        i_noise = []
        for i in range(noise_n_neighborfreqs):
            i_noise.append(i_freq + noise_skip_neighborfreqs + i + 1)
            i_noise.append(i_freq - noise_skip_neighborfreqs - i - 1)
        noise = psd[:, :, i_noise].mean(axis=2)

        snr[:, :, i_freq] = signal / noise

    # reshape not epoched / single channel data to original dimensions
    if is_2d:
        snr = snr.reshape(snr.shape[1], snr.shape[2])

    return snr


###############################################################################
# Calculate SNR
# ^^^^^^^^^^^^^
# Now we call the function to compute our snr spectrum.
# SNR is a relative measure: it's the ratio of power in a given frequency bin
# compared to a baseline - the average power in the surrounding frequency bins.
# Hence, we need to define some parameters for this 'baseline' - how many
# neighboring bins should be taken for this computation, and do we want to skip
# the direct neighbors (this can make sense if the stimulation frequency is not
# super constant, or frequency bands are very narrow).

snrs = snr_spectrum(psds, noise_n_neighborfreqs=3,
                    noise_skip_neighborfreqs=1)

###############################################################################
# Find frequency bin containing stimulation frequency
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Ideally, this bin should should have the stimulation frequency exactly in the
# center.

stim_freq = 12.
tmp_distlist = abs(np.subtract(freqs, stim_freq))
i_signal = np.where(tmp_distlist == min(tmp_distlist))[0][0]
# could be updated to support multiple frequencies

###############################################################################
# Calculate SNR
# ^^^^^^^^^^^^^
# Extract and average SNRs at this frequency
snrs_stim = snrs[:, :, i_signal]  # trial subselection can be done here
print('average SNR at %iHz (all channels, all trials): %.3f '
      % (stim_freq, snrs_stim.mean()))


###############################################################################
# Visualization
# -------------


##############################################################################
# Plot power spectral density
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# code snippet from
# https://martinos.org/mne/stable/auto_examples/time_frequency/plot_compute_raw_data_spectrum.html  # noqa E501

fig, axes = plt.subplots(1, 1, sharex='all', sharey='all', dpi=300)
rng = range(np.where(np.floor(freqs) == 1.)[0][0],
            np.where(np.ceil(freqs) == fmax - 1)[0][0])

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean((0, 1))[rng]
psds_std = psds_plot.std((0, 1))[rng]
axes.plot(freqs[rng], psds_mean, color='b')
axes.fill_between(freqs[rng], psds_mean - psds_std, psds_mean + psds_std,
                  color='b', alpha=.5)
axes.set(title="PSD spectrum", xlabel='Frequency [Hz]',
         ylabel='Power Spectral Density [dB]')
plt.xlim([0, fmax])
fig.show()

##############################################################################
# SNR spectrum - trial average
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 1, sharex='all', sharey='all', dpi=300)
# Average over trials
axes.plot(freqs, snrs.mean(axis=0).T, color='b')
# Average over trials and channels
axes.plot(freqs, snrs.mean(axis=0).mean(axis=0), color='r')
axes.set(title="SNR spectrum - trial average", xlabel='Frequency [Hz]',
         ylabel='SNR', ylim=[0, 20])
fig.show()

##############################################################################
# SNR spectrum - channel average
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 1, sharex='all', sharey='all', dpi=300)
# Average over channels
axes.plot(freqs, snr_spectrum(psds[:, :, :], 3, 1).mean(axis=1).T, color='b')
# Average over trials and channels
axes.plot(freqs, snr_spectrum(psds[:, :, :], 3, 1).mean(axis=0).mean(axis=0),
          color='r')
axes.set(title="SNR spectrum - channel average", xlabel='Frequency [Hz]',
         ylabel='SNR', ylim=[0, 20])
fig.show()


###############################################################################
# Subsetting data
# ---------------
#
# For statistical comparison you probably want specific subsets of the SNR
# array. Either some channels, or - obviously - different trials depending on
# the stimuli.
#
# - So far, one needs to define the indices of the channels / trials by hand -
#   not nice.
# - Alternatively, one can subset trials already at the epoch level using MNEs
#   event information, and create individual PSD and SNR objects.
#
# Here we have already sub-setted trials before snr calculation (only 12Hz
# stimulation) and will now compare SNR in different channel subsets.
#
# For illustration purposes, we will still subset the first 5 and last 5 of the
# 10 trials with 12hz stimulation.


##############################################################################
# Define ROIs
roi_temporal = ['T7', 'F7', 'T8', 'F8']  # temporal
roi_aud = ['AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'F1', 'FC1',
           'C1', 'CP1', 'F2', 'FC2', 'C2', 'CP2']  # auditory roi
roi_vis = ['POz', 'Oz', 'O1', 'O2', 'PO3', 'PO4', 'PO7',
           'PO8', 'PO9', 'PO10', 'O9', 'O10']  # visual roi

##############################################################################
# Create corresponding picks
picks_roi_temp = mne.pick_types(raw.info, eeg=True, stim=False,
                                exclude='bads', selection=roi_temporal)
picks_roi_aud = mne.pick_types(raw.info, eeg=True, stim=False,
                               exclude='bads', selection=roi_aud)
picks_roi_vis = mne.pick_types(raw.info, eeg=True, stim=False,
                               exclude='bads', selection=roi_vis)

##############################################################################
# Subset data based on ROIs
snrs_trialwise_roi_aud = snrs_stim[:, picks_roi_aud]
snrs_trialwise_roi_vis = snrs_stim[:, picks_roi_vis]
snrs_trialwise_temp = snrs_stim[:, picks_roi_temp]

##############################################################################
# SNR for different ROIs
print('mean SNR (all channels, all trials) at %iHz = %.3f '
      % (stim_freq, snrs_stim.mean()))
print('mean SNR (auditory ROI) at %iHz = %.3f '
      % (stim_freq, snrs_trialwise_roi_aud.mean()))
print('mean SNR (visual ROI) at %iHz = %.3f '
      % (stim_freq, snrs_trialwise_roi_vis.mean()))
print('mean SNR (temporal chans) at %iHz = %.3f '
      % (stim_freq, snrs_trialwise_temp.mean()))


##############################################################################
# Define trial subsets
i_cat1_1 = [i for i in range(5)]
i_cat1_2 = [i for i in range(5, 10)]

##############################################################################
# Subset data trial-wise
snrs_trialwise_cat1_1 = snrs_stim[i_cat1_1, :]
snrs_trialwise_cat1_2 = snrs_stim[i_cat1_2, :]

##############################################################################
# SNR for different subsets of trials
print('mean SNR (trial subset 1) at %iHz = %.3f '
      % (stim_freq, snrs_trialwise_cat1_1.mean()))
print('mean SNR (trial subset 2) at %iHz = %.3f '
      % (stim_freq, snrs_trialwise_cat1_2.mean()))


##############################################################################
# Statistics
# ----------
# Just a toy t-test example to test whether:
#
# - SNR in visual ROI is significantly different from full scalp montage at p
#   < 0.05
# - SNR in first 5 trials does not significantly differ from 5 last trials

##############################################################################
# Compare SNR in ROIs after averaging over channels
tstat_roi = ttest_rel(snrs_trialwise_roi_vis.mean(axis=1),
                      snrs_stim.mean(axis=1))
print("trial-wise SNR in visual ROI is significantly different from full scalp"
      " montage: t = %.3f, p = %f" % tstat_roi)

##############################################################################
# Compare SNR in subsets of trials after averaging over channels
tstat_trials = ttest_ind(snrs_trialwise_cat1_1.mean(axis=1),
                         snrs_trialwise_cat1_2.mean(axis=1))
print("trial-wise SNR in trial subset 1 is NOT significantly different from"
      " trial subset 2: t = %.3f, p = %f" % tstat_trials)
