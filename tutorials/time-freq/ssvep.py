"""
=======================================
Basic analysis of an SSVEP/vSSR dataset
=======================================

In this tutorial we compute the frequency spectrum and quantify signal-to-noise 
ratio (SNR) at a target frequency in EEG data recorded during fast periodic 
visual stimulation (FPVS). 
Extracting SNR at stimulation frequency is a simple way to quantify frequency tagged 
responses in MEEG (a.k.a. steady state visually evoked potentials, SSVEP, or visual 
steady-state responses, vSSR in the visual domain, 
or auditory steady-state responses, aSSR in the auditory domain).

DATA:

We use a simple example dataset with frequency tagged visual stimulation:
N=2 participants observed checkerboards patterns inverting with a constant frequency
of either 12Hz of 15Hz. 10 trials of 30s length each. 32ch wet EEG was recorded.

Data can be downloaded at https://osf.io/7ne6y/ 
Data format: BrainVision .eeg/.vhdr/.vmrk files organized according to BIDS standard.

OUTLINE:

- We will visualize both the PSD spectrum and the SNR spectrum of our epoched data.
- We will extract SNR at stimulation frequency for all trials and channels.
- We will show, that we can statistically separate 12hz and 15 hz responses in our data.   

Since the evoked response is mainly generated in early visual areas of the brain we 
will stick with an ROI analysis and extract SNR from occipital channels. 
"""  # noqa: E501
# Authors: Dominik Welke <dominik.welke@web.de>
#          Evgenii Kalenkovich <e.kalenkovich@gmail.com>
#
# License: BSD (3-clause)

import warnings
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import read_raw_bids, BIDSPath
from scipy.stats import ttest_rel, ttest_ind

###############################################################################
# Data preprocessing
# ------------------
# Due to a generally high SNR in SSVEP/vSSR, typical preprocessing steps
# are considered optional. this doesnt mean, that a proper cleaning would not
# increase your signal quality!
#
# - Raw data comes with FCz recording reference, so we will apply common-average rereferencing.
# - We will apply a 50 Hz notch-filter to remove line-noise,
# - and a 0.1 - 250 Hz bandpass filter.
# - Lastly we will cut the data in 30 s epochs according to the trials.

# Load raw data
data_path = mne.datasets.ssvep.data_path()
bids_path = BIDSPath(subject='02', session='01', task='ssvep', root=data_path)

# read_raw_bids issues warnings about missing electrodes.tsv and coordsystem.json.
# These warning prevent successful building of the tutorial.
# As a quick workaround, we just suppress the warnings here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = read_raw_bids(bids_path, verbose=False)
raw.load_data()


# Set montage
montage_style = 'easycap-M1'
montage = mne.channels.make_standard_montage(
    montage_style,
    head_size=0.095)  # head_size parameter default = 0.095
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference('average', projection=False, verbose=True)

# Apply notch filter to remove line-noise
notch = np.arange(raw.info['line_freq'], raw.info['lowpass'] / 2,
                  raw.info['line_freq'])
raw.notch_filter(notch, filter_length='auto', phase='zero')

# Apply bandpass filter
hp = .1
lp = 250.
raw.filter(hp, lp, fir_design='firwin')

# Construct epochs
event_id = {
    '12hz': 10001,
    '15hz': 10002
}
events, _ = mne.events_from_annotations(raw, verbose=False)
raw.info["events"] = events
tmin, tmax = -1., 30.  # in s
baseline = None
epochs = mne.Epochs(raw, events=events, event_id=[event_id['12hz'], event_id['15hz']], tmin=tmin,
                    tmax=tmax, baseline=baseline, verbose=True)

###############################################################################
# Frequency analysis
# ------------------
# Now we compute the frequency spectrum of the EEG data.
# You will already see the peaks at the stimulation frequencies and some of
# their harmonics, without any further processing.
#
# The 'classical' PSD plot will be compared to a plot of the SNR spectrum.
# SNR will be computed as the power in a given frequency bin
# relative to the average power in it's neighboring bins.
# This procedure has two advantages over using the raw PSD:
#
# * it normalizes the spectrum and accounts for 1/f power decay.
# * power modulations which are not very narrow band will disappear.
#

###############################################################################
# Calculate power spectral density (PSD)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use Welch's method for frequency decomposition, since it is really fast.
# You could compare it with, e.g., multitaper to get an impression of the
# influence on SNR. All the other methods implemented in MNE can be used as
# well.

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
# Calculate signal to noise ratio (SNR)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# SNR - as we define it here - is a measure of relative power:
# it's the ratio of power in a given frequency bin - the 'signal' -
# compared to a 'noise' baseline - the average power in the surrounding frequency bins.
#
# Hence, we need to set some parameters for this baseline - how many
# neighboring bins should be taken for this computation, and do we want to skip
# the direct neighbors (this can make sense if the stimulation frequency is not
# super constant, or frequency bands are very narrow).
#
# The function below does what we want.
#

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
    # Construct a kernel that calculates the mean of the neighboring frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighborfreqs),
        np.zeros(2 * noise_skip_neighborfreqs + 1),
        np.ones(noise_n_neighborfreqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the averaging kernel.
    mean_noise = np.apply_along_axis(lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
                                     axis=-1, arr=psd)

    # The mean is not defined on the edges so we will pad it with nas. The padding needs to be done for the last
    # dimension only so we set it to (0, 0) for the other ones.
    edge_width = noise_n_neighborfreqs + noise_skip_neighborfreqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


###############################################################################
# Now we call the function to compute our SNR spectrum.
#
# As described above, we have to define two parameters.
#
# * how many noise bins do you want?
# * do you want to skip n bins directly next to the target bin?
#
# Tweaking these parameters *can* drastically impact the resulting spectrum,
# but mainly if you choose extremes.
# E.g. if you'd skip very many neighboring bins, broad band power modulations
# (such as the alpha peak) should reappear in the SNR spectrum.
# On the other hand, if you skip none you might miss or smear peaks if the
# induced power is distributed over two or more frequency bins (e.g. if the
# stimulation frequency isn't perfectly constant, or you have very narrow bins).
#
# Here, we want to compare power at each bin with average power of the
# **three neighboring bins** (on each side) and **skip one bin** directly next to it.
#


snrs = snr_spectrum(psds, noise_n_neighborfreqs=3,
                    noise_skip_neighborfreqs=1)

##############################################################################
# Plot PSD and SNR spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we will plot grand average PSD (in blue) and SNR (in red) +- STD
# for every frequency bin.
# PSD is plotted on a log scale.
#

# PSD code snippet from
# https://martinos.org/mne/stable/auto_examples/time_frequency/plot_compute_raw_data_spectrum.html  # noqa E501
fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', dpi=300)
rng = range(np.where(np.floor(freqs) == 1.)[0][0],
            np.where(np.ceil(freqs) == fmax - 1)[0][0])

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[rng]
psds_std = psds_plot.std(axis=(0, 1))[rng]
axes[0].plot(freqs[rng], psds_mean, color='b')
axes[0].fill_between(freqs[rng], psds_mean - psds_std, psds_mean + psds_std,
                  color='b', alpha=.2)
axes[0].set(title="PSD spectrum", ylabel='Power Spectral Density [dB]')

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[rng]
snr_std = snrs.std(axis=(0, 1))[rng]


#axes[1].plot(freqs, snrs.mean(axis=0).T, color='grey', alpha=0.1)
#axes[1].axvline(x=stim_freq, ls=':')
#axes[1].plot(freqs, snrs.mean(axis=(0, 1)), color='r')

axes[1].plot(freqs[rng], snr_mean, color='r')
axes[1].fill_between(freqs[rng], snr_mean - snr_std, snr_mean + snr_std,
                  color='r', alpha=.2)
axes[1].set(
    title="SNR spectrum", xlabel='Frequency [Hz]',
    ylabel='SNR', ylim=[-1, 15], xlim=[fmin, fmax])
fig.show()

###############################################################################
# You can see that the peaks at the stimulation frequencies (12 hz, 15 hz)
# and their harmonics are visible in both plots.
# Yet, the SNR spectrum shows them more prominently as peaks from a
# noisy but more or less constant baseline of SNR = 1.
# You can further see that the SNR processing removes any broad-band power
# differences (such as the increased power in alpha band around 10 hz),
# and also removes the 1/f decay in the PSD.
#
# Note, that while the SNR plot implies the possibility of values below 0
# (mean minus STD) such values do not make sense.
# Each SNR value is a ratio of positive PSD values, and the lowest possible PSD
# value is 0 (negative Y-axis values in the upper panel only result from
# plotting PSD on a log scale).
# Hence SNR values must be positive and can minimally go towards 0.
# You can nicely see this at 50hz with the artifact induced by notch
# filtering the line noise:
# The PSD spectrum shows a prominent negative peak and average SNR approaches 0.
#

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





##############################################################################
# SNR spectrum - averaged over channels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plt.figure()
plt.semilogy(freqs, snrs.mean(axis=1).T, color='grey', alpha=0.1)
plt.axvline(x=stim_freq, ls=':')
plt.semilogy(freqs, snrs.mean(axis=(0, 1)), color='r')
plt.gca().set(title="SNR spectrum - averaged over channels", xlabel='Frequency [Hz]',
              ylabel='SNR', ylim=[0, 20])
plt.show()

##############################################################################
# SNR topography - grand average per channel
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# create montage (here the default)
montage = mne.channels.make_standard_montage('easycap-M1', head_size=0.095)  # head_size parameter default = 0.095

# convert digitization to xyz coordinates
montage.positions = montage._get_ch_pos()  # luckily i dug this out in the mne code!

# plot montage, if wanted
# montage.plot(show=True)

# snr topography-plot grand average (all subs, all trials)

# get grand average SNR per channel (all subs, all trials) and electrode labels
snr_grave = snrs_stim.mean(axis=0)

# select only present channels from the standard montage
topo_pos_grave = []
[topo_pos_grave.append(montage.positions[ch][:2]) for ch in epochs.info['ch_names']]
topo_pos_grave = np.array(topo_pos_grave)

# plot SNR topography
f, ax = plt.subplots()
mne.viz.plot_topomap(snr_grave, topo_pos_grave, vmin=1., axes=ax)
print("sub 2, all trials")
print("average SNR: %f" % snr_grave.mean())

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
# Here we have already subsetted trials before snr calculation (only 12Hz
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
