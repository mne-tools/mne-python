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
of either 12Hz of 15Hz. 10 trials of 20s length each. 32ch wet EEG was recorded.

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
# - Raw data come with FCz recording reference, so we will apply common-average rereferencing.
# - We will apply a 50 Hz notch-filter to remove line-noise,
# - and a 0.1 - 250 Hz bandpass filter.
# - Lastly we will cut the data in 20 s epochs according to the trials.

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
raw.set_eeg_reference('average', projection=False, verbose=False)

# Apply notch filter to remove line-noise
notch = np.arange(raw.info['line_freq'], raw.info['lowpass'] / 2,
                  raw.info['line_freq'])
raw.notch_filter(notch, filter_length='auto', phase='zero', verbose=False)

# Apply bandpass filter
hp = .1
lp = 250.
raw.filter(hp, lp, fir_design='firwin', verbose=False)

# Construct epochs
event_id = {
    '12hz': 10001,
    '15hz': 10002
}
events, _ = mne.events_from_annotations(raw, verbose=False)
raw.info["events"] = events
tmin, tmax = -1., 20.  # in s
baseline = None
epochs = mne.Epochs(raw, events=events, event_id=[event_id['12hz'], event_id['15hz']], tmin=tmin,
                    tmax=tmax, baseline=baseline, verbose=False)

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
# We chose a frequency resolution of 0.1 hz.
# You could compare it with, e.g., multitaper to get an impression of the
# influence on SNR. All the other methods implemented in MNE can be used as
# well.

tmin = 0.
tmax = 20.
fmin = 1.
fmax = 90.
sf = epochs.info['sfreq']

psds, freqs = mne.time_frequency.psd_welch(
    epochs,
    n_fft=int(sf * 10), n_overlap=int(sf * .5), n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax, verbose=False)


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
# Subsetting data
# ---------------
#
# Our processing yielded a large array of many SNR values for each trial x
# channel x frequency-bin of the PSD array.
#
# For statistical analysis we obviously need to define specific subsets of this
# array. First of all, we are only interested in SNR at the stimulation frequency,
# but we also wanted to restrict analysis to a spatial ROI. The most interesting
# questions, however, will probably rely on comparing SNR in different trials.
#
# Since we here have a large SNR array with all conditions, we will have to find
# the indices of trials, channels, etc.
# Alternatively, one could subselect the trials already at the epoching step,
# using MNE's event information, and process different epoch structures
# individually.
#
# Let's have a look at the trials with 12 hz stimulation, for now.
#

# define stimulation frequency
stim_freq = 12.

###############################################################################
# get index for the stimulation frequency (12hz)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Ideally, there would be a bin with the stimulation frequency exactly in its
# center. However, depending on your Spectral decomposition this is not
# always the case. We will find the bin closest to it (this one should contain
# our frequency tagged response.
#

# find index of frequency bin closest to stimulation frequency
i_bin_12hz = np.argmin(abs(np.subtract(freqs, stim_freq)))
# could be updated to support multiple frequencies

# for later, we will already find the 15 hz bin and the 1st and 2nd harmonic for both.
i_bin_24hz = np.argmin(abs(np.subtract(freqs, 24)))
i_bin_36hz = np.argmin(abs(np.subtract(freqs, 36)))
i_bin_15hz = np.argmin(abs(np.subtract(freqs, 15)))
i_bin_30hz = np.argmin(abs(np.subtract(freqs, 30)))
i_bin_45hz = np.argmin(abs(np.subtract(freqs, 45)))


###############################################################################
# get indices for the different trial types
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

i_trial_12hz = np.where(epochs.events[:, 2] == event_id['12hz'])
i_trial_15hz = np.where(epochs.events[:, 2] == event_id['15hz'])


###############################################################################
# get indices for the EEG channels forming the ROI
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Define different ROIs
roi_temporal = ['T7', 'F7', 'T8', 'F8']  # temporal
roi_aud = ['AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'F1', 'FC1',
           'C1', 'CP1', 'F2', 'FC2', 'C2', 'CP2']  # auditory roi
roi_vis = ['POz', 'Oz', 'O1', 'O2', 'PO3', 'PO4', 'PO7',
           'PO8', 'PO9', 'PO10', 'O9', 'O10']  # visual roi

# Find corresponding indices using mne.pick_types()
picks_roi_temp = mne.pick_types(epochs.info, eeg=True, stim=False,
                                exclude='bads', selection=roi_temporal)
picks_roi_aud = mne.pick_types(epochs.info, eeg=True, stim=False,
                               exclude='bads', selection=roi_aud)
picks_roi_vis = mne.pick_types(epochs.info, eeg=True, stim=False,
                               exclude='bads', selection=roi_vis)

###############################################################################
# Apply the subset, and check the result
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we simply need to apply our selection and yield a result. Therefore,
# we typically report grand average SNR over the subselection.
#

snrs_target = snrs[i_trial_12hz, :, i_bin_12hz][0][:, picks_roi_vis]
print("sub 2, 12hz trials, SNR at 12hz")
print('average SNR (occipital ROI): %f' % snrs_target.mean())


##############################################################################
# Topography of the vSSR
# ----------------------
# But wait...
# As described in the intro, we have decided *a priori* to work with average
# SNR over a subset of occipital channels - a visual region of interest (ROI)
# - because we expect SNR to be higher on these channels than in other channels.
#
# Let's check out, whether this was a good decision!
#
# Here we will plot average SNR for each channel location as a topoplot.
# Then we will do a simple paired T-test to check, whether average SNRs over
# the two sets of channels are significantly different.

# get average SNR at 12hz for ALL channels
snrs_12hz = snrs[i_trial_12hz, :, i_bin_12hz][0]
snrs_12hz_chaverage = snrs_12hz.mean(axis=0)

# create a standard montage
montage = mne.channels.make_standard_montage('easycap-M1', head_size=0.095)  # head_size parameter default = 0.095

# add xyz coordinates for all channels
montage.positions = montage._get_ch_pos()  # luckily i dug this out in the mne code!

# select only channels from the standard montage that are present in our data
topo_pos_grave = []
[topo_pos_grave.append(montage.positions[ch][:2]) for ch in epochs.info['ch_names']]
topo_pos_grave = np.array(topo_pos_grave)

# plot SNR topography, eventually
f, ax = plt.subplots()
mne.viz.plot_topomap(snrs_12hz_chaverage, topo_pos_grave, vmin=1., axes=ax)

print("sub 2, 12hz trials, SNR at 12hz")
print("average SNR (all channels): %f" % snrs_12hz_chaverage.mean())
print("average SNR (occipital ROI): %f" % snrs_target.mean())

tstat_roi_vs_scalp = ttest_rel(snrs_target.mean(axis=1), snrs_12hz.mean(axis=1))
print("12 hz SNR in occipital ROI is significantly larger than 12 hz SNR over all channels"
      ": t = %.3f, p = %f" % tstat_roi_vs_scalp)


##############################################################################
# We can see, that 1) this participant indeed exhibits a cluster of chanels
# with high SNR in the occipital region and 2) that the average SNR over all
# channels is smaller than the average of the visual ROI computed above.
# The difference is statistically significant. Great!
#
# Such a topoplot can be a nice tool to explore and play with your data - e.g.
# you could try how changing the reference will affect the spatial
# distribution of SNR values.
#
# However, we also wanted to show this plot to illustrate
# a large problem with frequency-tagged or any other brain imaging data:
# there are many channels and somewhere you will likely find some significant
# effects. It's very easy - even unintended - to end up
# double-dipping or p-hacking.
# Avoid this either by selecting your ROI or channels for analysis *a priori*
# (and ideally preregister this decision, so people will believe you), or if
# you select an ROI or individual channel for reporting *because this channel
# or ROI shows an effect* do so transparently and correct for multiple
# comparison.


##############################################################################
# Statistical analysis
# --------------------
# Now that we finished this edgy little open science lesson let's move on and
# do the analyses we actually wanted to do:
#
# We will show that we can well discriminate the two responses in the
# different trials.
#
# In the frequency and SNR spectrum plot above, we had all trials mixed up.
# Now we will extract 12 and 15 hz SNR as well as the 1st and 2nd harmonic
# in both types of trials individually, and compare the values with a
# simple t-test.
#

snrs_roi = snrs[:, picks_roi_vis, :].mean(axis=1)

freq_plot = [12,15,24,30,36,45]
color_plot = ['darkblue', 'darkgreen', 'mediumblue', 'green', 'blue', 'seagreen']
xpos_plot = [-5./12, -3./12, -1./12, 1./12, 3./12, 5./12]
fig, ax = plt.subplots()
labels = ['12hz trials', '15hz trials']
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
res = dict()

# loop to plot SNRs at stimulation frequencies and harmonics
for i, f in enumerate(freq_plot):
    # extract snrs
    stim_12hz_tmp = snrs_roi[i_trial_12hz, np.argmin(abs(np.subtract(freqs, f)))][0]
    stim_15hz_tmp = snrs_roi[i_trial_15hz, np.argmin(abs(np.subtract(freqs, f)))][0]
    SNR_tmp = [stim_12hz_tmp.mean(), stim_15hz_tmp.mean()]
    # plot (with std)
    ax.bar(
        x + width * xpos_plot[i], SNR_tmp, width / len(freq_plot),
        yerr=np.std(SNR_tmp),
        label='%ihz SNR' % f, color=color_plot[i])
    # store results for statistical comparison
    res['stim_12hz_snrs_%ihz' % f] = stim_12hz_tmp
    res['stim_15hz_snrs_%ihz' % f] = stim_15hz_tmp

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SNR')
ax.set_title('Average SNR at target frequencies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(['%ihz' % f for f in freq_plot], title='SNR at:')
ax.set_ylim([0, 20])
ax.axhline(1, ls='--', c='r')
# fig.tight_layout()
fig.show()

##############################################################################
# As you can easily see there are striking differences.
# Let's verify this using a series of two-tailed paired T-Tests.
#

# Compare 12 hz and 15 hz SNR in trials after averaging over channels

tstat_12hz_trial_stim = ttest_rel(res['stim_12hz_snrs_12hz'], res['stim_12hz_snrs_15hz'])
print("12 hz Trials: 12 hz SNR is significantly higher than 15 hz SNR"
      ": t = %.3f, p = %f" % tstat_12hz_trial_stim)

tstat_12hz_trial_1st_harmonic = ttest_rel(res['stim_12hz_snrs_24hz'], res['stim_12hz_snrs_30hz'])
print("12 hz Trials: 24 hz SNR is significantly higher than 30 hz SNR"
      ": t = %.3f, p = %f" % tstat_12hz_trial_1st_harmonic)

tstat_12hz_trial_2nd_harmonic = ttest_rel(res['stim_12hz_snrs_36hz'], res['stim_12hz_snrs_45hz'])
print("12 hz Trials: 36 hz SNR is significantly higher than 45 hz SNR"
      ": t = %.3f, p = %f" % tstat_12hz_trial_2nd_harmonic)

print()
tstat_15hz_trial_stim = ttest_rel(res['stim_15hz_snrs_12hz'], res['stim_15hz_snrs_15hz'])
print("15 hz trials: 12 hz SNR is significantly lower than 15 hz SNR"
      ": t = %.3f, p = %f" % tstat_15hz_trial_stim)

tstat_15hz_trial_1st_harmonic = ttest_rel(res['stim_15hz_snrs_24hz'], res['stim_15hz_snrs_30hz'])
print("15 hz trials: 24 hz SNR is significantly lower than 30 hz SNR"
      ": t = %.3f, p = %f" % tstat_15hz_trial_1st_harmonic)

tstat_15hz_trial_2nd_harmonic = ttest_rel(res['stim_15hz_snrs_36hz'], res['stim_15hz_snrs_45hz'])
print("15 hz trials: 36 hz SNR is significantly lower than 45 hz SNR"
      ": t = %.3f, p = %f" % tstat_15hz_trial_2nd_harmonic)

##############################################################################
# Debriefing
# ----------
# tbd
#

##############################################################################
# Bonus exercises
# ---------------
# For the overly motivated amongst you, let's see what else we can show with
# these our data..
#
# Using Welch's method as implemented in MNE makes it very easy to change
# the amount of data that is actually used in the spectrum
# estimation. We can tweak both the length of the actual trial, and the length
# of the Welch window.
#
# Here we employ this to show you some features of frequency
# tagging data that you might or might not have already intuitively expected:
#

##############################################################################
# Effect of trial duration on SNR
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First we will simulate shorter trials by taking the first x s of our 20s trials
# (2,4,6,8, ..,20 s), and compute the SNR using a welch window that covers the
# entire epoch:
#

stim_bandwidth = .5

# shorten data and welch window
window_lengths = [i for i in range(2, 21, 2)]
window_snrs = [[]] * len(window_lengths)
for i_win, win in enumerate(window_lengths):
    # compute spectrogram
    windowed_psd, windowed_freqs = mne.time_frequency.psd_welch(
        epochs[str(event_id['12hz'])],
        n_fft=int(sf * win), n_overlap=0, n_per_seg=None,
        tmin=0, tmax=win,
        fmin=fmin, fmax=fmax, verbose=False)
    # define a bandwidth of 1hz around stimfreq for SNR computation
    bin_width = windowed_freqs[1] - windowed_freqs[0]
    noise_skip_neighborfreqs = round((stim_bandwidth/2)/bin_width - bin_width/2. - .5) if bin_width < stim_bandwidth else 0
    noise_n_neighborfreqs = int((sum((windowed_freqs <= 13) & (windowed_freqs >= 11)) - 1 - 2*noise_skip_neighborfreqs) / 2)
    # compute snr
    windowed_snrs = snr_spectrum(windowed_psd, noise_n_neighborfreqs=noise_n_neighborfreqs if noise_n_neighborfreqs > 0 else 1,
                    noise_skip_neighborfreqs=noise_skip_neighborfreqs)
    window_snrs[i_win] = windowed_snrs[:, picks_roi_vis, np.argmin(abs(np.subtract(windowed_freqs, 12.)))].mean(axis=1)

plt.boxplot(window_snrs, labels=window_lengths, vert=True)
plt.title('Effect of trial length on 12hz SNR')
plt.axhline(1, ls='--', c='r')
plt.ylabel('average SNR')
plt.xlabel('trial length [s]')
plt.show()

##############################################################################
# You can see that the signal estimate / our SNR measure increases with the
# trial duration.
#
# This should be easy to understand: in longer recordings there is simply more signal
# (one second of additional stimulation adds, in our case, 12 cycles of signal)
# while the noise is (hopefully) stochastic and not locked to the stimulation frequency.
# in other words: with more data the signal term grows faster than the noise term.
#

##############################################################################
# Effect of Welch window
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's now look at the effect of the Welch window from two different perspectives:
# For the first plot we kept the entire 20s of data, but changed the length of the Welch window.
# For the second plot we fixed the Welch window, and again simulated different trial lengths.
#


# fig 1 - shorten welch window, full data
window_lengths = [i for i in range(2,21,2)]
window_snrs = [[]] * len(window_lengths)
for i_win, win in enumerate(window_lengths):
    # compute spectrogram
    windowed_psd, windowed_freqs = mne.time_frequency.psd_welch(
        epochs[str(event_id['12hz'])],
        n_fft=int(sf * win), n_overlap=int(sf * (win - .1)), n_per_seg=None,
        tmin=tmin, tmax=tmax,
        fmin=fmin, fmax=fmax, verbose=False)
    # define a bandwidth of 1hz around stimfreq for SNR computation
    bin_width = windowed_freqs[1] - windowed_freqs[0]
    noise_skip_neighborfreqs = round((stim_bandwidth/2)/bin_width - bin_width/2. - .5) if bin_width < stim_bandwidth else 0
    noise_n_neighborfreqs = int((sum((windowed_freqs <= 13) & (windowed_freqs >= 11)) - 1 - 2*noise_skip_neighborfreqs) / 2)
    # compute snr
    windowed_snrs = snr_spectrum(windowed_psd, noise_n_neighborfreqs=noise_n_neighborfreqs if noise_n_neighborfreqs > 0 else 1,
                    noise_skip_neighborfreqs=1)
    window_snrs[i_win] = windowed_snrs[:, picks_roi_vis, np.argmin(abs(np.subtract(windowed_freqs, 12.)))].mean(axis=1)

plt.figure()
plt.boxplot(window_snrs, labels=window_lengths, vert=True)
plt.title('Effect of time window length on 12hz SNR - adapt only welch window')
plt.title('Varying Welch window, fixed trial length')
plt.ylabel('average SNR')
plt.xlabel('Welch window [s]')
plt.axhline(1, ls='--', c='r')
plt.show()


# fig 2 - shorten data, welch window fix
window_lengths = [i for i in range(2,21,2)]
window_lengths[0] = 3
window_snrs = [[]] * len(window_lengths)
for i_win, win in enumerate(window_lengths):
    # compute spectrogram
    windowed_psd, windowed_freqs = mne.time_frequency.psd_welch(
        epochs[str(event_id['12hz'])],
        n_fft=int(sf * 3.), n_overlap=int(sf * 2.9), n_per_seg=None,
        tmin=0, tmax=win,
        fmin=fmin, fmax=fmax, verbose=False)
    # define a bandwidth of 1hz around stimfreq for SNR computation
    bin_width = windowed_freqs[1] - windowed_freqs[0]
    noise_skip_neighborfreqs = round((stim_bandwidth/2)/bin_width - bin_width/2. - .5) if bin_width < stim_bandwidth else 0
    noise_n_neighborfreqs = int((sum((windowed_freqs <= 13) & (windowed_freqs >= 11)) - 1 - 2*noise_skip_neighborfreqs) / 2)
    # compute snr
    windowed_snrs = snr_spectrum(windowed_psd, noise_n_neighborfreqs=noise_n_neighborfreqs if noise_n_neighborfreqs > 0 else 1,
                    noise_skip_neighborfreqs=noise_skip_neighborfreqs)
    window_snrs[i_win] = windowed_snrs[:, picks_roi_vis, np.argmin(abs(np.subtract(windowed_freqs, 12.)))].mean(axis=1)

plt.figure()
plt.boxplot(window_snrs, labels=window_lengths, vert=True)
plt.title('3s Welch window, varying trial length')
plt.ylabel('average SNR')
plt.xlabel('trial length [s]')
plt.axhline(1, ls='--', c='r')
plt.show()

##############################################################################
# In these plots you can see that a shorter Welch window (relative to the amount of data)
# leads not only to lower SNR values (as described above)
# but also to lower variance / higher confidence in the means.
#
# This is the very idea of Welch's method, and is relevant to remember when
# you decide for your analysis parameters:
#
# there is no such thing as an optimal welch window, its always a tradeoff -
# - longer window => higher average SNR values
# - shorter window => lower values but more robust / higher confidence in the measure.
#
# Yet, one should also not aim too low: in these data we can see that the very
# short Welch windows < 2-3s are not great - here we've hit the noise bottom.


##############################################################################
# Time resolved SNR
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ..and finally we can trick MNE's PSD implementation to make it a
# sliding window analysis and come up with a time resolved SNR measure.
# This will reveal whether a participant blinked or scratched their head..
# differently colored lines for the ten different trials.
#


# 3s sliding window
window_length = 3
window_starts = [i for i in range(20-window_length)]
window_snrs = [[]] * len(window_starts)

for i_win, win in enumerate(window_starts):
    # compute spectrogram
    windowed_psd, windowed_freqs = mne.time_frequency.psd_welch(
        epochs[str(event_id['12hz'])],
        n_fft=int(sf * window_length)-1, n_overlap=0, n_per_seg=None,
        tmin=win, tmax=win+window_length,
        fmin=fmin, fmax=fmax, verbose=False)
    # define a bandwidth of 1hz around stimfreq for SNR computation
    bin_width = windowed_freqs[1] - windowed_freqs[0]
    noise_skip_neighborfreqs = round((stim_bandwidth/2)/bin_width - bin_width/2. - .5) if bin_width < stim_bandwidth else 0
    noise_n_neighborfreqs = int((sum((windowed_freqs <= 13) & (windowed_freqs >= 11)) - 1 - 2*noise_skip_neighborfreqs) / 2)
    # compute snr
    windowed_snrs = snr_spectrum(windowed_psd, noise_n_neighborfreqs=noise_n_neighborfreqs if noise_n_neighborfreqs > 0 else 1,
                    noise_skip_neighborfreqs=noise_skip_neighborfreqs)
    window_snrs[i_win] = windowed_snrs[:, picks_roi_vis, np.argmin(abs(np.subtract(windowed_freqs, 12.)))].mean(axis=1)

plt.figure()
plt.plot(window_starts, np.array(window_snrs))
plt.title('Time resolved 12hz SNR - %is sliding window' % window_length)
plt.legend(['individual trials'])
plt.ylabel('average SNR')
plt.xlabel('t0 of analysis window [s]')
plt.axhline(1, ls='--', c='r')
plt.show()

##############################################################################
# well.. turns out this was a bit too optimistic ;)
#
# but seriously: this was a nice idea, but we've simply reached the limit of
# what's possible with this single-subject example dataset.
# There are certainly data or applications where such an analysis makes more sense.



