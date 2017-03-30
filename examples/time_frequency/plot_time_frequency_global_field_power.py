"""
=============================================
Explore oscillatory activity in sensor space
=============================================

The objective is to show you how to explore spectrally localized
effects. For this purpose we adapt the method described in [1] and use it on
the somato dataset. The idea is to track the band-limited temporal evolution
of spatial patterns by using the Global Field Power (GFP).
We first bandpass filter the signals, then apply a Hilbert transform, then
subtract the avarage and finally rectify the signals prior to averaging.
Then the GFP is computed as described in [2], using the
sum of the squares devided by the true degrees of freedom.
Baselinging is
applied to make the GFPs comparabel between frequencies.
The procedure is then repeated for each frequency band of interest and
all GFPs are visualized. The non-parametric confidence intervals are computed
as described in [3].
The advantage of this method over summarizing the the Space x Time x Frequency
output of a Morlet Wavelet in frequency bands is relative speed and, more
importantly, the comparability of the spectral decomposition (the same type of
filter is used across all bands).

References
----------
.. [1] Haari R. and Salmelin R. Human cortical oscillations: a neuromagnetic
       view through the skull (1997). Trends in Neuroscience 20 (1),
       pp. 44-49.
.. [2] Engemann D. and Gramfort A. (2015) Automated model selection in
       covariance estimation and spatial whitening of MEG and EEG signals,
       vol. 108, 328-342, NeuroImage.
.. [3] Efron B. and Hastie T. Computer Age Statistical Inference (2016).
       Cambrdige University Press, Chapter 11.2.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import somato

###############################################################################
# Set parameters
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=False)
events = mne.find_events(raw, stim_channel='STI 014')

# let's explore some frequency bands
iter_freqs = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25),
    ('Gamma', 30, 45)
]

###############################################################################
# We create average power time courses for each frequency band

# set epoching parameters
event_id, tmin, tmax = 1, -1., 3.
baseline = None
frequency_map = list()
for band, fmin, fmax in iter_freqs:

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.pick_types(meg='grad', eog=True)  # we just look at gradiometers

    # bandpass filter and compute Hilbert
    raw.filter(fmin, fmax, n_jobs=1)  # use more jobs to speed up.
    raw.apply_hilbert(n_jobs=1, envelope=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=baseline,
                        reject=dict(grad=4000e-13, eog=350e-6),
                        preload=True)
    # remove evoked response and get analytic signal (envelope)
    data = epochs.get_data()
    mean = np.mean(data, axis=0, keepdims=True)
    epochs._data = np.abs(data - mean)
    # now average and move on
    frequency_map.append((band, epochs.average()))

###############################################################################
# Now we can compute the Global Field Power

# We first estimae the rank as this data is rank-reduced as SSS was applied.
# Therefore the degrees of freedom are less then the number of sensors.

rank = raw.estimate_rank()
print('The numerical rank is: %i' % rank)
rng = np.random.RandomState(42)
mne.utils.set_log_level('warning')

# Then we prepare a bootstrapping function to estimate confidence intervals


def get_gfp_ci(this_average, rank=rank):
    """get confidence intervals from non-parametric bootstrap"""
    indices = np.arange(len(this_average.ch_names), dtype=int)
    gfps_bs = list()
    for bootstrap_iteration in range(2000):
        bs_indices = rng.choice(indices, replace=True, size=len(indices))
        gfp_bs = np.sum(
            this_average.data[bs_indices] ** 2, 0) / rank
        gfps_bs.append(gfp_bs)
    gfps_bs = np.array(gfps_bs)
    gfps_bs = mne.baseline.rescale(
        gfps_bs, this_average.times, baseline=(None, 0))
    ci_low = np.percentile(gfps_bs, 2.5, axis=0)
    ci_up = np.percentile(gfps_bs, 97.5, axis=0)
    return ci_low, ci_up

# Now we can track the emergence of spatial patterns compared to baseline
# for each freqyenct band


fig, axes = plt.subplots(4, 1, figsize=(10, 7),
                         sharex=True, sharey=True)
colors = [plt.cm.viridis(ii) for ii in (0.1, 0.35, 0.75, 0.95)]
for (freq_name, average), color, ax in zip(
        frequency_map, colors, axes.ravel()[::-1]):
    this_average = average
    times = average.times * 1e3
    gfp = np.sum(this_average.data ** 2, 0) / rank
    gfp = mne.baseline.rescale(gfp, times, (None, 0))
    ax.plot(times,
            gfp,
            label=freq_name, color=color,
            linewidth=2.5)
    ax.plot(times, np.zeros_like(times), linestyle='--',
            color='red', linewidth=1)
    ci_low, ci_up = get_gfp_ci(this_average)
    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color,
                    alpha=0.3)
    ax.grid(True)
    ax.set_ylabel(freq_name)
    ax.annotate('%d-%dHz' % (fmin, fmax),
                xy=(0.8, 0.8),
                xycoords='axes fraction')
axes.ravel()[-1].set_xlabel('Time [ms]')


# We see dominant responses in the Alpha and Beta bands.
