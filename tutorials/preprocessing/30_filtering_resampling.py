# -*- coding: utf-8 -*-
"""
.. _tut-filter-resample:

=============================
Filtering and resampling data
=============================

This tutorial covers filtering and resampling, and gives examples of how
filtering can be used for artifact repair.

We begin as always by importing the necessary Python modules and loading some
:ref:`example data <sample-dataset>`. We'll also crop the data to 60 seconds
(to save memory on the documentation server):
"""

# %%

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
# use just 60 seconds of data and mag channels, to save memory
raw.crop(0, 60).pick_types(meg='mag', stim=True).load_data()

# %%
# Background on filtering
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# A filter removes or attenuates parts of a signal. Usually, filters act on
# specific *frequency ranges* of a signal — for example, suppressing all
# frequency components above or below a certain cutoff value. There are *many*
# ways of designing digital filters; see :ref:`disc-filtering` for a longer
# discussion of the various approaches to filtering physiological signals in
# MNE-Python.
#
#
# Repairing artifacts by filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Artifacts that are restricted to a narrow frequency range can sometimes
# be repaired by filtering the data. Two examples of frequency-restricted
# artifacts are slow drifts and power line noise. Here we illustrate how each
# of these can be repaired by filtering.
#
#
# Slow drifts
# ~~~~~~~~~~~
#
# Low-frequency drifts in raw data can usually be spotted by plotting a fairly
# long span of data with the :meth:`~mne.io.Raw.plot` method, though it is
# helpful to disable channel-wise DC shift correction to make slow drifts
# more readily visible. Here we plot 60 seconds, showing all the magnetometer
# channels:

raw.plot(duration=60, proj=False, n_channels=len(raw.ch_names),
         remove_dc=False)

# %%
# A half-period of this slow drift appears to last around 10 seconds, so a full
# period would be 20 seconds, i.e., :math:`\frac{1}{20} \mathrm{Hz}`. To be
# sure those components are excluded, we want our highpass to be *higher* than
# that, so let's try :math:`\frac{1}{10} \mathrm{Hz}` and :math:`\frac{1}{5}
# \mathrm{Hz}` filters to see which works best:

for cutoff in (0.1, 0.2):
    raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
    with mne.viz.use_browser_backend('matplotlib'):
        fig = raw_highpass.plot(duration=60, proj=False,
                                n_channels=len(raw.ch_names), remove_dc=False)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
                 weight='bold')

# %%
# Looks like 0.1 Hz was not quite high enough to fully remove the slow drifts.
# Notice that the text output summarizes the relevant characteristics of the
# filter that was created. If you want to visualize the filter, you can pass
# the same arguments used in the call to :meth:`raw.filter()
# <mne.io.Raw.filter>` above to the function :func:`mne.filter.create_filter`
# to get the filter parameters, and then pass the filter parameters to
# :func:`mne.viz.plot_filter`. :func:`~mne.filter.create_filter` also requires
# parameters ``data`` (a :class:`NumPy array <numpy.ndarray>`) and ``sfreq``
# (the sampling frequency of the data), so we'll extract those from our
# :class:`~mne.io.Raw` object:

filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'],
                                         l_freq=0.2, h_freq=None)

# %%
# Notice that the output is the same as when we applied this filter to the data
# using :meth:`raw.filter() <mne.io.Raw.filter>`. You can now pass the filter
# parameters (and the sampling frequency) to :func:`~mne.viz.plot_filter` to
# plot the filter:

mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.01, 5))

# %%
# .. _tut-section-line-noise:
#
# Power line noise
# ~~~~~~~~~~~~~~~~
#
# Power line noise is an environmental artifact that manifests as persistent
# oscillations centered around the `AC power line frequency`_. Power line
# artifacts are easiest to see on plots of the spectrum, so we'll use
# :meth:`~mne.io.Raw.plot_psd` to illustrate. We'll also write a little
# function that adds arrows to the spectrum plot to highlight the artifacts:


def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)


fig = raw.plot_psd(fmax=250, average=True)
add_arrows(fig.axes[:2])

# %%
# It should be evident that MEG channels are more susceptible to this kind of
# interference than EEG that is recorded in the magnetically shielded room.
# Removing power-line noise can be done with a notch filter,
# applied directly to the :class:`~mne.io.Raw` object, specifying an array of
# frequencies to be attenuated. Since the EEG channels are relatively
# unaffected by the power line noise, we'll also specify a ``picks`` argument
# so that only the magnetometers and gradiometers get filtered:

meg_picks = mne.pick_types(raw.info, meg=True)
freqs = (60, 120, 180, 240)
raw_notch = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)
for title, data in zip(['Un', 'Notch '], [raw, raw_notch]):
    fig = data.plot_psd(fmax=250, average=True)
    fig.subplots_adjust(top=0.85)
    fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
    add_arrows(fig.axes[:2])

# %%
# :meth:`~mne.io.Raw.notch_filter` also has parameters to control the notch
# width, transition bandwidth and other aspects of the filter. See the
# docstring for details.
#
# It's also possible to try to use a spectrum fitting routine to notch filter.
# In principle it can automatically detect the frequencies to notch, but our
# implementation generally does not do so reliably, so we specify the
# frequencies to remove instead, and it does a good job of removing the
# line noise at those frequencies:

raw_notch_fit = raw.copy().notch_filter(
    freqs=freqs, picks=meg_picks, method='spectrum_fit', filter_length='10s')
for title, data in zip(['Un', 'spectrum_fit '], [raw, raw_notch_fit]):
    fig = data.plot_psd(fmax=250, average=True)
    fig.subplots_adjust(top=0.85)
    fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
    add_arrows(fig.axes[:2])

# %%
# Resampling
# ^^^^^^^^^^
#
# EEG and MEG recordings are notable for their high temporal precision, and are
# often recorded with sampling rates around 1000 Hz or higher. This is good
# when precise timing of events is important to the experimental design or
# analysis plan, but also consumes more memory and computational resources when
# processing the data. In cases where high-frequency components of the signal
# are not of interest and precise timing is not needed (e.g., computing EOG or
# ECG projectors on a long recording), downsampling the signal can be a useful
# time-saver.
#
# In MNE-Python, the resampling methods (:meth:`raw.resample()
# <mne.io.Raw.resample>`, :meth:`epochs.resample() <mne.Epochs.resample>` and
# :meth:`evoked.resample() <mne.Evoked.resample>`) apply a low-pass filter to
# the signal to avoid `aliasing`_, so you don't need to explicitly filter it
# yourself first. This built-in filtering that happens when using
# :meth:`raw.resample() <mne.io.Raw.resample>`, :meth:`epochs.resample()
# <mne.Epochs.resample>`, or :meth:`evoked.resample() <mne.Evoked.resample>` is
# a brick-wall filter applied in the frequency domain at the `Nyquist
# frequency`_ of the desired new sampling rate. This can be clearly seen in the
# PSD plot, where a dashed vertical line indicates the filter cutoff; the
# original data had an existing lowpass at around 172 Hz (see
# ``raw.info['lowpass']``), and the data resampled from 600 Hz to 200 Hz gets
# automatically lowpass filtered at 100 Hz (the `Nyquist frequency`_ for a
# target rate of 200 Hz):

raw_downsampled = raw.copy().resample(sfreq=200)

for data, title in zip([raw, raw_downsampled], ['Original', 'Downsampled']):
    fig = data.plot_psd(average=True)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title)
    plt.setp(fig.axes, xlim=(0, 300))

# %%
# Because resampling involves filtering, there are some pitfalls to resampling
# at different points in the analysis stream:
#
# - Performing resampling on :class:`~mne.io.Raw` data (*before* epoching) will
#   negatively affect the temporal precision of Event arrays, by causing
#   `jitter`_ in the event timing. This reduced temporal precision will
#   propagate to subsequent epoching operations.
#
# - Performing resampling *after* epoching can introduce edge artifacts *on
#   every epoch*, whereas filtering the :class:`~mne.io.Raw` object will only
#   introduce artifacts at the start and end of the recording (which is often
#   far enough from the first and last epochs to have no affect on the
#   analysis).
#
# The following section suggests best practices to mitigate both of these
# issues.
#
#
# Best practices
# ~~~~~~~~~~~~~~
#
# To avoid the reduction in temporal precision of events that comes with
# resampling a :class:`~mne.io.Raw` object, and also avoid the edge artifacts
# that come with filtering an :class:`~mne.Epochs` or :class:`~mne.Evoked`
# object, the best practice is to:
#
# 1. low-pass filter the :class:`~mne.io.Raw` data at or below
#    :math:`\frac{1}{3}` of the desired sample rate, then
#
# 2. decimate the data after epoching, by either passing the ``decim``
#    parameter to the :class:`~mne.Epochs` constructor, or using the
#    :meth:`~mne.Epochs.decimate` method after the :class:`~mne.Epochs` have
#    been created.
#
# .. warning::
#    The recommendation for setting the low-pass corner frequency at
#    :math:`\frac{1}{3}` of the desired sample rate is a fairly safe rule of
#    thumb based on the default settings in :meth:`raw.filter()
#    <mne.io.Raw.filter>` (which are different from the filter settings used
#    inside the :meth:`raw.resample() <mne.io.Raw.resample>` method). If you
#    use a customized lowpass filter (specifically, if your transition
#    bandwidth is wider than 0.5× the lowpass cutoff), downsampling to 3× the
#    lowpass cutoff may still not be enough to avoid `aliasing`_, and
#    MNE-Python will not warn you about it (because the :class:`raw.info
#    <mne.Info>` object only keeps track of the lowpass cutoff, not the
#    transition bandwidth). Conversely, if you use a steeper filter, the
#    warning may be too sensitive. If you are unsure, plot the PSD of your
#    filtered data *before decimating* and ensure that there is no content in
#    the frequencies above the `Nyquist frequency`_ of the sample rate you'll
#    end up with *after* decimation.
#
# Note that this method of manually filtering and decimating is exact only when
# the original sampling frequency is an integer multiple of the desired new
# sampling frequency. Since the sampling frequency of our example data is
# 600.614990234375 Hz, ending up with a specific sampling frequency like (say)
# 90 Hz will not be possible:

current_sfreq = raw.info['sfreq']
desired_sfreq = 90  # Hz
decim = np.round(current_sfreq / desired_sfreq).astype(int)
obtained_sfreq = current_sfreq / decim
lowpass_freq = obtained_sfreq / 3.

raw_filtered = raw.copy().filter(l_freq=None, h_freq=lowpass_freq)
events = mne.find_events(raw_filtered)
epochs = mne.Epochs(raw_filtered, events, decim=decim)

print('desired sampling frequency was {} Hz; decim factor of {} yielded an '
      'actual sampling frequency of {} Hz.'
      .format(desired_sfreq, decim, epochs.info['sfreq']))

# %%
# If for some reason you cannot follow the above-recommended best practices,
# you should at the very least either:
#
# 1. resample the data *after* epoching, and make your epochs long enough that
#    edge effects from the filtering do not affect the temporal span of the
#    epoch that you hope to analyze / interpret; or
#
# 2. perform resampling on the :class:`~mne.io.Raw` object and its
#    corresponding Events array *simultaneously* so that they stay more or less
#    in synch. This can be done by passing the Events array as the
#    ``events`` parameter to :meth:`raw.resample() <mne.io.Raw.resample>`.
#
#
# .. LINKS
#
# .. _`AC power line frequency`:
#    https://en.wikipedia.org/wiki/Mains_electricity
# .. _`aliasing`: https://en.wikipedia.org/wiki/Anti-aliasing_filter
# .. _`jitter`: https://en.wikipedia.org/wiki/Jitter
# .. _`Nyquist frequency`: https://en.wikipedia.org/wiki/Nyquist_frequency
