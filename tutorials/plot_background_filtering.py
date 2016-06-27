# -*- coding: utf-8 -*-
r"""
.. _tut_background_filtering:

===================================
Background information on filtering
===================================

Here we give some background information on filtering in general,
and how it is done in MNE-Python in particular.
Recommended reading for practical applications of digital
filter design can be found in [1]_. To see how to use the default filters
in MNE-Python on actual data, see the :ref:`tut_artifacts_filter` tutorial.

.. contents::

.. _filtering-basics:

Filtering basics
================
Let's get some of the basic math down. In the frequency domain, digital
filters have a transfer function that is given by:

.. math::

    H(z) &= \frac{b_0 + b_1 z^{-1} + b_2 z^{-2} + ... + b_M z^{-M}}
                 {1 + a_1 z^{-1} + a_2 z^{-2} + ... + a_N z^{-M}} \\
         &= \frac{\sum_0^Mb_kz^{-k}}{\sum_1^Na_kz^{-k}}

In the time domain, the numerator coefficients :math:`b_k` and denominator
coefficients :math:`a_k` can be used to obtain our output data
:math:`y(n)` in terms of our input data :math:`x(n)` as:

.. math::
   :label: summations

    y(n) &= b_0 x(n) + b_1 x(n-1) + ... + b_M x(n-M)
            - a_1 y(n-1) - a_2 y(n - 2) - ... - a_N y(n - N)\\
         &= \sum_0^M b_k x(n-k) - \sum_1^N a_k y(n-k)

In other words, the output at time :math:`n` is determined by a sum over:

    1. The numerator coefficients :math:`b_k`, which get multiplied by
       the previous input :math:`x(n-k)` values, and
    2. The denominator coefficients :math:`a_k`, which get multiplied by
       the previous output :math:`y(n-k)` values.

Note that these summations in :eq:`summations` correspond nicely to
(1) a weighted `moving average`_ and (2) an autoregression_.

Filters are broken into two classes: FIR_ (finite impulse response) and
IIR_ (infinite impulse response) based on these coefficients.
FIR filters use a finite number of numerator
coefficients :math:`b_k` (:math:`\forall k, a_k=0`), and thus each output
value of :math:`y(n)` depends only on the :math:`M` previous input values.
IIR filters depend on the previous input and output values, and thus can have
effectively infinite impulse responses.

As outlined in [1]_, FIR and IIR have different tradeoffs:

    * A causal FIR filter can be linear-phase -- i.e., the same time delay
      across all frequencies -- whereas a causal IIR filter cannot. The phase
      and group delay characteristics are also usually better for FIR filters.
    * IIR filters can generally have a steeper cutoff than an FIR filter of
      equivalent order.
    * IIR filters are generally less numerically stable, in part due to
      accumulating error (due to its recursive calculations).

When designing a filter (FIR or IIR), there are always tradeoffs that
need to be considered, including but not limited to:

    1. Ripple in the pass-band
    2. Attenuation of the stop-band
    3. Steepness of roll-off
    4. Filter order (i.e., length for FIR filters)
    5. Time-domain ringing

In general, the sharper something is in frequency, the broader it is in time,
and vice-versa. This is a fundamental time-frequency tradeoff, and it will
show up below.

===========

First we will focus first on FIR filters, which are the default filters used by
MNE-Python.
"""

###############################################################################
# Designing FIR filters
# ---------------------
# Here we'll try designing a low-pass filter, and look at trade-offs in terms
# of time- and frequency-domain filter characteristics. Later, in
# :ref:`effect_on_signals`, we'll look at how such filters can affect
# signals when they are used.
#
# First let's import some useful tools for filtering, and set some default
# values for our data that are reasonable for M/EEG data.

import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt

from mne.time_frequency.tfr import morlet

import mne

sfreq = 1000.
f_p = 40.
ylim = [-60, 10]  # for dB plots
xlim = [2, sfreq / 2.]
blue = '#1f77b4'

###############################################################################
# Take for example an ideal low-pass filter, which would give a value of 1 in
# the pass-band (up to frequency :math:`f_p`) and a value of 0 in the stop-band
# (down to frequency :math:`f_s`) such that :math:`f_p=f_s=40` Hz here
# (shown to a lower limit of -60 dB for simplicity):

nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate
freq = [0, f_p, f_p, nyq]
gain = [1, 1, 0, 0]


def box_off(ax):
    ax.grid(zorder=0)
    for key in ('top', 'right'):
        ax.spines[key].set_visible(False)


def plot_ideal(freq, gain, ax):
    freq = np.maximum(freq, xlim[0])
    xs, ys = list(), list()
    for ii in range(len(freq)):
        xs.append(freq[ii])
        ys.append(ylim[0])
        if ii < len(freq) - 1 and gain[ii] != gain[ii + 1]:
            xs += [freq[ii], freq[ii + 1]]
            ys += [ylim[1]] * 2
    gain = 10 * np.log10(np.maximum(gain, 10 ** (ylim[0] / 10.)))
    ax.fill_between(xs, ylim[0], ys, color='r', alpha=0.1)
    ax.semilogx(freq, gain, 'r--', alpha=0.5, linewidth=4, zorder=3)
    xticks = [1, 2, 4, 10, 20, 40, 100, 200, 400]
    ax.set(xlim=xlim, ylim=ylim, xticks=xticks, xlabel='Frequency (Hz)',
           ylabel='Amplitude (dB)')
    ax.set(xticklabels=xticks)
    box_off(ax)

half_height = np.array(plt.rcParams['figure.figsize']) * [1, 0.5]
ax = plt.subplots(1, figsize=half_height)[1]
plot_ideal(freq, gain, ax)
ax.set(title='Ideal %s Hz lowpass' % f_p)
mne.viz.tight_layout()
plt.show()

###############################################################################
# This filter hypothetically achieves zero ripple in the frequency domain,
# perfect attenuation, and perfect steepness. However, due to the discontunity
# in the frequency response, the filter would require infinite ringing in the
# time domain (i.e., infinite order) to be realized. Another way to think of
# this is that a rectangular window in frequency is actually sinc_ function
# in time, which requires an infinite number of samples, and thus infinite
# time, to represent. So although this filter has ideal frequency suppression,
# it has poor time-domain characteristics.
#
# Let's try to naïvely make a brick-wall filter of length 0.1 sec, and look
# at the filter itself in the time domain and the frequency domain:

n = int(round(0.1 * sfreq)) + 1
t = np.arange(-n // 2, n // 2) / sfreq  # center our sinc
h = np.sinc(2 * f_p * t) / (4 * np.pi)


def plot_filter(h, title, freq, gain, show=True):
    if h.ndim == 2:  # second-order sections
        sos = h
        n = mne.filter.estimate_ringing_samples(sos)
        h = np.zeros(n)
        h[0] = 1
        h = signal.sosfilt(sos, h)
        H = np.ones(512, np.complex128)
        for section in sos:
            f, this_H = signal.freqz(section[:3], section[3:])
            H *= this_H
    else:
        f, H = signal.freqz(h)
    fig, axs = plt.subplots(2)
    t = np.arange(len(h)) / sfreq
    axs[0].plot(t, h, color=blue)
    axs[0].set(xlim=t[[0, -1]], xlabel='Time (sec)',
               ylabel='Amplitude h(n)', title=title)
    box_off(axs[0])
    f *= sfreq / (2 * np.pi)
    axs[1].semilogx(f, 10 * np.log10((H * H.conj()).real), color=blue,
                    linewidth=2, zorder=4)
    plot_ideal(freq, gain, axs[1])
    mne.viz.tight_layout()
    if show:
        plt.show()

plot_filter(h, 'Sinc (0.1 sec)', freq, gain)

###############################################################################
# This is not so good! Making the filter 10 times longer (1 sec) gets us a
# bit better stop-band suppression, but still has a lot of ringing in
# the time domain. Note the x-axis is an order of magnitude longer here:

n = int(round(1. * sfreq)) + 1
t = np.arange(-n // 2, n // 2) / sfreq
h = np.sinc(2 * f_p * t) / (4 * np.pi)
plot_filter(h, 'Sinc (1.0 sec)', freq, gain)

###############################################################################
# Let's make the stop-band tighter still with a longer filter (10 sec),
# with a resulting larger x-axis:

n = int(round(10. * sfreq)) + 1
t = np.arange(-n // 2, n // 2) / sfreq
h = np.sinc(2 * f_p * t) / (4 * np.pi)
plot_filter(h, 'Sinc (10.0 sec)', freq, gain)

###############################################################################
# Now we have very sharp frequency suppression, but our filter rings for the
# entire second. So this naïve method is probably not a good way to build
# our low-pass filter.
#
# Fortunately, there are multiple established methods to design FIR filters
# based on desired response characteristics. These include:
#
#     1. The Remez_ algorithm (`scipy remez`_, `MATLAB firpm`_)
#     2. Windowed FIR design (`scipy firwin2`_, `MATLAB fir2`_)
#     3. Least squares designs (`MATLAB firls`_; coming to scipy 0.18)
#
# If we relax our frequency-domain filter requirements a little bit, we can
# use these functions to construct a lowpass filter that instead has a
# *transition band*, or a region between the pass frequency :math:`f_p`
# and stop frequency :math:`f_s`, e.g.:

trans_bandwidth = 10  # 10 Hz transition band
f_s = f_p + trans_bandwidth  # = 50 Hz

freq = [0., f_p, f_s, nyq]
gain = [1., 1., 0., 0.]
ax = plt.subplots(1, figsize=half_height)[1]
plot_ideal(freq, gain, ax)
ax.set(title='%s Hz lowpass with a %s Hz transition' % (f_p, trans_bandwidth))
mne.viz.tight_layout()
plt.show()

###############################################################################
# Accepting a shallower roll-off of the filter in the frequency domain makes
# our time-domain response potentially much better. We end up with a
# smoother slope through the transition region, but a *much* cleaner time
# domain signal. Here again for the 1 sec filter:

h = signal.firwin2(n, freq, gain, nyq=nyq)
plot_filter(h, 'Windowed 10-Hz transition (1.0 sec)', freq, gain)

###############################################################################
# Since our lowpass is around 40 Hz with a 10 Hz transition, we can actually
# use a shorter filter (5 cycles at 10 Hz = 0.5 sec) and still get okay
# stop-band attenuation:

n = int(round(sfreq * 0.5)) + 1
h = signal.firwin2(n, freq, gain, nyq=nyq)
plot_filter(h, 'Windowed 10-Hz transition (0.5 sec)', freq, gain)

###############################################################################
# But then if we shorten the filter too much (2 cycles of 10 Hz = 0.2 sec),
# our effective stop frequency gets pushed out past 60 Hz:

n = int(round(sfreq * 0.2)) + 1
h = signal.firwin2(n, freq, gain, nyq=nyq)
plot_filter(h, 'Windowed 10-Hz transition (0.2 sec)', freq, gain)

###############################################################################
# If we want a filter that is only 0.1 seconds long, we should probably use
# something more like a 25 Hz transition band (0.2 sec = 5 cycles @ 25 Hz):

trans_bandwidth = 25
f_s = f_p + trans_bandwidth
freq = [0, f_p, f_s, nyq]
h = signal.firwin2(n, freq, gain, nyq=nyq)
plot_filter(h, 'Windowed 50-Hz transition (0.2 sec)', freq, gain)

###############################################################################
# .. _effect_on_signals:
#
# Applying FIR filters
# --------------------
# Now lets look at some practical effects of these filters by applying
# them to some data.
#
# Let's construct a Gaussian-windowed sinusoid (i.e., Morlet imaginary part)
# plus noise (random + line). Note that the original, clean signal contains
# frequency content in both the pass band and transition bands of our
# low-pass filter.

dur = 10.
center = 2.
morlet_freq = f_p
tlim = [center - 0.2, center + 0.2]
tticks = [tlim[0], center, tlim[1]]
flim = [20, 70]

x = np.zeros(int(sfreq * dur))
blip = morlet(sfreq, [morlet_freq], n_cycles=7)[0].imag / 20.
n_onset = int(center * sfreq) - len(blip) // 2
x[n_onset:n_onset + len(blip)] += blip
x_orig = x.copy()

rng = np.random.RandomState(0)
x += rng.randn(len(x)) / 1000.
x += np.sin(2. * np.pi * 60. * np.arange(len(x)) / sfreq) / 2000.

###############################################################################
# Filter it with a shallow cutoff, linear-phase FIR and compensate for
# the delay:

transition_band = 0.25 * f_p
f_s = f_p + transition_band
filter_dur = 5. / transition_band  # sec
n = int(sfreq * filter_dur)
freq = [0., f_p, f_s, sfreq / 2.]
gain = [1., 1., 0., 0.]
h = signal.firwin2(n, freq, gain, nyq=sfreq / 2.)
x_shallow = np.convolve(h, x)[len(h) // 2:]

###############################################################################
# Now let's filter it with the MNE-Python 0.12 defaults, which is a
# long-duration, steep cutoff FIR:

transition_band = 0.5  # Hz
f_s = f_p + transition_band
filter_dur = 10.  # sec
n = int(sfreq * filter_dur)
freq = [0., f_p, f_s, sfreq / 2.]
gain = [1., 1., 0., 0.]
h = signal.firwin2(n, freq, gain, nyq=sfreq / 2.)
x_steep = np.convolve(h, x)[len(h) // 2:]

plot_filter(h, 'MNE-Python 0.12 default', freq, gain)

###############################################################################
# It has excellent frequency attenuation, but this comes at a cost of potential
# ringing (long-lasting ripples) in the time domain. Ripple can occur with
# steep filters, especially on signals with frequency content around the
# transition band. Our Morlet wavelet signal has power in our transition band,
# and the time-domain ringing is thus more pronounced for the steep-slope,
# long-duration filter than the shorter, shallower-slope filter:

axs = plt.subplots(2)[1]


def plot_signal(x, offset):
    t = np.arange(len(x)) / sfreq
    axs[0].plot(t, x + offset)
    axs[0].set(xlabel='Time (sec)', xlim=t[[0, -1]])
    box_off(axs[0])
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x), 1. / sfreq)
    mask = freqs >= 0
    X = X[mask]
    freqs = freqs[mask]
    axs[1].plot(freqs, 20 * np.log10(np.abs(X)))
    axs[1].set(xlim=xlim)

yticks = np.arange(4) / -30.
yticklabels = ['Original', 'Noisy', 'FIR-shallow', 'FIR-steep']
plot_signal(x_orig, offset=yticks[0])
plot_signal(x, offset=yticks[1])
plot_signal(x_shallow, offset=yticks[2])
plot_signal(x_steep, offset=yticks[3])
axs[0].set(xlim=tlim, title='Lowpass=%d Hz' % f_p, xticks=tticks,
           ylim=[-0.125, 0.025], yticks=yticks, yticklabels=yticklabels,)
for text in axs[0].get_yticklabels():
    text.set(rotation=45, size=8)
axs[1].set(xlim=flim, ylim=ylim, xlabel='Frequency (Hz)',
           ylabel='Magnitude (dB)')
box_off(axs[0])
box_off(axs[1])
mne.viz.tight_layout()
plt.show()

###############################################################################
# IIR filters
# ===========
# MNE-Python also offers IIR filtering functionality that is based on the
# methods from :mod:`scipy.signal`. Specifically, we use the general-purpose
# functions :func:`scipy.signal.iirfilter` and :func:`scipy.signal.iirdesign`,
# which provide unified interfaces to IIR filter design.
#
# Designing IIR filters
# ---------------------
# Let's continue with our design of a 40 Hz low-pass filter, and look at
# some trade-offs of different IIR filters.
#
# Often the default IIR filter is a `Butterworth filter`_, which is designed
# to have a *maximally flat pass-band*. Let's look at a few orders of filter,
# i.e., a few different number of coefficients used and therefore steepness
# of the filter:

sos = signal.iirfilter(2, f_p / nyq, btype='low', ftype='butter', output='sos')
plot_filter(sos, 'Butterworth order=2', freq, gain)

# Eventually this will just be from scipy signal.sosfiltfilt, but 0.18 is
# not widely adopted yet (as of June 2016), so we use our wrapper...
sosfiltfilt = mne.fixes.get_sosfiltfilt()
x_shallow = sosfiltfilt(sos, x)

###############################################################################
# The falloff of this filter is not very steep.
#
# .. warning:: For brevity, we do not show the phase of these filters here.
#              In the FIR case, we can design linear-phase filters, and
#              compensate for the delay if necessary. This cannot be done
#              with IIR filters, and as the filter order increases, the
#              phase distortion near and in the transition band worsens.
#              However, if acausal (forward-backward) filtering can be used,
#              e.g. with :func:`scipy.signal.filtfilt`, these phase issues
#              can be mitigated.
#
# .. note:: Here we have made use of second-order sections (SOS)
#           by using :func:`scipy.signal.sosfilt` and, under the
#           hood, :func:`scipy.signal.zpk2sos` when passing the
#           ``output='sos'`` keyword argument to
#           :func:`scipy.signal.iirfilter`. The filter definitions
#           given in :ref:`filtering-basics` use the polynomial
#           numerator/denominator (sometimes called "tf") form ``(b, a)``,
#           which are theoretically equivalent to the SOS form used here.
#           In practice, however, the SOS form can give much better results
#           due to issues with numerical precision (see
#           :func:`scipy.signal.sosfilt` for an example), so SOS should be
#           used when possible to do IIR filtering.
#
# Let's increase the order, and note that now we have better attenuation,
# with a longer impulse response:

sos = signal.iirfilter(8, f_p / nyq, btype='low', ftype='butter', output='sos')
plot_filter(sos, 'Butterworth order=8', freq, gain)
x_steep = sosfiltfilt(sos, x)

###############################################################################
# There are other types of IIR filters that we can use. For a complete list,
# check out the documentation for :func:`scipy.signal.iirdesign`. Let's
# try a Chebychev (type I) filter, which trades off ripple in the pass-band
# to get better attenuation in the stop-band:

sos = signal.iirfilter(8, f_p / nyq, btype='low', ftype='cheby1', output='sos',
                       rp=1)  # dB of acceptable pass-band ripple
plot_filter(sos, 'Chebychev-1 order=8, ripple=1 dB', freq, gain)

###############################################################################
# And if we can live with even more ripple, we can get it slightly steeper,
# but the impulse response begins to ring substantially longer (note the
# different x-axis scale):

sos = signal.iirfilter(8, f_p / nyq, btype='low', ftype='cheby1', output='sos',
                       rp=6)
plot_filter(sos, 'Chebychev-1 order=8, ripple=6 dB', freq, gain)

###############################################################################
# Applying IIR filters
# --------------------
# Now let's look at how our shallow and steep Butterworth IIR filters
# perform on our morlet signal from before:

axs = plt.subplots(2)[1]
yticks = np.arange(4) / -30.
yticklabels = ['Original', 'Noisy', 'Butterworth-2', 'Butterworth-8']
plot_signal(x_orig, offset=yticks[0])
plot_signal(x, offset=yticks[1])
plot_signal(x_shallow, offset=yticks[2])
plot_signal(x_steep, offset=yticks[3])
axs[0].set(xlim=tlim, title='Lowpass=%d Hz' % f_p, xticks=tticks,
           ylim=[-0.125, 0.025], yticks=yticks, yticklabels=yticklabels,)
for text in axs[0].get_yticklabels():
    text.set(rotation=45, size=8)
axs[1].set(xlim=flim, ylim=ylim, xlabel='Frequency (Hz)',
           ylabel='Magnitude (dB)')
box_off(axs[0])
box_off(axs[1])
mne.viz.tight_layout()
plt.show()

###############################################################################
# Filtering in MNE-Python
# =======================
# Most often, filtering in MNE-Python is done at the :class:`mne.io.Raw` level,
# and thus :func:`mne.io.Raw.filter` is used. This function under the hood
# (among other things) calls :func:`mne.filter.filter_data` to actually
# filter the data.
#
# :func:`mne.filter.filter_data` by default applies a FIR filter designed using
# :func:`scipy.signal.firwin2`. For more information on how to use the
# MNE-Python filtering functions with real data, consult the preprocessing
# tutorial on :ref:`tut_artifacts_filter`.
#
# Summary
# =======
# When filtering, there are always tradeoffs that should be considered.
# One important tradeoff is between time-domain characteristics (like ringing)
# and frequency-domain attenuation characteristics (like effective transition
# bandwidth). Filters with sharp frequency cutoffs can produce outputs that
# ring for a long time when they operate on signals with frequency content
# in the transition band. In general, therefore, the wider a transition band
# that can be tolerated, the better behaved the filter will be in the time
# domain.

###############################################################################
# References
# ==========
# .. [1] Parks TW, Burrus CS. Digital Filter Design.
#    New York: Wiley-Interscience, 1987.
#
# .. _FIR: https://en.wikipedia.org/wiki/Finite_impulse_response
# .. _IIR: https://en.wikipedia.org/wiki/Infinite_impulse_response
# .. _sinc: https://en.wikipedia.org/wiki/Sinc_function
# .. _moving average: https://en.wikipedia.org/wiki/Moving_average
# .. _autoregression: https://en.wikipedia.org/wiki/Autoregressive_model
# .. _Remez: https://en.wikipedia.org/wiki/Remez_algorithm
# .. _scipy remez: http://scipy.github.io/devdocs/generated/scipy.signal.remez.html  # noqa
# .. _matlab firpm: http://www.mathworks.com/help/signal/ref/firpm.html
# .. _scipy firwin2: http://scipy.github.io/devdocs/generated/scipy.signal.firwin2.html  # noqa
# .. _matlab fir2: http://www.mathworks.com/help/signal/ref/fir2.html
# .. _matlab firls: http://www.mathworks.com/help/signal/ref/firls.html
# .. _Butterworth filter: https://en.wikipedia.org/wiki/Butterworth_filter
