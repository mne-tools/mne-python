# -*- coding: utf-8 -*-
r"""
.. _tut_background_filtering:

===================================
Background information on filtering
===================================

Here we give some background information on filtering in general,
and how it is done in MNE-Python in particular.
Recommended reading for practical applications of digital
filter design can be found in Parks & Burrus [1]_ and
Ifeachor and Jervis [2]_, and for filtering in an
M/EEG context we recommend reading Widmann *et al.* 2015 [7]_.
To see how to use the default filters in MNE-Python on actual data, see
the :ref:`tut_artifacts_filter` tutorial.

.. contents::
    :local:

Problem statement
=================

The practical issues with filtering electrophysiological data are covered
well by Widmann *et al.* in [7]_, in a follow-up to an article where they
conclude with this statement:

    Filtering can result in considerable distortions of the time course
    (and amplitude) of a signal as demonstrated by VanRullen (2011) [[3]_].
    Thus, filtering should not be used lightly. However, if effects of
    filtering are cautiously considered and filter artifacts are minimized,
    a valid interpretation of the temporal dynamics of filtered
    electrophysiological data is possible and signals missed otherwise
    can be detected with filtering.

In other words, filtering can increase SNR, but if it is not used carefully,
it can distort data. Here we hope to cover some filtering basics so
users can better understand filtering tradeoffs, and why MNE-Python has
chosen particular defaults.

.. _tut_filtering_basics:

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

In MNE-Python we default to using FIR filtering. As noted in Widmann *et al.*
2015 [7]_:

    Despite IIR filters often being considered as computationally more
    efficient, they are recommended only when high throughput and sharp
    cutoffs are required (Ifeachor and Jervis, 2002[2]_, p. 321),
    ...FIR filters are easier to control, are always stable, have a
    well-defined passband, can be corrected to zero-phase without
    additional computations, and can be converted to minimum-phase.
    We therefore recommend FIR filters for most purposes in
    electrophysiological data analysis.

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

FIR Filters
===========

First we will focus first on FIR filters, which are the default filters used by
MNE-Python.
"""

###############################################################################
# Designing FIR filters
# ---------------------
# Here we'll try designing a low-pass filter, and look at trade-offs in terms
# of time- and frequency-domain filter characteristics. Later, in
# :ref:`tut_effect_on_signals`, we'll look at how such filters can affect
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
    my_freq, my_gain = list(), list()
    for ii in range(len(freq)):
        xs.append(freq[ii])
        ys.append(ylim[0])
        if ii < len(freq) - 1 and gain[ii] != gain[ii + 1]:
            xs += [freq[ii], freq[ii + 1]]
            ys += [ylim[1]] * 2
            my_freq += np.linspace(freq[ii], freq[ii + 1], 20,
                                   endpoint=False).tolist()
            my_gain += np.linspace(gain[ii], gain[ii + 1], 20,
                                   endpoint=False).tolist()
        else:
            my_freq.append(freq[ii])
            my_gain.append(gain[ii])
    my_gain = 10 * np.log10(np.maximum(my_gain, 10 ** (ylim[0] / 10.)))
    ax.fill_between(xs, ylim[0], ys, color='r', alpha=0.1)
    ax.semilogx(my_freq, my_gain, 'r--', alpha=0.5, linewidth=4, zorder=3)
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
#     1. The Remez_ algorithm (:func:`scipy.signal.remez`, `MATLAB firpm`_)
#     2. Windowed FIR design (:func:`scipy.signal.firwin2`, `MATLAB fir2`_)
#     3. Least squares designs (:func:`scipy.signal.firls`, `MATLAB firls`_)
#     4. Frequency-domain design (construct filter in Fourier
#        domain and use an :func:`IFFT <scipy.fftpack.ifft>` to invert it)
#
# .. note:: Remez and least squares designs have advantages when there are
#           "do not care" regions in our frequency response. However, we want
#           well controlled responses in all frequency regions.
#           Frequency-domain construction is good when an arbitrary response
#           is desired, but generally less clean (due to sampling issues) than
#           a windowed approach for more straightfroward filter applications.
#           Since our filters (low-pass, high-pass, band-pass, band-stop)
#           are fairly simple and we require precisel control of all frequency
#           regions, here we will use and explore primarily windowed FIR
#           design.
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
# .. _tut_effect_on_signals:
#
# Applying FIR filters
# --------------------
#
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
filter_dur = 6.6 / transition_band  # sec
n = int(sfreq * filter_dur)
freq = [0., f_p, f_s, sfreq / 2.]
gain = [1., 1., 0., 0.]
h = signal.firwin2(n, freq, gain, nyq=sfreq / 2.)
x_shallow = np.convolve(h, x)[len(h) // 2:]

plot_filter(h, 'MNE-Python 0.14 default', freq, gain)

###############################################################################
# This is actually set to become the default type of filter used in MNE-Python
# in 0.14 (see :ref:`tut_filtering_in_python`).
#
# Let's also filter with the MNE-Python 0.13 default, which is a
# long-duration, steep cutoff FIR that gets applied twice:

transition_band = 0.5  # Hz
f_s = f_p + transition_band
filter_dur = 10.  # sec
n = int(sfreq * filter_dur)
freq = [0., f_p, f_s, sfreq / 2.]
gain = [1., 1., 0., 0.]
h = signal.firwin2(n, freq, gain, nyq=sfreq / 2.)
x_steep = np.convolve(np.convolve(h, x)[::-1], h)[::-1][len(h) - 1:-len(h) - 1]

plot_filter(h, 'MNE-Python 0.13 default', freq, gain)

###############################################################################
# Finally, Let's also filter it with the
# MNE-C default, which is a long-duration steep-slope FIR filter designed
# using frequency-domain techniques:

h = mne.filter.design_mne_c_filter(sfreq, l_freq=None, h_freq=f_p + 2.5)
x_mne_c = np.convolve(h, x)[len(h) // 2:]

transition_band = 5  # Hz (default in MNE-C)
f_s = f_p + transition_band
freq = [0., f_p, f_s, sfreq / 2.]
gain = [1., 1., 0., 0.]
plot_filter(h, 'MNE-C default', freq, gain)

###############################################################################
# Both the MNE-Python 0.13 and MNE-C filhters have excellent frequency
# attenuation, but it comes at a cost of potential
# ringing (long-lasting ripples) in the time domain. Ringing can occur with
# steep filters, especially on signals with frequency content around the
# transition band. Our Morlet wavelet signal has power in our transition band,
# and the time-domain ringing is thus more pronounced for the steep-slope,
# long-duration filter than the shorter, shallower-slope filter:

axs = plt.subplots(1, 2)[1]


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

yticks = np.arange(5) / -30.
yticklabels = ['Original', 'Noisy', 'FIR-shallow (0.14)', 'FIR-steep (0.13)',
               'FIR-steep (MNE-C)']
plot_signal(x_orig, offset=yticks[0])
plot_signal(x, offset=yticks[1])
plot_signal(x_shallow, offset=yticks[2])
plot_signal(x_steep, offset=yticks[3])
plot_signal(x_mne_c, offset=yticks[4])
axs[0].set(xlim=tlim, title='FIR, Lowpass=%d Hz' % f_p, xticks=tticks,
           ylim=[-0.150, 0.025], yticks=yticks, yticklabels=yticklabels,)
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
#
# MNE-Python also offers IIR filtering functionality that is based on the
# methods from :mod:`scipy.signal`. Specifically, we use the general-purpose
# functions :func:`scipy.signal.iirfilter` and :func:`scipy.signal.iirdesign`,
# which provide unified interfaces to IIR filter design.
#
# Designing IIR filters
# ---------------------
#
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
#              compensate for the delay (making the filter acausal) if
#              necessary. This cannot be done
#              with IIR filters, as they have a non-linear phase.
#              As the filter order increases, the
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
#           given in tut_filtering_basics_ use the polynomial
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
#
# Now let's look at how our shallow and steep Butterworth IIR filters
# perform on our Morlet signal from before:

axs = plt.subplots(1, 2)[1]
yticks = np.arange(4) / -30.
yticklabels = ['Original', 'Noisy', 'Butterworth-2', 'Butterworth-8']
plot_signal(x_orig, offset=yticks[0])
plot_signal(x, offset=yticks[1])
plot_signal(x_shallow, offset=yticks[2])
plot_signal(x_steep, offset=yticks[3])
axs[0].set(xlim=tlim, title='IIR, Lowpass=%d Hz' % f_p, xticks=tticks,
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
# Some pitfalls of filtering
# ==========================
#
# Multiple recent papers have noted potential risks of drawing
# errant inferences due to misapplication of filters.
#
# Low-pass problems
# -----------------
#
# Filters in general, especially those that are acausal (zero-phase), can make
# activity appear to occur earlier or later than it truly did. As
# mentioned in VanRullen 2011 [3]_, investigations of commonly (at the time)
# used low-pass filters created artifacts when they were applied to smulated
# data. However, such deleterious effects were minimal in many real-world
# examples in Rousselet 2012 [5]_.
#
# Perhaps more revealing, it was noted in Widmann & Schröger 2012 [6]_ that
# the problematic low-pass filters from VanRullen 2011 [3]_:
#
#    1. Used a least-squares design (like :func:`scipy.signal.firls`) that
#       included "do-not-care" transition regions, which can lead to
#       uncontrolled behavior.
#    2. Had a filter length that was independent of the transition bandwidth,
#       which can cause excessive ringing and signal distortion.
#
# .. _tut_filtering_hp_problems:
#
# High-pass problems
# ------------------
#
# When it comes to high-pass filtering, using corner frequencies above 0.1 Hz
# were found in Acunzo *et al.* 2012 [4]_ to:
#
#    "...generate a systematic bias easily leading to misinterpretations of
#    neural activity.”
#
# In a related paper, Widmann *et al.* 2015 [7]_ also came to suggest a 0.1 Hz
# highpass. And more evidence followed in Tanner *et al.* 2015 [8]_ of such
# distortions. Using data from language ERP studies of semantic and syntactic
# processing (i.e., N400 and P600), using a high-pass above 0.3 Hz caused
# significant effects to be introduced implausibly early when compared to the
# unfiltered data. From this, the authors suggested the optimal high-pass
# value for language processing to be 0.1 Hz.
#
# We can recreate a problematic simulation from Tanner *et al.* 2015 [8]_:
#
#    "The simulated component is a single-cycle cosine wave with an amplitude
#    of 5µV, onset of 500 ms poststimulus, and duration of 800 ms. The
#    simulated component was embedded in 20 s of zero values to avoid
#    filtering edge effects... Distortions [were] caused by 2 Hz low-pass and
#    high-pass filters... No visible distortion to the original waveform
#    [occurred] with 30 Hz low-pass and 0.01 Hz high-pass filters...
#    Filter frequencies correspond to the half-amplitude (-6 dB) cutoff
#    (12 dB/octave roll-off)."
#
# .. note:: This simulated signal contains energy not just within the
#           pass-band, but also within the transition and stop-bands -- perhaps
#           most easily understood because the signal has a non-zero DC value,
#           but also because it is a shifted cosine that has been
#           *windowed* (here multiplied by a rectangular window), which
#           makes the cosine and DC frequencies spread to other frequencies
#           (multiplication in time is convolution in frequency, so multiplying
#           by a rectangular window in the time domain means convolving a sinc
#           function with the impulses at DC and the cosine frequency in the
#           frequency domain).
#

x = np.zeros(int(2 * sfreq))
t = np.arange(0, len(x)) / sfreq - 0.2
onset = np.where(t >= 0.5)[0][0]
cos_t = np.arange(0, int(sfreq * 0.8)) / sfreq
sig = 2.5 - 2.5 * np.cos(2 * np.pi * (1. / 0.8) * cos_t)
x[onset:onset + len(sig)] = sig

iir_lp_30 = signal.iirfilter(2, 30. / sfreq, btype='lowpass')
iir_hp_p1 = signal.iirfilter(2, 0.1 / sfreq, btype='highpass')
iir_lp_2 = signal.iirfilter(2, 2. / sfreq, btype='lowpass')
iir_hp_2 = signal.iirfilter(2, 2. / sfreq, btype='highpass')
x_lp_30 = signal.filtfilt(iir_lp_30[0], iir_lp_30[1], x, padlen=0)
x_hp_p1 = signal.filtfilt(iir_hp_p1[0], iir_hp_p1[1], x, padlen=0)
x_lp_2 = signal.filtfilt(iir_lp_2[0], iir_lp_2[1], x, padlen=0)
x_hp_2 = signal.filtfilt(iir_hp_2[0], iir_hp_2[1], x, padlen=0)

xlim = t[[0, -1]]
ylim = [-2, 6]
xlabel = 'Time (sec)'
ylabel = 'Amplitude ($\mu$V)'
tticks = [0, 0.5, 1.3, t[-1]]
axs = plt.subplots(2, 2)[1].ravel()
for ax, x_f, title in zip(axs, [x_lp_2, x_lp_30, x_hp_2, x_hp_p1],
                          ['LP$_2$', 'LP$_{30}$', 'HP$_2$', 'LP$_{0.1}$']):
    ax.plot(t, x, color='0.5')
    ax.plot(t, x_f, color='k', linestyle='--')
    ax.set(ylim=ylim, xlim=xlim, xticks=tticks,
           title=title, xlabel=xlabel, ylabel=ylabel)
    box_off(ax)
mne.viz.tight_layout()
plt.show()

###############################################################################
# Similarly, in a P300 paradigm reported by Kappenman & Luck 2010 [12]_,
# they found that applying a 1 Hz high-pass decreased the probaility of
# finding a significant difference in the N100 response, likely because
# the P300 response was smeared (and inverted) in time by the high-pass
# filter such that it tended to cancel out the increased N100. However,
# they nonetheless note that some high-passing can still be useful to deal
# with drifts in the data.
#
# Even though these papers generally advise a 0.1 HZ or lower frequency for
# a high-pass, it is important to keep in mind (as most authors note) that
# filtering choices should depend on the frequency content of both the
# signal(s) of interest and the noise to be suppressed. For example, in
# some of the MNE-Python examples involving :ref:`ch_sample_data`,
# high-pass values of around 1 Hz are used when looking at auditory
# or visual N100 responses, because we analyze standard (not deviant) trials
# and thus expect that contamination by later or slower components will
# be limited.
#
# Baseline problems (or solutions?)
# ---------------------------------
#
# In an evolving discussion, Tanner *et al.* 2015 [8]_ suggest using baseline
# correction to remove slow drifts in data. However, Maess *et al.* 2016 [9]_
# suggest that baseline correction, which is a form of high-passing, does
# not offer substantial advantages over standard high-pass filtering.
# Tanner *et al.* [10]_ rebutted that baseline correction can correct for
# problems with filtering.
#
# To see what they mean, consider again our old simulated signal ``x`` from
# before:


def baseline_plot(x):
    all_axs = plt.subplots(3, 2)[1]
    for ri, (axs, freq) in enumerate(zip(all_axs, [0.1, 0.3, 0.5])):
        for ci, ax in enumerate(axs):
            if ci == 0:
                iir_hp = signal.iirfilter(4, freq / sfreq, btype='highpass',
                                          output='sos')
                x_hp = sosfiltfilt(iir_hp, x, padlen=0)
            else:
                x_hp -= x_hp[t < 0].mean()
            ax.plot(t, x, color='0.5')
            ax.plot(t, x_hp, color='k', linestyle='--')
            if ri == 0:
                ax.set(title=('No ' if ci == 0 else '') +
                       'Baseline Correction')
            box_off(ax)
            ax.set(xticks=tticks, ylim=ylim, xlim=xlim, xlabel=xlabel)
            ax.set_ylabel('%0.1f Hz' % freq, rotation=0,
                          horizontalalignment='right')
    mne.viz.tight_layout()
    plt.suptitle(title)
    plt.show()

baseline_plot(x)

###############################################################################
# In respose, Maess *et al.* 2016 [11]_ note that these simulations do not
# address cases of pre-stimulus activity that is shared across conditions, as
# applying baseline correction will effectively copy the topology outside the
# baseline period. We can see this if we give our signal ``x`` with some
# consistent pre-stimulus activity, which makes everything look bad.
#
# .. note:: An important thing to keep in mind with these plots is that they
#           are for a single simulated sensor. In multielectrode recordings
#           the topology (i.e., spatial pattiern) of the pre-stimulus activity
#           will leak into the post-stimulus period. This will likely create a
#           spatially varying distortion of the time-domain signals, as the
#           averaged pre-stimulus spatial pattern gets subtracted from the
#           sensor time courses.
#
# Putting some activity in the baseline period:

n_pre = (t < 0).sum()
sig_pre = 1 - np.cos(2 * np.pi * np.arange(n_pre) / (0.5 * n_pre))
x[:n_pre] += sig_pre
baseline_plot(x)

###############################################################################
# Both groups seem to acknowledge that the choices of filtering cutoffs, and
# perhaps even the application of baseline correction, depend on the
# characteristics of the data being investigated, especially when it comes to:
#
#    1. The frequency content of the underlying evoked activity relative
#       to the filtering parameters.
#    2. The validity of the assumption of no consistent evoked activity
#       in the baseline period.
#
# We thus recommend carefully applying baseline correction and/or high-pass
# values based on the characteristics of the data to be analyzed.
#
#
# Filtering defaults
# ==================
#
# .. _tut_filtering_in_python:
#
# Defaults in MNE-Python
# ----------------------
#
# Most often, filtering in MNE-Python is done at the :class:`mne.io.Raw` level,
# and thus :func:`mne.io.Raw.filter` is used. This function under the hood
# (among other things) calls :func:`mne.filter.filter_data` to actually
# filter the data, which by default applies a zero-phase FIR filter designed
# using :func:`scipy.signal.firwin2`. In Widmann *et al.* 2015 [7]_, they
# suggest a specific set of parameters to use for high-pass filtering,
# including:
#
#     "... providing a transition bandwidth of 25% of the lower passband
#     edge but, where possible, not lower than 2 Hz and otherwise the
#     distance from the passband edge to the critical frequency.”
#
# In practice, this means that for each high-pass value ``l_freq`` or
# low-pass value ``h_freq`` below, you would get this corresponding
# ``l_trans_bandwidth`` or ``h_trans_bandwidth``, respectively,
# if the sample rate were 100 Hz (i.e., Nyquist frequency of 50 Hz):
#
# +------------------+-------------------+-------------------+
# | l_freq or h_freq | l_trans_bandwidth | h_trans_bandwidth |
# +==================+===================+===================+
# |             0.01 |              0.01 |               2.0 |
# +------------------+-------------------+-------------------+
# |              0.1 |               0.1 |               2.0 |
# +------------------+-------------------+-------------------+
# |              1.0 |               1.0 |               2.0 |
# +------------------+-------------------+-------------------+
# |              2.0 |               2.0 |               2.0 |
# +------------------+-------------------+-------------------+
# |              4.0 |               2.0 |               2.0 |
# +------------------+-------------------+-------------------+
# |              8.0 |               2.0 |               2.0 |
# +------------------+-------------------+-------------------+
# |             10.0 |               2.5 |               2.5 |
# +------------------+-------------------+-------------------+
# |             20.0 |               5.0 |               5.0 |
# +------------------+-------------------+-------------------+
# |             40.0 |              10.0 |              10.0 |
# +------------------+-------------------+-------------------+
# |             45.0 |             11.25 |              5.0  |
# +------------------+-------------------+-------------------+
# |             48.0 |              12.0 |              2.0  |
# +------------------+-------------------+-------------------+
#
# MNE-Python has adopted this definition for its high-pass (and low-pass)
# transition bandwidth choices when using ``l_trans_bandwidth='auto'`` and
# ``h_trans_bandwidth='auto'``.
#
# To choose the filter length automatically with ``filter_length='auto'``,
# the reciprocal of the shortest transition bandwidth is used to ensure
# decent attenuation at the stop frequency. Specifically, the reciprocal
# (in samples) is multiplied by 6.2, 6.6, or 11.0 for the Hann, Hamming,
# or Blackman windows, respectively as selected by the ``fir_window``
# argument.
#
# .. note:: These multiplicative factors are double what is given in
#           Ifeachor and Jervis [2]_ (p. 357). The window functions have a
#           smearing effect on the frequency response; I&J thus take the
#           approach of setting the stop frequency as
#           :math:`f_s = f_p + f_{trans} / 2.`, but our stated definitions of
#           :math:`f_s` and :math:`f_{trans}` do not
#           allow us to do this in a nice way. Instead, we increase our filter
#           length to achieve acceptable (20+ dB) attenuation by
#           :math:`f_s = f_p + f_{trans}`, and excellent (50+ dB)
#           attenuation by :math:`f_s + f_{trans}` (and usually earlier).
#
# In 0.14, we default to using a Hamming window in filter design, as it
# provides up to 53 dB of stop-band attenuation with small pass-band ripple.
#
# .. note:: In band-pass applications, often a low-pass filter can operate
#           effectively with fewer samples than the high-pass filter, so
#           it is advisable to apply the high-pass and low-pass separately.
#
# For more information on how to use the
# MNE-Python filtering functions with real data, consult the preprocessing
# tutorial on :ref:`tut_artifacts_filter`.
#
# Defaults in MNE-C
# -----------------
# MNE-C by default uses:
#
#    1. 5 Hz transition band for low-pass filters.
#    2. 3-sample transition band for high-pass filters.
#    3. Filter length of 8197 samples.
#
# The filter is designed in the frequency domain, creating a linear-phase
# filter such that the delay is compensated for as is done with the MNE-Python
# ``phase='zero'`` filtering option.
#
# Squared-cosine ramps are used in the transition regions. Because these
# are used in place of more gradual (e.g., linear) transitions,
# a given transition width will result in more temporal ringing but also more
# rapid attenuation than the same transition width in windowed FIR designs.
#
# The default filter length will generally have excellent attenuation
# but long ringing for the sample rates typically encountered in M-EEG data
# (e.g. 500-2000 Hz).
#
# Defaults in other software
# --------------------------
# A good but possibly outdated comparison of filtering in various software
# packages is available in [7]_. Briefly:
#
# * EEGLAB
#     MNE-Python in 0.14 defaults to behavior very similar to that of EEGLAB,
#     see the `EEGLAB filtering FAQ`_ for more information.
# * Fieldrip
#     By default FieldTrip applies a forward-backward Butterworth IIR filter
#     of order 4 (band-pass and band-stop filters) or 2 (for low-pass and
#     high-pass filters). Similar filters can be achieved in MNE-Python when
#     filtering with :meth:`raw.filter(..., method='iir') <mne.io.Raw.filter>`
#     (see also :func:`mne.filter.construct_iir_filter` for options).
#     For more inforamtion, see e.g. `FieldTrip band-pass documentation`_.
#
# Summary
# =======
#
# When filtering, there are always tradeoffs that should be considered.
# One important tradeoff is between time-domain characteristics (like ringing)
# and frequency-domain attenuation characteristics (like effective transition
# bandwidth). Filters with sharp frequency cutoffs can produce outputs that
# ring for a long time when they operate on signals with frequency content
# in the transition band. In general, therefore, the wider a transition band
# that can be tolerated, the better behaved the filter will be in the time
# domain.
#
# References
# ==========
#
# .. [1] Parks TW, Burrus CS (1987). Digital Filter Design.
#        New York: Wiley-Interscience.
# .. [2] Ifeachor, E. C., & Jervis, B. W. (2002). Digital Signal Processing:
#        A Practical Approach. Prentice Hall.
# .. [3] Vanrullen, R. (2011). Four common conceptual fallacies in mapping
#        the time course of recognition. Perception Science, 2, 365.
# .. [4] Acunzo, D. J., MacKenzie, G., & van Rossum, M. C. W. (2012).
#        Systematic biases in early ERP and ERF components as a result
#        of high-pass filtering. Journal of Neuroscience Methods,
#        209(1), 212–218. http://doi.org/10.1016/j.jneumeth.2012.06.011
# .. [5] Rousselet, G. A. (2012). Does filtering preclude us from studying
#        ERP time-courses? Frontiers in Psychology, 3(131)
# .. [6] Widmann, A., & Schröger, E. (2012). Filter effects and filter
#        artifacts in the analysis of electrophysiological data.
#        Perception Science, 233.
# .. [7] Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter
#        design for electrophysiological data – a practical approach.
#        Journal of Neuroscience Methods, 250, 34–46.
# .. [8] Tanner, D., Morgan-Short, K., & Luck, S. J. (2015).
#        How inappropriate high-pass filters can produce artifactual effects
#        and incorrect conclusions in ERP studies of language and cognition.
#        Psychophysiology, 52(8), 997–1009. http://doi.org/10.1111/psyp.12437
# .. [9] Maess, B., Schröger, E., & Widmann, A. (2016).
#        High-pass filters and baseline correction in M/EEG analysis.
#        Commentary on: “How inappropriate high-pass filters can produce
#        artefacts and incorrect conclusions in ERP studies of language
#        and cognition.” Journal of Neuroscience Methods, 266, 164–165.
# .. [10] Tanner, D., Norton, J. J. S., Morgan-Short, K., & Luck, S. J. (2016).
#        On high-pass filter artifacts (they’re real) and baseline correction
#        (it’s a good idea) in ERP/ERMF analysis.
# .. [11] Maess, B., Schröger, E., & Widmann, A. (2016).
#        High-pass filters and baseline correction in M/EEG analysis-continued
#        discussion. Journal of Neuroscience Methods, 266, 171–172.
#        Journal of Neuroscience Methods, 266, 166–170.
# .. [12] Kappenman E. & Luck, S. (2010). The effects of impedance on data
#        quality and statistical significance in ERP recordings.
#        Psychophysiology, 47, 888-904.
#
# .. _FIR: https://en.wikipedia.org/wiki/Finite_impulse_response
# .. _IIR: https://en.wikipedia.org/wiki/Infinite_impulse_response
# .. _sinc: https://en.wikipedia.org/wiki/Sinc_function
# .. _moving average: https://en.wikipedia.org/wiki/Moving_average
# .. _autoregression: https://en.wikipedia.org/wiki/Autoregressive_model
# .. _Remez: https://en.wikipedia.org/wiki/Remez_algorithm
# .. _matlab firpm: http://www.mathworks.com/help/signal/ref/firpm.html
# .. _matlab fir2: http://www.mathworks.com/help/signal/ref/fir2.html
# .. _matlab firls: http://www.mathworks.com/help/signal/ref/firls.html
# .. _Butterworth filter: https://en.wikipedia.org/wiki/Butterworth_filter
# .. _eeglab filtering faq: https://sccn.ucsd.edu/wiki/Firfilt_FAQ
# .. _fieldtrip band-pass documentation: http://www.fieldtriptoolbox.org/reference/ft_preproc_bandpassfilter  # noqa
