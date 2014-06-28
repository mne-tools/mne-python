"""
===================================================================
Plot time-frequency representations on topographies for MEG sensors
===================================================================

Both induced power and phase locking values are displayed.
"""
print(__doc__)

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import io
from mne.time_frequency import induced_power
from mne.datasets import sample

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                        stim=False, include=include, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))

###############################################################################
# Calculate power and phase locking value

from mne.channels import ContainsMixin, PickDropChannelsMixin
from mne.viz import partial, _imshow_tfr, _plot_topo, rescale
from copy import deepcopy
from mne.utils import logger, verbose


def _preproc_tfr(data, times, freqs, tmin, tmax, fmin, fmax, mode,
                 baseline, vmin, vmax, dB):
    if mode is not None and baseline is not None:
        logger.info("Applying baseline correction '%s' during %s" %
                    (mode, baseline))
        data = rescale(data.copy(), times, baseline, mode)

    # crop time
    itmin, itmax = None, None
    if tmin is not None:
        itmin = np.where(times >= tmin)[0][0]
    if tmax is not None:
        itmax = np.where(times <= tmax)[0][-1]

    times = times[itmin:itmax]

    # crop freqs
    ifmin, ifmax = None, None
    if fmin is not None:
        ifmin = np.where(freqs >= fmin)[0][0]
    if fmax is not None:
        ifmax = np.where(freqs <= fmax)[0][-1]

    freqs = freqs[ifmin:ifmax]

    times *= 1e3
    if dB:
        data = 20 * np.log10(data)
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    return data, times, freqs


# XXX : todo IO of TFRs
class AverageTFR(ContainsMixin, PickDropChannelsMixin):
    """Container for Time-Frequency data

    Can for example store induced power at sensor level or intertrial
    coherence.

    Parameters
    ----------
    info : Info
        The measurement info.
    data : ndarray, shape (n_channels, n_freqs, n_times)
        The data.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.

    Attributes
    ----------
    ch_names : list
        The names of the channels.
    """
    @verbose
    def __init__(self, info, data, times, freqs, verbose=None):
        self.info = info
        if data.ndim != 3:
            raise ValueError('data should be 3d. Got %d.' % data.ndim)
        n_channels, n_freqs, n_times = data.shape
        if n_channels != len(info['chs']):
            raise ValueError("Number of channels and data size don't match"
                             " (%d != %d)." % (n_channels, len(info['chs'])))
        if n_freqs != len(freqs):
            raise ValueError("Number of frequencies and data size don't match"
                             " (%d != %d)." % (n_freqs, len(freqs)))
        if n_times != len(times):
            raise ValueError("Number of times and data size don't match"
                             " (%d != %d)." % (n_times, len(times)))
        self.data = data
        self.times = times
        self.freqs = freqs

    @property
    def ch_names(self):
        return self.info['ch_names']

    @verbose
    def plot(self, picks, baseline=None, mode='mean', tmin=None, tmax=None,
             fmin=None, fmax=None, vmin=None, vmax=None, cmap=None, dB=False,
             colorbar=True, show=True, verbose=None):
        """Plot TFRs in a topography with images

        Parameters
        ----------
        picks : array-like of int
            The indices of the channels to plot.
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal ot (None, None) all the time
            interval is used.
        mode : None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)).
            If None no baseline correction is applied.
        tmin : None | float
            The first time instant to display. If None the first time point
            available is used.
        tmax : None | float
            The last time instant to display. If None the last time point
            available is used.
        fmin : None | float
            The first frequency to display. If None the first frequency
            available is used.
        fmax : None | float
            The last frequency to display. If None the last frequency
            available is used.
        vmin : float | None
            The mininum value an the color scale. If vmin is None, the data minimum
            value is used.
        vmax : float | None
            The maxinum value an the color scale. If vmax is None, the data maximum
            value is used.
        layout : Layout | None
            Layout instance specifying sensor positions. If possible, the correct
            layout is inferred from the data.
        cmap : matplotlib colormap
            The colormap to use.
        dB : bool
            If True, 20*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas
        show : bool
            Call pyplot.show() at the end.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        """
        import matplotlib.pyplot as plt
        times, freqs = self.times, self.freqs
        data = self.data[picks]
        data, times, freqs = _preproc_tfr(data, times, freqs, tmin, tmax,
                                          fmin, fmax, mode, baseline, vmin,
                                          vmax, dB)

        if mode is not None and baseline is not None:
            logger.info("Applying baseline correction '%s' during %s" %
                        (mode, baseline))
            data = rescale(data, times, baseline, mode)

        tmin, tmax = times[0], times[-1]
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        for k, p in zip(range(len(data)), picks):
            plt.figure()
            _imshow_tfr(plt, 0, tmin, tmax, vmin, vmax, ylim=None,
                        tfr=data[k: k + 1], freq=freqs, x_label='Time (ms)',
                        y_label='Frequency (Hz)', colorbar=colorbar,
                        picker=False)

        if show:
            import matplotlib.pyplot as plt
            plt.show()

    def plot_topo(self, picks=None, baseline=None, mode='mean', tmin=None, tmax=None,
                  fmin=None, fmax=None, vmin=None, vmax=None, layout=None, cmap=None,
                  title=None, dB=False, colorbar=True, layout_scale=0.945, show=True):
        """Plot TFRs in a topography with images

        Parameters
        ----------
        picks : array-like of int | None
            The indices of the channels to plot. If None all available
            channels are displayed.
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal ot (None, None) all the time
            interval is used.
        mode : None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)).
            If None no baseline correction is applied.
        tmin : None | float
            The first time instant to display. If None the first time point
            available is used.
        tmax : None | float
            The last time instant to display. If None the last time point
            available is used.
        fmin : None | float
            The first frequency to display. If None the first frequency
            available is used.
        fmax : None | float
            The last frequency to display. If None the last frequency
            available is used.
        vmin : float | None
            The mininum value an the color scale. If vmin is None, the data minimum
            value is used.
        vmax : float | None
            The maxinum value an the color scale. If vmax is None, the data maximum
            value is used.
        layout : Layout | None
            Layout instance specifying sensor positions. If possible, the correct
            layout is inferred from the data.
        cmap : matplotlib colormap
            The colormap to use.
        title : str
            Title of the figure.
        dB : bool
            If True, 20*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas
        show : bool
            Call pyplot.show() at the end.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        """
        times = self.times.copy()
        freqs = self.freqs
        data = self.data
        info = self.info

        if picks is not None:
            data = data[picks]
            info = pick_info(info, picks)

        data, times, freqs = _preproc_tfr(data, times, freqs, tmin, tmax,
                                          fmin, fmax, mode, baseline, vmin,
                                          vmax, dB)

        if layout is None:
            from mne.layouts.layout import find_layout
            layout = find_layout(self.info)

        imshow = partial(_imshow_tfr, tfr=data, freq=freqs)

        fig = _plot_topo(info=info, times=times,
                         show_func=imshow, layout=layout,
                         colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                         layout_scale=layout_scale, title=title, border='w',
                         x_label='Time (ms)', y_label='Frequency (Hz)')

        if show:
            import matplotlib.pyplot as plt
            plt.show()

        return fig

    def _check_compat(self, tfr):
        assert np.all(tfr.times == self.times)
        assert np.all(tfr.freqs == self.freqs)

    def __add__(self, tfr):
        self._check_compat(tfr)
        out = self.copy()
        out.data += tfr.data
        return out

    def __iadd__(self, tfr):
        self._check_compat(tfr)
        self.data += tfr.data
        return self

    def __sub__(self, tfr):
        self._check_compat(tfr)
        out = self.copy()
        out.data -= tfr.data
        return out

    def __isub__(self, tfr):
        self._check_compat(tfr)
        self.data -= tfr.data
        return self

    def copy(self):
        """Return a copy of the instance."""
        return deepcopy(self)

    def __repr__(self):
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ', channels: %d' % self.data.shape[1]
        return "<AverageTFR  |  %s>" % s


def tfr_morlet(epochs, freqs, n_cycles, use_fft=False,
               return_itc=True, decim=1, n_jobs=1):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets

    Parameters
    ----------
    epochs : Epochs
        The epochs.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
    use_fft : bool
        The fft based convolution or not.
    return_itc : bool
        Return intertrial coherence (ITC) as well as averaged power.
    decim : int
        The decimation factor on the time axis. To reduce memory usage.
    n_jobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    power : AverageTFR
        The averaged power.
    itc : AverageTFR
        The intertrial coherence (ITC). Only returned if return_itc
        is True.
    """
    data = epochs.get_data()
    picks = mne.pick_types(epochs.info, meg=True, eeg=True)
    info = mne.pick_info(epochs.info, picks)
    data = data[:, picks, :]
    power, itc = induced_power(data, Fs=info['sfreq'], frequencies=freqs,
                               n_cycles=n_cycles, n_jobs=n_jobs,
                               use_fft=use_fft, decim=decim,
                               zero_mean=True)
    times = epochs.times[::decim].copy()
    out = AverageTFR(info, power, times, freqs)
    if return_itc:
        out = (out, AverageTFR(info, itc, times, freqs))
    return out

freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / 7.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                        return_itc=True, decim=3, n_jobs=1)

power.plot_topo(baseline=(None, 0), mode='ratio',
                title='Induced power - MNE sample data', vmin=0., vmax=14.)
power.plot([0, 4], baseline=(None, 0), mode='ratio')

itc.plot_topo(title='Inter-Trial coherence - MNE sample data', vmin=0., vmax=1.)

print itc
print itc.ch_names
itc = itc + power
itc -= power
assert 'meg' in power
assert not ('eeg' in power)