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

class AverageTFR(ContainsMixin, PickDropChannelsMixin):
    """docstring for TFR"""
    def __init__(self, info, data, times, freqs):
        self.info = info
        assert data.ndim == 3
        n_channels, n_freqs, n_times = data.shape
        assert n_channels == len(info['chs'])
        assert n_freqs == len(freqs)
        assert n_times == len(times)
        self.data = data
        self.times = times
        self.freqs = freqs

    @property
    def ch_names(self):
        return self.info['ch_names']

    def plot_topo(self, picks, tmin, tmax, fmin, fmax, clim, cmap, layout):
        pass

    def plot(self, picks=None, baseline=None, mode='mean', tmin=None, tmax=None,
             fmin=None, fmax=None, vmin=None, vmax=None, layout=None, cmap=None,
             layout_scale=0.945, title=None, dB=False, colorbar=True, show=True):

        times = self.times
        data = self.data

        if mode is not None:
            data = rescale(data.copy(), times, baseline, mode)

        times *= 1e3
        if dB:
            data = 20 * np.log10(data)
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        if layout is None:
            from mne.layouts.layout import find_layout
            layout = find_layout(self.info)

        imshow = partial(_imshow_tfr, tfr=data, freq=self.freqs)

        fig = _plot_topo(info=self.info, times=times,
                         show_func=imshow, layout=layout,
                         colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                         layout_scale=layout_scale, title=title, border='w',
                         x_label='Time (ms)', y_label='Frequency (Hz)')

        if show:
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

    def save(self, fname="foobar-tfr.fif"):
        pass

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ', channels: %d' % self.data.shape[1]
        return "<AverageTFR  |  %s>" % s


def tfr_morlet(epochs, freqs, n_cycles, use_fft=False,
               return_itc=True, zero_mean=True, decim=1, n_jobs=1):
    data = epochs.get_data()
    picks = mne.pick_types(epochs.info, meg=True, eeg=True)
    info = mne.pick_info(epochs.info, picks)
    data = data[:, picks, :]
    power, itc = induced_power(data, Fs=info['sfreq'], frequencies=freqs,
                               n_cycles=n_cycles, n_jobs=n_jobs,
                               use_fft=use_fft, decim=decim,
                               zero_mean=zero_mean)
    times = epochs.times[::decim].copy()
    out = AverageTFR(info, power, times, freqs)
    if return_itc:
        out = (out, AverageTFR(info, itc, times, freqs))
    return out

freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / 7.  # different number of cycle per frequency
decim=3
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                         return_itc=True, zero_mean=True, decim=decim, n_jobs=1)

power.plot(baseline=(None, 0), mode='ratio',
           title='Induced power - MNE sample data', vmin=0., vmax=14.)
itc.plot(title='Inter-Trial coherence - MNE sample data', vmin=0., vmax=1.)

print itc
itc = itc + power
itc -= power