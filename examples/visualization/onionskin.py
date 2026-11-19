"""
==============================================================
Advanced plotting customization by subclassing MNEBrowseFigure
==============================================================

This example shows how plot_epochs(...) and plot_raw(...) can be customized by
subclassing MNEBrowseFigure and using the ``figure_class`` argument.
It plots one EEG trace overlaid ("onion-skinned") on top of another.

This example is "bad code" in a few ways:

* Since the interface for MNEBrowseFigure is not public, it is liable to
  break between minor and even patch versions of MNE without warning
* Some functionality is reimplemented from MNEBrowseFigure in a more or
  less copy-paste style
* The code is backend-specific, in particular it is limited to the
  matplotlib backend, and will not work with the qt browser
* Since there is no way to pass another EEG directly to the MNEBrowseFigure,
  it is passed through a global variable

Nevertheless, the example shows that the "escape hatch" of using a subclass is
available when other customization possibilities offered by MNE are not
sufficient.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne.viz import set_browser_backend
from mne.viz._mpl_figure import MNEBrowseFigure as MNEBrowseFigureOrig

set_browser_backend("matplotlib")


onionskin_eeg = None


def _set_onionskin_eeg(eeg):
    global onionskin_eeg
    onionskin_eeg = eeg


class OnionskinMNEBrowseFigure(MNEBrowseFigureOrig):
    """
    Subclass of MNEBrowseFigure adding in onion-skin functionality,
    i.e. plotting one EEG trace overlaid on top of another.
    """

    def __init__(self, *args, **kwargs):
        import numpy as np

        super().__init__(*args, **kwargs)
        onionskin_kwargs = {
            **self.mne.trace_kwargs,
        }
        self.mne.onionskins = self.mne.ax_main.plot(
            np.full((1, self.mne.n_channels), np.nan), **onionskin_kwargs
        )

    def _update_data(self):
        import numpy as np

        from mne.io.base import BaseRaw

        super()._update_data()
        if not onionskin_eeg:
            self.mne.onionskin_data = None
            return
        start, stop = self._get_start_stop()
        if isinstance(onionskin_eeg, BaseRaw):
            if stop is None:
                data = onionskin_eeg[:, start:]
            else:
                data = onionskin_eeg[:, start:stop]
            data = data[0]
        else:
            ix_start = np.searchsorted(
                self.mne.boundary_times, self.mne.t_start - self.mne.sampling_period
            )
            ix_stop = ix_start + self.mne.n_epochs
            item = slice(ix_start, ix_stop)
            print(type(onionskin_eeg))
            data = np.concatenate(
                onionskin_eeg.get_data(item=item, copy=False), axis=-1
            )
        data = self._process_data(data, start, stop, picks=self.mne.picks)
        self.mne.onionskin_data = data

    def _draw_traces(self):
        import numpy as np
        from matplotlib.colors import to_rgba_array
        from matplotlib.patches import Rectangle

        super()._draw_traces()
        if self.mne.onionskin_data is None:
            return
        picks = self.mne.picks
        offset_ixs = (
            picks
            if self.mne.butterfly and self.mne.ch_selections is None
            else slice(None)
        )
        offsets = self.mne.trace_offsets[offset_ixs]

        ch_colors = to_rgba_array(self.mne.ch_colors)
        ch_colors[:, 3] *= 0.5

        decim = np.ones_like(picks)
        data_picks_mask = np.isin(picks, self.mne.picks_data)
        decim[data_picks_mask] = self.mne.decim
        # decim can vary by channel type, so compute different `times` vectors
        decim_times = {
            decim_value: self.mne.times[::decim_value] + self.mne.first_time
            for decim_value in set(decim)
        }

        time_range = (self.mne.times + self.mne.first_time)[[0, -1]]
        ylim = self.mne.ax_main.get_ylim()
        for ii, line in enumerate(self.mne.onionskins):
            this_offset = offsets[ii]
            this_times = decim_times[decim[ii]]
            this_data = (
                this_offset - self.mne.onionskin_data[ii] * self.mne.scale_factor
            )
            this_data = this_data[..., :: decim[ii]]
            clip = 0.2 if self.mne.butterfly else 0.5
            bottom = max(this_offset - clip, ylim[1])
            height = min(2 * clip, ylim[0] - bottom)
            rect = Rectangle(
                xy=np.array([time_range[0], bottom]),
                width=time_range[1] - time_range[0],
                height=height,
                transform=self.mne.ax_main.transData,
            )
            line.set_clip_path(rect)
            line.set_xdata(this_times)
            line.set_ydata(this_data)
            color = ch_colors[ii]
            line.set_color(color)
            line.set_zorder(self.mne.zorder["data"] - 1)


subjects = [1]
runs = [1, 2]
raw_fnames = eegbci.load_data(subjects, runs)
first_data = read_raw_edf(raw_fnames[0], preload=True)
second_data = read_raw_edf(raw_fnames[1], preload=True)

first_data.plot(title="First plot", n_channels=3)
second_data.plot(title="Second plot", n_channels=3)

_set_onionskin_eeg(first_data)
second_data.plot(
    title="Onionskinned plot",
    n_channels=3,
    block=True,
    figure_class=OnionskinMNEBrowseFigure,
)
