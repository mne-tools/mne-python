import sys
from abc import ABC

import numpy as np
from mne.viz._figure import _figure, MNEBrowseFigure


class BrowserParams:
    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        vars(self).update(**kwargs)


class BrowserBase(ABC):
    def __init__(self, **kwargs):
        from .. import BaseEpochs
        from ..io import BaseRaw
        from ..preprocessing import ICA

        self._data = None
        self._times = None

        self.mne = BrowserParams(**kwargs)

        inst = kwargs['inst'] if 'inst' in kwargs else None
        ica = kwargs['ica'] if 'ica' in kwargs else None

        # what kind of data are we dealing with?
        if inst is not None:
            if isinstance(ica, ICA):
                self.mne.instance_type = 'ica'
            elif isinstance(inst, BaseRaw):
                self.mne.instance_type = 'raw'
            elif isinstance(inst, BaseEpochs):
                self.mne.instance_type = 'epochs'
            else:
                raise TypeError('Expected an instance of Raw, Epochs, or ICA, '
                                f'got {type(inst)}.')
        else:
            self.mne.instance_type = None

        self.mne.ica_type = None
        if self.mne.instance_type == 'ica':
            if isinstance(self.mne.ica_inst, BaseRaw):
                self.mne.ica_type = 'raw'
            elif isinstance(self.mne.ica_inst, BaseEpochs):
                self.mne.ica_type = 'epochs'
        self.mne.is_epochs = 'epochs' in (self.mne.instance_type,
                                          self.mne.ica_type)

        # things that always start the same
        self.mne.ch_start = 0
        self.mne.projector = None
        if hasattr(self.mne, 'projs'):
            self.mne.projs_active = np.array([p['active']
                                              for p in self.mne.projs])
        self.mne.whitened_ch_names = list()
        if hasattr(self.mne, 'noise_cov'):
            self.mne.use_noise_cov = self.mne.noise_cov is not None
        self.mne.zorder = dict(patch=0, grid=1, ann=2, events=3, bads=4,
                               data=5, mag=6, grad=7, scalebar=8, vline=9)
        # additional params for epochs (won't affect raw / ICA)
        self.mne.epoch_traces = list()
        self.mne.bad_epochs = list()
        if inst is not None:
            self.mne.sampling_period = (np.diff(inst.times[:2])[0]
                                        / inst.info['sfreq'])
        # annotations
        self.mne.annotations = list()
        self.mne.hscroll_annotations = list()
        self.mne.annotation_segments = list()
        self.mne.annotation_texts = list()
        self.mne.new_annotation_labels = list()
        self.mne.annotation_segment_colors = dict()
        self.mne.annotation_hover_line = None
        self.mne.draggable_annotations = False
        # lines
        self.mne.event_lines = None
        self.mne.event_texts = list()
        self.mne.vline_visible = False
        # scalings
        if hasattr(self.mne, 'butterfly'):
            self.mne.scale_factor = 0.5 if self.mne.butterfly else 1.
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()
        # ancillary child figures
        self.mne.child_figs = list()
        self.mne.fig_help = None
        self.mne.fig_proj = None
        self.mne.fig_histogram = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        self._times = value

    def _load_data(self, start=None, stop=None):
        """Retrieve the bit of data we need for plotting."""
        if 'raw' in (self.mne.instance_type, self.mne.ica_type):
            # Add additional sample to cover the case sfreq!=1000
            # when the shown time-range wouldn't correspond to duration anymore
            return self.mne.inst[:, start:stop + 2]
        else:
            # subtract one sample from tstart before searchsorted, to make sure
            # we land on the left side of the boundary time (avoid precision
            # errors)
            ix = np.searchsorted(self.mne.boundary_times,
                                 self.mne.t_start - self.mne.sampling_period)
            item = slice(ix, ix + self.mne.n_epochs)
            data = np.concatenate(self.mne.inst.get_data(item=item), axis=-1)
            times = np.arange(len(self.mne.inst) * len(self.mne.inst.times)
                              )[start:stop] / self.mne.info['sfreq']
            return data, times

    def _update_data(self):
        """Update self.mne.data after user interaction."""
        from ..filter import _overlap_add_filter, _filtfilt
        # update time
        start_sec = self.mne.t_start - self.mne.first_time
        stop_sec = start_sec + self.mne.duration
        if self.mne.is_epochs:
            start, stop = np.round(np.array([start_sec, stop_sec])
                                   * self.mne.info['sfreq']).astype(int)
        else:
            start, stop = self.mne.inst.time_as_index((start_sec, stop_sec))
        # get the data
        data, times = self._load_data(start, stop)
        # apply projectors
        if self.mne.projector is not None:
            data = self.mne.projector @ data
        # get only the channels we're displaying
        picks = self.mne.picks
        data = data[picks]
        # remove DC
        if self.mne.remove_dc:
            data -= data.mean(axis=1, keepdims=True)
        # filter (with same defaults as raw.filter())
        if self.mne.filter_coefs is not None:
            starts, stops = self.mne.filter_bounds
            mask = (starts < stop) & (stops > start)
            starts = np.maximum(starts[mask], start) - start
            stops = np.minimum(stops[mask], stop) - start
            for _start, _stop in zip(starts, stops):
                _picks = np.where(np.in1d(picks, self.mne.picks_data))[0]
                if len(_picks) == 0:
                    break
                this_data = data[_picks, _start:_stop]
                if isinstance(self.mne.filter_coefs, np.ndarray):  # FIR
                    this_data = _overlap_add_filter(
                        this_data, self.mne.filter_coefs, copy=False)
                else:  # IIR
                    this_data = _filtfilt(
                        this_data, self.mne.filter_coefs, None, 1, False)
                data[_picks, _start:_stop] = this_data
        # scale the data for display in a 1-vertical-axis-unit slot
        this_names = self.mne.ch_names[picks]
        this_types = self.mne.ch_types[picks]
        stims = this_types == 'stim'
        white = np.logical_and(np.in1d(this_names, self.mne.whitened_ch_names),
                               np.in1d(this_names, self.mne.info['bads'],
                                       invert=True))
        norms = np.vectorize(self.mne.scalings.__getitem__)(this_types)
        norms[stims] = data[stims].max(axis=-1)
        norms[white] = self.mne.scalings['whitened']
        norms[norms == 0] = 1
        data /= 2 * norms[:, np.newaxis]
        self.mne.data = data
        self.mne.times = times


def _get_browser(inst, backend='matplotlib', **kwargs):
    """Instantiate a new MNE browse-style figure."""
    from .utils import _get_figsize_from_config
    figsize = kwargs.pop('figsize', _get_figsize_from_config())

    if backend == 'matplotlib':
        fig = _figure(inst=inst, toolbar=False, FigureClass=MNEBrowseFigure,
                      figsize=figsize, **kwargs)
        # initialize zen mode (can't do in __init__ due to get_position() calls)
        fig.canvas.draw()
        fig._update_zen_mode_offsets()
        fig._resize(None)  # needed for MPL >=3.4
        # if scrollbars are supposed to start hidden, set to True and then toggle
        if not fig.mne.scrollbars_visible:
            fig.mne.scrollbars_visible = True
            fig._toggle_scrollbars()
    else:
        import pyqtgraph as pg
        from prototypes.pyqtgraph_ptyp import PyQtGraphPtyp

        pg.setConfigOption('enableExperimental', True)

        app = pg.mkQApp()
        fig = PyQtGraphPtyp(inst=inst, figsize=figsize, **kwargs)
        fig.show()
        sys.exit(app.exec())

    return fig