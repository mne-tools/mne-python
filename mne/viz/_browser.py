import sys
from abc import ABC
from copy import deepcopy
from itertools import cycle

import numpy as np
from mne.annotations import _sync_onset
from mne.viz._figure import _figure, MNEBrowseFigure
from mne.viz.utils import _get_color_list, _merge_annotations, \
    _setup_plot_projector


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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ANNOTATIONS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _get_annotation_labels(self):
        """Get the unique labels in the raw object and added in the UI."""
        return sorted(set(self.mne.inst.annotations.description) |
                      set(self.mne.new_annotation_labels))

    def _toggle_draggable_annotations(self):
        """Enable/disable draggable annotation edges."""
        self.mne.draggable_annotations = not self.mne.draggable_annotations

    def _setup_annotation_colors(self):
        """Set up colors for annotations; init some annotation vars."""
        segment_colors = getattr(self.mne, 'annotation_segment_colors', dict())
        labels = self._get_annotation_labels()
        colors, red = _get_color_list(annotations=True)
        color_cycle = cycle(colors)
        for key, color in segment_colors.items():
            if color != red and key in labels:
                next(color_cycle)
        for idx, key in enumerate(labels):
            if key in segment_colors:
                continue
            elif key.lower().startswith('bad') or \
                    key.lower().startswith('edge'):
                segment_colors[key] = red
            else:
                segment_colors[key] = next(color_cycle)
        self.mne.annotation_segment_colors = segment_colors
        # init a couple other annotation-related variables
        self.mne.visible_annotations = {label: True for label in labels}
        self.mne.show_hide_annotation_checkboxes = None

    def _clear_annotations(self):
        """Clear all annotations from the figure."""
        for annot in list(self.mne.annotations):
            annot.remove()
            self.mne.annotations.remove(annot)
        for annot in list(self.mne.hscroll_annotations):
            annot.remove()
            self.mne.hscroll_annotations.remove(annot)
        for text in list(self.mne.annotation_texts):
            text.remove()
            self.mne.annotation_texts.remove(text)

    def _update_annotation_segments(self):
        """Update the array of annotation start/end times."""
        segments = list()
        raw = self.mne.inst
        if len(raw.annotations):
            for idx, annot in enumerate(raw.annotations):
                annot_start = _sync_onset(raw, annot['onset'])
                annot_end = annot_start + max(annot['duration'],
                                              1 / self.mne.info['sfreq'])
                segments.append((annot_start, annot_end))
        self.mne.annotation_segments = np.array(segments)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # PROJECTOR & BADS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_projector(self):
        """Update the data after projectors (or bads) have changed."""
        inds = np.where(self.mne.projs_on)[0]  # doesn't include "active" projs
        # copy projs from full list (self.mne.projs) to info object
        self.mne.info['projs'] = [deepcopy(self.mne.projs[ix]) for ix in inds]
        # compute the projection operator
        proj, wh_chs = _setup_plot_projector(self.mne.info, self.mne.noise_cov,
                                             True, self.mne.use_noise_cov)
        self.mne.whitened_ch_names = list(wh_chs)
        self.mne.projector = proj

    def _toggle_bad_channel(self, idx):
        """Mark/unmark bad channels; `idx` is index of *visible* channels."""
        pick = self.mne.picks[idx]
        ch_name = self.mne.ch_names[pick]
        # add/remove from bads list
        bads = self.mne.info['bads']
        marked_bad = ch_name not in bads
        if marked_bad:
            bads.append(ch_name)
            color = self.mne.ch_color_bad
        else:
            while ch_name in bads:  # to make sure duplicates are removed
                bads.remove(ch_name)
            color = self.mne.ch_colors[idx]
        self.mne.info['bads'] = bads

        return color, pick, marked_bad

    def _toggle_bad_epoch(self, xtime):
        epoch_num = self._get_epoch_num_from_time(xtime)
        epoch_ix = self.mne.inst.selection.tolist().index(epoch_num)
        if epoch_num in self.mne.bad_epochs:
            self.mne.bad_epochs.remove(epoch_num)
            color = 'none'
        else:
            self.mne.bad_epochs.append(epoch_num)
            self.mne.bad_epochs.sort()
            color = self.mne.epoch_color_bad

        return epoch_ix, color

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA TRACES
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_picks(self):
        """Compute which channel indices to show."""
        if self.mne.butterfly and self.mne.ch_selections is not None:
            selections_dict = self._make_butterfly_selections_dict()
            self.mne.picks = np.concatenate(tuple(selections_dict.values()))
        elif self.mne.butterfly:
            self.mne.picks = np.arange(self.mne.ch_names.shape[0])
        else:
            _slice = slice(self.mne.ch_start,
                           self.mne.ch_start + self.mne.n_channels)
            self.mne.picks = self.mne.ch_order[_slice]
            self.mne.n_channels = len(self.mne.picks)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # MANAGE DATA
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

    def _get_epoch_num_from_time(self, time):
        epoch_nums = self.mne.inst.selection
        return epoch_nums[np.searchsorted(self.mne.boundary_times[1:], time)]

    def _redraw(self, **kwargs):
        """This is usually not necessary for the pyqtgraph-backend as
        the redraw of objects is often handled by pyqtgraph internally."""
        pass


def _get_browser(inst, backend='matplotlib', **kwargs):
    """Instantiate a new MNE browse-style figure."""
    from .utils import _get_figsize_from_config
    figsize = kwargs.pop('figsize', _get_figsize_from_config())

    if backend == 'pyqtgraph':
        try:
            import pyqtgraph as pg
            from prototypes.pyqtgraph_ptyp import PyQtGraphPtyp
        except ModuleNotFoundError:
            backend = 'matplotlib'
        else:
            pg.setConfigOption('enableExperimental', True)

            app = pg.mkQApp()
            browser = PyQtGraphPtyp(inst=inst, figsize=figsize, **kwargs)
            browser.show()
            sys.exit(app.exec())

    if backend == 'matplotlib':
        browser = _figure(inst=inst, toolbar=False, FigureClass=MNEBrowseFigure,
                      figsize=figsize, **kwargs)
        # initialize zen mode (can't do in __init__ due to get_position() calls)
        browser.canvas.draw()
        browser._update_zen_mode_offsets()
        browser._resize(None)  # needed for MPL >=3.4
        # if scrollbars are supposed to start hidden, set to True and then toggle
        if not browser.mne.scrollbars_visible:
            browser.mne.scrollbars_visible = True
            browser._toggle_scrollbars()

    return browser