# -*- coding: utf-8 -*-
"""Base classes and functions for 2D browser backends."""

# Authors: Daniel McCloy <dan@mccloy.info>
#          Martin Schulz <dev@earthman-music.de>
#
# License: Simplified BSD
import importlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from itertools import cycle

import numpy as np

from .. import verbose, get_config, set_config
from ..annotations import _sync_onset
from ..defaults import _handle_default
from ..utils import logger, _validate_type, _check_option
from ..io.pick import _DATA_CH_TYPES_SPLIT
from .backends._utils import VALID_BROWSE_BACKENDS
from .utils import _get_color_list, _setup_plot_projector

MNE_BROWSER_BACKEND = None
backend = None


class BrowserParams:
    """Container object for 2D browser parameters."""

    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        vars(self).update(**kwargs)


class BrowserBase(ABC):
    """A base class containing for the 2D browser.

    This class contains all backend-independent attributes and methods.
    """

    def __init__(self, **kwargs):
        from .. import BaseEpochs
        from ..io import BaseRaw
        from ..preprocessing import ICA

        self.backend_name = None

        self._data = None
        self._times = None

        self.mne = BrowserParams(**kwargs)

        inst = kwargs.get('inst', None)
        ica = kwargs.get('ica', None)

        # what kind of data are we dealing with?
        if isinstance(ica, ICA):
            self.mne.instance_type = 'ica'
        elif isinstance(inst, BaseRaw):
            self.mne.instance_type = 'raw'
        elif isinstance(inst, BaseEpochs):
            self.mne.instance_type = 'epochs'
        else:
            raise TypeError('Expected an instance of Raw, Epochs, or ICA, '
                            f'got {type(inst)}.')

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
        self.mne.event_lines = list()
        self.mne.event_texts = list()
        self.mne.vline_visible = False
        # decim
        self.mne.decim_times = None
        self.mne.decim_data = None
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
        # extra attributes for epochs
        if self.mne.is_epochs:
            # add epoch boundaries & center epoch numbers between boundaries
            self.mne.midpoints = np.convolve(self.mne.boundary_times,
                                             np.ones(2), mode='valid') / 2

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
        with self.mne.info._unlock():
            self.mne.info['projs'] = [deepcopy(self.mne.projs[ix])
                                      for ix in inds]
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
            # Only mpl-backend has ch_colors
            if hasattr(self.mne, 'ch_colors'):
                color = self.mne.ch_colors[idx]
            else:
                color = None
        self.mne.info['bads'] = bads

        self._update_projector()

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

    def _toggle_whitening(self):
        if self.mne.noise_cov is not None:
            self.mne.use_noise_cov = not self.mne.use_noise_cov
            self._update_projector()
            self._update_yaxis_labels()  # add/remove italics
            self._redraw()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # MANAGE TRACES
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_picks(self):
        """Compute which channel indices to show."""
        if self.mne.butterfly and self.mne.ch_selections is not None:
            selections_dict = self._make_butterfly_selections_dict()
            self.mne.picks = np.concatenate(tuple(selections_dict.values()))
        elif self.mne.butterfly:
            self.mne.picks = self.mne.ch_order
        else:
            _slice = slice(self.mne.ch_start,
                           self.mne.ch_start + self.mne.n_channels)
            self.mne.picks = self.mne.ch_order[_slice]
            self.mne.n_channels = len(self.mne.picks)

    def _make_butterfly_selections_dict(self):
        """Make an altered copy of the selections dict for butterfly mode."""
        from ..utils import _get_stim_channel
        selections_dict = deepcopy(self.mne.ch_selections)
        # remove potential duplicates
        for selection_group in ('Vertex', 'Custom'):
            selections_dict.pop(selection_group, None)
        # if present, remove stim channel from non-misc selection groups
        stim_ch = _get_stim_channel(None, self.mne.info, raise_error=False)
        if len(stim_ch):
            stim_pick = self.mne.ch_names.tolist().index(stim_ch[0])
            for _sel, _picks in selections_dict.items():
                if _sel != 'Misc':
                    stim_mask = np.in1d(_picks, [stim_pick], invert=True)
                    selections_dict[_sel] = np.array(_picks)[stim_mask]
        return selections_dict

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # MANAGE DATA
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _get_start_stop(self):
        # update time
        start_sec = self.mne.t_start - self.mne.first_time
        stop_sec = start_sec + self.mne.duration
        if self.mne.is_epochs:
            start, stop = np.round(np.array([start_sec, stop_sec])
                                   * self.mne.info['sfreq']).astype(int)
        else:
            start, stop = self.mne.inst.time_as_index((start_sec, stop_sec))

        return start, stop

    def _load_data(self, start=None, stop=None):
        """Retrieve the bit of data we need for plotting."""
        if 'raw' in (self.mne.instance_type, self.mne.ica_type):
            # Add additional sample to cover the case sfreq!=1000
            # when the shown time-range wouldn't correspond to duration anymore
            if stop is None:
                return self.mne.inst[:, start:]
            else:
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

    def _apply_filter(self, data, start, stop, picks):
        """Filter (with same defaults as raw.filter())."""
        from ..filter import _overlap_add_filter, _filtfilt
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

    def _process_data(self, data, start, stop, picks,
                      signals=None):
        """Update self.mne.data after user interaction."""
        # apply projectors
        if self.mne.projector is not None:
            if signals:
                signals.processText.emit('Applying Projectors...')
            data = self.mne.projector @ data
        # get only the channels we're displaying
        data = data[picks]
        # remove DC
        if self.mne.remove_dc:
            if signals:
                signals.processText.emit('Removing DC...')
            data -= data.mean(axis=1, keepdims=True)
        # apply filter
        if self.mne.filter_coefs is not None:
            if signals:
                signals.processText.emit('Apply Filter...')
            self._apply_filter(data, start, stop, picks)
        # scale the data for display in a 1-vertical-axis-unit slot
        if signals:
            signals.processText.emit('Scale Data...')
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

        return data

    def _update_data(self):
        start, stop = self._get_start_stop()
        # get the data
        data, times = self._load_data(start, stop)
        # process the data
        data = self._process_data(data, start, stop, self.mne.picks)
        # set the data as attributes
        self.mne.data = data
        self.mne.times = times

    def _get_epoch_num_from_time(self, time):
        epoch_nums = self.mne.inst.selection
        return epoch_nums[np.searchsorted(self.mne.boundary_times[1:], time)]

    def _redraw(self, update_data=True, annotations=False):
        """Redraws backend if necessary."""
        if update_data:
            self._update_data()

        self._draw_traces()

        if annotations and not self.mne.is_epochs:
            self._draw_annotations()

    def _close(self, event):
        """Handle close events (via keypress or window [x])."""
        # write out bad epochs (after converting epoch numbers to indices)
        if self.mne.instance_type == 'epochs':
            bad_ixs = np.in1d(self.mne.inst.selection,
                              self.mne.bad_epochs).nonzero()[0]
            self.mne.inst.drop(bad_ixs)
        # write bad channels back to instance (don't do this for proj;
        # proj checkboxes are for viz only and shouldn't modify the instance)
        if self.mne.instance_type in ('raw', 'epochs'):
            self.mne.inst.info['bads'] = self.mne.info['bads']
            logger.info(
                f"Channels marked as bad: {self.mne.info['bads'] or 'none'}")
        # ICA excludes
        elif self.mne.instance_type == 'ica':
            self.mne.ica.exclude = [self.mne.ica._ica_names.index(ch)
                                    for ch in self.mne.info['bads']]
        # write window size to config
        str_size = ','.join([str(i) for i in self._get_size()])
        set_config('MNE_BROWSE_RAW_SIZE', str_size, set_env=False)
        # Clean up child figures (don't pop(), child figs remove themselves)
        while len(self.mne.child_figs):
            fig = self.mne.child_figs[-1]
            self._close_event(fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CHILD FIGURES
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @abstractmethod
    def _new_child_figure(self, fig_name, **kwargs):
        pass

    def _create_ch_context_fig(self, idx):
        """Show context figure; idx is index of **visible** channels."""
        inst = self.mne.instance_type
        pick = self.mne.picks[idx]
        if inst == 'raw':
            fig = self._create_ch_location_fig(pick)
        elif inst == 'ica':
            fig = self._create_ica_properties_fig(pick)
        else:
            fig = self._create_epoch_image_fig(pick)

        return fig

    def _create_ch_location_fig(self, pick):
        """Show channel location figure."""
        from .utils import _channel_type_prettyprint, plot_sensors
        ch_name = self.mne.ch_names[pick]
        ch_type = self.mne.ch_types[pick]
        if ch_type not in _DATA_CH_TYPES_SPLIT:
            return
        # create figure and axes
        title = f'Location of {ch_name}'
        fig = self._new_child_figure(figsize=(4, 4), fig_name=None,
                                     window_title=title)
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        title = f'{ch_name} position ({_channel_type_prettyprint[ch_type]})'
        _ = plot_sensors(self.mne.info, ch_type=ch_type, axes=ax,
                         title=title, kind='select', show=False)
        # highlight desired channel & disable interactivity
        inds = np.in1d(fig.lasso.ch_names, [ch_name])
        fig.lasso.disconnect()
        fig.lasso.alpha_other = 0.3
        fig.lasso.linewidth_selected = 3
        fig.lasso.style_sensors(inds)

        return fig

    def _create_ica_properties_fig(self, idx):
        """Show ICA properties for the selected component."""
        from mne.viz.ica import (_create_properties_layout,
                                 _prepare_data_ica_properties,
                                 _fast_plot_ica_properties)

        ch_name = self.mne.ch_names[idx]
        if ch_name not in self.mne.ica._ica_names:  # for EOG chans: do nothing
            return
        pick = self.mne.ica._ica_names.index(ch_name)
        title = f'{ch_name} properties'
        fig = self._new_child_figure(figsize=(7, 6), fig_name=None,
                                     window_title=title)
        fig.suptitle(title)
        fig, axes = _create_properties_layout(fig=fig)
        if not hasattr(self.mne, 'data_ica_properties'):
            # Precompute epoch sources only once
            self.mne.data_ica_properties = _prepare_data_ica_properties(
                self.mne.ica_inst, self.mne.ica)
        _fast_plot_ica_properties(
            self.mne.ica, self.mne.ica_inst, picks=pick, axes=axes,
            precomputed_data=self.mne.data_ica_properties, show=False)

        return fig

    def _create_epoch_image_fig(self, pick):
        """Show epochs image for the selected channel."""
        from matplotlib.gridspec import GridSpec
        from mne.viz import plot_epochs_image
        ch_name = self.mne.ch_names[pick]
        title = f'Epochs image ({ch_name})'
        fig = self._new_child_figure(figsize=(6, 4), fig_name=None,
                                     window_title=title)
        fig.suptitle = title
        gs = GridSpec(nrows=3, ncols=10)
        fig.add_subplot(gs[:2, :9])
        fig.add_subplot(gs[2, :9])
        fig.add_subplot(gs[:2, 9])
        plot_epochs_image(self.mne.inst, picks=pick, fig=fig, show=False)

        return fig

    def _toggle_epoch_histogram(self):
        """Show or hide peak-to-peak histogram of channel amplitudes."""
        if self.mne.instance_type == 'epochs':
            if self.mne.fig_histogram is None:
                self._create_epoch_histogram()
            else:
                from matplotlib.pyplot import close
                close(self.mne.fig_histogram)

    def _create_epoch_histogram(self):
        """Create peak-to-peak histogram of channel amplitudes."""
        epochs = self.mne.inst
        data = OrderedDict()
        ptp = np.ptp(epochs.get_data(), axis=2)
        for ch_type in ('eeg', 'mag', 'grad'):
            if ch_type in epochs:
                data[ch_type] = ptp.T[self.mne.ch_types == ch_type].ravel()
        units = _handle_default('units')
        titles = _handle_default('titles')
        colors = _handle_default('color')
        scalings = _handle_default('scalings')
        title = 'Histogram of peak-to-peak amplitudes'
        figsize = (4, 1 + 1.5 * len(data))
        fig = self._new_child_figure(figsize=figsize, fig_name='fig_histogram',
                                     window_title=title)
        for ix, (_ch_type, _data) in enumerate(data.items()):
            ax = fig.add_subplot(len(data), 1, ix + 1)
            ax.set(title=titles[_ch_type], xlabel=units[_ch_type],
                   ylabel='Count')
            # set histogram bin range based on rejection thresholds
            reject = None
            _range = None
            if epochs.reject is not None and _ch_type in epochs.reject:
                reject = epochs.reject[_ch_type] * scalings[_ch_type]
                _range = (0., reject * 1.1)
            # plot it
            ax.hist(_data * scalings[_ch_type], bins=100,
                    color=colors[_ch_type], range=_range)
            if reject is not None:
                ax.plot((reject, reject), (0, ax.get_ylim()[1]), color='r')
        # finalize
        fig.suptitle(title, y=0.99)
        kwargs = dict(bottom=fig._inch_to_rel(0.5, horiz=False),
                      top=1 - fig._inch_to_rel(0.5, horiz=False),
                      left=fig._inch_to_rel(0.75),
                      right=1 - fig._inch_to_rel(0.25))
        fig.subplots_adjust(hspace=0.7, **kwargs)
        self.mne.fig_histogram = fig

        return fig

    def _close_event(self, fig):
        # This method is a fix for mpl issue #18609, which still seems to
        # be a problem with matplotlib==3.4.
        pass
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TEST METHODS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @abstractmethod
    def _get_size(self):
        pass

    @abstractmethod
    def _fake_keypress(self, key, fig):
        pass

    @abstractmethod
    def _fake_click(self, point, fig, axis, xform, button, kind):
        pass

    @abstractmethod
    def _click_ch_name(self, ch_index, button):
        pass

    @abstractmethod
    def _resize_by_factor(self, factor):
        pass

    @abstractmethod
    def _get_ticklabels(self, orientation):
        pass

    @abstractmethod
    def _update_yaxis_labels(self):
        pass


def _load_backend(backend_name):
    global backend
    if backend_name == 'matplotlib':
        backend = importlib.import_module(name='._mpl_figure',
                                          package='mne.viz')
    else:
        from mne_qt_browser import _pg_figure as backend

    logger.info(f'Using {backend_name} as 2D backend.')

    return backend


def _get_browser(**kwargs):
    """Instantiate a new MNE browse-style figure."""
    from .utils import _get_figsize_from_config
    figsize = kwargs.setdefault('figsize', _get_figsize_from_config())
    if figsize is None or np.any(np.array(figsize) < 8):
        kwargs['figsize'] = (8, 8)
    # Initialize browser backend
    _init_browser_backend()

    # Initialize Browser
    browser = backend._init_browser(**kwargs)

    return browser


def _check_browser_backend_name(backend_name):
    _validate_type(backend_name, str, 'backend_name')
    _check_option('backend_name', backend_name, VALID_BROWSE_BACKENDS)
    return backend_name


@verbose
def set_browser_backend(backend_name, verbose=None):
    """Set the 2D browser backend for MNE.

    The backend will be set as specified and operations will use
    that backend.

    Parameters
    ----------
    backend_name : str
        The 2D browser backend to select. See Notes for the capabilities
        of each backend (``'matplotlib'``, ``'pyqtgraph'``).
    %(verbose)s

    Returns
    -------
    old_backend_name : str | None
        The old backend that was in use.

    Notes
    -----
    This table shows the capabilities of each backend ("✓" for full support,
    and "-" for partial support):

    .. table::
       :widths: auto

       +--------------------------------------+------------+-----------+
       | **2D browser function:**             | matplotlib | pyqtgraph |
       +======================================+============+===========+
       | :func:`plot_raw`                     | ✓          | ✓         |
       +--------------------------------------+------------+-----------+
       | :func:`plot_epochs`                  | ✓          |           |
       +--------------------------------------+------------+-----------+
       | :func:`plot_ica_sources`             | ✓          |           |
       +--------------------------------------+------------+-----------+
       +--------------------------------------+------------+-----------+
       | **Feature:**                                                  |
       +--------------------------------------+------------+-----------+
       | Show Events                          | ✓          | ✓         |
       +--------------------------------------+------------+-----------+
       | Add/Edit/Remove Annotations          | ✓          | ✓         |
       +--------------------------------------+------------+-----------+
       | Toggle Projections                   | ✓          | ✓         |
       +--------------------------------------+------------+-----------+
       | Butterfly Mode                       | ✓          | ✓         |
       +--------------------------------------+------------+-----------+
       | Selection Mode                       | ✓          | ✓         |
       +--------------------------------------+------------+-----------+
       | Smooth Scrolling                     |            | ✓         |
       +--------------------------------------+------------+-----------+
       | Overview-Bar (with Z-Score-Mode)     |            | ✓         |
       +--------------------------------------+------------+-----------+

    .. versionadded:: 0.24
    """
    global MNE_BROWSER_BACKEND
    old_backend_name = MNE_BROWSER_BACKEND
    backend_name = _check_browser_backend_name(backend_name)
    if MNE_BROWSER_BACKEND != backend_name:
        _load_backend(backend_name)
        MNE_BROWSER_BACKEND = backend_name

    return old_backend_name


def _init_browser_backend():
    global MNE_BROWSER_BACKEND

    # check if MNE_BROWSER_BACKEND is not None and valid or get it from config
    loaded_backend = (MNE_BROWSER_BACKEND or
                      get_config(key='MNE_BROWSER_BACKEND', default=None))
    if loaded_backend is not None:
        set_browser_backend(loaded_backend)
        return MNE_BROWSER_BACKEND
    else:
        errors = dict()
        # Try import of valid browser backends
        for name in VALID_BROWSE_BACKENDS:
            try:
                _load_backend(name)
            except ImportError as exc:
                errors[name] = str(exc)
            else:
                MNE_BROWSER_BACKEND = name
                break
        else:
            raise RuntimeError(
                'Could not load any valid 2D backend:\n' +
                '\n'.join(f'{key}: {val}' for key, val in errors.items()))

        return MNE_BROWSER_BACKEND


def get_browser_backend():
    """Return the 2D backend currently used.

    Returns
    -------
    backend_used : str | None
        The 2D browser backend currently in use. If no backend is found,
        returns ``None``.
    """
    try:
        backend_name = _init_browser_backend()
    except RuntimeError as exc:
        backend_name = None
        logger.info(str(exc))
    return backend_name


@contextmanager
def use_browser_backend(backend_name):
    """Create a 2D browser visualization context using the designated backend.

    See :func:`mne.viz.set_browser_backend` for more details on the available
    2D browser backends and their capabilities.

    Parameters
    ----------
    backend_name : {'matplotlib', 'pyqtgraph'}
        The 2D browser backend to use in the context.
    """
    old_backend = set_browser_backend(backend_name)
    try:
        yield backend
    finally:
        if old_backend is not None:
            try:
                set_browser_backend(old_backend)
            except Exception:
                pass
