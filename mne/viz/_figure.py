# -*- coding: utf-8 -*-
"""Figure classes for MNE-Python's 2D plots."""

# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

from functools import partial
import numpy as np
from matplotlib.figure import Figure
from .utils import plt_show
from ..utils import set_config


class MNEFigParams:
    """Container for MNE figure parameters."""
    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        for attr, value in kwargs.items():
            setattr(self, attr, value)


class MNEFigure(Figure):
    """Wrapper of matplotlib.figure.Figure; adds MNE-Python figure params."""
    def __init__(self, **kwargs):
        # figsize is the only kwarg we pass to matplotlib Figure()
        figsize = kwargs.pop('figsize', None)
        super().__init__(figsize=figsize)
        # remove matplotlib default keypress catchers
        default_cbs = list(
            self.canvas.callbacks.callbacks.get('key_press_event', {}))
        for callback in default_cbs:
            self.canvas.callbacks.disconnect(callback)
        # add our param object
        self.mne = MNEFigParams(**kwargs)

    def _get_dpi_ratio(self):
        """Get DPI ratio (to handle hi-DPI screens)."""
        dpi_ratio = 1.
        for key in ('_dpi_ratio', '_device_scale'):
            dpi_ratio = getattr(self.canvas, key, dpi_ratio)
        return dpi_ratio

    def _get_size_px(self):
        """Get figure size in pixels."""
        dpi_ratio = self._get_dpi_ratio()
        size = self.get_size_inches() * self.dpi / dpi_ratio
        return size

    def _inch_to_rel(self, dim_inches, horiz=True):
        """Convert inches to figure-relative distances."""
        fig_w, fig_h = self.get_size_inches()
        w_or_h = fig_w if horiz else fig_h
        return dim_inches / w_or_h


class MNEDialogFigure(MNEFigure):
    """Interactive dialog figure for annotations, projectors, etc."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _keypress(self, event):
        from matplotlib.pyplot import close
        if event.key == self.mne.close_key:
            close(self)


class MNEBrowseFigure(MNEFigure):
    """Interactive figure with scrollbars, for data browsing."""
    def __init__(self, inst, xlabel='Time (s)', **kwargs):
        from matplotlib.widgets import Button
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from .utils import _get_figsize_from_config
        from .. import BaseEpochs
        from ..io import BaseRaw
        from ..preprocessing import ICA

        # get figsize from config if not provided
        figsize = kwargs.pop('figsize', _get_figsize_from_config())
        kwargs.update(inst=inst)
        super().__init__(figsize=figsize, **kwargs)

        # what kind of data are we dealing with?
        if isinstance(inst, BaseRaw):
            self.mne.instance_type = 'raw'
        elif isinstance(inst, BaseEpochs):
            self.mne.instance_type = 'epochs'
        elif isinstance(inst, ICA):
            self.mne.instance_type = 'ica'
        else:
            raise TypeError('Expected an instance of Raw, Epochs, or ICA, '
                            f'got {type(inst)}.')

        # additional params for browse figures (comments indicate name changes)
        # self.mne.inst = inst                # raw

        # self.mne.info = None
        self.mne.projector = None
        # self.mne.proj = None
        # self.mne.noise_cov = None
        # self.mne.event_id_rev = None
        # # channel
        # self.mne.n_channels = None
        # self.mne.ch_types = None            # types
        # self.mne.group_by = None
        # self.mne.data_picks = None
        self.mne.whitened_ch_names = list()

        # # time
        # self.mne.n_times = None
        # self.mne.first_time = None
        # self.mne.event_times = None
        # self.mne.event_nums = None
        # self.mne.duration = None
        # self.mne.decim = None
        # self.mne.hsel_patch = None

        # # annotations
        # self.mne.annotations = None
        # self.mne.snap_annotations = None
        # self.mne.added_label = None
        # self.mne.annotation_segments      # segments

        # # traces
        # self.mne.traces = None            # lines
        # self.mne.trace_offsets = None     # offsets
        # self.mne.ch_order = None     # inds
        # self.mne.orig_indices = None      # orig_inds
        self.mne.segment_line = None
        # self.mne.clipping = None
        # self.mne.butterfly = None

        # # filters
        # self.mne.remove_dc = None
        # self.mne.filter_coefs = None      # ba
        # self.mne.filter_bounds = None     # filt_bounds

        # # scalings
        # self.mne.units = None
        # self.mne.scalings = None
        # self.mne.unit_scalings = None
        self.mne.scale_factor = 1.
        self.mne.scalebars = dict()           # (new)
        self.mne.scalebar_texts = dict()      # (new)

        # # ancillary figures
        # self.mne.fig_proj = None
        # self.mne.fig_help = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None

        # # UI state variables
        # self.mne.ch_start = None
        # self.mne.t_start = None
        # self.mne.scalebars_visible = None
        # self.mne.scrollbars_visible = scrollbars_visible

        # MAIN AXES: default sizes (inches)
        l_margin = 1.
        r_margin = 0.1
        b_margin = 0.45
        t_margin = 0.25
        scroll_width = 0.25
        hscroll_dist = 0.25
        vscroll_dist = 0.1
        help_width = scroll_width * 2

        # MAIN AXES: default margins (figure-relative coordinates)
        left = self._inch_to_rel(l_margin - vscroll_dist - help_width)
        right = 1 - self._inch_to_rel(r_margin)
        bottom = self._inch_to_rel(b_margin, horiz=False)
        top = 1 - self._inch_to_rel(t_margin, horiz=False)
        width = right - left
        height = top - bottom
        position = [left, bottom, width, height]

        # Main axes must be a subplot for subplots_adjust to work (so user can
        # adjust margins). That's why we don't use the Divider class directly.
        ax = self.add_subplot(1, 1, 1, position=position)
        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right)
        div = make_axes_locatable(ax)

        # SCROLLBARS
        ax_hscroll = div.append_axes(position='bottom',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(hscroll_dist))
        ax_vscroll = div.append_axes(position='right',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(vscroll_dist))
        ax_hscroll.get_yaxis().set_visible(False)
        ax_hscroll.set_xlabel(xlabel)
        ax_vscroll.set_axis_off()
        # HELP BUTTON: initialize in the wrong spot...
        ax_help = div.append_axes(position='left',
                                  size=Fixed(help_width),
                                  pad=Fixed(vscroll_dist))
        # HELP BUTTON: ...move it down by changing its locator
        loc = div.new_locator(nx=0, ny=0)
        ax_help.set_axes_locator(loc)
        # HELP BUTTON: make it a proper button
        button_help = Button(ax_help, 'Help')
        button_help.on_clicked(self._onclick_help)
        # PROJ BUTTON
        if len(inst.info['projs']) and not inst.proj:
            # PROJ BUTTON: compute position
            proj_button_pos = [
                1 - self._inch_to_rel(r_margin + scroll_width),  # left
                self._inch_to_rel(b_margin, horiz=False),        # bottom
                self._inch_to_rel(scroll_width),                 # width
                self._inch_to_rel(scroll_width, horiz=False)     # height
            ]
            loc = div.new_locator(nx=4, ny=0)
            ax_proj = self.add_axes(proj_button_pos)
            ax_proj.set_axes_locator(loc)
            # PROJ BUTTON: make it a proper button
            button_proj = Button(ax_proj, 'Prj')
            button_proj.on_clicked(self._toggle_proj_fig)
            # self.mne.apply_proj = proj  # TODO add proj to init signature?

        # SAVE PARAMS
        self.mne.ax_main = ax
        self.mne.ax_help = ax_help
        self.mne.ax_proj = ax_proj
        self.mne.ax_hscroll = ax_hscroll
        self.mne.ax_vscroll = ax_vscroll
        self.mne.button_help = button_help
        self.mne.button_proj = button_proj

    def _resize(self, event):
        """Handle resize event for mne_browse-style plots (Raw/Epochs/ICA)."""
        size = ','.join(self.get_size_inches().astype(str))
        set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
        old_width, old_height = self.mne.fig_size_px
        new_width, new_height = self._get_size_px()
        new_margins = dict()
        for side in ('left', 'right', 'bottom', 'top'):
            ratio = ((old_width / new_width) if side in ('left', 'right') else
                     (old_height / new_height))
            rel_dim = getattr(self.subplotpars, side)
            if side in ('right', 'top'):
                new_margins[side] = 1 - ratio * (1 - rel_dim)
            else:
                new_margins[side] = ratio * rel_dim
        self.subplots_adjust(**new_margins)
        # zen mode bookkeeping
        self.mne.zen_w *= old_width / new_width
        self.mne.zen_h *= old_height / new_height
        self.mne.fig_size_px = (new_width, new_height)

    def _toggle_scrollbars(self):
        """Show or hide scrollbars (A.K.A. zen mode)."""
        # grow/shrink main axes to take up space from (or make room for)
        # scrollbars. We can't use ax.set_position() because axes are
        # locatable, so we use subplots_adjust
        should_show = not self.mne.scrollbars_visible
        margins = {side: getattr(self.subplotpars, side)
                   for side in ('left', 'bottom', 'right', 'top')}
        # if should_show, bottom margin moves up; right margin moves left
        margins['bottom'] += (1 if should_show else -1) * self.mne.zen_h
        margins['right'] += (-1 if should_show else 1) * self.mne.zen_w
        # squeeze a bit more because we don't need space for xlabel now
        v_delta = self._inch_to_rel(0.16, horiz=False)
        margins['bottom'] += (1 if should_show else -1) * v_delta
        self.subplots_adjust(**margins)
        # show/hide
        for elem in ('ax_hscroll', 'ax_vscroll', 'ax_button', 'ax_help'):
            butterfly = getattr(self.mne, 'butterfly', False)
            if elem == 'ax_vscroll' and butterfly:
                continue
            # sometimes we don't have a proj button (ax_button)
            if getattr(self.mne, elem, None) is not None:
                getattr(self.mne, elem).set_visible(should_show)
        self.mne.scrollbars_visible = should_show
        self.canvas.draw()

    def _toggle_proj_fig(self, event):
        """Show/hide the projectors dialog window."""
        if self.mne.fig_proj is None:
            self._create_proj_fig(draw_current_state=False)
        else:
            self.mne.fig_proj.canvas.close_event()
            del self.mne.proj_checks
            self.mne.fig_proj = None

    def _toggle_proj(self, event, all_=False):
        """Perform operations when proj boxes clicked."""
        # TODO: get from viz/utils.py lines 308-336
        pass

    def _create_proj_fig(self, draw_current_state):
        """Create the projectors dialog window."""
        # TODO: partially incorporated from _draw_proj_checkbox; untested
        from matplotlib.widgets import Button, CheckButtons

        projs = self.inst.info['projs']
        labels = [p['desc'] for p in projs]
        actives = ([p['active'] for p in projs] if draw_current_state else
                   getattr(self.mne, 'proj_bools',
                           [self.mne.apply_proj] * len(projs)))
        # make figure
        width = max([4., max([len(p['desc']) for p in projs]) / 6.0 + 0.5])
        height = (len(projs) + 1) / 6.0 + 1.5
        self.mne.fig_proj = dialog_figure(figsize=(width, height))
        self.mne.fig_proj.canvas.set_window_title('SSP projection vectors')
        # make axes
        offset = (1. / 6. / height)
        position = (0, offset, 1, 0.8 - offset)
        ax_temp = self.mne.fig_proj.add_axes(position, frameon=False)
        ax_temp.set_title('Projectors marked with "X" are active')
        # draw checkboxes
        self.mne.proj_checks = CheckButtons(ax_temp, labels=labels,
                                            actives=actives)
        for rect in self.mne.proj_checks.rectangles:
            rect.set_edgecolor('0.5')
            rect.set_linewidth(1.)
        # change already-applied projectors to red
        for ii, p in enumerate(projs):
            if p['active']:
                for x in self.mne.proj_checks.lines[ii]:
                    x.set_color('#ff0000')
        # add event listeners
        self.mne.proj_checks.on_clicked(self._toggle_proj)
        # add "toggle all" button
        ax_all = self.mne.fig_proj.add_axes((0, 0, 1, offset), frameon=False)
        self.mne.proj_all = Button(ax_all, 'Toggle all')
        self.mne.proj_all.on_clicked(partial(self._toggle_proj, all_=True))
        # show figure (this should work for non-test cases)
        try:
            self.mne.fig_proj.canvas.draw()
            plt_show(fig=self.mne.fig_proj, warn=False)
        except Exception:
            pass
        # TODO: partially incorporated from _draw_proj_checkbox; untested

    def _setup_annotation_colors(self):
        pass

    def _plot_annotations(self):
        """."""
        from ..annotations import _sync_onset
        while len(self.mne.ax_hscroll.collections) > 0:
            self.mne.ax_hscroll.collections.pop()
        segments = list()
        self._setup_annotation_colors()
        for idx, annot in enumerate(self.mne.inst.annotations):
            annot_start = (_sync_onset(self.mne.inst, annot['onset']) +
                           self.mne.first_time)
            annot_end = annot_start + annot['duration']
            segments.append([annot_start, annot_end])
            self.mne.ax_hscroll.fill_betweenx(
                (0., 1.), annot_start, annot_end, alpha=0.3,
                color=self.mne.segment_colors[annot['description']])
        # Do not adjust ½ sample backward (even though it would clarify what
        # is included) because that would break click-drag functionality
        self.mne.segments = np.array(segments)

    def _create_annotation_fig(self):
        """."""
        pass
        #"""Initialize the annotation figure."""
        #import matplotlib.pyplot as plt
        #from matplotlib.widgets import RadioButtons, SpanSelector, Button
        #if params['fig_annotation'] is not None:
        #    params['fig_annotation'].canvas.close_event()
        #annotations = params['raw'].annotations
        #labels = list(set(annotations.description))
        #labels = np.union1d(labels, params['added_label'])
        #fig = figure_nobar(figsize=(4.5, 2.75 + len(labels) * 0.75))
        #fig.patch.set_facecolor('white')
        #len_labels = max(len(labels), 1)
        ## can't pass fig=fig here on matplotlib 2.0.2, need to wait for an update
        #ax = plt.subplot2grid((len_labels + 2, 2), (0, 0),
        #                      rowspan=len_labels,
        #                      colspan=2, frameon=False)
        #ax.set_title('Labels')
        #ax.set_aspect('equal')
        #button_ax = plt.subplot2grid((len_labels + 2, 2), (len_labels, 1),
        #                             rowspan=1, colspan=1)
        #label_ax = plt.subplot2grid((len_labels + 2, 2), (len_labels, 0),
        #                            rowspan=1, colspan=1)
        #plt.axis('off')
        #text_ax = plt.subplot2grid((len_labels + 2, 2), (len_labels + 1, 0),
        #                           rowspan=1, colspan=2)
        #text_ax.text(0.5, 0.9, 'Left click & drag - Create/modify annotation\n'
        #                       'Right click - Delete annotation\n'
        #                       'Letter/number keys - Add character\n'
        #                       'Backspace - Delete character\n'
        #                       'Esc - Close window/exit annotation mode', va='top',
        #             ha='center')
        #plt.axis('off')
        #annotations_closed = partial(_annotations_closed, params=params)
        #fig.canvas.mpl_connect('close_event', annotations_closed)
        #fig.canvas.set_window_title('Annotations')
        #fig.radio = RadioButtons(ax, labels, activecolor='#cccccc')
        #radius = 0.15
        #circles = fig.radio.circles
        #for circle, label in zip(circles, fig.radio.labels):
        #    circle.set_edgecolor(params['segment_colors'][label.get_text()])
        #    circle.set_linewidth(4)
        #    circle.set_radius(radius / (len(labels)))
        #    label.set_x(circle.center[0] + (radius + 0.1) / len(labels))
        #if len(fig.radio.circles) < 1:
        #    col = '#ff0000'
        #else:
        #    col = circles[0].get_edgecolor()
        #fig.canvas.mpl_connect('key_press_event', partial(
        #    _change_annotation_description, params=params))
        #fig.button = Button(button_ax, 'Add label')
        #fig.label = label_ax.text(0.5, 0.5, '"BAD_"', va='center', ha='center')
        #fig.button.on_clicked(partial(_onclick_new_label, params=params))
        #plt_show(fig=fig)
        #params['fig_annotation'] = fig
        #ax = params['ax']
        #cb_onselect = partial(_annotate_select, params=params)
        #selector = SpanSelector(ax, cb_onselect, 'horizontal', minspan=.1,
        #                        rectprops=dict(alpha=0.5, facecolor=col))
        #if len(labels) == 0:
        #    selector.active = False
        #params['ax'].selector = selector
        #hover_callback = partial(_on_hover, params=params)
        #params['hover_callback'] = params['fig'].canvas.mpl_connect(
        #    'motion_notify_event', hover_callback)
        #radio_clicked = partial(_annotation_radio_clicked, radio=fig.radio,
        #                        selector=selector)
        #fig.radio.on_clicked(radio_clicked)

    def _onclick_help(self, event):
        pass

    def _keypress(self, event):
        """Triage keypress events."""
        from matplotlib.pyplot import close, get_current_fig_manager
        from ..preprocessing import ICA
        key = event.key
        if key == self.mne.close_key:
            close(self)
            if self.mne.fig_annotation is not None:
                close(self.mne.fig_annotation)
        elif key in ('down', 'up'):
            if self.mne.butterfly:
                return
            elif self.mne.fig_selection is not None:
                pass  # TODO: change channel group
            else:
                ceiling = len(self.mne.ch_names) - self.mne.n_channels
                direction = -1 if key == 'up' else 1
                ch_start = self.mne.ch_start + direction * self.mne.n_channels
                self.mne.ch_start = np.clip(ch_start, 0, ceiling)
                self._update_data()
                self._draw_traces()
                self._update_vscroll()
            self.canvas.draw()
        elif key in ('right', 'left', 'shift+right', 'shift+left'):
            direction = 1 if key.endswith('right') else -1
            denom = 1 if key.startswith('shift') else 4
            t_max = self.mne.inst.times[-1] - self.mne.duration
            t_start = self.mne.t_start + direction * self.mne.duration / denom
            self.mne.t_start = np.clip(t_start, self.mne.first_time, t_max)
            self._update_data()
            self._draw_traces()
            self._update_hscroll()
            self.canvas.draw()
        elif key in ('=', '+', '-'):
            scaler = 1 / 1.1 if key == '-' else 1.1
            self.mne.scale_factor *= scaler
            self._draw_traces()
            self.canvas.draw()
        elif key in ('pageup', 'pagedown') and self.mne.fig_selection is None:
            n_ch_delta = 1 if key == 'pageup' else -1
            if self.mne.n_channels + n_ch_delta > 0:
                self.mne.n_channels += n_ch_delta
                # TODO: redraw
        elif key in ('home', 'end'):
            dur_delta = 1 if key == 'end' else -1
            if self.mne.duration + dur_delta > 0:
                if self.mne.duration + dur_delta > self.inst.times[-1]:
                    self.mne.duration = self.inst.times[-1]
                else:
                    self.mne.duration += dur_delta
                self.mne.hsel_patch.set_width(self.mne.duration)
                # TODO: redraw
        elif key == '?':
            self._onclick_help(event)
        elif key == 'f11':
            fig_manager = get_current_fig_manager()
            fig_manager.full_screen_toggle()
        elif key == 'a':
            if isinstance(self.mne.inst, ICA):
                return
            if self.mne.fig_annotation is None:
                self._setup_annotation_fig()
            else:
                self.mne.fig_annotation.canvas.close_event()
        elif key == 'b':  # TODO: toggle butterfly mode
            pass
        elif key == 'd':
            self.mne.remove_dc = not self.mne.remove_dc
            self._update_data()
            self._draw_traces()
        elif key == 'p':  # TODO: toggle snap annotations
            pass
        elif key == 's':
            self._toggle_scalebars(event)
        elif key == 'w':  # TODO: toggle noise cov / whitening
            pass
        elif key == 'z':  # zen mode: remove scrollbars and buttons
            self._toggle_scrollbars()

    def _update_vscroll(self):
        self.mne.vsel_patch.set_xy((0, self.mne.ch_start))
        self.mne.vsel_patch.set_height(self.mne.n_channels)

    def _update_hscroll(self):
        self.mne.hsel_patch.set_xy((self.mne.t_start, 0))
        self.mne.hsel_patch.set_width(self.mne.duration)

    def _toggle_scalebars(self, event):
        """Show/hide the scalebars."""
        if self.mne.scalebars_visible:
            self._hide_scalebars()
        else:
            if self.mne.butterfly:
                n_channels = len(self.mne.trace_offsets)
                ch_start = 0
            else:
                n_channels = self.mne.n_channels
                ch_start = self.mne.ch_start
            ch_indices = self.mne.ch_order[ch_start:(ch_start + n_channels)]
            self._show_scalebars(ch_indices)
        # toggle
        self.mne.scalebars_visible = not self.mne.scalebars_visible
        self.canvas.draw()

    def _hide_scalebars(self):
        """Remove channel scale bars."""
        for bar in self.mne.scalebars.values():
            self.mne.ax_main.lines.remove(bar)
        for text in self.mne.scalebar_texts.values():
            self.mne.ax_main.texts.remove(text)
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()

    def _show_scalebars(self, ch_indices):
        """Add channel scale bars."""
        offsets = (self.mne.trace_offsets[self.mne.ch_order]
                   if self.mne.butterfly else self.mne.trace_offsets)
        for ii, ch_ix in enumerate(ch_indices):
            this_name = self.mne.ch_names[ch_ix]
            this_type = self.mne.ch_types[ch_ix]
            if (this_name not in self.mne.info['bads'] and
                    this_name not in self.mne.whitened_ch_names and
                    this_type != 'stim' and
                    this_type in self.mne.scalings and
                    this_type in getattr(self.mne, 'units', {}) and
                    this_type in getattr(self.mne, 'unit_scalings', {}) and
                    this_type not in self.mne.scalebars):
                x = (self.mne.times[0] + self.mne.first_time,) * 2
                y = tuple(np.array([-0.5, 0.5]) + offsets[ii])
                self._draw_one_scalebar(x, y, this_type)

    def _draw_one_scalebar(self, x, y, ch_type):
        """Draw the scalebars."""
        from .utils import _simplify_float
        color = '#AA3377'  # purple
        kwargs = dict(color=color, zorder=5)
        # TODO: I changed trace spacing from 2 to 1; should this 2 be removed?
        inv_norm = (2 * self.mne.scalings[ch_type] *
                    self.mne.unit_scalings[ch_type] /
                    self.mne.scale_factor)
        bar = self.mne.ax_main.plot(x, y, lw=4, **kwargs)[0]
        label = f'{_simplify_float(inv_norm)} {self.mne.units[ch_type]} '
        text = self.mne.ax_main.text(x[1], y[1], label, va='baseline',
                                     ha='right', size='xx-small', **kwargs)
        self.mne.scalebars[ch_type] = bar
        self.mne.scalebar_texts[ch_type] = text

    def _update_trace_offsets(self):
        """Compute viewport height and adjust offsets."""
        n_channels = self.mne.n_channels
        ylim = (n_channels - 0.5, -0.5)  # inverted y axis → new chs at bottom
        offsets = np.arange(n_channels, dtype=float)
        # update ylim, ticks, vertline, and scrollbar patch
        self.mne.ax_main.set_ylim(ylim)
        self.mne.ax_main.set_yticks(offsets)
        self.mne.vsel_patch.set_height(n_channels)
        _x = self.mne.ax_vline._x
        self.mne.ax_vline.set_data(_x, np.array(ylim))
        # store new offsets
        self.mne.trace_offsets = offsets
        self.canvas.draw()

    def _load_data(self, picks, start=None, stop=None):
        """Retrieve the bit of data we need for plotting."""
        if self.mne.instance_type == 'raw':
            return self.mne.inst[picks, start:stop]
        else:
            raise NotImplementedError  # TODO: support epochs, ICA

    def _update_data(self):
        """Update self.mne.data after user interaction."""
        from ..filter import _overlap_add_filter, _filtfilt
        # update time
        start_sec = self.mne.t_start - self.mne.first_time
        stop_sec = start_sec + self.mne.duration
        start, stop = self.mne.inst.time_as_index((start_sec, stop_sec))
        # update picks
        _sl = slice(self.mne.ch_start, self.mne.ch_start + self.mne.n_channels)
        self.mne.picks = self.mne.ch_order[_sl]
        # get the data
        data, times = self._load_data(self.mne.picks, start, stop)
        # apply projectors
        if self.mne.projector is not None:
            data = self.mne.projector[self.mne.picks] @ data
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
                _picks = np.where(np.in1d(self.mne.picks, self.mne.data_picks))
                this_data = data[_picks, _start:_stop]
                if isinstance(self.mne.filter_coefs, np.ndarray):  # FIR
                    this_data = _overlap_add_filter(
                        this_data, self.mne.filter_coefs, copy=False)
                else:  # IIR
                    this_data = _filtfilt(
                        this_data, self.mne.filter_coefs, None, 1, False)
                data[_picks, _start:_stop] = this_data
        # scale the data for display in a 1-vertical-axis-unit slot
        for trace_ix, pick in enumerate(self.mne.picks):
            if self.mne.ch_types[pick] == 'stim':
                norm = max(data[trace_ix])
            elif self.mne.ch_names[pick] in self.mne.whitened_ch_names and \
                    self.mne.ch_names[pick] not in self.mne.info['bads']:
                norm = self.mne.scalings['whitened']
            else:
                norm = self.mne.scalings[self.mne.ch_types[pick]]
            data[trace_ix] /= norm if norm != 0 else 1
        # save
        self.mne.data = data
        self.mne.times = times

    def _draw_traces(self, event_lines=None, event_color=None):
        """Draw (or redraw) the channel data."""
        from matplotlib.patches import Rectangle

        if self.mne.butterfly:
            offsets = self.mne.trace_offsets[self.mne.ch_order]
        else:
            offsets = self.mne.trace_offsets
        labels = self.mne.ax_main.yaxis.get_ticklabels()
        # clear scalebars
        if self.mne.scalebars_visible:
            self._hide_scalebars()
        # TODO: clear event and annotation texts
        #
        # get indices of currently visible channels
        ch_names = self.mne.ch_names[self.mne.picks]
        ch_types = self.mne.ch_types[self.mne.picks]
        # colors
        bads = self.mne.info['bads']
        ch_colors = [self.mne.bad_color if _name in bads else
                     self.mne.color_dict[_type]
                     for _name, _type in zip(ch_names, ch_types)]
        if self.mne.butterfly:
            for label in labels:
                label.set_color('black')  # TODO make compat. w/ MPL dark style
        else:
            for label, color in zip(labels, ch_colors):
                label.set_color(color)
        # decim
        decim = np.ones_like(self.mne.picks)
        data_picks_mask = np.in1d(self.mne.picks, self.mne.data_picks)
        decim[data_picks_mask] = self.mne.decim
        # decim can vary by channel type, so compute different times vectors
        decim_times_dict = {decim_value:
                            self.mne.times[::decim_value] + self.mne.first_time
                            for decim_value in set(decim)}
        # update axis labels
        self.mne.ax_main.set_yticklabels(ch_names, rotation=0)
        # loop over channels
        for ii, ch_ix in enumerate(self.mne.picks):
            this_name = ch_names[ii]
            this_type = ch_types[ii]
            this_line = self.mne.traces[ii]
            this_times = decim_times_dict[decim[ii]]
            # do not update data in-place!
            this_data = self.mne.data[ii] * self.mne.scale_factor
            # clip
            if self.mne.clipping == 'clamp':
                np.clip(this_data, -0.5, 0.5, out=this_data)
            elif self.mne.clipping is not None:
                l, w = this_times[0], this_times[-1] - this_times[0]
                ylim = self.mne.ax_main.get_ylim()
                assert ylim[1] <= ylim[0]  # inverted
                b = offsets[ii] - self.mne.clipping
                h = 2 * self.mne.clipping
                b = max(b, ylim[1])
                h = min(h, ylim[0] - b)
                rect = Rectangle((l, b), w, h,
                                 transform=self.mne.ax_main.transData)
                this_line.set_clip_path(rect)
            # update trace data
            # subtraction yields correct orientation given inverted ylim
            this_line.set_xdata(this_times)
            self.mne.ax_main.set_xlim(this_times[0], this_times[-1])
            this_line.set_ydata(offsets[ii] - this_data[..., ::decim[ii]])
            this_line.set_color(ch_colors[ii])
            # add attributes to traces
            vars(this_line)['ch_name'] = this_name
            vars(this_line)['def_color'] = ch_colors[ii]
            # set z-order
            this_z = 0 if this_name in bads else 1
            if self.mne.butterfly:
                if this_name not in bads:
                    if this_type == 'mag':
                        this_z = 2
                    elif this_type == 'grad':
                        this_z = 3
            this_line.set_zorder(this_z)
        # draw scalebars maybe
        if self.mne.scalebars_visible:
            self._show_scalebars(self.mne.picks)
        # TODO: WIP event lines
        if self.mne.event_times is not None:
            mask = np.logical_and(self.mne.event_times >= self.mne.times[0],
                                  self.mne.event_times <= self.mne.times[-1])
            these_event_times = self.mne.event_times[mask]
            these_event_nums = self.mne.event_nums[mask]
            # plot them
            ylim = self.mne.ax_main.get_ylim()

    def _set_custom_selection(self):
        """Set custom selection by lasso selector."""
        chs = self.mne.fig_selection.lasso.selection
        if len(chs) == 0:
            return
        labels = [label._text for label in self.mne.fig_selection.radio.labels]
        inds = np.in1d(self.inst.ch_names, chs)
        self.mne.selections['Custom'] = np.where(inds)[0]
        # TODO: not tested; replaces _set_radio_button (old compatibility code)
        self.mne.fig_selection.radio.set_active(labels.index('Custom'))


def _figure(toolbar=True, FigureClass=MNEFigure, **kwargs):
    """Instantiate a new figure."""
    from matplotlib import rc_context
    from matplotlib.pyplot import figure
    if toolbar:
        fig = figure(FigureClass=FigureClass, **kwargs)
    else:
        with rc_context(rc=dict(toolbar='none')):
            fig = figure(FigureClass=FigureClass, **kwargs)
    return fig


def browse_figure(inst, **kwargs):
    """Instantiate a new MNE browse-style figure."""
    fig = _figure(inst=inst, toolbar=False, FigureClass=MNEBrowseFigure,
                  **kwargs)
    # initialize zen mode (can't do in __init__ due to get_position() calls)
    fig.canvas.draw()
    fig.mne.fig_size_px = fig._get_size_px()
    fig.mne.zen_w = (fig.mne.ax_vscroll.get_position().xmax -
                     fig.mne.ax_main.get_position().xmax)
    fig.mne.zen_h = (fig.mne.ax_main.get_position().ymin -
                     fig.mne.ax_hscroll.get_position().ymin)
    if not fig.mne.scrollbars_visible:
        fig.mne.scrollbars_visible = True
        fig._toggle_scrollbars()
    # add our custom callbacks
    callbacks = dict(resize_event=fig._resize,
                     key_press_event=fig._keypress)
    callback_ids = dict()
    for event, callback in callbacks.items():
        callback_ids[event] = fig.canvas.mpl_connect(event, callback)
    # store references so they aren't garbage-collected
    fig.mne.callback_ids = callback_ids
    return fig


def dialog_figure(**kwargs):
    """Instantiate a new MNE dialog figure."""
    fig = _figure(*args, toolbar=False, FigureClass=MNEDialogFigure, **kwargs)
    # add a close event callback
    callbacks = dict(key_press_event=fig._keypress)
    callback_ids = dict()
    for event, callback in callbacks.items():
        callback_ids[event] = fig.canvas.mpl_connect(event, callback)
    # store references so they aren't garbage-collected
    fig.mne.callback_ids = callback_ids
    return fig
