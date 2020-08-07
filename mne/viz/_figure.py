# -*- coding: utf-8 -*-
"""Figure classes for MNE-Python's 2D plots."""

# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

import platform
from copy import deepcopy
from functools import partial
from collections import OrderedDict
import numpy as np
from matplotlib.figure import Figure
from .utils import (plt_show, _setup_plot_projector, _events_off,
                    _set_window_title, _get_active_radio_idx,
                    _merge_annotations, DraggableLine, _get_color_list)
from ..utils import set_config
from ..annotations import _sync_onset


class MNEFigParams:
    """Container object for MNE figure parameters."""
    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        vars(self).update(**kwargs)


class MNEFigure(Figure):
    """Wrapper of matplotlib.figure.Figure; adds MNE-Python figure params."""
    def __init__(self, **kwargs):
        # figsize is the only kwarg we pass to matplotlib Figure()
        figsize = kwargs.pop('figsize', None)
        super().__init__(figsize=figsize)
        # add our param object
        self.mne = MNEFigParams(**kwargs)

    def _keypress(self, event):
        """Handle keypress events."""
        from matplotlib.pyplot import close
        if event.key == self.mne.close_key:
            close(self)

    def _buttonpress(self, event):
        """Handle buttonpress events."""
        pass

    def _resize(self, event):
        """Handle window resize events."""
        pass

    def _onpick(self, event):
        """Handle matplotlib artist picking events."""
        pass

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

    def _add_default_callbacks(self, **kwargs):
        """Remove matplotlib default callbacks and add MNE-Python ones."""
        # Remove matplotlib default keypress catchers
        default_cbs = list(
            self.canvas.callbacks.callbacks.get('key_press_event', {}))
        for callback in default_cbs:
            self.canvas.callbacks.disconnect(callback)
        # add our event callbacks
        callbacks = dict(resize_event=self._resize,
                         key_press_event=self._keypress,
                         button_press_event=self._buttonpress,
                         pick_event=self._onpick)
        callbacks.update(kwargs)
        callback_ids = dict()
        for event, callback in callbacks.items():
            callback_ids[event] = self.canvas.mpl_connect(event, callback)
        # store callback references so they aren't garbage-collected
        self.mne._callback_ids = callback_ids


class MNEAnnotationFigure(MNEFigure):
    """Interactive dialog figure for annotations."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _keypress(self, event):
        """Triage keypress events."""
        from matplotlib.pyplot import close
        text = self.label.get_text()
        key = event.key
        if key == self.mne.close_key:
            close(self)
        elif key == 'backspace':
            text = text[:-1]
        elif key == 'enter':
            self._parent_fig._add_annotation_label(event)
            return
        elif len(key) > 1 or key == ';':  # ignore modifier keys
            return
        else:
            text = text + key
        self.label.set_text(text)
        self.canvas.draw()

    def _radiopress(self, event=None):
        """Handle Radiobutton clicks for Annotation label selection."""
        # update which button looks active
        buttons = self.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        idx = labels.index(buttons.value_selected)
        self._set_active_button(idx)
        # update click-drag rectangle color
        color = buttons.circles[idx].get_edgecolor()
        selector = self._parent_fig.mne.ax_main.selector
        selector.rect.set_color(color)
        selector.rectprops.update(dict(facecolor=color))

    def _click_override(self, event):
        """Override MPL radiobutton click detector to use transData."""
        ax = self.radio_ax
        buttons = ax.buttons
        if (buttons.ignore(event) or event.button != 1 or event.inaxes != ax):
            return
        pclicked = ax.transData.inverted().transform((event.x, event.y))
        distances = {}
        for i, (p, t) in enumerate(zip(buttons.circles, buttons.labels)):
            if (t.get_window_extent().contains(event.x, event.y)
                    or np.linalg.norm(pclicked - p.center) < p.radius):
                distances[i] = np.linalg.norm(pclicked - p.center)
        if len(distances) > 0:
            closest = min(distances, key=distances.get)
            buttons.set_active(closest)

    def _set_active_button(self, idx):
        """Set active button in annotation dialog figure."""
        buttons = self.radio_ax.buttons
        with _events_off(buttons):
            buttons.set_active(idx)
        for circle in buttons.circles:
            circle.set_facecolor('w')
        # active circle gets filled in, partially transparent
        color = list(buttons.circles[idx].get_edgecolor())
        color[-1] = 0.5
        buttons.circles[idx].set_facecolor(color)
        self.canvas.draw()


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

        # additional params for browse figures
        self.mne.projector = None
        # self.mne.event_id_rev = None
        # # channel
        # self.mne.group_by = None
        self.mne.whitened_ch_names = list()
        # # annotations
        # self.mne.annotations = None
        self.mne.added_labels = list()
        # self.mne.annotation_segments      # segments
        self.mne.segment_line = None
        # # scalings
        self.mne.scale_factor = 1.
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()
        # # ancillary figures
        self.mne.fig_help = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None

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
        button_help.on_clicked(self._toggle_help_fig)
        # PROJ BUTTON
        if len(self.mne.projs) and not inst.proj:
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
            # PROJ BUTTON: make it a button. onclick handled by _buttonpress()
            button_proj = Button(ax_proj, 'Prj')

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

    def _hover(self, event):
        """Handle motion event when annotating."""
        if not self.mne.snap_annotations:  # don't snap to annotations
            self._remove_annotation_line()
            return
        from matplotlib.patheffects import Stroke, Normal
        if (event.button is not None or
                event.inaxes != self.mne.ax_main or event.xdata is None):
            return
        for coll in self.mne.ax_main.collections:
            if coll.contains(event)[0]:
                path = coll.get_paths()
                assert len(path) == 1
                path = path[0]
                color = coll.get_edgecolors()[0]
                mn = path.vertices[:, 0].min()
                mx = path.vertices[:, 0].max()
                # left/right line
                x = mn if abs(event.xdata - mn) < abs(event.xdata - mx) else mx
                mask = path.vertices[:, 0] == x
                ylim = self.mne.ax_main.get_ylim()

                def drag_callback(x0):
                    path.vertices[mask, 0] = x0

                if self.mne.segment_line is None:
                    line = self.mne.ax_main.plot([x, x], ylim, color=color,
                                                 linewidth=2., picker=True)[0]
                    line.set_pickradius(5.)
                    dl = DraggableLine(line, self._modify_annotation,
                                       drag_callback)
                    self.mne.segment_line = dl
                else:
                    self.mne.segment_line.set_x(x)
                    self.mne.segment_line.drag_callback = drag_callback
                line = self.mne.segment_line.line
                patheff = [Stroke(linewidth=4, foreground=color, alpha=0.5),
                           Normal()]
                line.set_path_effects(patheff if line.contains(event)[0] else
                                      patheff[1:])
                self.mne.ax_main.selector.active = False
                self.canvas.draw()
                return
        self._remove_annotation_line()

    def _keypress(self, event):
        """Triage keypress events."""
        from matplotlib.pyplot import close, get_current_fig_manager
        from ..preprocessing import ICA
        key = event.key
        if key == self.mne.close_key:
            close(self)
            if self.mne.fig_proj is not None:
                close(self.mne.fig_proj)
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
                self._update_vscroll()
                self._update_data()
                self._draw_traces()
            self.canvas.draw()
        elif key in ('right', 'left', 'shift+right', 'shift+left'):
            direction = 1 if key.endswith('right') else -1
            denom = 1 if key.startswith('shift') else 4
            t_max = self.mne.inst.times[-1] - self.mne.duration
            t_start = self.mne.t_start + direction * self.mne.duration / denom
            self.mne.t_start = np.clip(t_start, self.mne.first_time, t_max)
            # TODO: only update if changed
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
            n_ch = self.mne.n_channels + n_ch_delta
            self.mne.n_channels = np.clip(n_ch, 1, len(self.mne.ch_names))
            # TODO: only update if changed
            self._update_trace_offsets()
            self._update_data()
            self._draw_traces()
            self.canvas.draw()
        elif key in ('home', 'end'):  # change duration
            dur_delta = 1 if key == 'end' else -1
            old_dur = self.mne.duration
            new_dur = self.mne.duration + dur_delta
            min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]
            self.mne.duration = np.clip(new_dur, min_dur,
                                        self.mne.inst.times[-1])
            if self.mne.duration != old_dur:
                self._update_data()
                self._draw_traces()
                self._update_hscroll()
                self.canvas.draw()
        elif key == '?':  # help
            self._toggle_help_fig(event)
        elif key == 'f11':  # full screen
            fig_manager = get_current_fig_manager()
            fig_manager.full_screen_toggle()
        elif key == 'a':  # annotation mode
            if isinstance(self.mne.inst, ICA):
                return
            self._toggle_annotation_fig()
        elif key == 'b':  # toggle butterfly mode
            self.mne.butterfly = not self.mne.butterfly
            self._update_data()
            self._draw_traces()
            self.canvas.draw()
        elif key == 'd':  # DC shift
            self.mne.remove_dc = not self.mne.remove_dc
            self._update_data()
            self._draw_traces()
            self.canvas.draw()
        elif key == 'p':  # TODO: toggle snap annotations
            pass
        elif key == 's':  # scalebars
            self._toggle_scalebars(event)
        elif key == 'w':  # TODO: toggle noise cov / whitening
            pass
        elif key == 'z':  # zen mode: hide scrollbars and buttons
            self._toggle_scrollbars()

    def _buttonpress(self, event):
        """Triage mouse clicks."""
        # ignore middle clicks or scroll wheel events
        if event.button not in (1, 3):
            return
        elif event.button == 1:  # left-click (primary)
            if event.inaxes is None:  # clicked on channel name
                if self.mne.n_channels > 100:  # too dense to identify
                    return
                ylim = self.mne.ax_main.get_ylim()
                xy = (event.x, event.y)
                x, y = self.mne.ax_main.transData.inverted().transform(xy)
                if x > self.mne.t_start or y < 0 or y > ylim[0]:
                    return
                self._label_clicked(x, y)
            elif event.inaxes == self.mne.ax_main:
                if not self.mne.butterfly:
                    for line in self.mne.traces:
                        if line.contains(event)[0]:
                            idx = self.mne.traces.index(line)
                            self._toggle_bad_channel(idx)
                            return
                # click was not on a data trace, or in butterfly mode
                self._show_vline(event.xdata)
            elif event.inaxes == self.mne.ax_vscroll:
                if self.mne.fig_selection is not None:
                    # _handle_change_selection(event, params)
                    pass  # TODO FIXME
                else:
                    if self._check_update_vscroll_clicked(event):
                        self._update_data()
                        self._draw_traces()
                        self.canvas.draw()
            elif event.inaxes == self.mne.ax_hscroll:
                if self._check_update_hscroll_clicked(event):
                    self._update_data()
                    self._draw_traces()
                    self.canvas.draw()
            elif event.inaxes == self.mne.ax_proj:
                self._toggle_proj_fig(event)
        else:  # right-click (secondary)
            ax = self.mne.ax_main
            raw = self.mne.inst
            if self.mne.fig_annotation is not None:
                if any(c.contains(event)[0] for c in ax.collections):
                    xdata = event.xdata - self.mne.first_time
                    onset = _sync_onset(raw, raw.annotations.onset)
                    ends = onset + raw.annotations.duration
                    ann_idx = np.where((xdata > onset) & (xdata < ends))[0]
                    raw.annotations.delete(ann_idx)  # only first one deleted
                self._remove_annotation_line()
                self._draw_annotations()
                self.canvas.draw()
            elif event.inaxes == ax:  # hide green line
                self._hide_vline()

    def _label_clicked(self, x, y):
        """Handle left-click on channel names."""
        if self.mne.butterfly:
            return
        idx = np.searchsorted(self.mne.trace_offsets + 0.5, y)
        self._toggle_bad_channel(idx)

    def _toggle_bad_channel(self, idx):
        """Mark/unmark bad channels. `idx` is index of *visible* channels."""
        bads = self.mne.info['bads']
        pick = self.mne.picks[idx]
        line = self.mne.traces[idx]
        vscroll_idx = self.mne.ch_order.tolist().index(pick)
        ch_name = self.mne.ch_names[pick]
        if not len(ch_name):
            return
        if self.mne.fig_selection is not None:
            # ch_idx = _find_channel_idx(text, params)
            # _handle_topomap_bads(text, params)
            pass  # TODO FIXME
        if ch_name in bads:
            while ch_name in bads:  # to make sure duplicates are removed
                bads.remove(ch_name)
            color = vars(line)['def_color']
        else:
            bads.append(ch_name)
            color = self.mne.bad_color
        self.mne.ax_vscroll.patches[vscroll_idx].set_color(color)
        self.mne.info['bads'] = bads
        # redraw
        self._update_projector()
        self._update_data()
        self._draw_traces()
        self.canvas.draw()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # HELP DIALOG
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_help_fig(self):
        """Create help dialog window."""
        text = {key: val for key, val in self._help_text().items()
                if val is not None}
        keys = ''
        vals = ''
        for key, val in text.items():
            newsection = '\n' if key.startswith('_') else ''
            key = key[1:] if key.startswith('_') else key
            newlines = '\n' * len(val.split('\n'))  # handle multiline values
            keys += f'{newsection}{key}      {newlines}'
            vals += f'{newsection}{val}\n'

        # calc figure size
        n_lines = len(keys.split('\n'))
        longest_key = max(len(k) for k in text.keys())
        longest_val = max(max(len(w) for w in v.split('\n')) if '\n' in v else
                          len(v) for v in text.values())
        width = (longest_key + longest_val) / 12
        height = (n_lines) / 5
        # create figure and axes
        fig = dialog_figure(figsize=(width, height))
        _set_window_title(fig, 'Help')
        ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
        ax.set_axis_off()
        kwargs = dict(va='top', linespacing=1.5, usetex=False)
        ax.text(0.42, 1, keys, ma='right', ha='right', **kwargs)
        ax.text(0.42, 1, vals, ma='left', ha='left', **kwargs)
        # save
        self.mne.fig_help = fig

    def _clear_help_fig(self, event=None):
        """Close the help dialog window (via keypress or window [x])."""
        self.mne.fig_help = None

    def _toggle_help_fig(self, event):
        """Show/hide the annotation dialog window."""
        if self.mne.fig_help is None:
            self._create_help_fig()
        else:
            self.mne.fig_help.canvas.close_event()
            self._clear_help_fig()

    def _help_text(self):
        """Generate help dialog text, omitting `None`-valued entries."""
        is_mac = platform.system() == 'Darwin'
        inst = self.mne.instance_type
        ch_cmp = 'component' if inst == 'ica' else 'channel'
        ch_epo = 'epoch' if inst == 'epochs' else 'channel'
        ica_bad = 'Mark/unmark component for exclusion'
        # adapt keys to different platforms
        dur_keys = ('⌘ + ←', '⌘ + →') if is_mac else ('Home', 'End')
        ch_keys = ('⌘ + ↑', '⌘ + ↓') if is_mac else ('Page up', 'Page down')
        # adapt descriptions to different instance types
        dur_vals = ([f'Show {n} epochs' for n in ('fewer', 'more')]
                    if inst ==  'epochs' else
                    [f'Show {d} time window' for d in ('shorter', 'longer')])
        ch_vals = [f'{nd}crease visible {ch_cmp} count' for nd in ('In', 'De')]
        lclick_data = ica_bad if inst == 'ica' else f'Mark/unmark bad {ch_epo}'
        lclick_name = (ica_bad if inst == 'ica' else 'Mark/unmark bad channel')
        rclick_name = dict(ica='Show diagnostics for component',
                           epoch='Show imageplot for channel',
                           raw='Show channel location')[inst]
        ldrag = ('Show spectrum plot for selected time span;\nor (in '
                 'annotation mode) add annotation') if inst == 'raw' else None

        help_text = OrderedDict([
            # navigation
            ('_NAVIGATION', ' '),
            ('→', 'Scroll ¼ window right (scroll full window with Shift + →)'),
            ('←', 'Scroll ¼ window left (scroll full window with Shift + ←)'),
            (dur_keys[0], dur_vals[0]),
            (dur_keys[1], dur_vals[1]),
            ('↑', f'Scroll up ({ch_cmp}s)'),
            ('↓', f'Scroll down ({ch_cmp}s)'),
            (ch_keys[0], ch_vals[0]),
            (ch_keys[1], ch_vals[1]),
            # signal
            ('_SIGNAL TRANSFORMATIONS', ' '),
            ('+ or =', 'Increase signal scaling'),
            ('-', 'Decrease signal scaling'),
            ('b', 'Toggle butterfly mode' if inst != 'ica' else None),
            ('d', 'Toggle DC removal' if inst == 'raw' else None),
            ('w', 'Toggle signal whitening'),  # TODO only if noise_cov given?
            ('a', 'Toggle annotation mode' if inst == 'raw' else None),
            ('p', 'Toggle annotation snapping' if inst == 'raw' else None),
            ('h', 'Show peak-to-peak histogram' if inst == 'epochs' else None),
            # UI
            ('_USER INTERFACE', ' '),
            ('s', 'Toggle scalebars' if inst != 'ica' else None),
            ('z', 'Toggle scrollbars'),
            ('F11', 'Toggle fullscreen'),
            ('?', 'Open this help window'),
            ('esc', 'Close focused figure or dialog window'),
            # mouse
            ('_MOUSE INTERACTION', ' '),
            (f'Left-click {ch_cmp} name', lclick_name),
            (f'Left-click {ch_cmp} data', lclick_data),
            ('Left-click-and-drag on plot', ldrag),
            ('Left-click on plot background', 'Place vertical guide'),
            ('Right-click on plot background', 'Clear vertical guide'),
            ('Right-click on channel name', rclick_name)
        ])
        return help_text

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ANNOTATIONS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_annotation_fig(self):
        """Create the annotation dialog window."""
        from matplotlib.widgets import Button, SpanSelector
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # make figure
        labels = np.array(sorted(set(self.mne.inst.annotations.description)))
        width, var_height, fixed_height, pad = \
            self._compute_annotation_figsize(len(labels))
        fig = dialog_figure(figsize=(width, var_height + fixed_height),
                            FigureClass=MNEAnnotationFigure)
        fig._parent_fig = self
        _set_window_title(fig, 'Annotations')
        self.mne.fig_annotation = fig
        # make main axes
        left = fig._inch_to_rel(pad)
        bottom = fig._inch_to_rel(pad, horiz=False)
        width = 1 - 2 * left
        height = 1 - 2 * bottom
        fig.radio_ax = fig.add_axes((left, bottom, width, height),
                                    frameon=False, aspect='equal')
        div = make_axes_locatable(fig.radio_ax)
        self._update_annotation_fig()  # populate w/ radio buttons & labels
        # append instructions at top
        instructions_ax = div.append_axes(position='top', size=Fixed(1),
                                          pad=Fixed(5 * pad))
        instructions = '\n'.join(
            [r'$\mathbf{Left‐click~&~drag~on~plot:}$ create/modify annotation',
             r'$\mathbf{Right‐click~on~plot~annotation:}$ delete annotation',
             r'$\mathbf{Type~in~annotation~window:}$ modify new label name',
             r'$\mathbf{Enter~(or~click~button):}$ add new label to list',
             r'$\mathbf{Esc:}$ exit annotation mode & close window'])
        # in case user has usetex=True set in rcParams, pass usetex=False here
        instructions_ax.text(0, 1, instructions, va='top', ha='left',
                             usetex=False)
        instructions_ax.set_axis_off()
        # append text entry axes at bottom
        text_entry_ax = div.append_axes(position='bottom', size=Fixed(3 * pad),
                                        pad=Fixed(pad))
        text_entry_ax.text(0.4, 0.5, 'New label:', va='center', ha='right',
                           weight='bold')
        fig.label = text_entry_ax.text(0.5, 0.5, 'BAD_', va='center',
                                       ha='left')
        text_entry_ax.set_axis_off()
        # append button at bottom
        button_ax = div.append_axes(position='bottom', size=Fixed(3 * pad),
                                    pad=Fixed(pad))
        fig.button = Button(button_ax, 'Add new label')
        fig.button.on_clicked(self._add_annotation_label)
        plt_show(fig=fig)
        # setup interactivity in plot window
        col = ('#ff0000' if len(fig.radio_ax.buttons.circles) < 1 else
               fig.radio_ax.buttons.circles[0].get_edgecolor())
        selector = SpanSelector(self.mne.ax_main, self._select_annotation_span,
                                'horizontal', minspan=0.1,  # useblit=True, ?
                                rectprops=dict(alpha=0.5, facecolor=col))
        if len(labels) == 0:
            selector.active = False
        self.mne.ax_main.selector = selector
        # add event listeners
        fig.canvas.mpl_connect('close_event', self._clear_annotation_fig)
        self.mne._callback_ids['motion_notify_event'] = \
            self.canvas.mpl_connect('motion_notify_event', self._hover)

    def _update_annotation_fig(self):
        """Draw or redraw the radio buttons and annotation labels."""
        from matplotlib.widgets import RadioButtons
        # define shorthand variables
        fig = self.mne.fig_annotation
        ax = fig.radio_ax
        # get all the labels
        labels = list(set(self.mne.inst.annotations.description))
        labels = np.union1d(labels, self.mne.added_labels)
        # compute new figsize
        width, var_height, fixed_height, pad = \
            self._compute_annotation_figsize(len(labels))
        fig.set_size_inches(width, var_height + fixed_height, forward=True)
        # populate center axes with labels & radio buttons
        ax.clear()
        title = 'Existing labels:' if len(labels) else 'No existing labels'
        ax.set_title(title, size=None)
        ax.buttons = RadioButtons(ax, labels)
        # adjust xlim to keep equal aspect & full width (keep circles round)
        aspect = (width - 2 * pad) / var_height
        ax.set_xlim((0, aspect))
        # style the buttons & adjust spacing
        radius = 0.15
        circles = ax.buttons.circles
        for circle, label in zip(circles, ax.buttons.labels):
            circle.set_transform(ax.transData)
            center = ax.transData.inverted().transform(
                ax.transAxes.transform((0.1, 0)))
            circle.set_center((center[0], circle.center[1]))
            circle.set_edgecolor(self.mne.segment_colors[label.get_text()])
            circle.set_linewidth(4)
            circle.set_radius(radius / (len(labels)))
        # add event listeners
        ax.buttons.disconnect_events()  # clear MPL default listeners
        ax.buttons.on_clicked(fig._radiopress)
        ax.buttons.connect_event('button_press_event', fig._click_override)
        # activate the selector
        if len(labels):
            self.mne.ax_main.selector.active = True

    def _clear_annotation_fig(self, event=None):
        """Close the annotation dialog window (via keypress or window [x])."""
        self.mne.fig_annotation = None
        # disconnect hover callback
        callback_id = self.mne._callback_ids['motion_notify_event']
        self.canvas.callbacks.disconnect(callback_id)

    def _toggle_annotation_fig(self):
        """Show/hide the annotation dialog window."""
        if self.mne.fig_annotation is None:
            self._create_annotation_fig()
        else:
            self.mne.fig_annotation.canvas.close_event()
            self._clear_annotation_fig()

    def _compute_annotation_figsize(self, n_labels):
        """Adapt size of Annotation UI to accommodate the number of buttons.

        self._create_annotation_fig() implements the following:

        Fixed part of height:
        0.1  top margin
        1.0  instructions
        0.5  padding below instructions
        ---  (variable-height axis for label list)
        0.1  padding above text entry
        0.3  text entry
        0.1  padding above button
        0.3  button
        0.1  bottom margin
        ------------------------------------------
        2.5  total fixed height
        """
        pad = 0.1
        width = 4.5
        var_height = max(pad, 0.7 * n_labels)
        fixed_height = 2.5
        return (width, var_height, fixed_height, pad)

    def _add_annotation_label(self, event):
        """Add new annotation description."""
        text = self.mne.fig_annotation.label.get_text()
        self.mne.added_labels.append(text)
        self._setup_annotation_colors()
        self._update_annotation_fig()
        # automatically activate new label's radio button
        idx = [label.get_text() for label in
               self.mne.fig_annotation.radio_ax.buttons.labels].index(text)
        self.mne.fig_annotation._set_active_button(idx)
        # simulate a click on the radiobutton → update the span selector color
        self.mne.fig_annotation._radiopress()
        # reset the text entry box's text
        self.mne.fig_annotation.label.set_text('BAD_')

    def _setup_annotation_colors(self):
        """Set up colors for annotations."""
        from itertools import cycle

        # TODO disable for epochs/ica instance types
        raw = self.mne.inst
        segment_colors = getattr(self.mne, 'segment_colors', dict())
        # sort the segments by start time
        ann_order = raw.annotations.onset.argsort(axis=0)
        descriptions = raw.annotations.description[ann_order]
        color_keys = np.union1d(descriptions, self.mne.added_labels)
        color_cycle = cycle(_get_color_list(annotations=True))  # no red
        for key, color in segment_colors.items():
            if color != '#ff0000' and key in color_keys:
                next(color_cycle)
        for idx, key in enumerate(color_keys):
            if key in segment_colors:
                continue
            elif key.lower().startswith('bad') or \
                    key.lower().startswith('edge'):
                segment_colors[key] = '#ff0000'
            else:
                segment_colors[key] = next(color_cycle)
        self.mne.segment_colors = segment_colors

    def _select_annotation_span(self, vmin, vmax):
        """Handle annotation span selector."""
        onset = _sync_onset(self.mne.inst, vmin, True) - self.mne.first_time
        duration = vmax - vmin
        radio = self.mne.fig_annotation.radio_ax.buttons
        active_idx = _get_active_radio_idx(radio)
        description = radio.labels[active_idx].get_text()
        _merge_annotations(onset, onset + duration, description,
                           self.mne.inst.annotations)
        self._draw_annotations()

    def _remove_annotation_line(self):
        """Remove annotation line from the plot."""
        if self.mne.segment_line is not None:
            self.mne.segment_line.remove()
            self.mne.segment_line = None
            self.mne.ax_main.selector.active = True

    def _modify_annotation(self, old_x, new_x):
        """Modify annotation."""
        segment = np.array(np.where(self.mne.segments == old_x))
        if segment.shape[1] == 0:
            return
        raw = self.mne.inst
        annotations = raw.annotations
        first_time = self.mne.first_time
        idx = [segment[0][0], segment[1][0]]
        onset = _sync_onset(raw, self.mne.segments[idx[0]][0], True)
        ann_idx = np.where(annotations.onset == onset - first_time)[0]
        if idx[1] == 0:  # start of annotation
            onset = _sync_onset(raw, new_x, True) - first_time
            duration = annotations.duration[ann_idx] + old_x - new_x
        else:  # end of annotation
            onset = annotations.onset[ann_idx]
            duration = _sync_onset(raw, new_x, True) - onset - first_time

        if duration < 0:
            onset += duration
            duration *= -1.

        _merge_annotations(onset, onset + duration,
                           annotations.description[ann_idx],
                           annotations, ann_idx)
        self._draw_annotations()
        self._remove_annotation_line()
        self.canvas.draw()

    def _draw_annotations(self):
        """Draw (or redraw) the annotation spans."""
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # PROJECTORS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_proj_fig(self):
        """Create the projectors dialog window."""
        from matplotlib.widgets import Button, CheckButtons

        projs = self.mne.projs
        labels = [p['desc'] for p in projs]
        for ix, active in enumerate(self.mne.projs_active):
            if active:
                labels[ix] += ' (already applied)'
        # make figure
        width = max([4, max([len(label) for label in labels]) / 6 + 0.5])
        height = (len(projs) + 1) / 6 + 1.5
        fig = dialog_figure(figsize=(width, height))
        _set_window_title(fig, 'SSP projection vectors')
        # make axes
        offset = (1 / 6 / height)
        position = (0, offset, 1, 0.8 - offset)
        ax = fig.add_axes(position, frameon=False)
        # make title
        first_line = ('Projectors already applied to the data are dimmed.\n'
                      if any(self.mne.projs_active) else '')
        second_line = 'Projectors marked with "X" are active on the plot.'
        ax.set_title(f'{first_line}{second_line}')
        # draw checkboxes
        checkboxes = CheckButtons(ax, labels=labels, actives=self.mne.projs_on)
        # gray-out already applied projectors
        for label, rect, lines in zip(checkboxes.labels,
                                      checkboxes.rectangles,
                                      checkboxes.lines):
            if label.get_text().endswith('(already applied)'):
                label.set_color('0.5')
                rect.set_edgecolor('0.7')
                [x.set_color('0.7') for x in lines]
            rect.set_linewidth(1)
        # add "toggle all" button
        ax_all = fig.add_axes((0.25, 0.01, 0.5, offset), frameon=True)
        self.mne.proj_all = Button(ax_all, 'Toggle all')
        # add event listeners
        fig.canvas.mpl_connect('close_event', self._clear_proj_fig)
        checkboxes.on_clicked(self._toggle_proj_checkbox)
        self.mne.proj_all.on_clicked(partial(self._toggle_proj_checkbox,
                                             toggle_all=True))
        # save params
        self.mne.fig_proj = fig
        self.mne._proj_checkboxes = checkboxes
        # show figure (this should work for non-test cases)
        try:
            self.mne.fig_proj.canvas.draw()
            plt_show(fig=self.mne.fig_proj, warn=False)
        except Exception:
            pass

    def _clear_proj_fig(self, event=None):
        """Close the projectors dialog window (via keypress or window [x])."""
        self.mne.fig_proj = None

    def _toggle_proj_fig(self, event):
        """Show/hide the projectors dialog window."""
        if self.mne.fig_proj is None:
            self._create_proj_fig()
        else:
            self.mne.fig_proj.canvas.close_event()
            self._clear_proj_fig()

    def _toggle_proj_checkbox(self, event, toggle_all=False):
        """Perform operations when proj boxes clicked."""
        on = self.mne.projs_on
        applied = self.mne.projs_active
        new_state = (np.full_like(on, not all(on)) if toggle_all else
                     np.array(self.mne._proj_checkboxes.get_status()))
        # update Xs when toggling all
        if toggle_all:
            with _events_off(self.mne._proj_checkboxes):
                for ix in np.where(on != new_state)[0]:
                    self.mne._proj_checkboxes.set_active(ix)
        # don't allow disabling already-applied projs
        with _events_off(self.mne._proj_checkboxes):
            for ix in np.where(applied)[0]:
                if not new_state[ix]:
                    self.mne._proj_checkboxes.set_active(ix)
            new_state[applied] = True
        # update the data if necessary
        if not np.array_equal(on, new_state):
            self.mne.projs_on = new_state
            self._update_projector()
            self._update_data()
            self._draw_traces()
            self.canvas.draw()

    def _update_projector(self):
        """Update the data after projectors (or bads) have changed."""
        inds = np.where(self.mne.projs_on)[0]  # doesn't include "active" projs
        # copy projs from full list (self.mne.projs) to info object
        self.mne.info['projs'] = [deepcopy(self.mne.projs[ix]) for ix in inds]
        # compute the projection operator
        proj, wh_chs = _setup_plot_projector(self.mne.info,
                                             self.mne.whitened_ch_names,
                                             True, self.mne.use_noise_cov)
        self.mne.whitened_ch_names = wh_chs
        self.mne.projector = proj

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SCROLLBARS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

    def _update_vscroll(self):
        """Update the vertical scrollbar (channel) selection indicator."""
        self.mne.vsel_patch.set_xy((0, self.mne.ch_start))
        self.mne.vsel_patch.set_height(self.mne.n_channels)
        self._update_yaxis_labels()

    def _update_hscroll(self):
        """Update the horizontal scrollbar (time) selection indicator."""
        self.mne.hsel_patch.set_xy((self.mne.t_start, 0))
        self.mne.hsel_patch.set_width(self.mne.duration)

    def _check_update_hscroll_clicked(self, event):
        """Handle clicks on horizontal scrollbar."""
        time = event.xdata - self.mne.duration / 2
        max_time = (self.mne.n_times / self.mne.info['sfreq'] +
                    self.mne.first_time - self.mne.duration)
        time = np.clip(time, self.mne.first_time, max_time)
        if self.mne.t_start != time:
            self.mne.t_start = time
            self._update_hscroll()
            return True
        return False

    def _check_update_vscroll_clicked(self, event):
        """Update vscroll patch on click, return True if location changed."""
        new_ch_start = max(0, int(event.ydata - self.mne.n_channels / 2))
        if self.mne.ch_start != new_ch_start:
            self.mne.ch_start = new_ch_start
            self._update_vscroll()
            return True
        return False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SCALEBARS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

    def _hide_scalebars(self):
        """Remove channel scale bars."""
        for bar in self.mne.scalebars.values():
            self.mne.ax_main.lines.remove(bar)
        for text in self.mne.scalebar_texts.values():
            self.mne.ax_main.texts.remove(text)
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()

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

    def _draw_one_scalebar(self, x, y, ch_type):
        """Draw a scalebar."""
        from .utils import _simplify_float
        color = '#AA3377'  # purple
        kwargs = dict(color=color, zorder=5)
        inv_norm = (2 * self.mne.scalings[ch_type] *
                    self.mne.unit_scalings[ch_type] /
                    self.mne.scale_factor)
        bar = self.mne.ax_main.plot(x, y, lw=4, **kwargs)[0]
        label = f'{_simplify_float(inv_norm)} {self.mne.units[ch_type]} '
        text = self.mne.ax_main.text(x[1], y[1], label, va='baseline',
                                     ha='right', size='xx-small', **kwargs)
        self.mne.scalebars[ch_type] = bar
        self.mne.scalebar_texts[ch_type] = text

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA TRACES
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_picks(self):
        _sl = slice(self.mne.ch_start, self.mne.ch_start + self.mne.n_channels)
        self.mne.picks = self.mne.ch_order[_sl]

    def _load_data(self, start=None, stop=None):
        """Retrieve the bit of data we need for plotting."""
        if self.mne.instance_type == 'raw':
            return self.mne.inst[:, start:stop]
        else:
            raise NotImplementedError  # TODO: support epochs, ICA

    def _update_data(self):
        """Update self.mne.data after user interaction."""
        from ..filter import _overlap_add_filter, _filtfilt
        # update time
        start_sec = self.mne.t_start - self.mne.first_time
        stop_sec = start_sec + self.mne.duration
        start, stop = self.mne.inst.time_as_index((start_sec, stop_sec))
        # get the data
        data, times = self._load_data(start, stop)
        # apply projectors
        if self.mne.projector is not None:
            data = self.mne.projector @ data
        # get only the channels we're displaying
        self._update_picks()
        data = data[self.mne.picks]
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
                _picks = np.where(np.in1d(self.mne.picks, self.mne.picks_data))
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
            data[trace_ix] /= 2 * (norm if norm != 0 else 1)
        # save
        self.mne.data = data
        self.mne.times = times

    def _update_trace_offsets(self):
        """Compute viewport height and adjust offsets."""
        # update picks, ylim, and offsets
        n_channels = self.mne.n_channels
        self._update_picks()
        ylim = (n_channels - 0.5, -0.5)  # inverted y axis → new chs at bottom
        offsets = np.arange(n_channels, dtype=float)
        # update ylim, ticks, vertline, and scrollbar patch
        self.mne.ax_main.set_ylim(ylim)
        self.mne.ax_main.set_yticks(offsets)
        self.mne.vsel_patch.set_height(n_channels)
        _x = self.mne.vline._x
        self.mne.vline.set_data(_x, np.array(ylim))
        # store new offsets, update axis labels
        self.mne.trace_offsets = offsets
        self._update_yaxis_labels()

    def _draw_traces(self, event_lines=None, event_color=None):
        """Draw (or redraw) the channel data."""
        from matplotlib.patches import Rectangle

        if self.mne.butterfly:
            offsets = self.mne.trace_offsets[self.mne.ch_order]
        else:
            offsets = self.mne.trace_offsets
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
        labels = self.mne.ax_main.yaxis.get_ticklabels()
        def_colors = [self.mne.color_dict[_type] for _type in ch_types]
        ch_colors = [self.mne.bad_color if _name in bads else def_color
                     for _name, def_color in zip(ch_names, def_colors)]
        if self.mne.butterfly:
            for label in labels:
                label.set_color('black')  # TODO make compat. w/ MPL dark style
        else:
            for label, color in zip(labels, ch_colors):
                label.set_color(color)
        # decim
        decim = np.ones_like(self.mne.picks)
        data_picks_mask = np.in1d(self.mne.picks, self.mne.picks_data)
        decim[data_picks_mask] = self.mne.decim
        # decim can vary by channel type, so compute different times vectors
        decim_times_dict = {decim_value:
                            self.mne.times[::decim_value] + self.mne.first_time
                            for decim_value in set(decim)}
        # add more traces if needed
        if len(self.mne.picks) > len(self.mne.traces):
            n_new_chs = len(self.mne.picks) - len(self.mne.traces)
            new_traces = self.mne.ax_main.plot(np.full((1, n_new_chs), np.nan),
                                               antialiased=True, linewidth=0.5)
            self.mne.traces.extend(new_traces)
        # loop over channels
        for ii, this_line in enumerate(self.mne.traces):
            # remove extra traces if needed
            if ii >= len(self.mne.picks):
                self.mne.ax_main.lines.remove(this_line)
                _ = self.mne.traces.pop(ii)
                continue
            this_name = ch_names[ii]
            this_type = ch_types[ii]
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
            vars(this_line)['def_color'] = def_colors[ii]
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # MISCELLANY
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_yaxis_labels(self):
        self.mne.ax_main.set_yticklabels(self.mne.ch_names[self.mne.picks],
                                         rotation=0, picker=True)

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

    def _show_vline(self, xdata):
        """Show the vertical line."""
        self.mne.vline.set_xdata(xdata)
        self.mne.vline_hscroll.set_xdata(xdata)
        self.mne.vline.set_visible(True)
        self.mne.vline_hscroll.set_visible(True)
        self.mne.vline_text.set_text(f'{xdata:0.2f}  ')
        self.canvas.draw()

    def _hide_vline(self, xdata=None):
        """Hide the vertical line."""
        self.mne.vline.set_visible(False)
        self.mne.vline_hscroll.set_visible(False)
        self.mne.vline_text.set_text('')
        self.canvas.draw()


def _figure(toolbar=True, FigureClass=MNEFigure, **kwargs):
    """Instantiate a new figure."""
    from matplotlib import rc_context
    from matplotlib.pyplot import figure
    rc = dict() if toolbar else dict(toolbar='none')
    with rc_context(rc=rc):
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
    # if scrollbars are supposed to start hidden, set to True and then toggle
    if not fig.mne.scrollbars_visible:
        fig.mne.scrollbars_visible = True
        fig._toggle_scrollbars()
    # add event callbacks
    fig._add_default_callbacks()
    return fig


def dialog_figure(**kwargs):
    """Instantiate a new MNE dialog figure (with event listeners)."""
    fig = _figure(toolbar=False, **kwargs)
    # add our event callbacks
    fig._add_default_callbacks()
    return fig
