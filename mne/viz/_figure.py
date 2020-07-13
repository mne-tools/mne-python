# -*- coding: utf-8 -*-
"""Figure classes for MNE-Python's 2D plots."""

# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

from matplotlib.figure import Figure
from ..utils import set_config


class MNEFigParams:
    """Container for MNE figure parameters."""
    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        for key, val in kwargs.items():
            setattr(self, key, val)


class MNEFigure(Figure):
    """Wrapper of matplotlib.figure.Figure; adds MNE-Python figure params."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # add our param object
        self.mne = MNEFigParams(fig_size_px=self._get_size_px())

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


class MNEBrowseFigure(MNEFigure):
    """Interactive figure with scrollbars, for data browsing."""
    def __init__(self, inst, *args, xlabel='Time (s)', show_scrollbars=True,
                 **kwargs):
        from matplotlib.widgets import Button
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mne.viz.utils import _get_figsize_from_config

        # figsize is the first arg of Figure()
        figsize = args.pop(0) if len(args) else kwargs.pop('figsize', None)
        if figsize is None:
            figsize = _get_figsize_from_config()
        kwargs['figsize'] = figsize

        # init Figure
        super().__init__(*args, **kwargs)

        # additional params for browse figures (comments indicate name changes)
        self.mne.inst = inst                # raw
        # self.mne.info = None
        # self.mne.proj = None
        # self.mne.noise_cov = None
        # self.mne.event_id_rev = None
        # # channel
        self.mne.n_channels = None
        self.mne.ch_types = None            # types
        # self.mne.group_by = None
        # self.mne.picks = None             # data_picks

        # # time
        # self.mne.n_times = None
        # self.mne.first_time = None
        # self.mne.event_times = None
        # self.mne.event_nums = None
        self.mne.duration = None
        # self.mne.decim = None
        self.mne.hsel_patch = None

        # # annotations
        # self.mne.annotations = None
        # self.mne.snap_annotations = None
        # self.mne.added_label = None

        # # traces
        # self.mne.trace_indices = None     # inds
        # self.mne.orig_indices = None      # orig_inds
        # self.mne.clipping = None
        # self.mne.butterfly = None

        # # filters
        # self.mne.remove_dc = None
        # self.mne.filter_coefs_ba = None   # ba
        # self.mne.filter_bounds = None     # filt_bounds

        # # scalings
        # self.mne.units = None
        # self.mne.scalings = None
        # self.mne.unit_scalings = None
        self.mne.scale_factor = 1.

        # # ancillary figures
        # self.mne.fig_proj = None
        self.mne.fig_help = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None

        # # UI state variables
        # self.mne.ch_start = None
        # self.mne.t_start = None
        # self.mne.show_scalebars = None
        self.mne.show_scrollbars = show_scrollbars

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
        #ax_vscroll.set_axis_off()
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
            button_proj.on_clicked(self._toggle_proj_dialog)
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
        new_width, new_height = self._get_size_px()
        self._update_margins(new_width, new_height)
        self.mne.fig_size_px = (new_width, new_height)

    def _update_margins(self, new_width, new_height):
        """Update figure margins to maintain fixed size in inches/pixels."""
        old_width, old_height = self.mne.fig_size_px
        new_margins = dict()
        for side in ('left', 'right', 'bottom', 'top'):
            ratio = ((old_width / new_width) if side in ('left', 'right') else
                     (old_height / new_height))
            rel_dim = getattr(self.subplotpars, side)
            if side in ('right', 'top'):
                new_margins[side] = 1 - ratio * (1 - rel_dim)
            else:
                new_margins[side] = ratio * rel_dim
        # zen mode adjustment
        self.mne.zen_w *= old_width / new_width
        self.mne.zen_h *= old_height / new_height
        # apply the update
        self.subplots_adjust(**new_margins)

    def _toggle_scrollbars(self):
        """Show or hide scrollbars (A.K.A. zen mode)."""
        if getattr(self.mne, 'show_scrollbars', None) is not None:
            # grow/shrink main axes to take up space from (or make room for)
            # scrollbars. We  can't use ax.set_position() because axes are
            # locatable, so we have to fake it with subplots_adjust
            should_show = not self.mne.show_scrollbars
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
            self.mne.show_scrollbars = should_show
            self.canvas.draw()

    def _toggle_proj_dialog(self, event):
        """Show/hide projectors dialog."""
        if self.mne.fig_proj is None:
            self._draw_proj_checkbox(event, draw_current_state=False)
        else:
            self.mne.fig_proj.canvas.close_event()
            del self.mne.proj_checks
            self.mne.fig_proj = None

    def _setup_annotation_fig(self):
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
                pass  # TODO: change visible channels
        elif key == ('right', 'left', 'shift+right', 'shift+left'):
            direction = 1 if key.endswith('right') else -1
            denom = 1 if key.startswith('shift') else 4
            t_shift = direction * self.mne.duration / denom
            self.mne.t_start += t_shift
            # TODO: redraw
        elif key in ('=', '+', '-'):
            scaler = 1 / 1.1 if key == '-' else 1.1
            self.mne.scale_factor *= scaler
            # TODO: redraw
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
        elif key == 'd':  # TODO: toggle remove DC
            pass
        elif key == 'p':  # TODO: toggle snap annotations
            pass
        elif key == 's':  # TODO: toggle show scalebars
            pass
        elif key == 'w':  # TODO: toggle noise cov / whitening
            pass
        elif key == 'z':  # zen mode: remove scrollbars and buttons
            self._toggle_scrollbars()


def _figure(*args, toolbar=False, FigureClass=None, **kwargs):
    """Instantiate a new figure."""
    from matplotlib import rc_context
    from matplotlib.pyplot import figure
    if toolbar:
        fig = figure(*args, FigureClass=FigureClass, **kwargs)
    else:
        with rc_context(rc=dict(toolbar='none')):
            fig = figure(*args, FigureClass=FigureClass, **kwargs)
    return fig


def browse_figure(inst, *args, toolbar=False, FigureClass=MNEBrowseFigure,
                  **kwargs):
    """Instantiate a new MNE browse-style figure."""
    fig = _figure(*args, toolbar=toolbar, FigureClass=FigureClass, inst=inst,
                  **kwargs)
    # initialize zen mode (can't do in __init__ due to get_position() calls)
    fig.canvas.draw()
    fig.mne.zen_w = (fig.mne.ax_vscroll.get_position().xmax -
                     fig.mne.ax_main.get_position().xmax)
    fig.mne.zen_h = (fig.mne.ax_main.get_position().ymin -
                     fig.mne.ax_hscroll.get_position().ymin)
    if not fig.mne.show_scrollbars:
        fig.mne.show_scrollbars = True
        fig._toggle_scrollbars()
    # remove MPL default keypress catchers
    default_cbs = list(fig.canvas.callbacks.callbacks['key_press_event'])
    for callback in default_cbs:
        fig.canvas.callbacks.disconnect(callback)
    # now add our custom ones
    callbacks = dict(resize_event=fig._resize,
                     key_press_event=fig._keypress)
    callback_ids = dict()
    for event, callback in callbacks.items():
        callback_ids[event] = fig.canvas.mpl_connect(event, callback)
    # store references so they aren't garbage-collected
    fig.mne.callback_ids = callback_ids
    return fig
