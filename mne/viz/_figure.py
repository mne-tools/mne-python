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

    def _onclick_help():
        pass


class MNEBrowseFigure(MNEFigure):
    """Interactive figure with scrollbars, for data browsing."""
    def __init__(self, *args, xlabel='Time (s)', show_scrollbars=True,
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
        # self.mne.data = None              # raw
        # self.mne.info = None
        # self.mne.proj = None
        # self.mne.noise_cov = None
        # self.mne.rev_event_id = None      # event_id_rev
        # # channel
        # self.mne.n_channels = None
        # self.mne.ch_types = None          # types
        # self.mne.group_by = None
        # self.mne.picks = None             # data_picks
        # # time
        # self.mne.n_times = None
        # self.mne.first_time = None
        # self.mne.event_times = None
        # self.mne.event_nums = None
        # self.mne.duration = None
        # self.mne.decim = None
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
        # # ancillary figures
        # self.mne.fig_proj = None
        # self.mne.fig_help = None
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
        bottom = self._inch_to_rel(b_margin, horiz=False)
        width = 1 - self._inch_to_rel(r_margin) - left
        height = 1 - self._inch_to_rel(t_margin, horiz=False) - bottom
        position = [left, bottom, width, height]

        # Main axes must be a subplot for subplots_adjust to work (so user can
        # adjust margins). That's why we don't use the Divider class directly.
        ax = self.add_subplot(1, 1, 1, position=position)
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
        help_button = Button(ax_help, 'Help')
        help_button.on_clicked(self._onclick_help)

        # PROJ BUTTON: (optionally) added later, easier to compute position now
        proj_button_pos = [
            1 - self._inch_to_rel(r_margin + scroll_width),  # left
            self._inch_to_rel(b_margin, horiz=False),        # bottom
            self._inch_to_rel(scroll_width),                 # width
            self._inch_to_rel(scroll_width, horiz=False)     # height
        ]
        self.mne.button_proj_position = proj_button_pos
        self.mne.button_proj_locator = div.new_locator(nx=2, ny=0)

        # ZEN MODE: (show/hide scrollbars)
        self.canvas.draw()  # otherwise the get_position() calls are inaccurate
        self.mne.zen_w = (ax_vscroll.get_position().xmax -
                          ax.get_position().xmax)
        self.mne.zen_h = (ax.get_position().ymin -
                          ax_hscroll.get_position().ymin)
        if not self.mne.show_scrollbars:
            self.mne.show_scrollbars = True
            self._toggle_scrollbars()

        # SAVE PARAMS
        self.mne.ax_main = ax
        self.mne.ax_help = ax_help
        self.mne.ax_hscroll = ax_hscroll
        self.mne.ax_vscroll = ax_vscroll
        self.mne.button_help = help_button

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
            # squeeze a bit more because we don't need space for "Time (s)" now
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

    def _keypress(self, event):
        """Triage keypress events."""
        from matplotlib.pyplot import close
        key = event.key
        if key == self.mne.close_key:
            close(self)
        elif key == 'z':  # zen mode: remove scrollbars and buttons
            self._toggle_scrollbars()


def mne_figure(*args, toolbar=False, FigureClass=MNEBrowseFigure, **kwargs):
    """Instantiate a new MNE browse-style figure."""
    from matplotlib import rc_context
    from matplotlib.pyplot import figure
    if toolbar:
        fig = figure(*args, FigureClass=FigureClass, **kwargs)
    else:
        with rc_context(rc=dict(toolbar='none')):
            fig = figure(*args, FigureClass=FigureClass, **kwargs)
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
