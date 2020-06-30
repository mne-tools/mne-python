# -*- coding: utf-8 -*-
"""Figure classes for MNE-Python's 2D plots."""

# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

from matplotlib.figure import Figure
from ..utils import set_config


class MNEFigParams:
    """Container for MNE figure parameters."""
    def __init__(self, *args, **kwargs):
        # default key to close window
        self.close_key = 'escape'


class MNEFigure(Figure):
    """Wrapper of matplotlib.figure.Figure; adds MNE-Python figure params."""
    def __init__(self, *args, **kwargs):
        from matplotlib import rc_context
        # pop the one constructor kwarg used by MNE-Python
        toolbar = kwargs.pop('toolbar', True)
        if toolbar:
            super().__init__(*args, **kwargs)
        else:
            with rc_context(toolbar='none'):
                super().__init__(*args, **kwargs)
                # remove button press catchers (for toolbar)
                cbs = list(self.canvas.callbacks.callbacks['key_press_event'])
                for callback in cbs:
                    self.canvas.callbacks.disconnect(callback)
        # add our param object
        self.mne = MNEFigParams()
        return self

    def _get_dpi_ratio(self):
        """Get DPI ratio (to handle hi-DPI screens)."""
        dpi_ratio = 1.
        for key in ('_dpi_ratio', '_device_scale'):
            dpi_ratio = getattr(self.canvas, key, dpi_ratio)
        return dpi_ratio

    def _get_size_px(self):
        """Get figure size in pixels."""
        dpi_ratio = self._get_dpi_ratio(self)
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
    def __init__(self, *args, xlabel='Time (s)', **kwargs):
        from matplotlib.widgets import Button
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mne.viz.utils import _get_figsize_from_config

        # figsize is the first arg of Figure()
        figsize = args[0] if len(args) else kwargs.pop('figsize', None)
        if figsize is None:
            figsize = _get_figsize_from_config()
        if len(args):
            args[0] = figsize
        else:
            kwargs['figsize'] = figsize

        # init Figure
        super().__init__(*args, **kwargs)

        # additional params for browse figures (comments indicate name changes)
        self.data = None              # raw
        self.info = None
        self.proj = None
        self.noise_cov = None
        self.rev_event_id = None      # event_id_rev
        # channel
        self.n_channels = None
        self.ch_types = None          # types
        self.group_by = None
        self.picks = None             # data_picks
        # time
        self.n_times = None
        self.first_time = None
        self.event_times = None
        self.event_nums = None
        self.duration = None
        self.decim = None
        # annotations
        self.annotations = None
        self.snap_annotations = None
        self.added_label = None
        # traces
        self.trace_indices = None     # inds
        self.orig_indices = None      # orig_inds
        self.clipping = None
        self.butterfly = None
        # filters
        self.remove_dc = None
        self.filter_coefs_ba = None   # ba
        self.filter_bounds = None     # filt_bounds
        # scalings
        self.units = None
        self.scalings = None
        self.unit_scalings = None
        # ancillary figures
        self.fig_proj = None
        self.fig_help = None
        # UI state variables
        self.ch_start = None
        self.t_start = None
        self.show_scalebars = None

        # MAIN AXES: default sizes (inches)
        l_border = 1.
        r_border = 0.1
        b_border = 0.45
        t_border = 0.25
        scroll_width = 0.25
        hscroll_dist = 0.25
        vscroll_dist = 0.1
        help_width = scroll_width * 2

        # MAIN AXES: default borders (figure-relative coordinates)
        left = self._inch_to_rel(self, l_border - vscroll_dist - help_width)
        bottom = self._inch_to_rel(self, b_border, horiz=False)
        width = 1 - self._inch_to_rel(self, r_border) - left
        height = 1 - self._inch_to_rel(self, t_border, horiz=False) - bottom
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
            1 - self._inch_to_rel(self, r_border + scroll_width),  # left
            self._inch_to_rel(self, b_border, horiz=False),        # bottom
            self._inch_to_rel(self, scroll_width),                 # width
            self._inch_to_rel(self, scroll_width, horiz=False)     # height
        ]
        self.mne.proj_button_pos = proj_button_pos
        self.mne.proj_button_locator = div.new_locator(nx=2, ny=0)

        # ZEN MODE: (show/hide scrollbars)
        self.canvas.draw()  # otherwise the get_position() calls are inaccurate
        self.mne.zen_w_delta = (ax_vscroll.get_position().xmax -
                                ax.get_position().xmax)
        self.mne.zen_h_delta = (ax.get_position().ymin -
                                ax_hscroll.get_position().ymin)
        if not getattr(self.mne, 'show_scrollbars', True):
            self.mne.show_scrollbars = True
            self._toggle_scrollbars()

        # add resize callback (it's the same for Raw/Epochs/ICA)
        self.canvas.mpl_connect('resize_event', self._resize_event)

        # SAVE PARAMS
        self.mne.ax_hscroll = ax_hscroll
        self.mne.ax_vscroll = ax_vscroll
        self.mne.ax_help = ax_help
        self.mne.help_button = help_button

    def _resize_event(self, event):
        """Handle resize event for mne_browse-style plots (Raw/Epochs/ICA)."""
        size = ','.join([str(s) for s in self.get_size_inches()])
        set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
        new_width, new_height = self._get_size_px()
        self._update_borders(new_width, new_height)
        self.mne.fig_size_px = (new_width, new_height)

    def _update_borders(self, new_width, new_height):
        """Update figure borders to maintain fixed size in inches/pixels."""
        old_width, old_height = self.mne.fig_size_px
        new_borders = dict()
        for side in ('left', 'right', 'bottom', 'top'):
            horiz = side in ('left', 'right')
            ratio = ((old_width / new_width) if horiz else
                     (old_height / new_height))
            rel_dim = getattr(self.subplotpars, side)
            if side in ('right', 'top'):
                new_borders[side] = 1 - ratio * (1 - rel_dim)
            else:
                new_borders[side] = ratio * rel_dim
        # zen mode adjustment
        self.mne.zen_w_delta *= old_width / new_width
        self.mne.zen_h_delta *= old_height / new_height
        # apply the update
        self.subplots_adjust(**new_borders)

    def _toggle_scrollbars(self):
        """Show or hide scrollbars (A.K.A. zen mode)."""
        if getattr(self.mne, 'show_scrollbars', None) is not None:
            # grow/shrink main axes to take up space from (or make room for)
            # scrollbars. We  can't use ax.set_position() because axes are
            # locatable, so we have to fake it with subplots_adjust
            should_show = not self.mne.show_scrollbars
            borders = {side: getattr(self.subplotpars, side)
                       for side in ('left', 'bottom', 'right', 'top')}
            # if should_show, bottom margin moves up; right margin moves left
            borders['bottom'] += (1 if should_show else -1) * self.mne.zen_h
            borders['right'] += (-1 if should_show else 1) * self.mne.zen_w
            # squeeze a bit more because we don't need space for "Time (s)" now
            v_delta = self._inch_to_rel(0.16, horiz=False)
            borders['bottom'] += (1 if should_show else -1) * v_delta
            self.subplots_adjust(**borders)
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
