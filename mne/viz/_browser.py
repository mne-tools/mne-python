from abc import ABC

import numpy as np


class MNEBrowserParams:
    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        vars(self).update(**kwargs)


class MNEDataBrowser(ABC):
    def __init__(self, inst, ica=None, **kwargs):
        from .. import BaseEpochs
        from ..io import BaseRaw
        from ..preprocessing import ICA

        self._data = None
        self._times = None

        self.mne = MNEBrowserParams(**kwargs)

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
        self.mne.projs_active = np.array([p['active'] for p in self.mne.projs])
        self.mne.whitened_ch_names = list()
        self.mne.use_noise_cov = self.mne.noise_cov is not None
        self.mne.zorder = dict(patch=0, grid=1, ann=2, events=3, bads=4,
                               data=5, mag=6, grad=7, scalebar=8, vline=9)
        # additional params for epochs (won't affect raw / ICA)
        self.mne.epoch_traces = list()
        self.mne.bad_epochs = list()
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

        # MAIN AXES: default sizes (inches)
        # XXX simpler with constrained_layout? (when it's no longer "beta")
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
        ax_main = self.add_subplot(1, 1, 1, position=position)
        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right)
        div = make_axes_locatable(ax_main)
        # this only gets shown in zen mode
        self.mne.zen_xlabel = ax_main.set_xlabel(xlabel)
        self.mne.zen_xlabel.set_visible(not self.mne.scrollbars_visible)

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
        # HORIZONTAL SCROLLBAR PATCHES (FOR MARKING BAD EPOCHS)
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            for ix, _ in enumerate(epoch_nums):
                start = self.mne.boundary_times[ix]
                width = np.diff(self.mne.boundary_times[:2])[0]
                ax_hscroll.add_patch(
                    Rectangle((start, 0), width, 1, color='none',
                              zorder=self.mne.zorder['patch']))
            # add epoch boundaries & center epoch numbers between boundaries
            midpoints = np.convolve(self.mne.boundary_times, np.ones(2),
                                    mode='valid') / 2
            # both axes, major ticks: gridlines
            for _ax in (ax_main, ax_hscroll):
                _ax.xaxis.set_major_locator(
                    FixedLocator(self.mne.boundary_times[1:-1]))
                _ax.xaxis.set_major_formatter(NullFormatter())
            grid_kwargs = dict(color=self.mne.fgcolor, axis='x',
                               zorder=self.mne.zorder['grid'])
            ax_main.grid(linewidth=2, linestyle='dashed', **grid_kwargs)
            ax_hscroll.grid(alpha=0.5, linewidth=0.5, linestyle='solid',
                            **grid_kwargs)
            # main axes, minor ticks: ticklabel (epoch number) for every epoch
            ax_main.xaxis.set_minor_locator(FixedLocator(midpoints))
            ax_main.xaxis.set_minor_formatter(FixedFormatter(epoch_nums))
            # hscroll axes, minor ticks: up to 20 ticklabels (epoch numbers)
            ax_hscroll.xaxis.set_minor_locator(
                FixedLocator(midpoints, nbins=20))
            ax_hscroll.xaxis.set_minor_formatter(
                FuncFormatter(lambda x, pos: self._get_epoch_num_from_time(x)))
            # hide some ticks
            ax_main.tick_params(axis='x', which='major', bottom=False)
            ax_hscroll.tick_params(axis='x', which='both', bottom=False)
        else:
            # RAW / ICA X-AXIS TICK & LABEL FORMATTING
            ax_main.xaxis.set_major_formatter(
                FuncFormatter(partial(self._xtick_formatter,
                                      ax_type='main')))
            ax_hscroll.xaxis.set_major_formatter(
                FuncFormatter(partial(self._xtick_formatter,
                                      ax_type='hscroll')))
            if self.mne.time_format != 'float':
                for _ax in (ax_main, ax_hscroll):
                    _ax.set_xlabel('Time (HH:MM:SS)')

        # VERTICAL SCROLLBAR PATCHES (COLORED BY CHANNEL TYPE)
        ch_order = self.mne.ch_order
        for ix, pick in enumerate(ch_order):
            this_color = (self.mne.ch_color_bad
                          if self.mne.ch_names[pick] in self.mne.info['bads']
                          else self.mne.ch_color_dict)
            if isinstance(this_color, dict):
                this_color = this_color[self.mne.ch_types[pick]]
            ax_vscroll.add_patch(
                Rectangle((0, ix), 1, 1, color=this_color,
                          zorder=self.mne.zorder['patch']))
        ax_vscroll.set_ylim(len(ch_order), 0)
        ax_vscroll.set_visible(not self.mne.butterfly)
        # SCROLLBAR VISIBLE SELECTION PATCHES
        sel_kwargs = dict(alpha=0.3, linewidth=4, clip_on=False,
                          edgecolor=self.mne.fgcolor)
        vsel_patch = Rectangle((0, 0), 1, self.mne.n_channels,
                               facecolor=self.mne.bgcolor, **sel_kwargs)
        ax_vscroll.add_patch(vsel_patch)
        hsel_facecolor = np.average(
            np.vstack((to_rgba_array(self.mne.fgcolor),
                       to_rgba_array(self.mne.bgcolor))),
            axis=0, weights=(3, 1))  # 75% foreground, 25% background
        hsel_patch = Rectangle((self.mne.t_start, 0), self.mne.duration, 1,
                               facecolor=hsel_facecolor, **sel_kwargs)
        ax_hscroll.add_patch(hsel_patch)
        ax_hscroll.set_xlim(self.mne.first_time, self.mne.first_time +
                            self.mne.n_times / self.mne.info['sfreq'])
        # VLINE
        vline_color = (0., 0.75, 0.)
        vline_kwargs = dict(visible=False, zorder=self.mne.zorder['vline'])
        if self.mne.is_epochs:
            x = np.arange(self.mne.n_epochs)
            vline = ax_main.vlines(
                x, 0, 1, colors=vline_color, **vline_kwargs)
            vline.set_transform(blended_transform_factory(ax_main.transData,
                                                          ax_main.transAxes))
            vline_hscroll = None
        else:
            vline = ax_main.axvline(0, color=vline_color, **vline_kwargs)
            vline_hscroll = ax_hscroll.axvline(0, color=vline_color,
                                               **vline_kwargs)
        vline_text = ax_main.annotate(
            '', xy=(0, 0), xycoords='axes fraction', xytext=(-2, 0),
            textcoords='offset points', fontsize=10, ha='right', va='center',
            color=vline_color, **vline_kwargs)

        # HELP BUTTON: initialize in the wrong spot...
        ax_help = div.append_axes(position='left',
                                  size=Fixed(help_width),
                                  pad=Fixed(vscroll_dist))
        # HELP BUTTON: ...move it down by changing its locator
        loc = div.new_locator(nx=0, ny=0)
        ax_help.set_axes_locator(loc)
        # HELP BUTTON: make it a proper button
        with _patched_canvas(ax_help.figure):
            self.mne.button_help = Button(ax_help, 'Help')
        # PROJ BUTTON
        ax_proj = None
        if len(self.mne.projs) and not inst.proj:
            proj_button_pos = [
                1 - self._inch_to_rel(r_margin + scroll_width),  # left
                self._inch_to_rel(b_margin, horiz=False),  # bottom
                self._inch_to_rel(scroll_width),  # width
                self._inch_to_rel(scroll_width, horiz=False)  # height
            ]
            loc = div.new_locator(nx=4, ny=0)
            ax_proj = self.add_axes(proj_button_pos)
            ax_proj.set_axes_locator(loc)
            with _patched_canvas(ax_help.figure):
                self.mne.button_proj = Button(ax_proj, 'Prj')

        # INIT TRACES
        self.mne.trace_kwargs = dict(antialiased=True, linewidth=0.5)
        self.mne.traces = ax_main.plot(
            np.full((1, self.mne.n_channels), np.nan), **self.mne.trace_kwargs)

        # SAVE UI ELEMENT HANDLES
        vars(self.mne).update(
            ax_main=ax_main, ax_help=ax_help, ax_proj=ax_proj,
            ax_hscroll=ax_hscroll, ax_vscroll=ax_vscroll,
            vsel_patch=vsel_patch, hsel_patch=hsel_patch, vline=vline,
            vline_hscroll=vline_hscroll, vline_text=vline_text)