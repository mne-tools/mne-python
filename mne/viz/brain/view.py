# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from collections import namedtuple
import sys

import numpy as np


View = namedtuple('View', 'elev azim')

views_dict = {'lateral': View(elev=5, azim=0),
              'medial': View(elev=5, azim=180),
              'rostral': View(elev=5, azim=90),
              'caudal': View(elev=5, azim=-90),
              'dorsal': View(elev=90, azim=0),
              'ventral': View(elev=-90, azim=0),
              'frontal': View(elev=5, azim=110),
              'parietal': View(elev=5, azim=-110)}
# add short-size version entries into the dict
_views_dict = dict()
for k, v in views_dict.items():
    _views_dict[k[:3]] = v
views_dict.update(_views_dict)


class TimeViewer(object):
    u"""Creates time viewer widget.

    Parameters
    ----------
    brain : Brain
        Object with cortex mesh to be controlled by time
        viewer.
    """

    def __init__(self, brain):
        import ipyvolume as ipv
        import ipywidgets as widgets

        if brain.data['time'] is None:
            raise ValueError('Brain class instance does not have time data.')

        self._brain = brain
        time = brain.data['time']
        time_idx = brain.data['time_idx']
        overlays = tuple(brain.overlays.values())
        control = ipv.animation_control(overlays,
                                        len(time),
                                        add=False,
                                        interval=500)
        slider = control.children[1]
        slider.readout = False
        slider.value = time_idx
        label = widgets.Label(self._get_label(time[time_idx]))

        # hadler for changing of selected time moment
        def slider_handler(change):
            time_idx_new = int(change.new)

            for v in brain.views:
                for h in brain.hemis:
                    smooth_mat = brain.data[h + '_smooth_mat']
                    act_data = brain.data[h + '_array'][:, time_idx_new]
                    act_data = smooth_mat.dot(act_data)
                    act_data = brain.data['k'] * act_data + brain.data['b']
                    act_data = np.clip(act_data, 0, 1)
                    brain.overlays[h + '_' + v].color = \
                        brain.data['lut'](act_data)

            # change label value
            label.value = self._get_label(time[time_idx_new])

        slider.observe(slider_handler, names='value')
        control = widgets.HBox((*control.children, label))
        ipv.gcc().children += (control,)

    def show(self):
        u"""Display widget."""
        import ipyvolume as ipv
        ipv.show()

    def _get_label(self, time):
        u"""Return time label string.

        Parameters
        ----------
        time : float | int
            Time value to show.

        Returns
        -------
        label : str
            Time label as string.
        """
        time_label = self._brain.data['time_label']

        if isinstance(time_label, str):
            label = time_label % time
        elif callable(time_label):
            label = time_label

        return label


class ColorBar(object):
    u"""Helper class for visualizing a color bar.

    Parameters
    ----------
    brain : Brain
        Object with cortex mesh to be controlled by time
        viewer.
    """

    def __init__(self, brain):
        from bqplot import Axis, ColorScale, Figure, HeatMap, LinearScale
        import ipyvolume as ipv
        import ipywidgets as widgets

        self._brain = brain
        self._input_fmin = None
        self._input_fmid = None
        self._input_fmax = None
        self._btn_upd_mesh = None
        self._colors = None

        if brain.data['center'] is None:
            dt_min = brain.data['fmin']
        else:
            dt_min = -brain.data['fmax']
        dt_max = brain.data['fmax']

        self._lut = brain.data['lut']
        self._cbar_data = np.linspace(0, 1, self._lut.N)
        cbar_ticks = np.linspace(dt_min, dt_max, self._lut.N)
        color = np.array((self._cbar_data, self._cbar_data))
        cbar_w = 500
        cbar_fig_margin = {'top': 15, 'bottom': 15, 'left': 5, 'right': 5}
        self._update_colors()

        x_sc, col_sc = LinearScale(), ColorScale(colors=self._colors)
        ax_x = Axis(scale=x_sc)
        heat = HeatMap(x=cbar_ticks,
                       color=color,
                       scales={'x': x_sc, 'color': col_sc})

        self._add_inputs()
        fig_layout = widgets.Layout(width='%dpx' % cbar_w,
                                    height='60px')
        cbar_fig = Figure(axes=[ax_x],
                          marks=[heat],
                          fig_margin=cbar_fig_margin,
                          layout=fig_layout)

        def on_update(but_event):
            u"""Update button click event handler."""
            val_min = self._input_fmin.value
            val_mid = self._input_fmid.value
            val_max = self._input_fmax.value
            center = brain.data['center']
            time_idx = brain.data['time_idx']
            time_arr = brain.data['time']

            if not val_min < val_mid < val_max:
                raise ValueError('Incorrect relationship between' +
                                 ' fmin, fmid, fmax. Given values ' +
                                 '{0}, {1}, {2}'
                                 .format(val_min, val_mid, val_max))
            if center is None:
                # 'hot' or another linear color map
                dt_min = val_min
            else:
                # 'mne' or another divergent color map
                dt_min = -val_max
            dt_max = val_max

            self._lut = self._brain.update_lut(fmin=val_min, fmid=val_mid,
                                               fmax=val_max)
            k = 1 / (dt_max - dt_min)
            b = 1 - k * dt_max
            self._brain.data['k'] = k
            self._brain.data['b'] = b

            for v in brain.views:
                for h in brain.hemis:
                    if (time_arr is None) or (time_idx is None):
                        act_data = brain.data[h + '_array']
                    else:
                        act_data = brain.data[h + '_array'][:, time_idx]

                    smooth_mat = brain.data[h + '_smooth_mat']
                    act_data = smooth_mat.dot(act_data)

                    act_data = k * act_data + b
                    act_data = np.clip(act_data, 0, 1)
                    act_color_new = self._lut(act_data)
                    brain.overlays[h + '_' + v].color = act_color_new
            self._update_colors()
            x_sc, col_sc = LinearScale(), ColorScale(colors=self._colors)
            ax_x = Axis(scale=x_sc)

            heat = HeatMap(x=cbar_ticks,
                           color=color,
                           scales={'x': x_sc, 'color': col_sc})
            cbar_fig.axes = [ax_x]
            cbar_fig.marks = [heat]

        self._btn_upd_mesh.on_click(on_update)

        info_widget = widgets.VBox((cbar_fig,
                                    self._input_fmin,
                                    self._input_fmid,
                                    self._input_fmax,
                                    self._btn_upd_mesh))

        ipv.gcc().children += (info_widget,)

    def _update_colors(self):
        u"""Update or prepare list of colors for plotting."""
        colors = self._lut(self._cbar_data)
        # transform to [0, 255] range taking into account transparency
        alphas = colors[:, -1]
        bg_color = 0.5 * np.ones((len(alphas), 3))
        colors = 255 * (alphas * colors[:, :-1].transpose() +
                        (1 - alphas) * bg_color.transpose())
        colors = colors.transpose()

        colors = colors.astype(int)
        colors = ['#%02x%02x%02x' % tuple(c) for c in colors]

        self._colors = colors

    def _add_inputs(self):
        u"""Add inputs and update button."""
        import ipywidgets as widgets

        val_min = self._brain.data['fmin']
        val_mid = self._brain.data['fmid']
        val_max = self._brain.data['fmax']

        if self._brain.data['center'] is None:
            # 'hot' color map
            sl_min = -sys.float_info.max
        else:
            # 'mne' color map
            sl_min = self._brain.data['center']

        self._input_fmin = widgets.BoundedFloatText(value=round(val_min, 2),
                                                    min=sl_min,
                                                    description='Fmin:',
                                                    step=0.1,
                                                    disabled=False)
        self._input_fmid = widgets.BoundedFloatText(value=round(val_mid, 2),
                                                    min=sl_min,
                                                    description='Fmid:',
                                                    step=0.1,
                                                    disabled=False)
        self._input_fmax = widgets.BoundedFloatText(value=round(val_max, 2),
                                                    min=sl_min,
                                                    description='Fmax:',
                                                    step=0.1,
                                                    disabled=False)

        self._btn_upd_mesh = widgets.Button(description='Update mesh',
                                            disabled=False,
                                            button_style='',
                                            tooltip='Update mesh')
