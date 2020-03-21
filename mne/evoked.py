# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Mads Jensen <mje.mads@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np

from .baseline import rescale
from .channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                SetChannelsMixin, InterpolationMixin)
from .channels.layout import _merge_ch_data, _pair_grad_sensors
from .filter import detrend, FilterMixin
from .utils import (check_fname, logger, verbose, _time_mask, warn, sizeof_fmt,
                    SizeMixin, copy_function_doc_to_method_doc, _validate_type,
                    fill_doc, _check_option, ShiftTimeMixin, _build_data_frame,
                    _check_pandas_installed, _check_pandas_index_arguments,
                    _convert_times, _scale_dataframe_data, _check_time_format,
                    _check_scaling_time)
from .viz import (plot_evoked, plot_evoked_topomap, plot_evoked_field,
                  plot_evoked_image, plot_evoked_topo)
from .viz.evoked import plot_evoked_white, plot_evoked_joint
from .viz.topomap import _topomap_animation

from .io.constants import FIFF
from .io.open import fiff_open
from .io.tag import read_tag
from .io.tree import dir_tree_find
from .io.pick import pick_types, _picks_to_idx
from .io.meas_info import read_meas_info, write_meas_info
from .io.proj import ProjMixin
from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_string, write_float_matrix,
                       write_id, write_float)
from .io.base import TimeMixin, _check_maxshield

_aspect_dict = {
    'average': FIFF.FIFFV_ASPECT_AVERAGE,
    'standard_error': FIFF.FIFFV_ASPECT_STD_ERR,
    'single_epoch': FIFF.FIFFV_ASPECT_SINGLE,
    'partial_average': FIFF.FIFFV_ASPECT_SUBAVERAGE,
    'alternating_subaverage': FIFF.FIFFV_ASPECT_ALTAVERAGE,
    'sample_cut_out_by_graph': FIFF.FIFFV_ASPECT_SAMPLE,
    'power_density_spectrum': FIFF.FIFFV_ASPECT_POWER_DENSITY,
    'dipole_amplitude_cuvre': FIFF.FIFFV_ASPECT_DIPOLE_WAVE,
    'squid_modulation_lower_bound': FIFF.FIFFV_ASPECT_IFII_LOW,
    'squid_modulation_upper_bound': FIFF.FIFFV_ASPECT_IFII_HIGH,
    'squid_gate_setting': FIFF.FIFFV_ASPECT_GATE,
}
_aspect_rev = {val: key for key, val in _aspect_dict.items()}


@fill_doc
class Evoked(ProjMixin, ContainsMixin, UpdateChannelsMixin, SetChannelsMixin,
             InterpolationMixin, FilterMixin, TimeMixin, SizeMixin,
             ShiftTimeMixin):
    """Evoked data.

    Parameters
    ----------
    fname : str
        Name of evoked/average FIF file to load.
        If None no data is loaded.
    condition : int, or str
        Dataset ID number (int) or comment/name (str). Optional if there is
        only one data set in file.
    proj : bool, optional
        Apply SSP projection vectors.
    kind : str
        Either 'average' or 'standard_error'. The type of data to read.
        Only used if 'condition' is a str.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    %(verbose)s

    Attributes
    ----------
    info : dict
        Measurement info.
    ch_names : list of str
        List of channels' names.
    nave : int
        Number of averaged epochs.
    kind : str
        Type of data, either average or standard_error.
    first : int
        First time sample.
    last : int
        Last time sample.
    comment : str
        Comment on dataset. Can be the condition.
    data : array of shape (n_channels, n_times)
        Evoked response.
    times :  array
        Time vector in seconds. Goes from `tmin` to `tmax`. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.
    %(verbose)s

    Notes
    -----
    Evoked objects contain a single condition only.
    """

    @verbose
    def __init__(self, fname, condition=None, proj=True,
                 kind='average', allow_maxshield=False,
                 verbose=None):  # noqa: D102
        _validate_type(proj, bool, "'proj'")
        # Read the requested data
        self.info, self.nave, self._aspect_kind, self.comment, self.times, \
            self.data = _read_evoked(fname, condition, kind, allow_maxshield)
        self._update_first_last()
        self.verbose = verbose
        self.preload = True
        # project and baseline correct
        if proj:
            self.apply_proj()

    @property
    def kind(self):
        """The data kind."""
        return _aspect_rev[self._aspect_kind]

    @kind.setter
    def kind(self, kind):
        _check_option('kind', kind, list(_aspect_dict.keys()))
        self._aspect_kind = _aspect_dict[kind]

    @property
    def data(self):
        """The data matrix."""
        return self._data

    @data.setter
    def data(self, data):
        """Set the data matrix."""
        self._data = data

    @verbose
    def apply_baseline(self, baseline=(None, 0), verbose=None):
        """Baseline correct evoked data.

        Parameters
        ----------
        baseline : tuple of length 2
            The time interval to apply baseline correction. If None do not
            apply it. If baseline is (a, b) the interval is between "a (s)" and
            "b (s)". If a is None the beginning of the data is used and if b is
            None then b is set to the end of the interval. If baseline is equal
            to (None, None) all the time interval is used. Correction is
            applied by computing mean of the baseline period and subtracting it
            from the data. The baseline (a, b) includes both endpoints, i.e.
            all timepoints t such that a <= t <= b.
        %(verbose_meth)s

        Returns
        -------
        evoked : instance of Evoked
            The baseline-corrected Evoked object.

        Notes
        -----
        Baseline correction can be done multiple times.

        .. versionadded:: 0.13.0
        """
        self.data = rescale(self.data, self.times, baseline, copy=False)
        return self

    def save(self, fname):
        """Save dataset to file.

        Parameters
        ----------
        fname : str
            The name of the file, which should end with -ave.fif or
            -ave.fif.gz.

        Notes
        -----
        To write multiple conditions into a single file, use
        :func:`mne.write_evokeds`.
        """
        write_evokeds(fname, self)

    def __repr__(self):  # noqa: D105
        s = "'%s' (%s, N=%s)" % (self.comment, self.kind, self.nave)
        s += ", [%0.5g, %0.5g] sec" % (self.times[0], self.times[-1])
        s += ", %s ch" % self.data.shape[0]
        s += ", ~%s" % (sizeof_fmt(self._size),)
        return "<Evoked  |  %s>" % s

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    @fill_doc
    def crop(self, tmin=None, tmax=None, include_tmax=True):
        """Crop data to a given time interval.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        %(include_tmax)s

        Returns
        -------
        evoked : instance of Evoked
            The cropped Evoked object.

        Notes
        -----
        Unlike Python slices, MNE time intervals include both their end points;
        crop(tmin, tmax) returns the interval tmin <= t <= tmax.
        """
        mask = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'],
                          include_tmax=include_tmax)
        self.times = self.times[mask]
        self._update_first_last()
        self.data = self.data[:, mask]
        return self

    def decimate(self, decim, offset=0):
        """Decimate the evoked data.

        .. note:: No filtering is performed. To avoid aliasing, ensure
                  your data are properly lowpassed.

        Parameters
        ----------
        decim : int
            The amount to decimate data.
        offset : int
            Apply an offset to where the decimation starts relative to the
            sample corresponding to t=0. The offset is in samples at the
            current sampling rate.

        Returns
        -------
        evoked : instance of Evoked
            The decimated Evoked object.

        See Also
        --------
        Epochs.decimate
        Epochs.resample
        mne.io.Raw.resample

        Notes
        -----
        Decimation can be done multiple times. For example,
        ``evoked.decimate(2).decimate(2)`` will be the same as
        ``evoked.decimate(4)``.

        .. versionadded:: 0.13.0
        """
        decim, offset, new_sfreq = _check_decim(self.info, decim, offset)
        start_idx = int(round(self.times[0] * (self.info['sfreq'] * decim)))
        i_start = start_idx % decim + offset
        decim_slice = slice(i_start, None, decim)
        self.info['sfreq'] = new_sfreq
        self.data = self.data[:, decim_slice].copy()
        self.times = self.times[decim_slice].copy()
        self._update_first_last()
        return self

    @copy_function_doc_to_method_doc(plot_evoked)
    def plot(self, picks=None, exclude='bads', unit=True, show=True, ylim=None,
             xlim='tight', proj=False, hline=None, units=None, scalings=None,
             titles=None, axes=None, gfp=False, window_title=None,
             spatial_colors=False, zorder='unsorted', selectable=True,
             noise_cov=None, time_unit='s', sphere=None, verbose=None):
        return plot_evoked(
            self, picks=picks, exclude=exclude, unit=unit, show=show,
            ylim=ylim, proj=proj, xlim=xlim, hline=hline, units=units,
            scalings=scalings, titles=titles, axes=axes, gfp=gfp,
            window_title=window_title, spatial_colors=spatial_colors,
            zorder=zorder, selectable=selectable, noise_cov=noise_cov,
            time_unit=time_unit, sphere=sphere, verbose=verbose)

    @copy_function_doc_to_method_doc(plot_evoked_image)
    def plot_image(self, picks=None, exclude='bads', unit=True, show=True,
                   clim=None, xlim='tight', proj=False, units=None,
                   scalings=None, titles=None, axes=None, cmap='RdBu_r',
                   colorbar=True, mask=None, mask_style=None,
                   mask_cmap='Greys', mask_alpha=.25, time_unit='s',
                   show_names=None, group_by=None, sphere=None):
        return plot_evoked_image(
            self, picks=picks, exclude=exclude, unit=unit, show=show,
            clim=clim, xlim=xlim, proj=proj, units=units, scalings=scalings,
            titles=titles, axes=axes, cmap=cmap, colorbar=colorbar, mask=mask,
            mask_style=mask_style, mask_cmap=mask_cmap, mask_alpha=mask_alpha,
            time_unit=time_unit, show_names=show_names, group_by=group_by,
            sphere=sphere)

    @copy_function_doc_to_method_doc(plot_evoked_topo)
    def plot_topo(self, layout=None, layout_scale=0.945, color=None,
                  border='none', ylim=None, scalings=None, title=None,
                  proj=False, vline=[0.0], fig_background=None,
                  merge_grads=False, legend=True, axes=None,
                  background_color='w', noise_cov=None, show=True):
        """
        Notes
        -----
        .. versionadded:: 0.10.0
        """
        return plot_evoked_topo(
            self, layout=layout, layout_scale=layout_scale, color=color,
            border=border, ylim=ylim, scalings=scalings, title=title,
            proj=proj, vline=vline, fig_background=fig_background,
            merge_grads=merge_grads, legend=legend, axes=axes,
            background_color=background_color, noise_cov=noise_cov, show=show)

    @copy_function_doc_to_method_doc(plot_evoked_topomap)
    def plot_topomap(self, times="auto", ch_type=None, layout=None, vmin=None,
                     vmax=None, cmap=None, sensors=True, colorbar=True,
                     scalings=None, units=None, res=64,
                     size=1, cbar_fmt="%3.1f",
                     time_unit='s', time_format=None,
                     proj=False, show=True, show_names=False, title=None,
                     mask=None, mask_params=None, outlines='head',
                     contours=6, image_interp='bilinear', average=None,
                     head_pos=None, axes=None, extrapolate='box', sphere=None,
                     border=0, nrows=1, ncols='auto'):
        return plot_evoked_topomap(
            self, times=times, ch_type=ch_type, layout=layout, vmin=vmin,
            vmax=vmax, cmap=cmap, sensors=sensors, colorbar=colorbar,
            scalings=scalings, units=units, res=res,
            size=size, cbar_fmt=cbar_fmt, time_unit=time_unit,
            time_format=time_format, proj=proj, show=show,
            show_names=show_names, title=title, mask=mask,
            mask_params=mask_params, outlines=outlines, contours=contours,
            image_interp=image_interp, average=average, head_pos=head_pos,
            axes=axes, extrapolate=extrapolate, sphere=sphere, border=border,
            nrows=nrows, ncols=ncols)

    @copy_function_doc_to_method_doc(plot_evoked_field)
    def plot_field(self, surf_maps, time=None, time_label='t = %0.0f ms',
                   n_jobs=1, fig=None, verbose=None):
        return plot_evoked_field(self, surf_maps, time=time,
                                 time_label=time_label, n_jobs=n_jobs,
                                 fig=fig, verbose=verbose)

    @copy_function_doc_to_method_doc(plot_evoked_white)
    def plot_white(self, noise_cov, show=True, rank=None, time_unit='s',
                   sphere=None, verbose=None):
        return plot_evoked_white(
            self, noise_cov=noise_cov, rank=rank, show=show,
            time_unit=time_unit, sphere=sphere, verbose=verbose)

    @copy_function_doc_to_method_doc(plot_evoked_joint)
    def plot_joint(self, times="peaks", title='', picks=None,
                   exclude='bads', show=True, ts_args=None,
                   topomap_args=None):
        return plot_evoked_joint(self, times=times, title=title, picks=picks,
                                 exclude=exclude, show=show, ts_args=ts_args,
                                 topomap_args=topomap_args)

    @fill_doc
    def animate_topomap(self, ch_type=None, times=None, frame_rate=None,
                        butterfly=False, blit=True, show=True, time_unit='s',
                        sphere=None):
        """Make animation of evoked data as topomap timeseries.

        The animation can be paused/resumed with left mouse button.
        Left and right arrow keys can be used to move backward or forward
        in time.

        Parameters
        ----------
        ch_type : str | None
            Channel type to plot. Accepted data types: 'mag', 'grad', 'eeg',
            'hbo', 'hbr', 'fnirs_od, and 'fnirs_raw'.
            If None, first available channel type from ('mag', 'grad', 'eeg',
            'hbo', 'hbr', 'fnirs_od, 'fnirs_raw') is used. Defaults to None.
        times : array of float | None
            The time points to plot. If None, 10 evenly spaced samples are
            calculated over the evoked time series. Defaults to None.
        frame_rate : int | None
            Frame rate for the animation in Hz. If None,
            frame rate = sfreq / 10. Defaults to None.
        butterfly : bool
            Whether to plot the data as butterfly plot under the topomap.
            Defaults to False.
        blit : bool
            Whether to use blit to optimize drawing. In general, it is
            recommended to use blit in combination with ``show=True``. If you
            intend to save the animation it is better to disable blit.
            Defaults to True.
        show : bool
            Whether to show the animation. Defaults to True.
        time_unit : str
            The units for the time axis, can be "ms" (default in 0.16)
            or "s" (will become the default in 0.17).

            .. versionadded:: 0.16
        %(topomap_sphere_auto)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        anim : instance of matplotlib.animation.FuncAnimation
            Animation of the topomap.

        Notes
        -----
        .. versionadded:: 0.12.0
        """
        return _topomap_animation(
            self, ch_type=ch_type, times=times, frame_rate=frame_rate,
            butterfly=butterfly, blit=blit, show=show, time_unit=time_unit,
            sphere=sphere)

    def as_type(self, ch_type='grad', mode='fast'):
        """Compute virtual evoked using interpolated fields.

        .. Warning:: Using virtual evoked to compute inverse can yield
            unexpected results. The virtual channels have `'_v'` appended
            at the end of the names to emphasize that the data contained in
            them are interpolated.

        Parameters
        ----------
        ch_type : str
            The destination channel type. It can be 'mag' or 'grad'.
        mode : str
            Either `'accurate'` or `'fast'`, determines the quality of the
            Legendre polynomial expansion used. `'fast'` should be sufficient
            for most applications.

        Returns
        -------
        evoked : instance of mne.Evoked
            The transformed evoked object containing only virtual channels.

        Notes
        -----
        This method returns a copy and does not modify the data it
        operates on. It also returns an EvokedArraw instance.

        .. versionadded:: 0.9.0
        """
        from .forward import _as_meg_type_inst
        return _as_meg_type_inst(self, ch_type=ch_type, mode=mode)

    @fill_doc
    def detrend(self, order=1, picks=None):
        """Detrend data.

        This function operates in-place.

        Parameters
        ----------
        order : int
            Either 0 or 1, the order of the detrending. 0 is a constant
            (DC) detrend, 1 is a linear detrend.
        %(picks_good_data)s

        Returns
        -------
        evoked : instance of Evoked
            The detrended evoked object.
        """
        picks = _picks_to_idx(self.info, picks)
        self.data[picks] = detrend(self.data[picks], order, axis=-1)
        return self

    def copy(self):
        """Copy the instance of evoked.

        Returns
        -------
        evoked : instance of Evoked
            A copy of the object.
        """
        evoked = deepcopy(self)
        return evoked

    def __neg__(self):
        """Negate channel responses.

        Returns
        -------
        evoked_neg : instance of Evoked
            The Evoked instance with channel data negated and '-'
            prepended to the comment.
        """
        out = self.copy()
        out.data *= -1
        out.comment = '-' + (out.comment or 'unknown')
        return out

    def get_peak(self, ch_type=None, tmin=None, tmax=None,
                 mode='abs', time_as_index=False, merge_grads=False,
                 return_amplitude=False):
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        ch_type : 'mag', 'grad', 'eeg', 'seeg', 'ecog', 'hbo', hbr', 'misc', None
            The channel type to use. Defaults to None. If more than one sensor
            Type is present in the data the channel type has to be explicitly
            set.
        tmin : float | None
            The minimum point in time to be considered for peak getting.
            If None (default), the beginning of the data is used.
        tmax : float | None
            The maximum point in time to be considered for peak getting.
            If None (default), the end of the data is used.
        mode : {'pos', 'neg', 'abs'}
            How to deal with the sign of the data. If 'pos' only positive
            values will be considered. If 'neg' only negative values will
            be considered. If 'abs' absolute values will be considered.
            Defaults to 'abs'.
        time_as_index : bool
            Whether to return the time index instead of the latency in seconds.
        merge_grads : bool
            If True, compute peak from merged gradiometer data.
        return_amplitude : bool
            If True, return also the amplitude at the maximum response.

            .. versionadded:: 0.16

        Returns
        -------
        ch_name : str
            The channel exhibiting the maximum response.
        latency : float | int
            The time point of the maximum response, either latency in seconds
            or index.
        amplitude : float
            The amplitude of the maximum response. Only returned if
            return_amplitude is True.

            .. versionadded:: 0.16
        """  # noqa: E501
        supported = ('mag', 'grad', 'eeg', 'seeg', 'ecog', 'misc', 'hbo',
                     'hbr', 'None', 'fnirs_raw', 'fnirs_od')
        types_used = self.get_channel_types(unique=True, only_data_chs=True)

        _check_option('ch_type', str(ch_type), supported)

        if ch_type is not None and ch_type not in types_used:
            raise ValueError('Channel type `{ch_type}` not found in this '
                             'evoked object.'.format(ch_type=ch_type))

        elif len(types_used) > 1 and ch_type is None:
            raise RuntimeError('More than one sensor type found. `ch_type` '
                               'must not be `None`, pass a sensor type '
                               'value instead')

        if merge_grads:
            if ch_type != 'grad':
                raise ValueError('Channel type must be grad for merge_grads')
            elif mode == 'neg':
                raise ValueError('Negative mode (mode=neg) does not make '
                                 'sense with merge_grads=True')

        meg = eeg = misc = seeg = ecog = fnirs = False
        picks = None
        if ch_type in ('mag', 'grad'):
            meg = ch_type
        elif ch_type == 'eeg':
            eeg = True
        elif ch_type == 'misc':
            misc = True
        elif ch_type == 'seeg':
            seeg = True
        elif ch_type == 'ecog':
            ecog = True
        elif ch_type in ('hbo', 'hbr', 'fnirs_raw', 'fnirs_od'):
            fnirs = ch_type

        if ch_type is not None:
            if merge_grads:
                picks = _pair_grad_sensors(self.info, topomap_coords=False)
            else:
                picks = pick_types(self.info, meg=meg, eeg=eeg, misc=misc,
                                   seeg=seeg, ecog=ecog, ref_meg=False,
                                   fnirs=fnirs)
        data = self.data
        ch_names = self.ch_names

        if picks is not None:
            data = data[picks]
            ch_names = [ch_names[k] for k in picks]

        if merge_grads:
            data, _ = _merge_ch_data(data, ch_type, [])
            ch_names = [ch_name[:-1] + 'X' for ch_name in ch_names[::2]]

        ch_idx, time_idx, max_amp = _get_peak(data, self.times, tmin,
                                              tmax, mode)

        out = (ch_names[ch_idx], time_idx if time_as_index else
               self.times[time_idx])

        if return_amplitude:
            out += (max_amp,)

        return out

    @fill_doc
    def to_data_frame(self, picks=None, index=None, scaling_time=None,
                      scalings=None, copy=True, long_format=False,
                      time_format='ms'):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        an additional column "time" is added, unless ``index='time'``
        (in which case time values form the DataFrame's index).

        Parameters
        ----------
        %(picks_all)s
        %(df_index_evk)s
            Defaults to ``None``.
        %(df_scaling_time_deprecated)s
        %(df_scalings)s
        %(df_copy)s
        %(df_longform_raw)s
        %(df_time_format)s

            .. versionadded:: 0.20

        Returns
        -------
        %(df_return)s
        """
        # check deprecation
        _check_scaling_time(scaling_time)
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ['time']
        valid_time_formats = ['ms', 'timedelta']
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        data = self.data[picks, :]
        times = self.times
        data = data.T
        if copy:
            data = data.copy()
        data = _scale_dataframe_data(self, data, picks, scalings)
        # prepare extra columns / multiindex
        mindex = list()
        times = _convert_times(self, times, time_format)
        mindex.append(('time', times))
        # build DataFrame
        df = _build_data_frame(self, data, picks, long_format, mindex, index,
                               default_index=['time'])
        return df


def _check_decim(info, decim, offset):
    """Check decimation parameters."""
    if decim < 1 or decim != int(decim):
        raise ValueError('decim must be an integer > 0')
    decim = int(decim)
    new_sfreq = info['sfreq'] / float(decim)
    lowpass = info['lowpass']
    if decim > 1 and lowpass is None:
        warn('The measurement information indicates data is not low-pass '
             'filtered. The decim=%i parameter will result in a sampling '
             'frequency of %g Hz, which can cause aliasing artifacts.'
             % (decim, new_sfreq))
    elif decim > 1 and new_sfreq < 3 * lowpass:
        warn('The measurement information indicates a low-pass frequency '
             'of %g Hz. The decim=%i parameter will result in a sampling '
             'frequency of %g Hz, which can cause aliasing artifacts.'
             % (lowpass, decim, new_sfreq))  # > 50% nyquist lim
    offset = int(offset)
    if not 0 <= offset < decim:
        raise ValueError('decim must be at least 0 and less than %s, got '
                         '%s' % (decim, offset))
    return decim, offset, new_sfreq


@fill_doc
class EvokedArray(Evoked):
    """Evoked object from numpy array.

    Parameters
    ----------
    data : array of shape (n_channels, n_times)
        The channels' evoked response. See notes for proper units of measure.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    tmin : float
        Start time before event. Defaults to 0.
    comment : str
        Comment on dataset. Can be the condition. Defaults to ''.
    nave : int
        Number of averaged epochs. Defaults to 1.
    kind : str
        Type of data, either average or standard_error. Defaults to 'average'.
    %(verbose)s

    See Also
    --------
    EpochsArray, io.RawArray, create_info

    Notes
    -----
    Proper units of measure:
    * V: eeg, eog, seeg, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc
    """

    @verbose
    def __init__(self, data, info, tmin=0., comment='', nave=1, kind='average',
                 verbose=None):  # noqa: D102
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)

        if data.ndim != 2:
            raise ValueError('Data must be a 2D array of shape (n_channels, '
                             'n_samples), got shape %s' % (data.shape,))

        if len(info['ch_names']) != np.shape(data)[0]:
            raise ValueError('Info (%s) and data (%s) must have same number '
                             'of channels.' % (len(info['ch_names']),
                                               np.shape(data)[0]))

        self.data = data

        self.first = int(round(tmin * info['sfreq']))
        self.last = self.first + np.shape(data)[-1] - 1
        self.times = np.arange(self.first, self.last + 1,
                               dtype=np.float) / info['sfreq']
        self.info = info.copy()  # do not modify original info
        self.nave = nave
        self.kind = kind
        self.comment = comment
        self.picks = None
        self.verbose = verbose
        self.preload = True
        self._projector = None
        _validate_type(self.kind, "str", "kind")
        if self.kind not in _aspect_dict:
            raise ValueError('unknown kind "%s", should be "average" or '
                             '"standard_error"' % (self.kind,))
        self._aspect_kind = _aspect_dict[self.kind]


def _get_entries(fid, evoked_node, allow_maxshield=False):
    """Get all evoked entries."""
    comments = list()
    aspect_kinds = list()
    for ev in evoked_node:
        for k in range(ev['nent']):
            my_kind = ev['directory'][k].kind
            pos = ev['directory'][k].pos
            if my_kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comments.append(tag.data)
        my_aspect = _get_aspect(ev, allow_maxshield)[0]
        for k in range(my_aspect['nent']):
            my_kind = my_aspect['directory'][k].kind
            pos = my_aspect['directory'][k].pos
            if my_kind == FIFF.FIFF_ASPECT_KIND:
                tag = read_tag(fid, pos)
                aspect_kinds.append(int(tag.data))
    comments = np.atleast_1d(comments)
    aspect_kinds = np.atleast_1d(aspect_kinds)
    if len(comments) != len(aspect_kinds) or len(comments) == 0:
        fid.close()
        raise ValueError('Dataset names in FIF file '
                         'could not be found.')
    t = [_aspect_rev[a] for a in aspect_kinds]
    t = ['"' + c + '" (' + tt + ')' for tt, c in zip(t, comments)]
    t = '\n'.join(t)
    return comments, aspect_kinds, t


def _get_aspect(evoked, allow_maxshield):
    """Get Evoked data aspect."""
    is_maxshield = False
    aspect = dir_tree_find(evoked, FIFF.FIFFB_ASPECT)
    if len(aspect) == 0:
        _check_maxshield(allow_maxshield)
        aspect = dir_tree_find(evoked, FIFF.FIFFB_IAS_ASPECT)
        is_maxshield = True
    if len(aspect) > 1:
        logger.info('Multiple data aspects found. Taking first one.')
    return aspect[0], is_maxshield


def _get_evoked_node(fname):
    """Get info in evoked file."""
    f, tree, _ = fiff_open(fname)
    with f as fid:
        _, meas = read_meas_info(fid, tree, verbose=False)
        evoked_node = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
    return evoked_node


def _check_evokeds_ch_names_times(all_evoked):
    evoked = all_evoked[0]
    ch_names = evoked.ch_names
    for ii, ev in enumerate(all_evoked[1:]):
        if ev.ch_names != ch_names:
            if set(ev.ch_names) != set(ch_names):
                raise ValueError(
                    "%s and %s do not contain the same channels." % (evoked,
                                                                     ev))
            else:
                warn("Order of channels differs, reordering channels ...")
                ev = ev.copy()
                ev.reorder_channels(ch_names)
                all_evoked[ii + 1] = ev
        if not np.max(np.abs(ev.times - evoked.times)) < 1e-7:
            raise ValueError("%s and %s do not contain the same time instants"
                             % (evoked, ev))
    return all_evoked


def combine_evoked(all_evoked, weights):
    """Merge evoked data by weighted addition or subtraction.

    Data should have the same channels and the same time instants.
    Subtraction can be performed by calling
    ``combine_evoked([evoked1, -evoked2], 'equal')``

    .. Warning::
        If you provide an array of weights instead of using `'equal'` or
        `'nave'`, strange things may happen with your resulting signal
        amplitude and/or `.nave` attribute.

    Parameters
    ----------
    all_evoked : list of Evoked
        The evoked datasets.
    weights : list of float | str
        The weights to apply to the data of each evoked instance.
        Can also be ``'nave'`` to weight according to evoked.nave,
        or ``"equal"`` to use equal weighting (each weighted as ``1/N``).

    Returns
    -------
    evoked : Evoked
        The new evoked data.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    naves = np.array([evk.nave for evk in all_evoked], float)
    if isinstance(weights, str):
        _check_option('weights', weights, ['nave', 'equal'])
        if weights == 'nave':
            weights = naves / naves.sum()
        else:
            weights = np.ones_like(naves)
    else:
        weights = np.array(weights, float)

    if weights.ndim != 1 or weights.size != len(all_evoked):
        raise ValueError('weights must be the same size as all_evoked')

    # cf. https://en.wikipedia.org/wiki/Weighted_arithmetic_mean, section on
    # how variances change when summing Gaussian random variables. The variance
    # of a weighted sample mean is:
    #
    #    σ² = w₁² σ₁² + w₂² σ₂² + ... + wₙ² σₙ²
    #
    # We estimate the variance of each evoked instance as 1 / nave to get:
    #
    #    σ² = w₁² / nave₁ + w₂² / nave₂ + ... + wₙ² / naveₙ
    #
    # And our resulting nave is the reciprocal of this:
    new_nave = 1. / np.sum(weights ** 2 / naves)
    # This general formula is equivalent to formulae in Matti's manual
    # (pp 128-129), where:
    # new_nave = sum(naves) when weights='nave' and
    # new_nave = 1. / sum(1. / naves) when weights='equal'

    all_evoked = _check_evokeds_ch_names_times(all_evoked)
    evoked = all_evoked[0].copy()

    # use union of bad channels
    bads = list(set(evoked.info['bads']).union(*(ev.info['bads']
                                                 for ev in all_evoked[1:])))
    evoked.info['bads'] = bads
    evoked.data = sum(w * e.data for w, e in zip(weights, all_evoked))
    evoked.nave = new_nave
    evoked.comment = ' + '.join('%0.3f * %s' % (w, e.comment or 'unknown')
                                for w, e in zip(weights, all_evoked))
    return evoked


@verbose
def read_evokeds(fname, condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    """Read evoked dataset(s).

    Parameters
    ----------
    fname : str
        The file name, which should end with -ave.fif or -ave.fif.gz.
    condition : int or str | list of int or str | None
        The index or list of indices of the evoked dataset to read. FIF files
        can contain multiple datasets. If None, all datasets are returned as a
        list.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    kind : str
        Either 'average' or 'standard_error', the type of data to read.
    proj : bool
        If False, available projectors won't be applied to the data.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    %(verbose)s

    Returns
    -------
    evoked : Evoked or list of Evoked
        The evoked dataset(s); one Evoked if condition is int or str,
        or list of Evoked if condition is None or list.

    See Also
    --------
    write_evokeds
    """
    check_fname(fname, 'evoked', ('-ave.fif', '-ave.fif.gz',
                                  '_ave.fif', '_ave.fif.gz'))
    logger.info('Reading %s ...' % fname)
    return_list = True
    if condition is None:
        evoked_node = _get_evoked_node(fname)
        condition = range(len(evoked_node))
    elif not isinstance(condition, list):
        condition = [condition]
        return_list = False

    out = [Evoked(fname, c, kind=kind, proj=proj,
                  allow_maxshield=allow_maxshield,
                  verbose=verbose).apply_baseline(baseline)
           for c in condition]

    return out if return_list else out[0]


def _read_evoked(fname, condition=None, kind='average', allow_maxshield=False):
    """Read evoked data from a FIF file."""
    if fname is None:
        raise ValueError('No evoked filename specified')

    f, tree, _ = fiff_open(fname)
    with f as fid:
        #   Read the measurement info
        info, meas = read_meas_info(fid, tree, clean_bads=True)

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        if len(processed) == 0:
            raise ValueError('Could not find processed data')

        evoked_node = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
        if len(evoked_node) == 0:
            raise ValueError('Could not find evoked data')

        # find string-based entry
        if isinstance(condition, str):
            if kind not in _aspect_dict.keys():
                raise ValueError('kind must be "average" or '
                                 '"standard_error"')

            comments, aspect_kinds, t = _get_entries(fid, evoked_node,
                                                     allow_maxshield)
            goods = (np.in1d(comments, [condition]) &
                     np.in1d(aspect_kinds, [_aspect_dict[kind]]))
            found_cond = np.where(goods)[0]
            if len(found_cond) != 1:
                raise ValueError('condition "%s" (%s) not found, out of '
                                 'found datasets:\n%s'
                                 % (condition, kind, t))
            condition = found_cond[0]
        elif condition is None:
            if len(evoked_node) > 1:
                _, _, conditions = _get_entries(fid, evoked_node,
                                                allow_maxshield)
                raise TypeError("Evoked file has more than one "
                                "condition, the condition parameters "
                                "must be specified from:\n%s" % conditions)
            else:
                condition = 0

        if condition >= len(evoked_node) or condition < 0:
            raise ValueError('Data set selector out of range')

        my_evoked = evoked_node[condition]

        # Identify the aspects
        my_aspect, info['maxshield'] = _get_aspect(my_evoked, allow_maxshield)

        # Now find the data in the evoked block
        nchan = 0
        sfreq = -1
        chs = []
        comment = last = first = first_time = nsamp = None
        for k in range(my_evoked['nent']):
            my_kind = my_evoked['directory'][k].kind
            pos = my_evoked['directory'][k].pos
            if my_kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comment = tag.data
            elif my_kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, pos)
                first = int(tag.data)
            elif my_kind == FIFF.FIFF_LAST_SAMPLE:
                tag = read_tag(fid, pos)
                last = int(tag.data)
            elif my_kind == FIFF.FIFF_NCHAN:
                tag = read_tag(fid, pos)
                nchan = int(tag.data)
            elif my_kind == FIFF.FIFF_SFREQ:
                tag = read_tag(fid, pos)
                sfreq = float(tag.data)
            elif my_kind == FIFF.FIFF_CH_INFO:
                tag = read_tag(fid, pos)
                chs.append(tag.data)
            elif my_kind == FIFF.FIFF_FIRST_TIME:
                tag = read_tag(fid, pos)
                first_time = float(tag.data)
            elif my_kind == FIFF.FIFF_NO_SAMPLES:
                tag = read_tag(fid, pos)
                nsamp = int(tag.data)

        if comment is None:
            comment = 'No comment'

        #   Local channel information?
        if nchan > 0:
            if chs is None:
                raise ValueError('Local channel information was not found '
                                 'when it was expected.')

            if len(chs) != nchan:
                raise ValueError('Number of channels and number of '
                                 'channel definitions are different')

            info['chs'] = chs
            logger.info('    Found channel information in evoked data. '
                        'nchan = %d' % nchan)
            if sfreq > 0:
                info['sfreq'] = sfreq

        # Read the data in the aspect block
        nave = 1
        epoch = []
        for k in range(my_aspect['nent']):
            kind = my_aspect['directory'][k].kind
            pos = my_aspect['directory'][k].pos
            if kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comment = tag.data
            elif kind == FIFF.FIFF_ASPECT_KIND:
                tag = read_tag(fid, pos)
                aspect_kind = int(tag.data)
            elif kind == FIFF.FIFF_NAVE:
                tag = read_tag(fid, pos)
                nave = int(tag.data)
            elif kind == FIFF.FIFF_EPOCH:
                tag = read_tag(fid, pos)
                epoch.append(tag)

        nepoch = len(epoch)
        if nepoch != 1 and nepoch != info['nchan']:
            raise ValueError('Number of epoch tags is unreasonable '
                             '(nepoch = %d nchan = %d)'
                             % (nepoch, info['nchan']))

        if nepoch == 1:
            # Only one epoch
            data = epoch[0].data
            # May need a transpose if the number of channels is one
            if data.shape[1] == 1 and info['nchan'] == 1:
                data = data.T
        else:
            # Put the old style epochs together
            data = np.concatenate([e.data[None, :] for e in epoch], axis=0)
        data = data.astype(np.float)

        if first_time is not None and nsamp is not None:
            times = first_time + np.arange(nsamp) / info['sfreq']
        elif first is not None:
            nsamp = last - first + 1
            times = np.arange(first, last + 1) / info['sfreq']
        else:
            raise RuntimeError('Could not read time parameters')
        del first, last
        if nsamp is not None and data.shape[1] != nsamp:
            raise ValueError('Incorrect number of samples (%d instead of '
                             ' %d)' % (data.shape[1], nsamp))
        logger.info('    Found the data of interest:')
        logger.info('        t = %10.2f ... %10.2f ms (%s)'
                    % (1000 * times[0], 1000 * times[-1], comment))
        if info['comps'] is not None:
            logger.info('        %d CTF compensation matrices available'
                        % len(info['comps']))
        logger.info('        nave = %d - aspect type = %d'
                    % (nave, aspect_kind))

    # Calibrate
    cals = np.array([info['chs'][k]['cal'] *
                     info['chs'][k].get('scale', 1.0)
                     for k in range(info['nchan'])])
    data *= cals[:, np.newaxis]

    return info, nave, aspect_kind, comment, times, data


def write_evokeds(fname, evoked):
    """Write an evoked dataset to a file.

    Parameters
    ----------
    fname : str
        The file name, which should end with -ave.fif or -ave.fif.gz.
    evoked : Evoked instance, or list of Evoked instances
        The evoked dataset, or list of evoked datasets, to save in one file.
        Note that the measurement info from the first evoked instance is used,
        so be sure that information matches.

    See Also
    --------
    read_evokeds
    """
    _write_evokeds(fname, evoked)


def _write_evokeds(fname, evoked, check=True):
    """Write evoked data."""
    if check:
        check_fname(fname, 'evoked', ('-ave.fif', '-ave.fif.gz',
                                      '_ave.fif', '_ave.fif.gz'))

    if not isinstance(evoked, list):
        evoked = [evoked]

    warned = False
    # Create the file and save the essentials
    with start_file(fname) as fid:

        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        if evoked[0].info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, evoked[0].info['meas_id'])

        # Write measurement info
        write_meas_info(fid, evoked[0].info)

        # One or more evoked data sets
        start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        for e in evoked:
            start_block(fid, FIFF.FIFFB_EVOKED)

            # Comment is optional
            if e.comment is not None and len(e.comment) > 0:
                write_string(fid, FIFF.FIFF_COMMENT, e.comment)

            # First time, num. samples, first and last sample
            write_float(fid, FIFF.FIFF_FIRST_TIME, e.times[0])
            write_int(fid, FIFF.FIFF_NO_SAMPLES, len(e.times))
            write_int(fid, FIFF.FIFF_FIRST_SAMPLE, e.first)
            write_int(fid, FIFF.FIFF_LAST_SAMPLE, e.last)

            # The epoch itself
            if e.info.get('maxshield'):
                aspect = FIFF.FIFFB_IAS_ASPECT
            else:
                aspect = FIFF.FIFFB_ASPECT
            start_block(fid, aspect)

            write_int(fid, FIFF.FIFF_ASPECT_KIND, e._aspect_kind)
            # convert nave to integer to comply with FIFF spec
            nave_int = int(round(e.nave))
            if nave_int != e.nave and not warned:
                warn('converting "nave" to integer before saving evoked; this '
                     'can have a minor effect on the scale of source '
                     'estimates that are computed using "nave".')
                warned = True
            write_int(fid, FIFF.FIFF_NAVE, nave_int)
            del nave_int

            decal = np.zeros((e.info['nchan'], 1))
            for k in range(e.info['nchan']):
                decal[k] = 1.0 / (e.info['chs'][k]['cal'] *
                                  e.info['chs'][k].get('scale', 1.0))

            write_float_matrix(fid, FIFF.FIFF_EPOCH, decal * e.data)
            end_block(fid, aspect)
            end_block(fid, FIFF.FIFFB_EVOKED)

        end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        end_block(fid, FIFF.FIFFB_MEAS)
        end_file(fid)


def _get_peak(data, times, tmin=None, tmax=None, mode='abs'):
    """Get feature-index and time of maximum signal from 2D array.

    Note. This is a 'getter', not a 'finder'. For non-evoked type
    data and continuous signals, please use proper peak detection algorithms.

    Parameters
    ----------
    data : instance of numpy.ndarray (n_locations, n_times)
        The data, either evoked in sensor or source space.
    times : instance of numpy.ndarray (n_times)
        The times in seconds.
    tmin : float | None
        The minimum point in time to be considered for peak getting.
    tmax : float | None
        The maximum point in time to be considered for peak getting.
    mode : {'pos', 'neg', 'abs'}
        How to deal with the sign of the data. If 'pos' only positive
        values will be considered. If 'neg' only negative values will
        be considered. If 'abs' absolute values will be considered.
        Defaults to 'abs'.

    Returns
    -------
    max_loc : int
        The index of the feature with the maximum value.
    max_time : int
        The time point of the maximum response, index.
    max_amp : float
        Amplitude of the maximum response.
    """
    _check_option('mode', mode, ['abs', 'neg', 'pos'])

    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    if tmin < times.min():
        raise ValueError('The tmin value is out of bounds. It must be '
                         'within {} and {}'.format(times.min(), times.max()))
    if tmax > times.max():
        raise ValueError('The tmax value is out of bounds. It must be '
                         'within {} and {}'.format(times.min(), times.max()))
    if tmin > tmax:
        raise ValueError('The tmin must be smaller or equal to tmax')

    time_win = (times >= tmin) & (times <= tmax)
    mask = np.ones_like(data).astype(np.bool)
    mask[:, time_win] = False

    maxfun = np.argmax
    if mode == 'pos':
        if not np.any(data > 0):
            raise ValueError('No positive values encountered. Cannot '
                             'operate in pos mode.')
    elif mode == 'neg':
        if not np.any(data < 0):
            raise ValueError('No negative values encountered. Cannot '
                             'operate in neg mode.')
        maxfun = np.argmin

    masked_index = np.ma.array(np.abs(data) if mode == 'abs' else data,
                               mask=mask)

    max_loc, max_time = np.unravel_index(maxfun(masked_index), data.shape)

    return max_loc, max_time, data[max_loc, max_time]
