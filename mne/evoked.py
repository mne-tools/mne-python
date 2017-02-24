# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Mads Jensen <mje.mads@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np

from .baseline import rescale
from .channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                SetChannelsMixin, InterpolationMixin,
                                equalize_channels)
from .filter import resample, detrend, FilterMixin
from .utils import (check_fname, logger, verbose, _time_mask, warn, sizeof_fmt,
                    SizeMixin, copy_function_doc_to_method_doc)
from .viz import (plot_evoked, plot_evoked_topomap, plot_evoked_field,
                  plot_evoked_image, plot_evoked_topo)
from .viz.evoked import (_plot_evoked_white, plot_evoked_joint,
                         _animate_evoked_topomap)

from .externals.six import string_types

from .io.constants import FIFF
from .io.open import fiff_open
from .io.tag import read_tag
from .io.tree import dir_tree_find
from .io.pick import channel_type, pick_types, _pick_data_channels
from .io.meas_info import read_meas_info, write_meas_info
from .io.proj import ProjMixin
from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_string, write_float_matrix,
                       write_id)
from .io.base import ToDataFrameMixin, TimeMixin, _check_maxshield

_aspect_dict = {'average': FIFF.FIFFV_ASPECT_AVERAGE,
                'standard_error': FIFF.FIFFV_ASPECT_STD_ERR}
_aspect_rev = {str(FIFF.FIFFV_ASPECT_AVERAGE): 'average',
               str(FIFF.FIFFV_ASPECT_STD_ERR): 'standard_error'}


class Evoked(ProjMixin, ContainsMixin, UpdateChannelsMixin,
             SetChannelsMixin, InterpolationMixin, FilterMixin,
             ToDataFrameMixin, TimeMixin, SizeMixin):
    """Evoked data.

    Parameters
    ----------
    fname : string
        Name of evoked/average FIF file to load.
        If None no data is loaded.
    condition : int, or str
        Dataset ID number (int) or comment/name (str). Optional if there is
        only one data set in file.
    proj : bool, optional
        Apply SSP projection vectors
    kind : str
        Either 'average' or 'standard_error'. The type of data to read.
        Only used if 'condition' is a str.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Attributes
    ----------
    info : dict
        Measurement info.
    ch_names : list of string
        List of channels' names.
    nave : int
        Number of averaged epochs.
    kind : str
        Type of data, either average or standard_error.
    first : int
        First time sample.
    last : int
        Last time sample.
    comment : string
        Comment on dataset. Can be the condition.
    times : array
        Array of time instants in seconds.
    data : array of shape (n_channels, n_times)
        Evoked response.
    verbose : bool, str, int, or None.
        See above.

    Notes
    -----
    Evoked objects contain a single condition only.
    """

    @verbose
    def __init__(self, fname, condition=None, proj=True,
                 kind='average', allow_maxshield=False,
                 verbose=None):  # noqa: D102
        if not isinstance(proj, bool):
            raise ValueError(r"'proj' must be 'True' or 'False'")
        # Read the requested data
        self.info, self.nave, self._aspect_kind, self.first, self.last, \
            self.comment, self.times, self.data = _read_evoked(
                fname, condition, kind, allow_maxshield)
        self.kind = _aspect_rev.get(str(self._aspect_kind), 'Unknown')
        self.verbose = verbose
        self.preload = True
        # project and baseline correct
        if proj:
            self.apply_proj()

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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

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
        fname : string
            Name of the file where to save the data.

        Notes
        -----
        To write multiple conditions into a single file, use
        :func:`mne.write_evokeds`.
        """
        write_evokeds(fname, self)

    def __repr__(self):  # noqa: D105
        s = "comment : '%s'" % self.comment
        s += ', kind : %s' % self.kind
        s += ", time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", n_epochs : %d" % self.nave
        s += ", n_channels x n_times : %s x %s" % self.data.shape
        s += ", ~%s" % (sizeof_fmt(self._size),)
        return "<Evoked  |  %s>" % s

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    def crop(self, tmin=None, tmax=None):
        """Crop data to a given time interval.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.

        Returns
        -------
        evoked : instance of Evoked
            The cropped Evoked object.

        Notes
        -----
        Unlike Python slices, MNE time intervals include both their end points;
        crop(tmin, tmax) returns the interval tmin <= t <= tmax.
        """
        mask = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'])
        self.times = self.times[mask]
        self.first = int(self.times[0] * self.info['sfreq'])
        self.last = len(self.times) + self.first - 1
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
        return self

    def shift_time(self, tshift, relative=True):
        """Shift time scale in evoked data.

        Parameters
        ----------
        tshift : float
            The amount of time shift to be applied if relative is True
            else the first time point. When relative is True, positive value
            of tshift moves the data forward while negative tshift moves it
            backward.
        relative : bool
            If true, move the time backwards or forwards by specified amount.
            Else, set the starting time point to the value of tshift.

        Notes
        -----
        Maximum accuracy of time shift is 1 / evoked.info['sfreq']
        """
        times = self.times
        sfreq = self.info['sfreq']

        offset = self.first if relative else 0

        self.first = int(tshift * sfreq) + offset
        self.last = self.first + len(times) - 1
        self.times = np.arange(self.first, self.last + 1,
                               dtype=np.float) / sfreq

    @copy_function_doc_to_method_doc(plot_evoked)
    def plot(self, picks=None, exclude='bads', unit=True, show=True, ylim=None,
             xlim='tight', proj=False, hline=None, units=None, scalings=None,
             titles=None, axes=None, gfp=False, window_title=None,
             spatial_colors=False, zorder='unsorted', selectable=True):
        return plot_evoked(
            self, picks=picks, exclude=exclude, unit=unit, show=show,
            ylim=ylim, proj=proj, xlim=xlim, hline=hline, units=units,
            scalings=scalings, titles=titles, axes=axes, gfp=gfp,
            window_title=window_title, spatial_colors=spatial_colors,
            zorder=zorder, selectable=selectable)

    @copy_function_doc_to_method_doc(plot_evoked_image)
    def plot_image(self, picks=None, exclude='bads', unit=True, show=True,
                   clim=None, xlim='tight', proj=False, units=None,
                   scalings=None, titles=None, axes=None, cmap='RdBu_r'):
        return plot_evoked_image(self, picks=picks, exclude=exclude, unit=unit,
                                 show=show, clim=clim, proj=proj, xlim=xlim,
                                 units=units, scalings=scalings,
                                 titles=titles, axes=axes, cmap=cmap)

    @copy_function_doc_to_method_doc(plot_evoked_topo)
    def plot_topo(self, layout=None, layout_scale=0.945, color=None,
                  border='none', ylim=None, scalings=None, title=None,
                  proj=False, vline=[0.0], fig_facecolor='k',
                  fig_background=None, axis_facecolor='k', font_color='w',
                  merge_grads=False, show=True):
        """

        Notes
        -----
        .. versionadded:: 0.10.0
        """
        return plot_evoked_topo(self, layout=layout, layout_scale=layout_scale,
                                color=color, border=border, ylim=ylim,
                                scalings=scalings, title=title, proj=proj,
                                vline=vline, fig_facecolor=fig_facecolor,
                                fig_background=fig_background,
                                axis_facecolor=axis_facecolor,
                                font_color=font_color, merge_grads=merge_grads,
                                show=show)

    @copy_function_doc_to_method_doc(plot_evoked_topomap)
    def plot_topomap(self, times="auto", ch_type=None, layout=None, vmin=None,
                     vmax=None, cmap=None, sensors=True, colorbar=True,
                     scale=None, scale_time=1e3, unit=None, res=64, size=1,
                     cbar_fmt="%3.1f", time_format='%01d ms', proj=False,
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None,
                     axes=None):
        return plot_evoked_topomap(self, times=times, ch_type=ch_type,
                                   layout=layout, vmin=vmin, vmax=vmax,
                                   cmap=cmap, sensors=sensors,
                                   colorbar=colorbar, scale=scale,
                                   scale_time=scale_time, unit=unit, res=res,
                                   proj=proj, size=size, cbar_fmt=cbar_fmt,
                                   time_format=time_format, show=show,
                                   show_names=show_names, title=title,
                                   mask=mask, mask_params=mask_params,
                                   outlines=outlines, contours=contours,
                                   image_interp=image_interp, average=average,
                                   head_pos=head_pos, axes=axes)

    @copy_function_doc_to_method_doc(plot_evoked_field)
    def plot_field(self, surf_maps, time=None, time_label='t = %0.0f ms',
                   n_jobs=1):
        return plot_evoked_field(self, surf_maps, time=time,
                                 time_label=time_label, n_jobs=n_jobs)

    def plot_white(self, noise_cov, show=True):
        """Plot whitened evoked response.

        Plots the whitened evoked response and the whitened GFP as described in
        [1]_. If one single covariance object is passed, the GFP panel (bottom)
        will depict different sensor types. If multiple covariance objects are
        passed as a list, the left column will display the whitened evoked
        responses for each channel based on the whitener from the noise
        covariance that has the highest log-likelihood. The left column will
        depict the whitened GFPs based on each estimator separately for each
        sensor type. Instead of numbers of channels the GFP display shows the
        estimated rank. The rank estimation will be printed by the logger for
        each noise covariance estimator that is passed.


        Parameters
        ----------
        noise_cov : list | instance of Covariance | str
            The noise covariance as computed by ``mne.cov.compute_covariance``.
        show : bool
            Whether to show the figure or not. Defaults to True.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure object containing the plot.

        References
        ----------
        .. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
               covariance estimation and spatial whitening of MEG and EEG
               signals, vol. 108, 328-342, NeuroImage.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        return _plot_evoked_white(self, noise_cov=noise_cov, scalings=None,
                                  rank=None, show=show)

    @copy_function_doc_to_method_doc(plot_evoked_joint)
    def plot_joint(self, times="peaks", title='', picks=None,
                   exclude='bads', show=True, ts_args=None,
                   topomap_args=None):
        return plot_evoked_joint(self, times=times, title=title, picks=picks,
                                 exclude=exclude, show=show, ts_args=ts_args,
                                 topomap_args=topomap_args)

    def animate_topomap(self, ch_type='mag', times=None, frame_rate=None,
                        butterfly=False, blit=True, show=True):
        """Make animation of evoked data as topomap timeseries.

        The animation can be paused/resumed with left mouse button.
        Left and right arrow keys can be used to move backward or forward
        in time.

        Parameters
        ----------
        ch_type : str | None
            Channel type to plot. Accepted data types: 'mag', 'grad', 'eeg'.
            If None, first available channel type from ('mag', 'grad', 'eeg')
            is used. Defaults to None.
        times : array of floats | None
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

        Returns
        -------
        fig : instance of matplotlib figure
            The figure.
        anim : instance of matplotlib FuncAnimation
            Animation of the topomap.

        Notes
        -----
        .. versionadded:: 0.12.0
        """
        return _animate_evoked_topomap(self, ch_type=ch_type, times=times,
                                       frame_rate=frame_rate,
                                       butterfly=butterfly, blit=blit,
                                       show=show)

    def as_type(self, ch_type='grad', mode='fast'):
        """Compute virtual evoked using interpolated fields.

        .. Warning:: Using virtual evoked to compute inverse can yield
            unexpected results. The virtual channels have `'_virtual'` appended
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
        .. versionadded:: 0.9.0
        """
        from .forward import _as_meg_type_evoked
        return _as_meg_type_evoked(self, ch_type=ch_type, mode=mode)

    def resample(self, sfreq, npad='auto', window='boxcar'):
        """Resample data.

        This function operates in-place.

        Parameters
        ----------
        sfreq : float
            New sample rate to use
        npad : int | str
            Amount to pad the start and end of the data.
            Can also be "auto" to use a padding that will result in
            a power-of-two size (can be much faster).
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.

        Returns
        -------
        evoked : instance of mne.Evoked
            The resampled evoked object.
        """
        sfreq = float(sfreq)
        o_sfreq = self.info['sfreq']
        self.data = resample(self.data, sfreq, o_sfreq, npad, -1, window)
        # adjust indirectly affected variables
        self.info['sfreq'] = sfreq
        self.times = (np.arange(self.data.shape[1], dtype=np.float) / sfreq +
                      self.times[0])
        self.first = int(self.times[0] * self.info['sfreq'])
        self.last = len(self.times) + self.first - 1
        return self

    def detrend(self, order=1, picks=None):
        """Detrend data.

        This function operates in-place.

        Parameters
        ----------
        order : int
            Either 0 or 1, the order of the detrending. 0 is a constant
            (DC) detrend, 1 is a linear detrend.
        picks : array-like of int | None
            If None only MEG, EEG, SEEG, ECoG and fNIRS channels are detrended.

        Returns
        -------
        evoked : instance of Evoked
            The detrended evoked object.
        """
        if picks is None:
            picks = _pick_data_channels(self.info)
        self.data[picks] = detrend(self.data[picks], order, axis=-1)
        return self

    def copy(self):
        """Copy the instance of evoked.

        Returns
        -------
        evoked : instance of Evoked
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

    def get_peak(self, ch_type=None, tmin=None, tmax=None, mode='abs',
                 time_as_index=False):
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        ch_type : 'mag', 'grad', 'eeg', 'seeg', 'ecog', 'hbo', hbr', 'misc', None  # noqa
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

        Returns
        -------
        ch_name : str
            The channel exhibiting the maximum response.
        latency : float | int
            The time point of the maximum response, either latency in seconds
            or index.
        """
        supported = ('mag', 'grad', 'eeg', 'seeg', 'ecog', 'misc', 'hbo',
                     'hbr', 'None')
        data_picks = _pick_data_channels(self.info, with_ref_meg=False)
        types_used = set([channel_type(self.info, idx) for idx in data_picks])

        if str(ch_type) not in supported:
            raise ValueError('Channel type must be `{supported}`. You gave me '
                             '`{ch_type}` instead.'
                             .format(ch_type=ch_type,
                                     supported='` or `'.join(supported)))

        elif ch_type is not None and ch_type not in types_used:
            raise ValueError('Channel type `{ch_type}` not found in this '
                             'evoked object.'.format(ch_type=ch_type))

        elif len(types_used) > 1 and ch_type is None:
            raise RuntimeError('More than one sensor type found. `ch_type` '
                               'must not be `None`, pass a sensor type '
                               'value instead')

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
        elif ch_type in ('hbo', 'hbr'):
            fnirs = ch_type

        if ch_type is not None:
            picks = pick_types(self.info, meg=meg, eeg=eeg, misc=misc,
                               seeg=seeg, ecog=ecog, ref_meg=False,
                               fnirs=fnirs)

        data = self.data
        ch_names = self.ch_names
        if picks is not None:
            data = data[picks]
            ch_names = [ch_names[k] for k in picks]
        ch_idx, time_idx = _get_peak(data, self.times, tmin,
                                     tmax, mode)

        return (ch_names[ch_idx],
                time_idx if time_as_index else self.times[time_idx])


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
    elif decim > 1 and new_sfreq < 2.5 * lowpass:
        warn('The measurement information indicates a low-pass frequency '
             'of %g Hz. The decim=%i parameter will result in a sampling '
             'frequency of %g Hz, which can cause aliasing artifacts.'
             % (lowpass, decim, new_sfreq))  # > 50% nyquist lim
    offset = int(offset)
    if not 0 <= offset < decim:
        raise ValueError('decim must be at least 0 and less than %s, got '
                         '%s' % (decim, offset))
    return decim, offset, new_sfreq


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
    comment : string
        Comment on dataset. Can be the condition. Defaults to ''.
    nave : int
        Number of averaged epochs. Defaults to 1.
    kind : str
        Type of data, either average or standard_error. Defaults to 'average'.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    Proper units of measure:
    * V: eeg, eog, seeg, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc

    See Also
    --------
    EpochsArray, io.RawArray, create_info
    """

    @verbose
    def __init__(self, data, info, tmin=0., comment='', nave=1, kind='average',
                 verbose=None):  # noqa: D102
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)

        if data.ndim != 2:
            raise ValueError('Data must be a 2D array of shape (n_channels, '
                             'n_samples)')

        if len(info['ch_names']) != np.shape(data)[0]:
            raise ValueError('Info (%s) and data (%s) must have same number '
                             'of channels.' % (len(info['ch_names']),
                                               np.shape(data)[0]))

        self.data = data

        # XXX: this should use round and be tested
        self.first = int(tmin * info['sfreq'])
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
        if not isinstance(self.kind, string_types):
            raise TypeError('kind must be a string, not "%s"' % (type(kind),))
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
    t = [_aspect_rev.get(str(a), 'Unknown') for a in aspect_kinds]
    t = ['"' + c + '" (' + tt + ')' for tt, c in zip(t, comments)]
    t = '  ' + '\n  '.join(t)
    return comments, aspect_kinds, t


def _get_aspect(evoked, allow_maxshield):
    """Get Evoked data aspect."""
    is_maxshield = False
    aspect = dir_tree_find(evoked, FIFF.FIFFB_ASPECT)
    if len(aspect) == 0:
        _check_maxshield(allow_maxshield)
        aspect = dir_tree_find(evoked, FIFF.FIFFB_SMSH_ASPECT)
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


def grand_average(all_evoked, interpolate_bads=True):
    """Make grand average of a list evoked data.

    The function interpolates bad channels based on `interpolate_bads`
    parameter. If `interpolate_bads` is True, the grand average
    file will contain good channels and the bad channels interpolated
    from the good MEG/EEG channels.

    The grand_average.nave attribute will be equal the number
    of evoked datasets used to calculate the grand average.

    Note: Grand average evoked shall not be used for source localization.

    Parameters
    ----------
    all_evoked : list of Evoked data
        The evoked datasets.
    interpolate_bads : bool
        If True, bad MEG and EEG channels are interpolated.

    Returns
    -------
    grand_average : Evoked
        The grand average data.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    # check if all elements in the given list are evoked data
    if not all(isinstance(e, Evoked) for e in all_evoked):
        raise ValueError("Not all the elements in list are evoked data")

    # Copy channels to leave the original evoked datasets intact.
    all_evoked = [e.copy() for e in all_evoked]

    # Interpolates if necessary
    if interpolate_bads:
        all_evoked = [e.interpolate_bads() if len(e.info['bads']) > 0
                      else e for e in all_evoked]

    equalize_channels(all_evoked)  # apply equalize_channels
    # make grand_average object using combine_evoked
    grand_average = combine_evoked(all_evoked, weights='equal')
    # change the grand_average.nave to the number of Evokeds
    grand_average.nave = len(all_evoked)
    # change comment field
    grand_average.comment = "Grand average (n = %d)" % grand_average.nave
    return grand_average


def combine_evoked(all_evoked, weights):
    """Merge evoked data by weighted addition or subtraction.

    Data should have the same channels and the same time instants.
    Subtraction can be performed by passing negative weights (e.g., [1, -1]).

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
    evoked = all_evoked[0].copy()
    if isinstance(weights, string_types):
        if weights not in ('nave', 'equal'):
            raise ValueError('weights must be a list of float, or "nave" or '
                             '"equal"')
        if weights == 'nave':
            weights = np.array([e.nave for e in all_evoked], float)
            weights /= weights.sum()
        else:  # == 'equal'
            weights = [1. / len(all_evoked)] * len(all_evoked)
    weights = np.array(weights, float)
    if weights.ndim != 1 or weights.size != len(all_evoked):
        raise ValueError('weights must be the same size as all_evoked')

    ch_names = evoked.ch_names
    for e in all_evoked[1:]:
        assert e.ch_names == ch_names, ValueError("%s and %s do not contain "
                                                  "the same channels"
                                                  % (evoked, e))
        assert np.max(np.abs(e.times - evoked.times)) < 1e-7, \
            ValueError("%s and %s do not contain the same time instants"
                       % (evoked, e))

    # use union of bad channels
    bads = list(set(evoked.info['bads']).union(*(ev.info['bads']
                                                 for ev in all_evoked[1:])))
    evoked.info['bads'] = bads

    evoked.data = sum(w * e.data for w, e in zip(weights, all_evoked))
    # We should set nave based on how variances change when summing Gaussian
    # random variables. From:
    #
    #    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    #
    # We know that the variance of a weighted sample mean is:
    #
    #    σ^2 = w_1^2 σ_1^2 + w_2^2 σ_2^2 + ... + w_n^2 σ_n^2
    #
    # We estimate the variance of each evoked instance as 1 / nave to get:
    #
    #    σ^2 = w_1^2 / nave_1 + w_2^2 / nave_2 + ... + w_n^2 / nave_n
    #
    # And our resulting nave is the reciprocal of this:
    evoked.nave = max(int(round(
        1. / sum(w ** 2 / e.nave for w, e in zip(weights, all_evoked)))), 1)
    evoked.comment = ' + '.join('%0.3f * %s' % (w, e.comment or 'unknown')
                                for w, e in zip(weights, all_evoked))
    return evoked


@verbose
def read_evokeds(fname, condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    """Read evoked dataset(s).

    Parameters
    ----------
    fname : string
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    evoked : Evoked (if condition is int or str) or list of Evoked (if
        condition is None or list)
        The evoked dataset(s).

    See Also
    --------
    write_evokeds
    """
    check_fname(fname, 'evoked', ('-ave.fif', '-ave.fif.gz'))
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
        if isinstance(condition, string_types):
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
                                 'found datasets:\n  %s'
                                 % (condition, kind, t))
            condition = found_cond[0]
        elif condition is None:
            if len(evoked_node) > 1:
                _, _, conditions = _get_entries(fid, evoked_node,
                                                allow_maxshield)
                raise TypeError("Evoked file has more than one "
                                "conditions, the condition parameters "
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

        if first is not None:
            nsamp = last - first + 1
        elif first_time is not None:
            first = int(round(first_time * info['sfreq']))
            last = first + nsamp
        else:
            raise RuntimeError('Could not read time parameters')
        if nsamp is not None and data.shape[1] != nsamp:
            raise ValueError('Incorrect number of samples (%d instead of '
                             ' %d)' % (data.shape[1], nsamp))
        nsamp = data.shape[1]
        last = first + nsamp - 1
        logger.info('    Found the data of interest:')
        logger.info('        t = %10.2f ... %10.2f ms (%s)'
                    % (1000 * first / info['sfreq'],
                       1000 * last / info['sfreq'], comment))
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

    times = np.arange(first, last + 1, dtype=np.float) / info['sfreq']
    return info, nave, aspect_kind, first, last, comment, times, data


def write_evokeds(fname, evoked):
    """Write an evoked dataset to a file.

    Parameters
    ----------
    fname : string
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
        check_fname(fname, 'evoked', ('-ave.fif', '-ave.fif.gz'))

    if not isinstance(evoked, list):
        evoked = [evoked]

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

            # First and last sample
            write_int(fid, FIFF.FIFF_FIRST_SAMPLE, e.first)
            write_int(fid, FIFF.FIFF_LAST_SAMPLE, e.last)

            # The epoch itself
            if e.info.get('maxshield'):
                aspect = FIFF.FIFFB_SMSH_ASPECT
            else:
                aspect = FIFF.FIFFB_ASPECT
            start_block(fid, aspect)

            write_int(fid, FIFF.FIFF_ASPECT_KIND, e._aspect_kind)
            write_int(fid, FIFF.FIFF_NAVE, e.nave)

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
    """
    modes = ('abs', 'neg', 'pos')
    if mode not in modes:
        raise ValueError('The `mode` parameter must be `{modes}`. You gave '
                         'me `{mode}`'.format(modes='` or `'.join(modes),
                                              mode=mode))

    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    if tmin < times.min():
        raise ValueError('The tmin value is out of bounds. It must be '
                         'within {0} and {1}'.format(times.min(), times.max()))
    if tmax > times.max():
        raise ValueError('The tmin value is out of bounds. It must be '
                         'within {0} and {1}'.format(times.min(), times.max()))
    if tmin >= tmax:
        raise ValueError('The tmin must be smaller than tmax')

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

    return max_loc, max_time
