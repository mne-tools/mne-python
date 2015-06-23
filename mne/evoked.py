# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Mads Jensen <mje.mads@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np
import warnings

from .baseline import rescale
from .channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                SetChannelsMixin, InterpolationMixin,
                                equalize_channels)
from .filter import resample, detrend, FilterMixin
from .fixes import in1d
from .utils import check_fname, logger, verbose, object_hash, _time_mask
from .viz import (plot_evoked, plot_evoked_topomap, plot_evoked_field,
                  plot_evoked_image)
from .viz.evoked import _plot_evoked_white
from .externals.six import string_types

from .io.constants import FIFF
from .io.open import fiff_open
from .io.tag import read_tag
from .io.tree import dir_tree_find
from .io.pick import channel_type, pick_types
from .io.meas_info import read_meas_info, write_meas_info
from .io.proj import ProjMixin
from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_string, write_float_matrix,
                       write_id)
from .io.base import ToDataFrameMixin

_aspect_dict = {'average': FIFF.FIFFV_ASPECT_AVERAGE,
                'standard_error': FIFF.FIFFV_ASPECT_STD_ERR}
_aspect_rev = {str(FIFF.FIFFV_ASPECT_AVERAGE): 'average',
               str(FIFF.FIFFV_ASPECT_STD_ERR): 'standard_error'}


class Evoked(ProjMixin, ContainsMixin, UpdateChannelsMixin,
             SetChannelsMixin, InterpolationMixin, FilterMixin,
             ToDataFrameMixin):
    """Evoked data

    Parameters
    ----------
    fname : string
        Name of evoked/average FIF file to load.
        If None no data is loaded.
    condition : int, or str
        Dataset ID number (int) or comment/name (str). Optional if there is
        only one data set in file.
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used. If None, no correction is applied.
    proj : bool, optional
        Apply SSP projection vectors
    kind : str
        Either 'average' or 'standard_error'. The type of data to read.
        Only used if 'condition' is a str.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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
    """
    @verbose
    def __init__(self, fname, condition=None, baseline=None, proj=True,
                 kind='average', verbose=None):

        if fname is None:
            raise ValueError('No evoked filename specified')

        self.verbose = verbose
        logger.info('Reading %s ...' % fname)
        f, tree, _ = fiff_open(fname)
        with f as fid:
            if not isinstance(proj, bool):
                raise ValueError(r"'proj' must be 'True' or 'False'")

            #   Read the measurement info
            info, meas = read_meas_info(fid, tree)
            info['filename'] = fname

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

                comments, aspect_kinds, t = _get_entries(fid, evoked_node)
                goods = np.logical_and(in1d(comments, [condition]),
                                       in1d(aspect_kinds,
                                            [_aspect_dict[kind]]))
                found_cond = np.where(goods)[0]
                if len(found_cond) != 1:
                    raise ValueError('condition "%s" (%s) not found, out of '
                                     'found datasets:\n  %s'
                                     % (condition, kind, t))
                condition = found_cond[0]

            if condition >= len(evoked_node) or condition < 0:
                fid.close()
                raise ValueError('Data set selector out of range')

            my_evoked = evoked_node[condition]

            # Identify the aspects
            aspects = dir_tree_find(my_evoked, FIFF.FIFFB_ASPECT)
            if len(aspects) > 1:
                logger.info('Multiple aspects found. Taking first one.')
            my_aspect = aspects[0]

            # Now find the data in the evoked block
            nchan = 0
            sfreq = -1
            chs = []
            comment = None
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
                info['nchan'] = nchan
                logger.info('    Found channel information in evoked data. '
                            'nchan = %d' % nchan)
                if sfreq > 0:
                    info['sfreq'] = sfreq

            nsamp = last - first + 1
            logger.info('    Found the data of interest:')
            logger.info('        t = %10.2f ... %10.2f ms (%s)'
                        % (1000 * first / info['sfreq'],
                           1000 * last / info['sfreq'], comment))
            if info['comps'] is not None:
                logger.info('        %d CTF compensation matrices available'
                            % len(info['comps']))

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

            logger.info('        nave = %d - aspect type = %d'
                        % (nave, aspect_kind))

            nepoch = len(epoch)
            if nepoch != 1 and nepoch != info['nchan']:
                raise ValueError('Number of epoch tags is unreasonable '
                                 '(nepoch = %d nchan = %d)'
                                 % (nepoch, info['nchan']))

            if nepoch == 1:
                # Only one epoch
                all_data = epoch[0].data.astype(np.float)
                # May need a transpose if the number of channels is one
                if all_data.shape[1] == 1 and info['nchan'] == 1:
                    all_data = all_data.T.astype(np.float)
            else:
                # Put the old style epochs together
                all_data = np.concatenate([e.data[None, :] for e in epoch],
                                          axis=0).astype(np.float)

            if all_data.shape[1] != nsamp:
                raise ValueError('Incorrect number of samples (%d instead of '
                                 ' %d)' % (all_data.shape[1], nsamp))

        # Calibrate
        cals = np.array([info['chs'][k]['cal'] *
                         info['chs'][k].get('scale', 1.0)
                         for k in range(info['nchan'])])
        all_data *= cals[:, np.newaxis]

        times = np.arange(first, last + 1, dtype=np.float) / info['sfreq']
        self.info = info

        # Put the rest together all together
        self.nave = nave
        self._aspect_kind = aspect_kind
        self.kind = _aspect_rev.get(str(self._aspect_kind), 'Unknown')
        self.first = first
        self.last = last
        self.comment = comment
        self.times = times
        self.data = all_data

        # bind info, proj, data to self so apply_proj can be used
        self.data = all_data
        if proj:
            self.apply_proj()
        # Run baseline correction
        self.data = rescale(self.data, times, baseline, 'mean', copy=False)

    def save(self, fname):
        """Save dataset to file.

        Parameters
        ----------
        fname : string
            Name of the file where to save the data.
        """
        write_evokeds(fname, self)

    def __repr__(self):
        s = "comment : '%s'" % self.comment
        s += ", time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", n_epochs : %d" % self.nave
        s += ", n_channels x n_times : %s x %s" % self.data.shape
        return "<Evoked  |  %s>" % s

    @property
    def ch_names(self):
        """Channel names"""
        return self.info['ch_names']

    def crop(self, tmin=None, tmax=None, copy=False):
        """Crop data to a given time interval

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        copy : bool
            If False epochs is cropped in place.
        """
        inst = self if not copy else self.copy()
        mask = _time_mask(inst.times, tmin, tmax)
        inst.times = inst.times[mask]
        inst.first = int(inst.times[0] * inst.info['sfreq'])
        inst.last = len(inst.times) + inst.first - 1
        inst.data = inst.data[:, mask]
        return inst

    def shift_time(self, tshift, relative=True):
        """Shift time scale in evoked data

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

    def plot(self, picks=None, exclude='bads', unit=True, show=True, ylim=None,
             xlim='tight', proj=False, hline=None, units=None, scalings=None,
             titles=None, axes=None):
        """Plot evoked data as butterfly plots

        Note: If bad channels are not excluded they are shown in red.

        Parameters
        ----------
        picks : array-like of int | None
            The indices of channels to plot. If None show all.
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the
            bad channels are excluded.
        unit : bool
            Scale plot with channel (SI) unit.
        show : bool
            Call pyplot.show() at the end or not.
        ylim : dict
            ylim for plots. e.g. ylim = dict(eeg=[-200e-6, 200e-6])
            Valid keys are eeg, mag, grad
        xlim : 'tight' | tuple | None
            xlim for plots.
        proj : bool | 'interactive'
            If true SSP projections are applied before display. If
            'interactive', a check box for reversible selection of SSP
            projection vectors will be shown.
        hline : list of floats | None
            The values at which show an horizontal line.
        units : dict | None
            The units of the channel types used for axes lables. If None,
            defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
        scalings : dict | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
        titles : dict | None
            The titles associated with the channels. If None, defaults to
            `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
        axes : instance of Axes | list | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as the number of channel types. If instance of
            Axes, there must be only one channel type plotted.
        """
        return plot_evoked(self, picks=picks, exclude=exclude, unit=unit,
                           show=show, ylim=ylim, proj=proj, xlim=xlim,
                           hline=hline, units=units, scalings=scalings,
                           titles=titles, axes=axes)

    def plot_image(self, picks=None, exclude='bads', unit=True, show=True,
                   clim=None, xlim='tight', proj=False, units=None,
                   scalings=None, titles=None, axes=None, cmap='RdBu_r'):
        """Plot evoked data as images

        Parameters
        ----------
        picks : array-like of int | None
            The indices of channels to plot. If None show all.
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the
            bad channels are excluded.
        unit : bool
            Scale plot with channel (SI) unit.
        show : bool
            Call pyplot.show() at the end or not.
        clim : dict
            clim for images. e.g. clim = dict(eeg=[-200e-6, 200e6])
            Valid keys are eeg, mag, grad
        xlim : 'tight' | tuple | None
            xlim for plots.
        proj : bool | 'interactive'
            If true SSP projections are applied before display. If
            'interactive', a check box for reversible selection of SSP
            projection vectors will be shown.
        units : dict | None
            The units of the channel types used for axes lables. If None,
            defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
        scalings : dict | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
        titles : dict | None
            The titles associated with the channels. If None, defaults to
            `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
        axes : instance of Axes | list | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as the number of channel types. If instance of
            Axes, there must be only one channel type plotted.
        cmap : matplotlib colormap
            Colormap.
        """
        return plot_evoked_image(self, picks=picks, exclude=exclude, unit=unit,
                                 show=show, clim=clim, proj=proj, xlim=xlim,
                                 units=units, scalings=scalings,
                                 titles=titles, axes=axes, cmap=cmap)

    def plot_topomap(self, times=None, ch_type=None, layout=None, vmin=None,
                     vmax=None, cmap='RdBu_r', sensors=True, colorbar=True,
                     scale=None, scale_time=1e3, unit=None, res=64, size=1,
                     cbar_fmt="%3.1f", time_format='%01d ms', proj=False,
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None,
                     axes=None):
        """Plot topographic maps of specific time points

        Parameters
        ----------
        times : float | array of floats | None.
            The time point(s) to plot. If None, the number of ``axes``
            determines the amount of time point(s). If ``axes`` is also None,
            10 topographies will be shown with a regular time spacing between
            the first and last time instant.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are collec-
            ted in pairs and the RMS for each pair is plotted.
            If None, then channels are chosen in the order given above.
        layout : None | Layout
            Layout instance specifying sensor positions (does not need to
            be specified for Neuromag data). If possible, the correct
            layout file is inferred from the data; if no appropriate layout
            file was found, the layout is automatically generated from the
            sensor locations.
        vmin : float | callable
            The value specfying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            If callable, the output equals vmin(data).
        vmax : float | callable
            The value specfying the upper bound of the color range.
            If None, the maximum absolute value is used. If vmin is None,
            but vmax is not, defaults to np.max(data).
            If callable, the output equals vmax(data).
        cmap : matplotlib colormap
            Colormap. Defaults to 'RdBu_r'.
        sensors : bool | str
            Add markers for sensor locations to the plot. Accepts matplotlib
            plot format string (e.g., 'r+' for red plusses). If True, a circle
            will be used (via .add_artist). Defaults to True.
        colorbar : bool
            Plot a colorbar.
        scale : dict | float | None
            Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
            for grad and 1e15 for mag.
        scale_time : float | None
            Scale the time labels. Defaults to 1e3 (ms).
        unit : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : scalar
            Side length of the topomaps in inches (only applies when plotting
            multiple topomaps at a time).
        cbar_fmt : str
            String format for colorbar values.
        time_format : str
            String format for topomap values. Defaults to ``"%01d ms"``.
        proj : bool | 'interactive'
            If true SSP projections are applied before display. If
            'interactive', a check box for reversible selection of SSP
            projection vectors will be shown.
        show : bool
            Call pyplot.show() at the end.
        show_names : bool | callable
            If True, show channel names on top of the map. If a callable is
            passed, channel names will be formatted using the callable; e.g.,
            to delete the prefix 'MEG ' from all channel names, pass the
            function
            lambda x: x.replace('MEG ', ''). If `mask` is not None, only
            significant sensors will be shown.
        title : str | None
            Title. If None (default), no title is displayed.
        mask : ndarray of bool, shape (n_channels, n_times) | None
            The channels to be marked as significant at a given time point.
            Indicies set to `True` will be considered. Defaults to None.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals:
            ``dict(marker='o', markerfacecolor='w', markeredgecolor='k',
            linewidth=0, markersize=4)``.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', a head scheme will be drawn.
            If dict, each key refers to a tuple of x and y positions.
            The values in 'mask_pos' will serve as image mask. If None, nothing
            will be drawn. Defaults to 'head'. If dict, the 'autoshrink' (bool)
            field will trigger automated shrinking of the positions due to
            points outside the outline. Moreover, a matplotlib patch object can
            be passed for advanced masking options, either directly or as a
            function that returns patches (required for multi-axis plots).
        contours : int | False | None
            The number of contour lines to draw. If 0, no contours will be
            drawn.
        image_interp : str
            The image interpolation to be used. All matplotlib options are
            accepted.
        average : float | None
            The time window around a given time to be used for averaging
            (seconds). For example, 0.01 would translate into window that
            starts 5 ms before and ends 5 ms after a given time point.
            Defaults to None, which means no averaging.
        head_pos : dict | None
            If None (default), the sensors are positioned such that they span
            the head circle. If dict, can have entries 'center' (tuple) and
            'scale' (tuple) for what the center and scale of the head should be
            relative to the electrode locations.
        axes : instance of Axes | list | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as ``times`` (unless ``times`` is None). If
            instance of Axes, ``times`` must be a float or a list of one float.
            Defaults to None.
        """
        return plot_evoked_topomap(self, times=times, ch_type=ch_type,
                                   layout=layout, vmin=vmin,
                                   vmax=vmax, cmap=cmap, sensors=sensors,
                                   colorbar=colorbar, scale=scale,
                                   scale_time=scale_time,
                                   unit=unit, res=res, proj=proj, size=size,
                                   cbar_fmt=cbar_fmt, time_format=time_format,
                                   show=show, show_names=show_names,
                                   title=title, mask=mask,
                                   mask_params=mask_params,
                                   outlines=outlines, contours=contours,
                                   image_interp=image_interp,
                                   average=average, head_pos=head_pos,
                                   axes=axes)

    def plot_field(self, surf_maps, time=None, time_label='t = %0.0f ms',
                   n_jobs=1):
        """Plot MEG/EEG fields on head surface and helmet in 3D

        Parameters
        ----------
        surf_maps : list
            The surface mapping information obtained with make_field_map.
        time : float | None
            The time point at which the field map shall be displayed. If None,
            the average peak latency (across sensor types) is used.
        time_label : str
            How to print info about the time instant visualized.
        n_jobs : int
            Number of jobs to run in parallel.

        Returns
        -------
        fig : instance of mlab.Figure
            The mayavi figure.
        """
        return plot_evoked_field(self, surf_maps, time=time,
                                 time_label=time_label, n_jobs=n_jobs)

    def plot_white(self, noise_cov, show=True):
        """Plot whitened evoked response

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
        noise_cov : list | instance of Covariance
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

    def as_type(self, ch_type='grad', mode='fast'):
        """Compute virtual evoked using interpolated fields in mag/grad channels.

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

    def resample(self, sfreq, npad=100, window='boxcar'):
        """Resample data

        This function operates in-place.

        Parameters
        ----------
        sfreq : float
            New sample rate to use
        npad : int
            Amount to pad the start and end of the data.
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        """
        o_sfreq = self.info['sfreq']
        self.data = resample(self.data, sfreq, o_sfreq, npad, -1, window)
        # adjust indirectly affected variables
        self.info['sfreq'] = sfreq
        self.times = (np.arange(self.data.shape[1], dtype=np.float) / sfreq +
                      self.times[0])
        self.first = int(self.times[0] * self.info['sfreq'])
        self.last = len(self.times) + self.first - 1

    def detrend(self, order=1, picks=None):
        """Detrend data

        This function operates in-place.

        Parameters
        ----------
        order : int
            Either 0 or 1, the order of the detrending. 0 is a constant
            (DC) detrend, 1 is a linear detrend.
        picks : array-like of int | None
            If None only MEG and EEG channels are detrended.
        """
        if picks is None:
            picks = pick_types(self.info, meg=True, eeg=True, ref_meg=False,
                               stim=False, eog=False, ecg=False, emg=False,
                               exclude='bads')
        self.data[picks] = detrend(self.data[picks], order, axis=-1)

    def copy(self):
        """Copy the instance of evoked

        Returns
        -------
        evoked : instance of Evoked
        """
        evoked = deepcopy(self)
        return evoked

    def __add__(self, evoked):
        """Add evoked taking into account number of epochs"""
        out = combine_evoked([self, evoked])
        out.comment = self.comment + " + " + evoked.comment
        return out

    def __sub__(self, evoked):
        """Add evoked taking into account number of epochs"""
        this_evoked = deepcopy(evoked)
        this_evoked.data *= -1.
        out = combine_evoked([self, this_evoked])
        if self.comment is None or this_evoked.comment is None:
            warnings.warn('evoked.comment expects a string but is None')
            out.comment = 'unknown'
        else:
            out.comment = self.comment + " - " + this_evoked.comment
        return out

    def __hash__(self):
        return object_hash(dict(info=self.info, data=self.data))

    def get_peak(self, ch_type=None, tmin=None, tmax=None, mode='abs',
                 time_as_index=False):
        """Get location and latency of peak amplitude

        Parameters
        ----------
        ch_type : {'mag', 'grad', 'eeg', 'misc', None}
            The channel type to use. Defaults to None. If more than one sensor
            Type is present in the data the channel type has to be explicitly
            set.
        tmin : float | None
            The minimum point in time to be considered for peak getting.
        tmax : float | None
            The maximum point in time to be considered for peak getting.
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
        supported = ('mag', 'grad', 'eeg', 'misc', 'None')

        data_picks = pick_types(self.info, meg=True, eeg=True, ref_meg=False)
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

        meg, eeg, misc, picks = False, False, False, None

        if ch_type == 'mag':
            meg = ch_type
        elif ch_type == 'grad':
            meg = ch_type
        elif ch_type == 'eeg':
            eeg = True
        elif ch_type == 'misc':
            misc = True

        if ch_type is not None:
            picks = pick_types(self.info, meg=meg, eeg=eeg, misc=misc,
                               ref_meg=False)

        data = self.data if picks is None else self.data[picks]
        ch_idx, time_idx = _get_peak(data, self.times, tmin,
                                     tmax, mode)

        return (self.ch_names[ch_idx],
                time_idx if time_as_index else self.times[time_idx])


class EvokedArray(Evoked):
    """Evoked object from numpy array

    Parameters
    ----------
    data : array of shape (n_channels, n_times)
        The channels' evoked response.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    tmin : float
        Start time before event.
    comment : string
        Comment on dataset. Can be the condition. Defaults to ''.
    nave : int
        Number of averaged epochs. Defaults to 1.
    kind : str
        Type of data, either average or standard_error. Defaults to 'average'.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    See Also
    --------
    EpochsArray, io.RawArray
    """

    @verbose
    def __init__(self, data, info, tmin, comment='', nave=1, kind='average',
                 verbose=None):

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
        self.info = info
        self.nave = nave
        self.kind = kind
        self.comment = comment
        self.picks = None
        self.verbose = verbose
        self._projector = None
        if self.kind == 'average':
            self._aspect_kind = _aspect_dict['average']
        else:
            self._aspect_kind = _aspect_dict['standard_error']


def _get_entries(fid, evoked_node):
    """Helper to get all evoked entries"""
    comments = list()
    aspect_kinds = list()
    for ev in evoked_node:
        for k in range(ev['nent']):
            my_kind = ev['directory'][k].kind
            pos = ev['directory'][k].pos
            if my_kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comments.append(tag.data)
        my_aspect = dir_tree_find(ev, FIFF.FIFFB_ASPECT)[0]
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


def _get_evoked_node(fname):
    """Helper to get info in evoked file"""
    f, tree, _ = fiff_open(fname)
    with f as fid:
        _, meas = read_meas_info(fid, tree)
        evoked_node = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
    return evoked_node


def grand_average(all_evoked, interpolate_bads=True):
    """Make grand average of a list evoked data

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


def combine_evoked(all_evoked, weights='nave'):
    """Merge evoked data by weighted addition

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
    evoked.nave = max(int(1. / sum(w ** 2 / e.nave
                                   for w, e in zip(weights, all_evoked))), 1)
    return evoked


@verbose
def read_evokeds(fname, condition=None, baseline=None, kind='average',
                 proj=True, verbose=None):
    """Read evoked dataset(s)

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
        If a is None the beginning of the data is used and if b is None then
        b is set to the end of the interval. If baseline is equal to
        (None, None) all the time interval is used.
    kind : str
        Either 'average' or 'standard_error', the type of data to read.
    proj : bool
        If False, available projectors won't be applied to the data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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

    return_list = True
    if condition is None:
        evoked_node = _get_evoked_node(fname)
        condition = range(len(evoked_node))
    elif not isinstance(condition, list):
        condition = [condition]
        return_list = False

    out = [Evoked(fname, c, baseline=baseline, kind=kind, proj=proj,
           verbose=verbose) for c in condition]

    return out if return_list else out[0]


def write_evokeds(fname, evoked):
    """Write an evoked dataset to a file

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
            start_block(fid, FIFF.FIFFB_ASPECT)

            write_int(fid, FIFF.FIFF_ASPECT_KIND, e._aspect_kind)
            write_int(fid, FIFF.FIFF_NAVE, e.nave)

            decal = np.zeros((e.info['nchan'], 1))
            for k in range(e.info['nchan']):
                decal[k] = 1.0 / (e.info['chs'][k]['cal'] *
                                  e.info['chs'][k].get('scale', 1.0))

            write_float_matrix(fid, FIFF.FIFF_EPOCH, decal * e.data)
            end_block(fid, FIFF.FIFFB_ASPECT)
            end_block(fid, FIFF.FIFFB_EVOKED)

        end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        end_block(fid, FIFF.FIFFB_MEAS)
        end_file(fid)


def _get_peak(data, times, tmin=None, tmax=None, mode='abs'):
    """Get feature-index and time of maximum signal from 2D array

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
