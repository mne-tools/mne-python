"""Tools for working with epoched data"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Denis Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from .externals.six import string_types

import copy as cp
import warnings
import json

import numpy as np

from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_float_matrix, write_float,
                       write_id, write_string)
from .io.meas_info import read_meas_info, write_meas_info, _merge_info
from .io.open import fiff_open
from .io.tree import dir_tree_find
from .io.tag import read_tag
from .io.constants import FIFF
from .io.pick import (pick_types, channel_indices_by_type, channel_type,
                      pick_channels, pick_info)
from .io.proj import setup_proj, ProjMixin
from .io.base import _BaseRaw, _time_as_index, _index_as_time
from .evoked import EvokedArray, aspect_rev
from .baseline import rescale
from .utils import (check_random_state, _check_pandas_index_arguments,
                    _check_pandas_installed, object_hash)
from .channels import ContainsMixin, PickDropChannelsMixin
from .filter import resample, detrend
from .event import _read_events_fif
from .fixes import in1d
from .viz import _mutable_defaults, plot_epochs, _drop_log_stats
from .utils import check_fname, logger, verbose
from .externals import six
from .externals.six.moves import zip
from .utils import deprecated, _check_type_picks


class _BaseEpochs(ProjMixin, ContainsMixin, PickDropChannelsMixin):
    """Abstract base class for Epochs-type classes

    This class provides basic functionality and should never be instantiated
    directly. See Epochs below for an explanation of the parameters.
    """
    def __init__(self, info, event_id, tmin, tmax, baseline=(None, 0),
                 picks=None, name='Unknown', reject=None, flat=None,
                 decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 add_eeg_ref=True, verbose=None):

        self.verbose = verbose
        self.name = name

        if isinstance(event_id, dict):
            if not all([isinstance(v, int) for v in event_id.values()]):
                raise ValueError('Event IDs must be of type integer')
            if not all([isinstance(k, string_types) for k in event_id]):
                raise ValueError('Event names must be of type str')
            self.event_id = event_id
        elif isinstance(event_id, list):
            if not all([isinstance(v, int) for v in event_id]):
                raise ValueError('Event IDs must be of type integer')
            self.event_id = dict(zip((str(i) for i in event_id), event_id))
        elif isinstance(event_id, int):
            self.event_id = {str(event_id): event_id}
        else:
            raise ValueError('event_id must be dict or int.')

        # check reject_tmin and reject_tmax
        if (reject_tmin is not None) and (reject_tmin < tmin):
            raise ValueError("reject_tmin needs to be None or >= tmin")
        if (reject_tmax is not None) and (reject_tmax > tmax):
            raise ValueError("reject_tmax needs to be None or <= tmax")
        if (reject_tmin is not None) and (reject_tmax is not None):
            if reject_tmin >= reject_tmax:
                raise ValueError('reject_tmin needs to be < reject_tmax')
        if not detrend in [None, 0, 1]:
            raise ValueError('detrend must be None, 0, or 1')

        # check that baseline is in available data
        if baseline is not None:
            baseline_tmin, baseline_tmax = baseline
            tstep = 1. / info['sfreq']
            if baseline_tmin is not None:
                if baseline_tmin < tmin - tstep:
                    err = ("Baseline interval (tmin = %s) is outside of epoch "
                           "data (tmin = %s)" % (baseline_tmin, tmin))
                    raise ValueError(err)
            if baseline_tmax is not None:
                if baseline_tmax > tmax + tstep:
                    err = ("Baseline interval (tmax = %s) is outside of epoch "
                           "data (tmax = %s)" % (baseline_tmax, tmax))
                    raise ValueError(err)

        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.reject = reject
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        self.flat = flat
        self.decim = decim = int(decim)
        self._bad_dropped = False
        self.drop_log = None
        self.selection = None
        self.detrend = detrend

        # Handle measurement info
        self.info = info
        if picks is None:
            picks = list(range(len(self.info['ch_names'])))
        else:
            self.info['chs'] = [self.info['chs'][k] for k in picks]
            self.info['ch_names'] = [self.info['ch_names'][k] for k in picks]
            self.info['nchan'] = len(picks)
        self.picks = _check_type_picks(picks)

        if len(picks) == 0:
            raise ValueError("Picks cannot be empty.")

        # Handle times
        if tmin >= tmax:
            raise ValueError('tmin has to be smaller than tmax')
        sfreq = float(self.info['sfreq'])
        n_times_min = int(round(tmin * sfreq))
        n_times_max = int(round(tmax * sfreq))
        times = np.arange(n_times_min, n_times_max + 1, dtype=np.float) / sfreq
        self.times = times
        self._raw_times = times  # times before decimation
        self._epoch_stop = ep_len = len(self.times)
        if decim > 1:
            new_sfreq = sfreq / decim
            lowpass = self.info['lowpass']
            if new_sfreq < 2.5 * lowpass:  # nyquist says 2 but 2.5 is safer
                msg = ('The measurement information indicates a low-pass '
                       'frequency of %g Hz. The decim=%i parameter will '
                       'result in a sampling frequency of %g Hz, which can '
                       'cause aliasing artifacts.'
                       % (lowpass, decim, new_sfreq))
                warnings.warn(msg)

            i_start = n_times_min % decim
            self._decim_idx = slice(i_start, ep_len, decim)
            self.times = self.times[self._decim_idx]
            self.info['sfreq'] = new_sfreq

        self.preload = False
        self._data = None
        self._offset = None

        # setup epoch rejection
        self._reject_setup()

    def _reject_setup(self):
        """Sets self._reject_time and self._channel_type_idx (called from
        __init__)
        """
        if self.reject is None and self.flat is None:
            return

        idx = channel_indices_by_type(self.info)
        for key in idx.keys():
            if (self.reject is not None and key in self.reject) \
                    or (self.flat is not None and key in self.flat):
                if len(idx[key]) == 0:
                    raise ValueError("No %s channel found. Cannot reject based"
                                     " on %s." % (key.upper(), key.upper()))

        self._channel_type_idx = idx

        if (self.reject_tmin is None) and (self.reject_tmax is None):
            self._reject_time = None
        else:
            if self.reject_tmin is None:
                reject_imin = None
            else:
                idxs = np.nonzero(self.times >= self.reject_tmin)[0]
                reject_imin = idxs[0]
            if self.reject_tmax is None:
                reject_imax = None
            else:
                idxs = np.nonzero(self.times <= self.reject_tmax)[0]
                reject_imax = idxs[-1]

            self._reject_time = slice(reject_imin, reject_imax)

    @verbose
    def _is_good_epoch(self, data, verbose=None):
        """Determine if epoch is good"""
        if data is None:
            return False, ['NO_DATA']
        n_times = len(self.times)
        if data.shape[1] < n_times:
            # epoch is too short ie at the end of the data
            return False, ['TOO_SHORT']
        if self.reject is None and self.flat is None:
            return True, None
        else:
            if self._reject_time is not None:
                data = data[:, self._reject_time]

            return _is_good(data, self.ch_names, self._channel_type_idx,
                            self.reject, self.flat, full_report=True,
                            ignore_chs=self.info['bads'])

    @verbose
    def _preprocess(self, epoch, verbose=None):
        """ Aux Function
        """
        # Detrend
        if self.detrend is not None:
            picks = pick_types(self.info, meg=True, eeg=True, stim=False,
                               ref_meg=False, eog=False, ecg=False,
                               emg=False, exclude=[])
            epoch[picks] = detrend(epoch[picks], self.detrend, axis=1)

        # Baseline correct
        picks = pick_types(self.info, meg=True, eeg=True, stim=False,
                           ref_meg=True, eog=True, ecg=True,
                           emg=True, exclude=[])
        epoch[picks] = rescale(epoch[picks], self._raw_times, self.baseline,
                               'mean', copy=False, verbose=verbose)

        # handle offset
        if self._offset is not None:
            epoch += self._offset

        # Decimate
        if self.decim > 1:
            epoch = epoch[:, self._decim_idx]
        return epoch

    def get_data(self):
        """Get all epochs as a 3D array

        Returns
        -------
        data : array of shape [n_epochs, n_channels, n_times]
            The epochs data
        """
        if self.preload:
            return self._data
        else:
            data = self._get_data_from_disk()
            return data

    def iter_evoked(self):
        """Iterate over Evoked objects with nave=1
        """
        self._current = 0

        while True:
            data, event_id = self.next(True)
            tmin = self.times[0]
            info = cp.deepcopy(self.info)

            yield EvokedArray(data, info, tmin, comment=str(event_id))

    def subtract_evoked(self, evoked=None):
        """Subtract an evoked response from each epoch

        Can be used to exclude the evoked response when analyzing induced
        activity, see e.g. [1].

        References
        ----------
        [1] David et al. "Mechanisms of evoked and induced responses in
        MEG/EEG", NeuroImage, vol. 31, no. 4, pp. 1580-1591, July 2006.

        Parameters
        ----------
        evoked : instance of Evoked | None
            The evoked response to subtract. If None, the evoked response
            is computed from Epochs itself.

        Returns
        -------
        self : instance of Epochs
            The modified instance (instance is also modified inplace).
        """
        logger.info('Subtracting Evoked from Epochs')
        if evoked is None:
            picks = pick_types(self.info, meg=True, eeg=True,
                               stim=False, eog=False, ecg=False,
                               emg=False, exclude=[])
            evoked = self.average(picks)

        # find the indices of the channels to use
        picks = pick_channels(evoked.ch_names, include=self.ch_names)

        # make sure the omitted channels are not data channels
        if len(picks) < len(self.ch_names):
            sel_ch = [evoked.ch_names[ii] for ii in picks]
            diff_ch = list(set(self.ch_names).difference(sel_ch))
            diff_idx = [self.ch_names.index(ch) for ch in diff_ch]
            diff_types = [channel_type(self.info, idx) for idx in diff_idx]
            bad_idx = [diff_types.index(t) for t in diff_types if t in
                       ['grad', 'mag', 'eeg']]
            if len(bad_idx) > 0:
                bad_str = ', '.join([diff_ch[ii] for ii in bad_idx])
                raise ValueError('The following data channels are missing '
                                 'in the evoked response: %s' % bad_str)
            logger.info('    The following channels are not included in the '
                        'subtraction: %s' % ', '.join(diff_ch))

        # make sure the times match
        if (len(self.times) != len(evoked.times) or
                np.max(np.abs(self.times - evoked.times)) >= 1e-7):
            raise ValueError('Epochs and Evoked object do not contain '
                             'the same time points.')

        # handle SSPs
        if not self.proj and evoked.proj:
            warnings.warn('Evoked has SSP applied while Epochs has not.')
        if self.proj and not evoked.proj:
            evoked = evoked.copy().apply_proj()

        # find the indices of the channels to use in Epochs
        ep_picks = [self.ch_names.index(evoked.ch_names[ii]) for ii in picks]

        # do the subtraction
        if self.preload:
            self._data[:, ep_picks, :] -= evoked.data[picks][None, :, :]
        else:
            if self._offset is None:
                self._offset = np.zeros((len(self.ch_names), len(self.times)),
                                        dtype=np.float)
            self._offset[ep_picks] -= evoked.data[picks]
        logger.info('[done]')

        return self

    def _get_data_from_disk(self, out=True, verbose=None):
        raise NotImplementedError('_get_data_from_disk() must be implemented '
                                  'in derived class.')

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self, return_event_id=False):
        raise NotImplementedError('next() must be implemented in derived '
                                  'class.')

    def __next__(self, *args, **kwargs):
        """Wrapper for Py3k"""
        return self.next(*args, **kwargs)

    def __hash__(self):
        if not self.preload:
            raise RuntimeError('Cannot hash epochs unless preloaded')
        return object_hash(dict(info=self.info, data=self._data))

    def average(self, picks=None):
        """Compute average of epochs

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        evoked : Evoked instance
            The averaged epochs
        """

        return self._compute_mean_or_stderr(picks, 'ave')

    def standard_error(self, picks=None):
        """Compute standard error over epochs

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        evoked : Evoked instance
            The standard error over epochs
        """
        return self._compute_mean_or_stderr(picks, 'stderr')

    def _compute_mean_or_stderr(self, picks, mode='ave'):
        """Compute the mean or std over epochs and return Evoked"""

        _do_std = True if mode == 'stderr' else False

        n_channels = len(self.ch_names)
        n_times = len(self.times)
        if self.preload:
            n_events = len(self.events)
            if not _do_std:
                data = np.mean(self._data, axis=0)
            else:
                data = np.std(self._data, axis=0)
            assert len(self.events) == len(self._data)
        else:
            data = np.zeros((n_channels, n_times))
            n_events = 0
            for e in self:
                data += e
                n_events += 1

            if n_events > 0:
                data /= n_events
            else:
                data.fill(np.nan)

            # convert to stderr if requested, could do in one pass but do in
            # two (slower) in case there are large numbers
            if _do_std:
                data_mean = cp.copy(data)
                data.fill(0.)
                for e in self:
                    data += (e - data_mean) ** 2
                data = np.sqrt(data / n_events)

        if not _do_std:
            _aspect_kind = FIFF.FIFFV_ASPECT_AVERAGE
        else:
            _aspect_kind = FIFF.FIFFV_ASPECT_STD_ERR
            data /= np.sqrt(n_events)
        kind = aspect_rev.get(str(_aspect_kind), 'Unknown')

        info = cp.deepcopy(self.info)
        evoked = EvokedArray(data, info, tmin=self.times[0],
                             comment=self.name, nave=n_events, kind=kind,
                             verbose=self.verbose)
        # XXX: above constructor doesn't recreate the times object precisely
        evoked.times = self.times.copy()
        evoked._aspect_kind = _aspect_kind

        # pick channels
        if picks is None:
            picks = pick_types(evoked.info, meg=True, eeg=True, ref_meg=True,
                               stim=False, eog=False, ecg=False,
                               emg=False, exclude=[])

        ch_names = [evoked.ch_names[p] for p in picks]
        evoked.pick_channels(ch_names)

        if len(evoked.info['ch_names']) == 0:
            raise ValueError('No data channel found when averaging.')

        # otherwise the apply_proj will be confused
        evoked.proj = True if self.proj is True else None

        if evoked.nave < 1:
            warnings.warn('evoked object is empty (based on less '
                          'than 1 epoch)', RuntimeWarning)

        return evoked

    @property
    def ch_names(self):
        return self.info['ch_names']

    def plot(self, epoch_idx=None, picks=None, scalings=None,
             title_str='#%003i', show=True, block=False):
        """ Visualize single trials using Trellis plot.

        Parameters
        ----------
        epoch_idx : array-like | int | None
            The epochs to visualize. If None, the frist 20 epochs are shoen.
            Defaults to None.
        picks : array-like of int | None
            Channels to be included. If None only good data channels are used.
            Defaults to None
        scalings : dict | None
            Scale factors for the traces. If None, defaults to:
            `dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                  emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1,
                  chpi=1e-4)`
        title_str : None | str
            The string formatting to use for axes titles. If None, no titles
            will be shown. Defaults expand to ``#001, #002, ...``
        show : bool
            Whether to show the figure or not.
        block : bool
            Whether to halt program execution until the figure is closed.
            Useful for rejecting bad trials on the fly by clicking on a
            sub plot.

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            The figure.
        """
        return plot_epochs(self, epoch_idx=epoch_idx, picks=picks,
                           scalings=scalings, title_str=title_str,
                           show=show, block=block)


class Epochs(_BaseEpochs):
    """List of Epochs

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    events : array, of shape [n_events, 3]
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    name : string
        Comment that describes the Evoked data created.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
        The baseline (a, b) includes both endpoints, i.e. all
        timepoints t such that a <= t <= b.
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels
        are used).
    preload : boolean
        Load all epochs from disk when creating the object
        or wait before accessing each epoch (more memory
        efficient but can be slower).
    reject : dict
        Epoch rejection parameters based on peak to peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done.
        Values are float. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )

    flat : dict
        Epoch rejection parameters based on flatness of signal
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'
        If flat is None then no rejection is done.
    proj : bool | 'delayed'
        Apply SSP projection vectors. If proj is 'delayed' and reject is not
        None the single epochs will be projected before the rejection
        decision, but used in unprojected state if they are kept.
        This way deciding which projection vectors are good can be postponed
        to the evoked stage without resulting in lower epoch counts and
        without producing results different from early SSP application
        given comparable parameters. Note that in this case baselining,
        detrending and temporal decimation will be postponed.
        If proj is False no projections will be applied which is the
        recommended value if SSPs are not used for cleaning the data.
    decim : int
        Factor by which to downsample the data from the raw file upon import.
        Warning: This simply selects every nth sample, data is not filtered
        here. If data is not properly filtered, aliasing artifacts may occur.
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
    detrend : int | None
        If 0 or 1, the data channels (MEG and EEG) will be detrended when
        loaded. 0 is a constant (DC) detrend, 1 is a linear detrend. None
        is no detrending. Note that detrending is performed before baseline
        correction. If no DC offset is preferred (zeroth order detrending),
        either turn off baseline correction, as this may introduce a DC
        shift, or set baseline correction to use the entire time interval
        (will yield equivalent results but be slower).
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless one
        already exists).
    on_missing : str
        What to do if an event id is not found in the recording.
        Valid keys are 'error' | 'warning' | 'ignore'
        Default is 'error'. If on_missing is 'warning' it will proceed but
        warn, if 'ignore' it will proceed silently.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    Attributes
    ----------
    info: dict
        Measurement info.
    event_id : dict
        Names of  of conditions corresponding to event_ids.
    ch_names : list of string
        List of channels' names.
    selection : array
        List of indices of selected events (not dropped or ignored etc.). For
        example, if the original event array had 4 events and the second event
        has been dropped, this attribute would be np.array([0, 2, 3]).
    preload : bool
        Indicates whether epochs are in memory.
    drop_log : list of lists
        A list of the same length as the event array used to initialize the
        Epochs object. If the i-th original event is still part of the
        selection, drop_log[i] will be an empty list; otherwise it will be
        a list of the reasons the event is not longer in the selection, e.g.:

        'IGNORED' if it isn't part of the current subset defined by the user;
        'NO DATA' or 'TOO SHORT' if epoch didn't contain enough data;
        names of channels that exceeded the amplitude threshold;
        'EQUALIZED_COUNTS' (see equalize_event_counts);
        or user-defined reasons (see drop_epochs).
    verbose : bool, str, int, or None
        See above.

    Notes
    -----
    For indexing and slicing:

    epochs[idx] : Epochs
        Return Epochs object with a subset of epochs (supports single
        index and python-style slicing)

    For subset selection using categorial labels:

    epochs['name'] : Epochs
        Return Epochs object with a subset of epochs corresponding to an
        experimental condition as specified by 'name'.

    epochs[['name_1', 'name_2', ... ]] : Epochs
        Return Epochs object with a subset of epochs corresponding to multiple
        experimental conditions as specified by 'name_1', 'name_2', ... .

    See also
    --------
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts
    """
    @verbose
    def __init__(self, raw, events, event_id, tmin, tmax, baseline=(None, 0),
                 picks=None, name='Unknown', preload=False, reject=None,
                 flat=None, proj=True, decim=1, reject_tmin=None,
                 reject_tmax=None, detrend=None, add_eeg_ref=True,
                 on_missing='error', verbose=None):
        if raw is None:
            return
        elif not isinstance(raw, _BaseRaw):
            raise ValueError('The first argument to `Epochs` must be `None` '
                             'or an instance of `mne.io.Raw`')
        if on_missing not in ['error', 'warning', 'ignore']:
            raise ValueError('on_missing must be one of: error, '
                             'warning, ignore. Got: %s' % on_missing)

        # prepare for calling the base constructor

        # Handle measurement info
        info = cp.deepcopy(raw.info)
        # make sure projs are really copied.
        info['projs'] = [cp.deepcopy(p) for p in info['projs']]

        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))

        proj = proj or raw.proj  # proj is on when applied in Raw

        # call _BaseEpochs constructor
        super(Epochs, self).__init__(info, event_id, tmin, tmax,
                                     baseline=baseline, picks=picks, name=name,
                                     reject=reject, flat=flat, decim=decim,
                                     reject_tmin=reject_tmin,
                                     reject_tmax=reject_tmax, detrend=detrend,
                                     add_eeg_ref=add_eeg_ref, verbose=verbose)

        # do the rest
        self.raw = raw
        proj = proj or raw.proj  # proj is on when applied in Raw
        if proj not in [True, 'delayed', False]:
            raise ValueError(r"'proj' must either be 'True', 'False' or "
                             "'delayed'")
        self.proj = proj
        if self._check_delayed():
            logger.info('Entering delayed SSP mode.')

        activate = False if self._check_delayed() else self.proj
        self._projector, self.info = setup_proj(self.info, add_eeg_ref,
                                                activate=activate)

        for key, val in self.event_id.items():
            if val not in events[:, 2]:
                msg = ('No matching events found for %s '
                       '(event id %i)' % (key, val))
                if on_missing == 'error':
                    raise ValueError(msg)
                elif on_missing == 'warning':
                    logger.warning(msg)
                    warnings.warn(msg)
                else:  # on_missing == 'ignore':
                    pass

        # Select the desired events
        values = list(self.event_id.values())
        selected = in1d(events[:, 2], values)
        self.events = events[selected]

        n_events = len(self.events)
        if n_events > 1:
            if np.diff(self.events.astype(np.int64)[:, 0]).min() <= 0:
                warnings.warn('The events passed to the Epochs constructor '
                              'are not chronologically ordered.',
                              RuntimeWarning)

        if n_events > 0:
            logger.info('%d matching events found' % n_events)
        else:
            raise ValueError('No desired events found.')

        self.selection = np.where(selected)[0]
        self.drop_log = []
        for k in range(len(events)):
            if events[k, 2] in values:
                self.drop_log.append([])
            else:
                self.drop_log.append(['IGNORED'])

        self.preload = preload
        if self.preload:
            self._data = self._get_data_from_disk()
            self.raw = None
        else:
            self._data = None

    @deprecated('drop_picks will be removed in v0.9. Use drop_channels.')
    def drop_picks(self, bad_picks):
        """Drop some picks

        Allows to discard some channels.
        """
        idx = [k for k, p in enumerate(self.picks) if p not in bad_picks]
        self.picks = self.picks[idx]

        self.info = pick_info(self.info, idx, copy=False)

        if self._projector is not None:
            self._projector = self._projector[idx][:, idx]

        if self.preload:
            self._data = self._data[:, idx, :]

    def drop_bad_epochs(self):
        """Drop bad epochs without retaining the epochs data.

        Should be used before slicing operations.

        .. Warning:: Operation is slow since all epochs have to be read from
            disk. To avoid reading epochs form disk multiple times, initialize
            Epochs object with preload=True.

        """
        self._get_data_from_disk(out=False)

    def drop_log_stats(self, ignore=['IGNORED']):
        """Compute the channel stats based on a drop_log from Epochs.

        Parameters
        ----------
        ignore : list
            The drop reasons to ignore.

        Returns
        -------
        perc : float
            Total percentage of epochs dropped.
        """
        return _drop_log_stats(self.drop_log, ignore)

    def plot_drop_log(self, threshold=0, n_max_plot=20, subject='Unknown',
                      color=(0.9, 0.9, 0.9), width=0.8, ignore=['IGNORED'],
                      show=True, return_fig=False):
        """Show the channel stats based on a drop_log from Epochs

        Parameters
        ----------
        threshold : float
            The percentage threshold to use to decide whether or not to
            plot. Default is zero (always plot).
        n_max_plot : int
            Maximum number of channels to show stats for.
        subject : str
            The subject name to use in the title of the plot.
        color : tuple | str
            Color to use for the bars.
        width : float
            Width of the bars.
        ignore : list
            The drop reasons to ignore.
        show : bool
            Show figure if True.
        return_fig : bool
            Return only figure handle if True. This argument will default
            to True in v0.9 and then be removed.

        Returns
        -------
        perc : float
            Total percentage of epochs dropped.
        fig : Instance of matplotlib.figure.Figure
            The figure.
        """
        if not self._bad_dropped:
            print("Bad epochs have not yet been dropped.")
            return

        from .viz import plot_drop_log
        return plot_drop_log(self.drop_log, threshold, n_max_plot, subject,
                             color=color, width=width, ignore=ignore,
                             show=show, return_fig=return_fig)

    def _check_delayed(self):
        """ Aux method
        """
        is_delayed = False
        if self.proj == 'delayed':
            if self.reject is None:
                raise RuntimeError('The delayed SSP mode was requested '
                                   'but no rejection parameters are present. '
                                   'Please add rejection parameters before '
                                   'using this option.')
            is_delayed = True
        return is_delayed

    @verbose
    def drop_epochs(self, indices, reason='USER', verbose=None):
        """Drop epochs based on indices or boolean mask

        Note that the indices refer to the current set of undropped epochs
        rather than the complete set of dropped and undropped epochs.
        They are therefore not necessarily consistent with any external indices
        (e.g., behavioral logs). To drop epochs based on external criteria,
        do not use the preload=True flag when constructing an Epochs object,
        and call this method before calling the drop_bad_epochs method.

        Parameters
        ----------
        indices : array of ints or bools
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are
            correspondingly modified.
        reason : str
            Reason for dropping the epochs ('ECG', 'timeout', 'blink' etc).
            Default: 'USER'.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to raw.verbose.
        """
        indices = np.atleast_1d(indices)

        if indices.ndim > 1:
            raise ValueError("indices must be a scalar or a 1-d array")

        if indices.dtype == bool:
            indices = np.where(indices)[0]

        out_of_bounds = (indices < 0) | (indices >= len(self.events))
        if out_of_bounds.any():
            first = indices[out_of_bounds][0]
            raise IndexError("Epoch index %d is out of bounds" % first)

        for ii in indices:
            self.drop_log[self.selection[ii]].append(reason)

        self.selection = np.delete(self.selection, indices)
        self.events = np.delete(self.events, indices, axis=0)
        if self.preload:
            self._data = np.delete(self._data, indices, axis=0)

        count = len(indices)
        logger.info('Dropped %d epoch%s' % (count, '' if count == 1 else 's'))

    @verbose
    def _get_epoch_from_disk(self, idx, proj, verbose=None):
        """Load one epoch from disk"""
        if self.raw is None:
            # This should never happen, as raw=None only if preload=True
            raise ValueError('An error has occurred, no valid raw file found.'
                             ' Please report this to the mne-python '
                             'developers.')
        sfreq = self.raw.info['sfreq']

        if self.events.ndim == 1:
            # single event
            event_samp = self.events[0]
        else:
            event_samp = self.events[idx, 0]

        # Read a data segment
        first_samp = self.raw.first_samp
        start = int(round(event_samp + self.tmin * sfreq)) - first_samp
        stop = start + self._epoch_stop
        if start < 0:
            return None, None

        epoch_raw, _ = self.raw[self.picks, start:stop]

        # setup list of epochs to handle delayed SSP
        epochs = []
        # whenever requested, the first epoch is being projected.
        if self._projector is not None and proj is True:
            epochs += [np.dot(self._projector, epoch_raw)]
        else:
            epochs += [epoch_raw]

        # in case the proj passed is True but self proj is not we
        # have delayed SSP
        if self.proj != proj:  # so append another unprojected epoch
            epochs += [epoch_raw.copy()]

        # only preprocess first candidate, to make delayed SSP working
        # we need to postpone the preprocessing since projection comes
        # first.
        epochs[0] = self._preprocess(epochs[0])

        # return a second None if nothing is projected
        if len(epochs) == 1:
            epochs += [None]

        return epochs

    @verbose
    def _get_data_from_disk(self, out=True, verbose=None):
        """Load all data from disk

        Parameters
        ----------
        out : bool
            Return the data. Setting this to False is used to reject bad
            epochs without caching all the data, which saves memory.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        n_events = len(self.events)
        data = np.array([])
        if self._bad_dropped:
            proj = False if self._check_delayed() else self.proj
            if not out:
                return
            for ii in range(n_events):
                # faster to pre-allocate memory here
                epoch, epoch_raw = self._get_epoch_from_disk(ii, proj=proj)
                if ii == 0:
                    data = np.empty((n_events, epoch.shape[0],
                                     epoch.shape[1]), dtype=epoch.dtype)
                if self._check_delayed():
                    epoch = epoch_raw
                data[ii] = epoch
        else:
            proj = True if self._check_delayed() else self.proj
            good_events = []
            n_out = 0
            for idx, sel in zip(range(n_events), self.selection):
                epoch, epoch_raw = self._get_epoch_from_disk(idx, proj=proj)
                is_good, offenders = self._is_good_epoch(epoch)
                if is_good:
                    good_events.append(idx)
                    if self._check_delayed():
                        epoch = epoch_raw
                    if out:
                        # faster to pre-allocate, then trim as necessary
                        if n_out == 0:
                            data = np.empty((n_events, epoch.shape[0],
                                             epoch.shape[1]),
                                            dtype=epoch.dtype, order='C')
                        data[n_out] = epoch
                        n_out += 1
                else:
                    self.drop_log[sel] += offenders

            self.selection = self.selection[good_events]
            self.events = np.atleast_2d(self.events[good_events])
            self._bad_dropped = True
            logger.info("%d bad epochs dropped"
                        % (n_events - len(good_events)))
            if not out:
                return
            # just take the good events
            assert len(good_events) == n_out
            if n_out > 0:
                # slicing won't free the space, so we resize
                # we have ensured the C-contiguity of the array in allocation
                # so this operation will be safe unless np is very broken
                data.resize((n_out,) + data.shape[1:], refcheck=False)
        return data

    @verbose
    def _is_good_epoch(self, data, verbose=None):
        """Determine if epoch is good"""
        if data is None:
            return False, ['NO_DATA']
        n_times = len(self.times)
        if data.shape[1] < n_times:
            # epoch is too short ie at the end of the data
            return False, ['TOO_SHORT']
        if self.reject is None and self.flat is None:
            return True, None
        else:
            if self._reject_time is not None:
                data = data[:, self._reject_time]

            return _is_good(data, self.ch_names, self._channel_type_idx,
                            self.reject, self.flat, full_report=True,
                            ignore_chs=self.info['bads'])

    def get_data(self):
        """Get all epochs as a 3D array

        Returns
        -------
        data : array of shape [n_epochs, n_channels, n_times]
            The epochs data
        """
        if self.preload:
            data_ = self._data
        else:
            data_ = self._get_data_from_disk()
        if self._check_delayed():
            data = np.zeros_like(data_)
            for ii, e in enumerate(data_):
                data[ii] = self._preprocess(e.copy(), self.verbose)
        else:
            data = data_

        return data

    def _reject_setup(self):
        """Sets self._reject_time and self._channel_type_idx (called from
        __init__)
        """
        if self.reject is None and self.flat is None:
            return

        idx = channel_indices_by_type(self.info)
        for key in idx.keys():
            if (self.reject is not None and key in self.reject) \
                    or (self.flat is not None and key in self.flat):
                if len(idx[key]) == 0:
                    raise ValueError("No %s channel found. Cannot reject based"
                                     " on %s." % (key.upper(), key.upper()))

        self._channel_type_idx = idx

        if (self.reject_tmin is None) and (self.reject_tmax is None):
            self._reject_time = None
        else:
            if self.reject_tmin is None:
                reject_imin = None
            else:
                idxs = np.nonzero(self.times >= self.reject_tmin)[0]
                reject_imin = idxs[0]
            if self.reject_tmax is None:
                reject_imax = None
            else:
                idxs = np.nonzero(self.times <= self.reject_tmax)[0]
                reject_imax = idxs[-1]

            self._reject_time = slice(reject_imin, reject_imax)

    def __len__(self):
        """Number of epochs.
        """
        if not self._bad_dropped:
            err = ("Since bad epochs have not been dropped, the length of the "
                   "Epochs is not known. Load the Epochs with preload=True, "
                   "or call Epochs.drop_bad_epochs(). To find the number of "
                   "events in the Epochs, use len(Epochs.events).")
            raise RuntimeError(err)
        return len(self.events)

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self, return_event_id=False):
        """To make iteration over epochs easy.
        """
        if self.preload:
            if self._current >= len(self._data):
                raise StopIteration
            epoch = self._data[self._current]
            if self._check_delayed():
                epoch = self._preprocess(epoch.copy(), self.verbose)
            self._current += 1
        else:
            proj = True if self._check_delayed() else self.proj
            is_good = False
            while not is_good:
                if self._current >= len(self.events):
                    raise StopIteration
                epoch, epoch_raw = self._get_epoch_from_disk(self._current,
                                                             proj=proj)
                self._current += 1
                is_good, _ = self._is_good_epoch(epoch)
            # If delayed-ssp mode, pass 'virgin' data after rejection decision.
            if self._check_delayed():
                epoch = self._preprocess(epoch_raw, self.verbose)

        if not return_event_id:
            return epoch
        else:
            return epoch, self.events[self._current - 1][-1]

        return epoch if not return_event_id else epoch, self.event_id

    def __repr__(self):
        """ Build string representation
        """
        if not self._bad_dropped:
            s = 'n_events : %s (good & bad)' % len(self.events)
        else:
            s = 'n_events : %s (all good)' % len(self.events)
        s += ', tmin : %s (s)' % self.tmin
        s += ', tmax : %s (s)' % self.tmax
        s += ', baseline : %s' % str(self.baseline)
        if len(self.event_id) > 1:
            counts = ['%r: %i' % (k, sum(self.events[:, 2] == v))
                      for k, v in sorted(self.event_id.items())]
            s += ',\n %s' % ', '.join(counts)

        return '<Epochs  |  %s>' % s

    def _key_match(self, key):
        """Helper function for event dict use"""
        if key not in self.event_id:
            raise KeyError('Event "%s" is not in Epochs.' % key)
        return self.events[:, 2] == self.event_id[key]

    def __getitem__(self, key):
        """Return an Epochs object with a subset of epochs
        """
        data = self._data
        del self._data
        epochs = self.copy()
        self._data, epochs._data = data, data

        if isinstance(key, string_types):
            key = [key]

        if isinstance(key, (list, tuple)) and isinstance(key[0], string_types):
            select = np.any(np.atleast_2d([epochs._key_match(k)
                                           for k in key]), axis=0)
            epochs.name = ('+'.join(key) if epochs.name == 'Unknown'
                           else 'epochs_%s' % '+'.join(key))
        else:
            select = key if isinstance(key, slice) else np.atleast_1d(key)

        key_selection = epochs.selection[select]
        for k in np.setdiff1d(epochs.selection, key_selection):
            epochs.drop_log[k] = ['IGNORED']
        epochs.selection = key_selection
        epochs.events = np.atleast_2d(epochs.events[select])
        if epochs.preload:
            epochs._data = epochs._data[select]

        # update event id to reflect new content of epochs
        epochs.event_id = dict((k, v) for k, v in epochs.event_id.items()
                               if v in epochs.events[:, 2])

        return epochs

    def crop(self, tmin=None, tmax=None, copy=False):
        """Crops a time interval from epochs object.

        Parameters
        ----------
        tmin : float
            Start time of selection in seconds.
        tmax : float
            End time of selection in seconds.
        copy : bool
            If False epochs is cropped in place.

        Returns
        -------
        epochs : Epochs instance
            The cropped epochs.

        Note
        ----
        Unlike Python slices, MNE time intervals include both their end points;
        crop(tmin, tmax) returns the interval tmin <= t <= tmax.
        """
        if not self.preload:
            raise RuntimeError('Modifying data of epochs is only supported '
                               'when preloading is used. Use preload=True '
                               'in the constructor.')

        if tmin is None:
            tmin = self.tmin
        elif tmin < self.tmin:
            warnings.warn("tmin is not in epochs' time interval."
                          "tmin is set to epochs.tmin")
            tmin = self.tmin

        if tmax is None:
            tmax = self.tmax
        elif tmax > self.tmax:
            warnings.warn("tmax is not in epochs' time interval."
                          "tmax is set to epochs.tmax")
            tmax = self.tmax

        tmask = (self.times >= tmin) & (self.times <= tmax)
        tidx = np.where(tmask)[0]

        this_epochs = self if not copy else self.copy()
        this_epochs.tmin = this_epochs.times[tidx[0]]
        this_epochs.tmax = this_epochs.times[tidx[-1]]
        this_epochs.times = this_epochs.times[tmask]
        this_epochs._data = this_epochs._data[:, :, tmask]
        return this_epochs

    @verbose
    def resample(self, sfreq, npad=100, window='boxcar', n_jobs=1,
                 verbose=None):
        """Resample preloaded data

        Parameters
        ----------
        sfreq : float
            New sample rate to use
        npad : int
            Amount to pad the start and end of the data.
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        n_jobs : int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        if self.preload:
            o_sfreq = self.info['sfreq']
            self._data = resample(self._data, sfreq, o_sfreq, npad,
                                  n_jobs=n_jobs)
            # adjust indirectly affected variables
            self.info['sfreq'] = sfreq
            self.times = (np.arange(self._data.shape[2], dtype=np.float)
                          / sfreq + self.times[0])
        else:
            raise RuntimeError('Can only resample preloaded data')

    def copy(self):
        """Return copy of Epochs instance"""
        raw = self.raw
        del self.raw
        new = cp.deepcopy(self)
        self.raw = raw
        new.raw = raw

        return new

    def save(self, fname):
        """Save epochs in a fif file

        Parameters
        ----------
        fname : str
            The name of the file, which should end with -epo.fif or
            -epo.fif.gz.
        """
        check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz'))

        # Create the file and save the essentials
        fid = start_file(fname)

        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        if self.info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, self.info['meas_id'])

        # Write measurement info
        write_meas_info(fid, self.info)

        # One or more evoked data sets
        start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        start_block(fid, FIFF.FIFFB_EPOCHS)

        # write events out after getting data to ensure bad events are dropped
        data = self.get_data()
        start_block(fid, FIFF.FIFFB_MNE_EVENTS)
        write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, self.events.T)
        mapping_ = ';'.join([k + ':' + str(v) for k, v in
                             self.event_id.items()])
        write_string(fid, FIFF.FIFF_DESCRIPTION, mapping_)
        end_block(fid, FIFF.FIFFB_MNE_EVENTS)

        # First and last sample
        first = int(self.times[0] * self.info['sfreq'])
        last = first + len(self.times) - 1
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first)
        write_int(fid, FIFF.FIFF_LAST_SAMPLE, last)

        # save baseline
        if self.baseline is not None:
            bmin, bmax = self.baseline
            bmin = self.times[0] if bmin is None else bmin
            bmax = self.times[-1] if bmax is None else bmax
            write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, bmin)
            write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX, bmax)

        # The epochs itself
        decal = np.empty(self.info['nchan'])
        for k in range(self.info['nchan']):
            decal[k] = 1.0 / (self.info['chs'][k]['cal']
                              * self.info['chs'][k].get('scale', 1.0))

        data *= decal[np.newaxis, :, np.newaxis]

        write_float_matrix(fid, FIFF.FIFF_EPOCH, data)

        # undo modifications to data
        data /= decal[np.newaxis, :, np.newaxis]

        write_string(fid, FIFF.FIFFB_MNE_EPOCHS_DROP_LOG,
                     json.dumps(self.drop_log))

        write_int(fid, FIFF.FIFFB_MNE_EPOCHS_SELECTION,
                  self.selection)

        end_block(fid, FIFF.FIFFB_EPOCHS)

        end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        end_block(fid, FIFF.FIFFB_MEAS)
        end_file(fid)

    def as_data_frame(self, picks=None, index=None, scale_time=1e3,
                      scalings=None, copy=True):
        """Get the epochs as Pandas DataFrame

        Export epochs data in tabular structure with MEG channels as columns
        and three additional info columns 'epoch', 'condition', and 'time'.
        The format matches a long table format commonly used to represent
        repeated measures in within-subject designs.

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.
        index : tuple of str | None
            Column to be used as index for the data. Valid string options
            are 'epoch', 'time' and 'condition'. If None, all three info
            columns will be included in the table as categorial data.
        scale_time : float
            Scaling to be applied to time units.
        scalings : dict | None
            Scaling to be applied to the channels picked. If None, defaults to
            ``scalings=dict(eeg=1e6, grad=1e13, mag=1e15, misc=1.0)`.
        copy : bool
            If true, data will be copied. Else data may be modified in place.

        Returns
        -------
        df : instance of pandas.core.DataFrame
            Epochs exported into tabular data structure.
        """

        pd = _check_pandas_installed()

        default_index = ['condition', 'epoch', 'time']
        if index is not None:
            _check_pandas_index_arguments(index, default_index)
        else:
            index = default_index

        if picks is None:
            picks = list(range(self.info['nchan']))
        else:
            if not in1d(picks, np.arange(len(self.events))).all():
                raise ValueError('At least one picked channel is not present '
                                 'in this eppochs instance.')

        data = self.get_data()[:, picks, :]
        shape = data.shape
        data = np.hstack(data).T
        if copy:
            data = data.copy()

        types = [channel_type(self.info, idx) for idx in picks]
        n_channel_types = 0
        ch_types_used = []

        scalings = _mutable_defaults(('scalings', scalings))[0]
        for t in scalings.keys():
            if t in types:
                n_channel_types += 1
                ch_types_used.append(t)

        for t in ch_types_used:
            scaling = scalings[t]
            idx = [picks[i] for i in range(len(picks)) if types[i] == t]
            if len(idx) > 0:
                data[:, idx] *= scaling

        id_swapped = dict((v, k) for k, v in self.event_id.items())
        names = [id_swapped[k] for k in self.events[:, 2]]

        mindex = list()
        mindex.append(('condition', np.repeat(names, shape[2])))
        mindex.append(('time', np.tile(self.times, shape[0]) *
                      scale_time))  # if 'epoch' in index:
        mindex.append(('epoch', np.repeat(np.arange(shape[0]),
                      shape[2])))

        assert all(len(mdx) == len(mindex[0]) for mdx in mindex)
        col_names = [self.ch_names[k] for k in picks]

        df = pd.DataFrame(data, columns=col_names)
        [df.insert(i, k, v) for i, (k, v) in enumerate(mindex)]
        if index is not None:
            with warnings.catch_warnings(record=True):
                if 'time' in index:
                    df['time'] = df['time'].astype(np.int64)
                df.set_index(index, inplace=True)

        return df

    def to_nitime(self, picks=None, epochs_idx=None, collapse=False,
                  copy=True, first_samp=0):
        """ Export epochs as nitime TimeSeries

        Parameters
        ----------
        picks : array-like of int | None
            Indices for exporting subsets of the epochs channels. If None
            all good channels will be used.
        epochs_idx : slice | array-like | None
            Epochs index for single or selective epochs exports. If None, all
            epochs will be used.
        collapse : boolean
            If True export epochs and time slices will be collapsed to 2D
            array. This may be required by some nitime functions.
        copy : boolean
            If True exports copy of epochs data.
        first_samp : int
            Number of samples to offset the times by. Use raw.first_samp to
            have the time returned relative to the session onset, or zero
            (default) for time relative to the recording onset.

        Returns
        -------
        epochs_ts : instance of nitime.TimeSeries
            The Epochs as nitime TimeSeries object.
        """
        try:
            from nitime import TimeSeries  # to avoid strong dependency
        except ImportError:
            raise Exception('the nitime package is missing')

        if picks is None:
            picks = pick_types(self.info, include=self.ch_names,
                               exclude='bads')
        if epochs_idx is None:
            epochs_idx = slice(len(self.events))

        data = self.get_data()[epochs_idx, picks]

        if copy is True:
            data = data.copy()

        if collapse is True:
            data = np.hstack(data).copy()

        offset = _time_as_index(abs(self.tmin), self.info['sfreq'],
                                first_samp, True)
        t0 = _index_as_time(self.events[0, 0] - offset, self.info['sfreq'],
                            first_samp, True)[0]
        epochs_ts = TimeSeries(data, sampling_rate=self.info['sfreq'], t0=t0)
        epochs_ts.ch_names = np.array(self.ch_names)[picks].tolist()

        return epochs_ts

    def equalize_event_counts(self, event_ids, method='mintime', copy=True):
        """Equalize the number of trials in each condition

        It tries to make the remaining epochs occurring as close as possible in
        time. This method works based on the idea that if there happened to be
        some time-varying (like on the scale of minutes) noise characteristics
        during a recording, they could be compensated for (to some extent) in
        the equalization process. This method thus seeks to reduce any of
        those effects by minimizing the differences in the times of the events
        in the two sets of epochs. For example, if one had event times
        [1, 2, 3, 4, 120, 121] and the other one had [3.5, 4.5, 120.5, 121.5],
        it would remove events at times [1, 2] in the first epochs and not
        [20, 21].

        Parameters
        ----------
        event_ids : list
            The event types to equalize. Each entry in the list can either be
            a str (single event) or a list of str. In the case where one of
            the entries is a list of str, event_ids in that list will be
            grouped together before equalizing trial counts across conditions.
        method : str
            If 'truncate', events will be truncated from the end of each event
            list. If 'mintime', timing differences between each event list will
            be minimized.
        copy : bool
            If True, a copy of epochs will be returned. Otherwise, the
            function will operate in-place.

        Returns
        -------
        epochs : instance of Epochs
            The modified Epochs instance.
        indices : array of int
            Indices from the original events list that were dropped.

        Notes
        ----
        For example (if epochs.event_id was {'Left': 1, 'Right': 2,
        'Nonspatial':3}:

            epochs.equalize_event_counts([['Left', 'Right'], 'Nonspatial'])

        would equalize the number of trials in the 'Nonspatial' condition with
        the total number of trials in the 'Left' and 'Right' conditions.
        """
        if copy is True:
            epochs = self.copy()
        else:
            epochs = self
        if len(event_ids) == 0:
            raise ValueError('event_ids must have at least one element')
        if not epochs._bad_dropped:
            epochs.drop_bad_epochs()
        # figure out how to equalize
        eq_inds = list()
        for eq in event_ids:
            eq = np.atleast_1d(eq)
            # eq is now a list of types
            key_match = np.zeros(epochs.events.shape[0])
            for key in eq:
                key_match = np.logical_or(key_match, epochs._key_match(key))
            eq_inds.append(np.where(key_match)[0])

        event_times = [epochs.events[e, 0] for e in eq_inds]
        indices = _get_drop_indices(event_times, method)
        # need to re-index indices
        indices = np.concatenate([e[idx] for e, idx in zip(eq_inds, indices)])
        epochs.drop_epochs(indices, reason='EQUALIZED_COUNT')
        # actually remove the indices
        return epochs, indices


class EpochsArray(Epochs):
    """Epochs object from numpy array

    Parameters
    ----------
    data : array, shape (n_epochs, n_channels, n_times)
        The channels' time series for each epoch.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
    tmin : float
        Start time before event.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    reject : dict
        Epoch rejection parameters based on peak to peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done.
        Values are float. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )

    flat : dict
        Epoch rejection parameters based on flatness of signal
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'
        If flat is None then no rejection is done.
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.
    """

    @verbose
    def __init__(self, data, info, events, tmin=0, event_id=None,
                 reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, verbose=None):

        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)

        if data.ndim != 3:
            raise ValueError('Data must be a 3D array of shape (n_epochs, '
                             'n_channels, n_samples)')

        if len(info['ch_names']) != np.shape(data)[1]:
            raise ValueError('Info and data must have same number of '
                             'channels.')

        self.info = info
        self._data = data
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
        self.event_id = event_id
        self.events = events

        for key, val in self.event_id.items():
            if val not in events[:, 2]:
                msg = ('No matching events found for %s '
                       '(event id %i)' % (key, val))
                raise ValueError(msg)

        self.proj = None
        self.baseline = None
        self.preload = True
        self.reject = None
        self.decim = 1
        self._decim_idx = slice(0, data.shape[-1], self.decim)
        self.raw = None
        self.drop_log = [[] for _ in range(len(events))]
        self._bad_dropped = True

        self.selection = np.arange(len(events))
        self.picks = None
        self.times = (np.arange(data.shape[-1], dtype=np.float) /
                      info['sfreq'] + tmin)
        self.tmin = self.times[0]
        self.tmax = self.times[-1]
        self.verbose = verbose
        self.name = 'Unknown'
        self._projector = None
        self.reject = reject
        self.flat = flat
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        self._reject_setup()
        drop_inds = list()
        if self.reject is not None or self.flat is not None:
            for i_epoch, epoch in enumerate(self):
                is_good, chan = self._is_good_epoch(epoch,
                                                    verbose=self.verbose)
                if not is_good:
                    drop_inds.append(i_epoch)
                    self.drop_log[i_epoch].extend(chan)
        if drop_inds:
            select = np.ones(len(events), dtype=np.bool)
            select[drop_inds] = False
            self.events = self.events[select]
            self._data = self._data[select]
            self.selection[select]


def combine_event_ids(epochs, old_event_ids, new_event_id, copy=True):
    """Collapse event_ids from an epochs instance into a new event_id

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to operate on.
    old_event_ids : str, or list
        Conditions to collapse together.
    new_event_id : dict, or int
        A one-element dict (or a single integer) for the new
        condition. Note that for safety, this cannot be any
        existing id (in epochs.event_id.values()).
    copy : bool
        If True, a copy of epochs will be returned. Otherwise, the
        function will operate in-place.

    Notes
    -----
    This For example (if epochs.event_id was {'Left': 1, 'Right': 2}:

        combine_event_ids(epochs, ['Left', 'Right'], {'Directional': 12})

    would create a 'Directional' entry in epochs.event_id replacing
    'Left' and 'Right' (combining their trials).
    """
    if copy:
        epochs = epochs.copy()
    old_event_ids = np.asanyarray(old_event_ids)
    if isinstance(new_event_id, int):
        new_event_id = {str(new_event_id): new_event_id}
    else:
        if not isinstance(new_event_id, dict):
            raise ValueError('new_event_id must be a dict or int')
        if not len(list(new_event_id.keys())) == 1:
            raise ValueError('new_event_id dict must have one entry')
    new_event_num = list(new_event_id.values())[0]
    if not isinstance(new_event_num, int):
        raise ValueError('new_event_id value must be an integer')
    if new_event_num in epochs.event_id.values():
        raise ValueError('new_event_id value must not already exist')
    # could use .pop() here, but if a latter one doesn't exist, we're
    # in trouble, so run them all here and pop() later
    old_event_nums = np.array([epochs.event_id[key] for key in old_event_ids])
    # find the ones to replace
    inds = np.any(epochs.events[:, 2][:, np.newaxis] ==
                  old_event_nums[np.newaxis, :], axis=1)
    # replace the event numbers in the events list
    epochs.events[inds, 2] = new_event_num
    # delete old entries
    [epochs.event_id.pop(key) for key in old_event_ids]
    # add the new entry
    epochs.event_id.update(new_event_id)
    return epochs


def equalize_epoch_counts(epochs_list, method='mintime'):
    """Equalize the number of trials in multiple Epoch instances

    It tries to make the remaining epochs occurring as close as possible in
    time. This method works based on the idea that if there happened to be some
    time-varying (like on the scale of minutes) noise characteristics during
    a recording, they could be compensated for (to some extent) in the
    equalization process. This method thus seeks to reduce any of those effects
    by minimizing the differences in the times of the events in the two sets of
    epochs. For example, if one had event times [1, 2, 3, 4, 120, 121] and the
    other one had [3.5, 4.5, 120.5, 121.5], it would remove events at times
    [1, 2] in the first epochs and not [20, 21].

    Note that this operates on the Epochs instances in-place.

    Example:

        equalize_epoch_counts(epochs1, epochs2)

    Parameters
    ----------
    epochs_list : list of Epochs instances
        The Epochs instances to equalize trial counts for.
    method : str
        If 'truncate', events will be truncated from the end of each event
        list. If 'mintime', timing differences between each event list will be
        minimized.
    """
    if not all([isinstance(e, Epochs) for e in epochs_list]):
        raise ValueError('All inputs must be Epochs instances')

    # make sure bad epochs are dropped
    [e.drop_bad_epochs() if not e._bad_dropped else None for e in epochs_list]
    event_times = [e.events[:, 0] for e in epochs_list]
    indices = _get_drop_indices(event_times, method)
    for e, inds in zip(epochs_list, indices):
        e.drop_epochs(inds, reason='EQUALIZED_COUNT')


def _get_drop_indices(event_times, method):
    """Helper to get indices to drop from multiple event timing lists"""
    small_idx = np.argmin([e.shape[0] for e in event_times])
    small_e_times = event_times[small_idx]
    if not method in ['mintime', 'truncate']:
        raise ValueError('method must be either mintime or truncate, not '
                         '%s' % method)
    indices = list()
    for e in event_times:
        if method == 'mintime':
            mask = _minimize_time_diff(small_e_times, e)
        else:
            mask = np.ones(e.shape[0], dtype=bool)
            mask[small_e_times.shape[0]:] = False
        indices.append(np.where(np.logical_not(mask))[0])

    return indices


def _minimize_time_diff(t_shorter, t_longer):
    """Find a boolean mask to minimize timing differences"""
    keep = np.ones((len(t_longer)), dtype=bool)
    scores = np.ones((len(t_longer)))
    for iter in range(len(t_longer) - len(t_shorter)):
        scores.fill(np.inf)
        # Check every possible removal to see if it minimizes
        for idx in np.where(keep)[0]:
            keep[idx] = False
            scores[idx] = _area_between_times(t_shorter, t_longer[keep])
            keep[idx] = True
        keep[np.argmin(scores)] = False
    return keep


def _area_between_times(t1, t2):
    """Quantify the difference between two timing sets"""
    x1 = list(range(len(t1)))
    x2 = list(range(len(t2)))
    xs = np.concatenate((x1, x2))
    return np.sum(np.abs(np.interp(xs, x1, t1) - np.interp(xs, x2, t2)))


@verbose
def _is_good(e, ch_names, channel_type_idx, reject, flat, full_report=False,
             ignore_chs=[], verbose=None):
    """Test if data segment e is good according to the criteria
    defined in reject and flat. If full_report=True, it will give
    True/False as well as a list of all offending channels.
    """
    bad_list = list()
    has_printed = False
    checkable = np.ones(len(ch_names), dtype=bool)
    checkable[np.array([c in ignore_chs
                        for c in ch_names], dtype=bool)] = False
    for refl, f, t in zip([reject, flat], [np.greater, np.less], ['', 'flat']):
        if refl is not None:
            for key, thresh in six.iteritems(refl):
                idx = channel_type_idx[key]
                name = key.upper()
                if len(idx) > 0:
                    e_idx = e[idx]
                    deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)
                    checkable_idx = checkable[idx]
                    idx_deltas = np.where(np.logical_and(f(deltas, thresh),
                                                         checkable_idx))[0]

                    if len(idx_deltas) > 0:
                        ch_name = [ch_names[idx[i]] for i in idx_deltas]
                        if (not has_printed):
                            logger.info('    Rejecting %s epoch based on %s : '
                                        '%s' % (t, name, ch_name))
                            has_printed = True
                        if not full_report:
                            return False
                        else:
                            bad_list.extend(ch_name)

    if not full_report:
        return True
    else:
        if bad_list == []:
            return True, None
        else:
            return False, bad_list


@verbose
def read_epochs(fname, proj=True, add_eeg_ref=True, verbose=None):
    """Read epochs from a fif file

    Parameters
    ----------
    fname : str
        The name of the file, which should end with -epo.fif or -epo.fif.gz.
    proj : bool | 'delayed'
        Apply SSP projection vectors. If proj is 'delayed' and reject is not
        None the single epochs will be projected before the rejection
        decision, but used in unprojected state if they are kept.
        This way deciding which projection vectors are good can be postponed
        to the evoked stage without resulting in lower epoch counts and
        without producing results different from early SSP application
        given comparable parameters. Note that in this case baselining,
        detrending and temporal decimation will be postponed.
        If proj is False no projections will be applied which is the
        recommended value if SSPs are not used for cleaning the data.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless one
        already exists).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    Returns
    -------
    epochs : instance of Epochs
        The epochs
    """
    check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz'))

    epochs = Epochs(None, None, None, None, None)

    logger.info('Reading %s ...' % fname)
    fid, tree, _ = fiff_open(fname)

    #   Read the measurement info
    info, meas = read_meas_info(fid, tree)
    info['filename'] = fname

    events, mappings = _read_events_fif(fid, tree)

    #   Locate the data of interest
    processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
    if len(processed) == 0:
        fid.close()
        raise ValueError('Could not find processed data')

    epochs_node = dir_tree_find(tree, FIFF.FIFFB_EPOCHS)
    if len(epochs_node) == 0:
        fid.close()
        raise ValueError('Could not find epochs data')

    my_epochs = epochs_node[0]

    # Now find the data in the block
    comment = None
    data = None
    bmin, bmax = None, None
    baseline = None
    selection = None
    drop_log = []
    for k in range(my_epochs['nent']):
        kind = my_epochs['directory'][k].kind
        pos = my_epochs['directory'][k].pos
        if kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, pos)
            first = int(tag.data)
        elif kind == FIFF.FIFF_LAST_SAMPLE:
            tag = read_tag(fid, pos)
            last = int(tag.data)
        elif kind == FIFF.FIFF_COMMENT:
            tag = read_tag(fid, pos)
            comment = tag.data
        elif kind == FIFF.FIFF_EPOCH:
            tag = read_tag(fid, pos)
            data = tag.data.astype(np.float)
        elif kind == FIFF.FIFF_MNE_BASELINE_MIN:
            tag = read_tag(fid, pos)
            bmin = float(tag.data)
        elif kind == FIFF.FIFF_MNE_BASELINE_MAX:
            tag = read_tag(fid, pos)
            bmax = float(tag.data)
        elif kind == FIFF.FIFFB_MNE_EPOCHS_SELECTION:
            tag = read_tag(fid, pos)
            selection = np.array(tag.data)
        elif kind == FIFF.FIFFB_MNE_EPOCHS_DROP_LOG:
            tag = read_tag(fid, pos)
            drop_log = json.loads(tag.data)

    if bmin is not None or bmax is not None:
        baseline = (bmin, bmax)

    nsamp = last - first + 1
    logger.info('    Found the data of interest:')
    logger.info('        t = %10.2f ... %10.2f ms (%s)'
                % (1000 * first / info['sfreq'],
                   1000 * last / info['sfreq'], comment))
    if info['comps'] is not None:
        logger.info('        %d CTF compensation matrices available'
                    % len(info['comps']))

    # Read the data
    if data is None:
        raise ValueError('Epochs data not found')

    if data.shape[2] != nsamp:
        fid.close()
        raise ValueError('Incorrect number of samples (%d instead of %d)'
                         % (data.shape[2], nsamp))

    # Calibrate
    cals = np.array([info['chs'][k]['cal'] * info['chs'][k].get('scale', 1.0)
                     for k in range(info['nchan'])])
    data *= cals[np.newaxis, :, np.newaxis]

    times = np.arange(first, last + 1, dtype=np.float) / info['sfreq']
    tmin = times[0]
    tmax = times[-1]

    # Put it all together
    epochs.preload = True
    epochs.raw = None
    epochs.picks = np.arange(data.shape[1])
    epochs._bad_dropped = True
    epochs.events = events
    epochs._data = data
    epochs.info = info
    epochs.tmin = tmin
    epochs.tmax = tmax
    epochs.name = comment
    epochs.times = times
    epochs._data = data
    epochs.proj = proj
    activate = False if epochs._check_delayed() else proj
    epochs._projector, epochs.info = setup_proj(info, add_eeg_ref,
                                                activate=activate)

    epochs.baseline = baseline
    epochs.event_id = (dict((str(e), e) for e in np.unique(events[:, 2]))
                       if mappings is None else mappings)
    epochs.verbose = verbose

    # In case epochs didn't have a FIFF.FIFFB_MNE_EPOCHS_SELECTION tag
    # (version < 0.8):
    if selection is None:
        selection = range(len(epochs))

    epochs.selection = selection
    epochs.drop_log = drop_log
    fid.close()

    return epochs


def bootstrap(epochs, random_state=None):
    """Compute epochs selected by bootstrapping

    Parameters
    ----------
    epochs : Epochs instance
        epochs data to be bootstrapped
    random_state : None | int | np.random.RandomState
        To specify the random generator state

    Returns
    -------
    epochs : Epochs instance
        The bootstrap samples
    """
    if not epochs.preload:
        raise RuntimeError('Modifying data of epochs is only supported '
                           'when preloading is used. Use preload=True '
                           'in the constructor.')

    rng = check_random_state(random_state)
    epochs_bootstrap = epochs.copy()
    n_events = len(epochs_bootstrap.events)
    idx = rng.randint(0, n_events, n_events)
    epochs_bootstrap = epochs_bootstrap[idx]
    return epochs_bootstrap


def _check_merge_epochs(epochs_list):
    """Aux function"""
    event_ids = set(tuple(epochs.event_id.items()) for epochs in epochs_list)
    if len(event_ids) == 1:
        event_id = dict(event_ids.pop())
    else:
        raise NotImplementedError("Epochs with unequal values for event_id")

    tmins = set(epochs.tmin for epochs in epochs_list)
    if len(tmins) == 1:
        tmin = tmins.pop()
    else:
        raise NotImplementedError("Epochs with unequal values for tmin")

    tmaxs = set(epochs.tmax for epochs in epochs_list)
    if len(tmaxs) == 1:
        tmax = tmaxs.pop()
    else:
        raise NotImplementedError("Epochs with unequal values for tmax")

    baselines = set(epochs.baseline for epochs in epochs_list)
    if len(baselines) == 1:
        baseline = baselines.pop()
    else:
        raise NotImplementedError("Epochs with unequal values for baseline")

    return event_id, tmin, tmax, baseline


@verbose
def add_channels_epochs(epochs_list, name='Unknown', add_eeg_ref=True,
                        verbose=None):
    """Concatenate channels, info and data from two Epochs objects

    Parameters
    ----------
    epochs_list : list of Epochs
        Epochs object to concatenate.
    name : str
        Comment that describes the Evoked data created.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless there is no
        EEG in the data).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to True if any of the input epochs have verbose=True.

    Returns
    -------
    epochs : Epochs
        Concatenated epochs.
    """
    if not np.all([e.preload for e in epochs_list]):
        raise ValueError('All epochs must be preloaded.')

    info = _merge_info([epochs.info for epochs in epochs_list])
    data = [epochs.get_data() for epochs in epochs_list]
    event_id, tmin, tmax, baseline = _check_merge_epochs(epochs_list)

    for d in data:
        if len(d) != len(data[0]):
            raise ValueError('all epochs must be of the same length')

    data = np.concatenate(data, axis=1)

    if len(info['chs']) != data.shape[1]:
        err = "Data shape does not match channel number in measurement info"
        raise RuntimeError(err)

    events = epochs_list[0].events.copy()
    all_same = np.all([events == epochs.events for epochs in epochs_list[1:]],
                      axis=0)
    if not np.all(all_same):
        raise ValueError('Events must be the same.')

    proj = any(e.proj for e in epochs_list) or add_eeg_ref

    if verbose is None:
        verbose = any(e.verbose for e in epochs_list)

    epochs = epochs_list[0].copy()
    epochs.info = info
    epochs.event_id = event_id
    epochs.tmin = tmin
    epochs.tmax = tmax
    epochs.baseline = baseline
    epochs.picks = None
    epochs.name = name
    epochs.verbose = verbose
    epochs.events = events
    epochs.preload = True
    epochs._bad_dropped = True
    epochs._data = data
    epochs.proj = proj
    epochs._projector, epochs.info = setup_proj(epochs.info, add_eeg_ref,
                                                activate=proj)
    return epochs
