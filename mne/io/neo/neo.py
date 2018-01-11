from copy import deepcopy
import numpy as np

from ..base import BaseRaw
from ... import create_info, Annotations
from ...channels import Montage


class BaseNeoRaw(BaseRaw):
    """Base class that wraps the NEO rawio.

    Parameters
    ----------
    fname : str
        The filename to be read. The extension supported depends on the
        actual subclass.
    preload : bool
        Whether to load the data into memory or not. Defaults to False.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    The `BaseNeoRaw` class is public to allow for stable type-checking in user
    code (i.e., ``isinstance(my_raw_object, BaseNeoRaw)``) but should not be
    used as a constructor for `Raw` objects (use instead one of the subclass
    constructors, or one of the ``mne.io.read_raw_*`` functions).

    Subclasses can optionall provide the following methods:

        * _trans_neo_montage_coords(self, coords) -> (np.ndarray, str)
          (only needed if the NEO actually parses electrode coordinates)

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    _neo_raw_class = None

    def __init__(self, fname, preload=False, verbose=None,
                 **neo_args):
        """Base class that wraps the NEO rawio."""
        if self._neo_raw_class is None:
            raise ValueError('I could not find NEO. Make sure to '
                             'have installed the latest version of NEO.')
        if self._neo_raw_class.rawmode in ('one-file', 'multi-file'):
            self._reader = self._neo_raw_class(filename=fname, **neo_args)
        elif self._neo_raw_class.rawmode == 'one-dir':
            self._reader = self._neo_raw_class(dirname=fname, **neo_args)

        self._reader.parse_header()
        self._neo_args = neo_args
        # make info OK
        ch_names = list(self._reader.header['signal_channels']['name'])
        sfreq = self._reader.get_signal_sampling_rate()
        units = self._reader.header['signal_channels']['units']

        self._volt_gains = []
        ch_types = []
        for uu in units:
            if uu.endswith('V'):
                ch_types.append('eeg')
                if uu == 'uV':
                    gain = 1e-6
                elif uu == 'mV':
                    gain = 1e-3
                elif uu == 'V':
                    gain = 1.
                else:
                    raise ValueError('Unit unkown!')
            else:
                ch_types.append('misc')
                gain = 1.
            self._volt_gains.append(gain)
        self._volt_gains = np.array(self._volt_gains)

        info = create_info(ch_names, sfreq, ch_types=ch_types)
        info['buffer_size_sec'] = 1.
        last_samps = self._reader.get_signal_size(
            block_index=0, seg_index=0,
            channel_indexes=None)
        last_samps = (last_samps - 1,)
        BaseRaw.__init__(self, info=info, preload=preload,
                         last_samps=last_samps)
        self.annotations = self._get_annotations_from_events()

        self._set_neo_montage()

    def _get_neo_coordinates(self):
        """Get coordinates from NEO reader if available."""
        sig_chs = self._reader.header['signal_channels']
        annot_chs = self._reader.raw_annotations['signal_channels']
        has_coords = np.array(['coordinates' in ch for ch in annot_chs])
        if np.any(has_coords):
            ch_names = list(sig_chs['name'][has_coords])
            coords = list()
            for c_idx in np.nonzero(has_coords)[0]:
                this_coords = annot_chs[c_idx]['coordinates']
                coords.append(this_coords)
            coords = np.array(coords)
        else:
            coords, ch_names = None, []

        return coords, ch_names

    def _set_neo_montage(self):
        """Set the montage from NEO if it can."""
        from mne.utils import logger
        coords, ch_names = self._get_neo_coordinates()
        if coords is not None:
            pos, montage_kind = self._trans_neo_montage_coords(coords)
            montage = Montage(
                pos=pos, ch_names=ch_names, kind=montage_kind,
                selection=np.arange(len(pos)).astype(np.int))
            logger.info('Setting the montage I found in your file.')
            self.set_montage(montage)
        else:
            logger.info('Did not find any montage in your data.')

    def _trans_neo_montage_coords(self, coords):
        """You have to implement coordinate transforms."""
        raise NotImplementedError('The transform is not implemented.')

    def _get_annotations_from_events(self):
        """Create annotations from events parsed by NEO."""
        reader = self._reader

        time_stamps, durations, values = list(), list(), list()
        for c_idx in range(self._reader.event_channels_count()):
            time_stamps_, durations_, values_ = reader.get_event_timestamps(
                block_index=0, seg_index=0, event_channel_index=c_idx)
            time_stamps_ = reader.rescale_event_timestamp(time_stamps_)
            time_stamps_ -= reader.get_signal_t_start(
                block_index=0, seg_index=0)
            if durations_ is None:
                durations_ = np.zeros(len(time_stamps_))
            event_ch_name = self._reader.header[
                'event_channels'][c_idx]['name']
            time_stamps.append(time_stamps_)
            durations.append(durations_)
            values.append(['%s/%s' % (event_ch_name, val) for val in values_])

        time_stamps, durations, values = [
            np.concatenate(aa) for aa in (time_stamps, durations, values)]
        return Annotations(
            onset=time_stamps,
            duration=durations, description=values, orig_time=None)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read segments from NEO io."""
        raw_sigs = self._reader.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            i_start=start,
            i_stop=stop,
            channel_indexes=idx)

        channels_hdr = self._reader.header['signal_channels']
        data[:] = raw_sigs.T
        gain = channels_hdr['gain'][idx] * self._volt_gains[idx]
        offset = channels_hdr['offset'][idx] * self._volt_gains[idx]

        data[:] *= gain[:, np.newaxis]
        data[:] += offset[:, np.newaxis]
        return data

    def copy(self):
        """Return copy of Raw instance."""
        reader_old = self._reader
        reader_new = self._neo_raw_class(
            self._reader.filename, **self._neo_args)
        delattr(self, '_reader')
        out = deepcopy(self)
        reader_new.parse_header()
        self._reader = reader_old
        out._reader = reader_new
        return out


class RawNeoMicroMed(BaseNeoRaw):
    """Read MicroMed EEG data.

    Parameters
    ----------
    fname : str
        The filename to be read. The extension supported depends on the
        actual subclass.
    preload : bool
        Whether to load the data into memory or not. Defaults to False.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    try:
        from neo.rawio import MicromedRawIO
        _neo_raw_class = MicromedRawIO
    except ImportError:
        pass


class RawNeoBrainVision(BaseNeoRaw):
    """Read Brainvision EEG data.

    Parameters
    ----------
    fname : str
        The filename to be read. The extension supported depends on the
        actual subclass.
    preload : bool
        Whether to load the data into memory or not. Defaults to False.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    try:
        from neo.rawio import BrainVisionRawIO
        _neo_raw_class = BrainVisionRawIO
    except ImportError:
        pass

    def _trans_neo_montage_coords(self, coords):
        """Transform spherical to cartesian coordinates for Bainvision EEG.

        Parameters
        ----------
        coords : np.ndarray, shape(n_sensors, 3)
            The spherical sensor coordinates.

        Returns
        -------
        pos : np.ndarray, shape(n_sensors, 3)
            The final sensor position in cartesian coordinates ase used
            for construction of Montages.
        """
        from mne.transforms import _sph_to_cart
        radius, theta, phi = coords.T
        # 1: radius, 2: theta, 3: phi
        pol = np.deg2rad(theta)
        az = np.deg2rad(phi)
        sph = np.array([radius * 85., az, pol]).T
        pos = _sph_to_cart(sph)
        return pos, 'Brainvision'


def read_raw_micromed_neo(fname, preload=False, verbose=None):
    """Read Micromed EEG data.

    Parameters
    ----------
    fname : str
        The filename to be read. The extension supported depends on the
        actual subclass.
    preload : bool
        Whether to load the data into memory or not. Defaults to False.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : instance of mne.RawMicromedNeo.
    """
    return RawNeoMicroMed(fname=fname, preload=preload, verbose=verbose)


def read_raw_brainvision_neo(fname, preload=False, verbose=None):
    """Read Micromed EEG data.

    Parameters
    ----------
    fname : str
        The filename to be read. The extension supported depends on the
        actual subclass.
    preload : bool
        Whether to load the data into memory or not. Defaults to False.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawBrainVisionNeo(fname=fname, preload=preload, verbose=verbose)
