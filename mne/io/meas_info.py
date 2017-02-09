# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from collections import Counter
from copy import deepcopy
from datetime import datetime as dt
import os.path as op
import re

import numpy as np
from scipy import linalg

from .pick import channel_type
from .constants import FIFF
from .open import fiff_open
from .tree import dir_tree_find
from .tag import read_tag, find_tag
from .proj import _read_proj, _write_proj, _uniquify_projs, _normalize_proj
from .ctf_comp import read_ctf_comp, write_ctf_comp
from .write import (start_file, end_file, start_block, end_block,
                    write_string, write_dig_points, write_float, write_int,
                    write_coord_trans, write_ch_info, write_name_list,
                    write_julian, write_float_matrix)
from .proc_history import _read_proc_history, _write_proc_history
from ..transforms import _to_const
from ..utils import logger, verbose, warn, object_diff
from .. import __version__
from ..externals.six import b, BytesIO, string_types, text_type


_kind_dict = dict(
    eeg=(FIFF.FIFFV_EEG_CH, FIFF.FIFFV_COIL_EEG, FIFF.FIFF_UNIT_V),
    mag=(FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_VV_MAG_T3, FIFF.FIFF_UNIT_T),
    grad=(FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_VV_PLANAR_T1, FIFF.FIFF_UNIT_T_M),
    ref_meg=(FIFF.FIFFV_REF_MEG_CH, FIFF.FIFFV_COIL_VV_MAG_T3,
             FIFF.FIFF_UNIT_T),
    misc=(FIFF.FIFFV_MISC_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_NONE),
    stim=(FIFF.FIFFV_STIM_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    eog=(FIFF.FIFFV_EOG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    ecg=(FIFF.FIFFV_ECG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    seeg=(FIFF.FIFFV_SEEG_CH, FIFF.FIFFV_COIL_EEG, FIFF.FIFF_UNIT_V),
    bio=(FIFF.FIFFV_BIO_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    ecog=(FIFF.FIFFV_ECOG_CH, FIFF.FIFFV_COIL_EEG, FIFF.FIFF_UNIT_V),
    hbo=(FIFF.FIFFV_FNIRS_CH, FIFF.FIFFV_COIL_FNIRS_HBO, FIFF.FIFF_UNIT_MOL),
    hbr=(FIFF.FIFFV_FNIRS_CH, FIFF.FIFFV_COIL_FNIRS_HBR, FIFF.FIFF_UNIT_MOL)
)


def _summarize_str(st):
    """Make summary string."""
    return st[:56][::-1].split(',', 1)[-1][::-1] + ', ...'


class Info(dict):
    """Information about the recording.

    This data structure behaves like a dictionary. It contains all meta-data
    that is available for a recording.

    The attributes listed below are the possible dictionary entries:

    Attributes
    ----------
    bads : list of str
        List of bad (noisy/broken) channels, by name. These channels will by
        default be ignored by many processing steps.
    ch_names : list-like of str (read-only)
        The names of the channels.
        This object behaves like a read-only Python list. Behind the scenes
        it iterates over the channels dictionaries in `info['chs']`:
        `info['ch_names'][x] == info['chs'][x]['ch_name']`
    chs : list of dict
        A list of channel information structures.
        See: :ref:`faq` for details.
    comps : list of dict
        CTF software gradient compensation data.
        See: :ref:`faq` for details.
    custom_ref_applied : bool
        Whether a custom (=other than average) reference has been applied to
        the EEG data. This flag is checked by some algorithms that require an
        average reference to be set.
    events : list of dict
        Event list, usually extracted from the stim channels.
        See: :ref:`faq` for details.
    hpi_results : list of dict
        Head position indicator (HPI) digitization points and fit information
        (e.g., the resulting transform). See: :ref:`faq` for details.
    meas_date : list of int
        The first element of this list is a POSIX timestamp (milliseconds since
        1970-01-01 00:00:00) denoting the date and time at which the
        measurement was taken. The second element is the number of
        microseconds.
    nchan : int
        Number of channels.
    projs : list of dict
        List of SSP operators that operate on the data.
        See: :ref:`faq` for details.
    sfreq : float
        Sampling frequency in Hertz.
        See: :ref:`faq` for details.
    acq_pars : str | None
        MEG system acquition parameters.
    acq_stim : str | None
        MEG system stimulus parameters.
    buffer_size_sec : float | None
        Buffer size (in seconds) when reading the raw data in chunks.
    ctf_head_t : dict | None
        The transformation from 4D/CTF head coordinates to Neuromag head
        coordinates. This is only present in 4D/CTF data.
        See: :ref:`faq` for details.
    description : str | None
        String description of the recording.
    dev_ctf_t : dict | None
        The transformation from device coordinates to 4D/CTF head coordinates.
        This is only present in 4D/CTF data.
        See: :ref:`faq` for details.
    dev_head_t : dict | None
        The device to head transformation.
        See: :ref:`faq` for details.
    dig : list of dict | None
        The Polhemus digitization data in head coordinates.
        See: :ref:`faq` for details.
    experimentor : str | None
        Name of the person that ran the experiment.
    file_id : dict | None
        The fif ID datastructure of the measurement file.
        See: :ref:`faq` for details.
    highpass : float | None
        Highpass corner frequency in Hertz. Zero indicates a DC recording.
    hpi_meas : list of dict | None
        HPI measurements that were taken at the start of the recording
        (e.g. coil frequencies).
    hpi_subsystem : dict | None
        Information about the HPI subsystem that was used (e.g., event
        channel used for cHPI measurements).
    line_freq : float | None
        Frequency of the power line in Hertz.
    lowpass : float | None
        Lowpass corner frequency in Hertz.
    meas_id : dict | None
        The ID assigned to this measurement by the acquisition system or during
        file conversion.
        See: :ref:`faq` for details.
    proj_id : int | None
        ID number of the project the experiment belongs to.
    proj_name : str | None
        Name of the project the experiment belongs to.
    subject_info : dict | None
        Information about the subject.
    proc_history : list of dict | None | not present in dict
        The SSS info, the CTC correction and the calibaraions from the SSS
        processing logs inside of a raw file.
        See: :ref:`faq` for details.
    """

    def copy(self):
        """Copy the instance.

        Returns
        -------
        info : instance of Info
            The copied info.
        """
        return Info(deepcopy(self))

    def normalize_proj(self):
        """(Re-)Normalize projection vectors after subselection.

        Applying projection after sub-selecting a set of channels that
        were originally used to compute the original projection vectors
        can be dangerous (e.g., if few channels remain, most power was
        in channels that are no longer picked, etc.). By default, mne
        will emit a warning when this is done.

        This function will re-normalize projectors to use only the
        remaining channels, thus avoiding that warning. Only use this
        function if you're confident that the projection vectors still
        adequately capture the original signal of interest.
        """
        _normalize_proj(self)

    def __repr__(self):
        """Summarize info instead of printing all."""
        strs = ['<Info | %s non-empty fields']
        non_empty = 0
        for k, v in self.items():
            if k in ['bads', 'ch_names']:
                entr = (', '.join(b for ii, b in enumerate(v) if ii < 10)
                        if v else '0 items')
                if len(v) > 10:
                    # get rid of of half printed ch names
                    entr = _summarize_str(entr)
            elif k == 'projs' and v:
                entr = ', '.join(p['desc'] + ': o%s' %
                                 {0: 'ff', 1: 'n'}[p['active']] for p in v)
                if len(entr) >= 56:
                    entr = _summarize_str(entr)
            elif k == 'meas_date' and np.iterable(v):
                # first entry in meas_date is meaningful
                entr = dt.fromtimestamp(v[0]).strftime('%Y-%m-%d %H:%M:%S')
            elif k == 'kit_system_id' and v is not None:
                from .kit.constants import SYSNAMES as KIT_SYSNAMES
                entr = '%i (%s)' % (v, KIT_SYSNAMES.get(v, 'unknown'))
            else:
                this_len = (len(v) if hasattr(v, '__len__') else
                            ('%s' % v if v is not None else None))
                entr = (('%d items' % this_len) if isinstance(this_len, int)
                        else ('%s' % this_len if this_len else ''))
            if entr:
                non_empty += 1
                entr = ' | ' + entr
            if k == 'chs':
                ch_types = [channel_type(self, idx) for idx in range(len(v))]
                ch_counts = Counter(ch_types)
                entr += " (%s)" % ', '.join("%s: %d" % (ch_type.upper(), count)
                                            for ch_type, count
                                            in ch_counts.items())
            strs.append('%s : %s%s' % (k, str(type(v))[7:-2], entr))
            if k in ['sfreq', 'lowpass', 'highpass']:
                strs[-1] += ' Hz'
        strs_non_empty = sorted(s for s in strs if '|' in s)
        strs_empty = sorted(s for s in strs if '|' not in s)
        st = '\n    '.join(strs_non_empty + strs_empty)
        st += '\n>'
        st %= non_empty
        return st

    def _check_consistency(self):
        """Do some self-consistency checks and datatype tweaks."""
        missing = [bad for bad in self['bads'] if bad not in self['ch_names']]
        if len(missing) > 0:
            raise RuntimeError('bad channel(s) %s marked do not exist in info'
                               % (missing,))

        chs = [ch['ch_name'] for ch in self['chs']]
        if len(self['ch_names']) != len(chs) or any(
                ch_1 != ch_2 for ch_1, ch_2 in zip(self['ch_names'], chs)) or \
                self['nchan'] != len(chs):
            raise RuntimeError('info channel name inconsistency detected, '
                               'please notify mne-python developers')

        # make sure we have the proper datatypes
        for key in ('sfreq', 'highpass', 'lowpass'):
            if self.get(key) is not None:
                self[key] = float(self[key])

        # make sure channel names are unique
        unique_ids = np.unique(self['ch_names'], return_index=True)[1]
        if len(unique_ids) != self['nchan']:
            dups = set(self['ch_names'][x]
                       for x in np.setdiff1d(range(self['nchan']), unique_ids))
            raise RuntimeError('Channel names are not unique, found '
                               'duplicates for: %s' % dups)
        if 'filename' in self:
            warn('the "filename" key is misleading\
                 and info should not have it')

    def _update_redundant(self):
        """Update the redundant entries."""
        self['ch_names'] = [ch['ch_name'] for ch in self['chs']]
        self['nchan'] = len(self['chs'])


def read_fiducials(fname):
    """Read fiducials from a fiff file.

    Parameters
    ----------
    fname : str
        The filename to read.

    Returns
    -------
    pts : list of dicts
        List of digitizer points (each point in a dict).
    coord_frame : int
        The coordinate frame of the points (one of
        mne.io.constants.FIFF.FIFFV_COORD_...)
    """
    fid, tree, _ = fiff_open(fname)
    with fid:
        isotrak = dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)
        isotrak = isotrak[0]
        pts = []
        coord_frame = FIFF.FIFFV_COORD_UNKNOWN
        for k in range(isotrak['nent']):
            kind = isotrak['directory'][k].kind
            pos = isotrak['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                pts.append(tag.data)
            elif kind == FIFF.FIFF_MNE_COORD_FRAME:
                tag = read_tag(fid, pos)
                coord_frame = tag.data[0]

    if coord_frame == FIFF.FIFFV_COORD_UNKNOWN:
        err = ("No coordinate frame was found in the file %r, it is probably "
               "not a valid fiducials file." % fname)
        raise ValueError(err)

    # coord_frame is not stored in the tag
    for pt in pts:
        pt['coord_frame'] = coord_frame

    return pts, coord_frame


def write_fiducials(fname, pts, coord_frame=FIFF.FIFFV_COORD_UNKNOWN):
    """Write fiducials to a fiff file.

    Parameters
    ----------
    fname : str
        Destination file name.
    pts : iterator of dict
        Iterator through digitizer points. Each point is a dictionary with
        the keys 'kind', 'ident' and 'r'.
    coord_frame : int
        The coordinate frame of the points (one of
        mne.io.constants.FIFF.FIFFV_COORD_...).
    """
    write_dig(fname, pts, coord_frame)


def write_dig(fname, pts, coord_frame=None):
    """Write digitization data to a FIF file.

    Parameters
    ----------
    fname : str
        Destination file name.
    pts : iterator of dict
        Iterator through digitizer points. Each point is a dictionary with
        the keys 'kind', 'ident' and 'r'.
    coord_frame : int | str | None
        If all the points have the same coordinate frame, specify the type
        here. Can be None (default) if the points could have varying
        coordinate frames.
    """
    if coord_frame is not None:
        coord_frame = _to_const(coord_frame)
        pts_frames = set((pt.get('coord_frame', coord_frame) for pt in pts))
        bad_frames = pts_frames - set((coord_frame,))
        if len(bad_frames) > 0:
            raise ValueError(
                'Points have coord_frame entries that are incompatible with '
                'coord_frame=%i: %s.' % (coord_frame, str(tuple(bad_frames))))

    with start_file(fname) as fid:
        write_dig_points(fid, pts, block=True, coord_frame=coord_frame)
        end_file(fid)


def _read_dig_fif(fid, meas_info):
    """Helper to read digitizer data from a FIFF file."""
    isotrak = dir_tree_find(meas_info, FIFF.FIFFB_ISOTRAK)
    dig = None
    if len(isotrak) == 0:
        logger.info('Isotrak not found')
    elif len(isotrak) > 1:
        warn('Multiple Isotrak found')
    else:
        isotrak = isotrak[0]
        dig = []
        for k in range(isotrak['nent']):
            kind = isotrak['directory'][k].kind
            pos = isotrak['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                dig.append(tag.data)
                dig[-1]['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    return dig


def _read_dig_points(fname, comments='%', unit='auto'):
    """Read digitizer data from a text file.

    If fname ends in .hsp or .esp, the function assumes digitizer files in [m],
    otherwise it assumes space-delimited text files in [mm].

    Parameters
    ----------
    fname : str
        The filepath of space delimited file with points, or a .mat file
        (Polhemus FastTrak format).
    comments : str
        The character used to indicate the start of a comment;
        Default: '%'.
    unit : 'auto' | 'm' | 'cm' | 'mm'
        Unit of the digitizer files (hsp and elp). If not 'm', coordinates will
        be rescaled to 'm'. Default is 'auto', which assumes 'm' for *.hsp and
        *.elp files and 'mm' for *.txt files, corresponding to the known
        Polhemus export formats.

    Returns
    -------
    dig_points : np.ndarray, shape (n_points, 3)
        Array of dig points in [m].
    """
    if unit not in ('auto', 'm', 'mm', 'cm'):
        raise ValueError('unit must be one of "auto", "m", "mm", or "cm"')

    _, ext = op.splitext(fname)
    if ext == '.elp' or ext == '.hsp':
        with open(fname) as fid:
            file_str = fid.read()
        value_pattern = "\-?\d+\.?\d*e?\-?\d*"
        coord_pattern = "({0})\s+({0})\s+({0})\s*$".format(value_pattern)
        if ext == '.hsp':
            coord_pattern = '^' + coord_pattern
        points_str = [m.groups() for m in re.finditer(coord_pattern, file_str,
                                                      re.MULTILINE)]
        dig_points = np.array(points_str, dtype=float)
    elif ext == '.mat':  # like FastScan II
        from scipy.io import loadmat
        dig_points = loadmat(fname)['Points'].T
    else:
        dig_points = np.loadtxt(fname, comments=comments, ndmin=2)
        if unit == 'auto':
            unit = 'mm'
        if dig_points.shape[1] > 3:
            warn('Found %d columns instead of 3, using first 3 for XYZ '
                 'coordinates' % (dig_points.shape[1],))
            dig_points = dig_points[:, :3]

    if dig_points.shape[-1] != 3:
        err = 'Data must be (n, 3) instead of %s' % (dig_points.shape,)
        raise ValueError(err)

    if unit == 'mm':
        dig_points /= 1000.
    elif unit == 'cm':
        dig_points /= 100.

    return dig_points


def _write_dig_points(fname, dig_points):
    """Write points to text file.

    Parameters
    ----------
    fname : str
        Path to the file to write. The kind of file to write is determined
        based on the extension: '.txt' for tab separated text file.
    dig_points : numpy.ndarray, shape (n_points, 3)
        Points.
    """
    _, ext = op.splitext(fname)
    dig_points = np.asarray(dig_points)
    if (dig_points.ndim != 2) or (dig_points.shape[1] != 3):
        err = ("Points must be of shape (n_points, 3), "
               "not %s" % (dig_points.shape,))
        raise ValueError(err)

    if ext == '.txt':
        with open(fname, 'wb') as fid:
            version = __version__
            now = dt.now().strftime("%I:%M%p on %B %d, %Y")
            fid.write(b("% Ascii 3D points file created by mne-python version "
                        "{version} at {now}\n".format(version=version,
                                                      now=now)))
            fid.write(b("% {N} 3D points, "
                        "x y z per line\n".format(N=len(dig_points))))
            np.savetxt(fid, dig_points, delimiter='\t', newline='\n')
    else:
        msg = "Unrecognized extension: %r. Need '.txt'." % ext
        raise ValueError(msg)


def _make_dig_points(nasion=None, lpa=None, rpa=None, hpi=None,
                     extra_points=None, dig_ch_pos=None):
    """Construct digitizer info for the info.

    Parameters
    ----------
    nasion : array-like | numpy.ndarray, shape (3,) | None
        Point designated as the nasion point.
    lpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the left auricular point.
    rpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the right auricular point.
    hpi : array-like | numpy.ndarray, shape (n_points, 3) | None
        Points designated as head position indicator points.
    extra_points : array-like | numpy.ndarray, shape (n_points, 3)
        Points designed as the headshape points.
    dig_ch_pos : dict
        Dict of EEG channel positions.

    Returns
    -------
    dig : list
        List of digitizer points to be added to the info['dig'].
    """
    dig = []
    if lpa is not None:
        lpa = np.asarray(lpa)
        if lpa.shape != (3,):
            raise ValueError('LPA should have the shape (3,) instead of %s'
                             % (lpa.shape,))
        dig.append({'r': lpa, 'ident': FIFF.FIFFV_POINT_LPA,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if nasion is not None:
        nasion = np.asarray(nasion)
        if nasion.shape != (3,):
            raise ValueError('Nasion should have the shape (3,) instead of %s'
                             % (nasion.shape,))
        dig.append({'r': nasion, 'ident': FIFF.FIFFV_POINT_NASION,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if rpa is not None:
        rpa = np.asarray(rpa)
        if rpa.shape != (3,):
            raise ValueError('RPA should have the shape (3,) instead of %s'
                             % (rpa.shape,))
        dig.append({'r': rpa, 'ident': FIFF.FIFFV_POINT_RPA,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if hpi is not None:
        hpi = np.asarray(hpi)
        if hpi.ndim != 2 or hpi.shape[1] != 3:
            raise ValueError('HPI should have the shape (n_points, 3) instead '
                             'of %s' % (hpi.shape,))
        for idx, point in enumerate(hpi):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_HPI,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if extra_points is not None:
        extra_points = np.asarray(extra_points)
        if extra_points.shape[1] != 3:
            raise ValueError('Points should have the shape (n_points, 3) '
                             'instead of %s' % (extra_points.shape,))
        for idx, point in enumerate(extra_points):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_EXTRA,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if dig_ch_pos is not None:
        keys = sorted(dig_ch_pos.keys())
        try:  # use the last 3 as int if possible (e.g., EEG001->1)
            idents = [int(key[-3:]) for key in keys]
        except ValueError:  # and if any conversion fails, simply use arange
            idents = np.arange(1, len(keys) + 1)
        for key, ident in zip(keys, idents):
            dig.append({'r': dig_ch_pos[key], 'ident': ident,
                        'kind': FIFF.FIFFV_POINT_EEG,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})
    return dig


@verbose
def read_info(fname, verbose=None):
    """Read measurement info from a file.

    Parameters
    ----------
    fname : str
        File name.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    info : instance of Info
       Measurement information for the dataset.
    """
    f, tree, _ = fiff_open(fname)
    with f as fid:
        info = read_meas_info(fid, tree)[0]
    return info


def read_bad_channels(fid, node):
    """Read bad channels.

    Parameters
    ----------
    fid : file
        The file descriptor.

    node : dict
        The node of the FIF tree that contains info on the bad channels.

    Returns
    -------
    bads : list
        A list of bad channel's names.
    """
    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = []
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None and tag.data is not None:
                bads = tag.data.split(':')
    return bads


@verbose
def read_meas_info(fid, tree, clean_bads=False, verbose=None):
    """Read the measurement info.

    Parameters
    ----------
    fid : file
        Open file descriptor.
    tree : tree
        FIF tree structure.
    clean_bads : bool
        If True, clean info['bads'] before running consistency check.
        Should only be needed for old files where we did not check bads
        before saving.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    info : instance of Info
       Info on dataset.
    meas : dict
        Node in tree that contains the info.
    """
    #   Find the desired blocks
    meas = dir_tree_find(tree, FIFF.FIFFB_MEAS)
    if len(meas) == 0:
        raise ValueError('Could not find measurement data')
    if len(meas) > 1:
        raise ValueError('Cannot read more that 1 measurement data')
    meas = meas[0]

    meas_info = dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    if len(meas_info) == 0:
        raise ValueError('Could not find measurement info')
    if len(meas_info) > 1:
        raise ValueError('Cannot read more that 1 measurement info')
    meas_info = meas_info[0]

    #   Read measurement info
    dev_head_t = None
    ctf_head_t = None
    dev_ctf_t = None
    meas_date = None
    highpass = None
    lowpass = None
    nchan = None
    sfreq = None
    chs = []
    experimenter = None
    description = None
    proj_id = None
    proj_name = None
    line_freq = None
    custom_ref_applied = False
    xplotter_layout = None
    kit_system_id = None
    for k in range(meas_info['nent']):
        kind = meas_info['directory'][k].kind
        pos = meas_info['directory'][k].pos
        if kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = int(tag.data)
        elif kind == FIFF.FIFF_SFREQ:
            tag = read_tag(fid, pos)
            sfreq = float(tag.data)
        elif kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
        elif kind == FIFF.FIFF_LOWPASS:
            tag = read_tag(fid, pos)
            lowpass = float(tag.data)
        elif kind == FIFF.FIFF_HIGHPASS:
            tag = read_tag(fid, pos)
            highpass = float(tag.data)
        elif kind == FIFF.FIFF_MEAS_DATE:
            tag = read_tag(fid, pos)
            meas_date = tag.data
        elif kind == FIFF.FIFF_COORD_TRANS:
            tag = read_tag(fid, pos)
            cand = tag.data

            if cand['from'] == FIFF.FIFFV_COORD_DEVICE and \
                    cand['to'] == FIFF.FIFFV_COORD_HEAD:
                dev_head_t = cand
            elif cand['from'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD and \
                    cand['to'] == FIFF.FIFFV_COORD_HEAD:
                ctf_head_t = cand
            elif cand['from'] == FIFF.FIFFV_MNE_COORD_CTF_DEVICE and \
                    cand['to'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
                dev_ctf_t = cand
        elif kind == FIFF.FIFF_EXPERIMENTER:
            tag = read_tag(fid, pos)
            experimenter = tag.data
        elif kind == FIFF.FIFF_DESCRIPTION:
            tag = read_tag(fid, pos)
            description = tag.data
        elif kind == FIFF.FIFF_PROJ_ID:
            tag = read_tag(fid, pos)
            proj_id = tag.data
        elif kind == FIFF.FIFF_PROJ_NAME:
            tag = read_tag(fid, pos)
            proj_name = tag.data
        elif kind == FIFF.FIFF_LINE_FREQ:
            tag = read_tag(fid, pos)
            line_freq = float(tag.data)
        elif kind in [FIFF.FIFF_MNE_CUSTOM_REF, 236]:  # 236 used before v0.11
            tag = read_tag(fid, pos)
            custom_ref_applied = bool(tag.data)
        elif kind == FIFF.FIFF_XPLOTTER_LAYOUT:
            tag = read_tag(fid, pos)
            xplotter_layout = str(tag.data)
        elif kind == FIFF.FIFF_MNE_KIT_SYSTEM_ID:
            tag = read_tag(fid, pos)
            kit_system_id = int(tag.data)

    # Check that we have everything we need
    if nchan is None:
        raise ValueError('Number of channels is not defined')

    if sfreq is None:
        raise ValueError('Sampling frequency is not defined')

    if len(chs) == 0:
        raise ValueError('Channel information not defined')

    if len(chs) != nchan:
        raise ValueError('Incorrect number of channel definitions found')

    if dev_head_t is None or ctf_head_t is None:
        hpi_result = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
        if len(hpi_result) == 1:
            hpi_result = hpi_result[0]
            for k in range(hpi_result['nent']):
                kind = hpi_result['directory'][k].kind
                pos = hpi_result['directory'][k].pos
                if kind == FIFF.FIFF_COORD_TRANS:
                    tag = read_tag(fid, pos)
                    cand = tag.data
                    if (cand['from'] == FIFF.FIFFV_COORD_DEVICE and
                            cand['to'] == FIFF.FIFFV_COORD_HEAD and
                            dev_head_t is None):
                        dev_head_t = cand
                    elif (cand['from'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD and
                          cand['to'] == FIFF.FIFFV_COORD_HEAD and
                          ctf_head_t is None):
                        ctf_head_t = cand

    #   Locate the Polhemus data
    dig = _read_dig_fif(fid, meas_info)

    #   Locate the acquisition information
    acqpars = dir_tree_find(meas_info, FIFF.FIFFB_DACQ_PARS)
    acq_pars = None
    acq_stim = None
    if len(acqpars) == 1:
        acqpars = acqpars[0]
        for k in range(acqpars['nent']):
            kind = acqpars['directory'][k].kind
            pos = acqpars['directory'][k].pos
            if kind == FIFF.FIFF_DACQ_PARS:
                tag = read_tag(fid, pos)
                acq_pars = tag.data
            elif kind == FIFF.FIFF_DACQ_STIM:
                tag = read_tag(fid, pos)
                acq_stim = tag.data

    #   Load the SSP data
    projs = _read_proj(fid, meas_info)

    #   Load the CTF compensation data
    comps = read_ctf_comp(fid, meas_info, chs)

    #   Load the bad channel list
    bads = read_bad_channels(fid, meas_info)

    #
    #   Put the data together
    #
    if tree['id'] is not None:
        info = Info(file_id=tree['id'])
    else:
        info = Info(file_id=None)

    #   Locate events list
    events = dir_tree_find(meas_info, FIFF.FIFFB_EVENTS)
    evs = list()
    for event in events:
        ev = dict()
        for k in range(event['nent']):
            kind = event['directory'][k].kind
            pos = event['directory'][k].pos
            if kind == FIFF.FIFF_EVENT_CHANNELS:
                ev['channels'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_EVENT_LIST:
                ev['list'] = read_tag(fid, pos).data
        evs.append(ev)
    info['events'] = evs

    #   Locate HPI result
    hpi_results = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
    hrs = list()
    for hpi_result in hpi_results:
        hr = dict()
        hr['dig_points'] = []
        for k in range(hpi_result['nent']):
            kind = hpi_result['directory'][k].kind
            pos = hpi_result['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                hr['dig_points'].append(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_DIGITIZATION_ORDER:
                hr['order'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_COILS_USED:
                hr['used'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_COIL_MOMENTS:
                hr['moments'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_FIT_GOODNESS:
                hr['goodness'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_FIT_GOOD_LIMIT:
                hr['good_limit'] = float(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_FIT_DIST_LIMIT:
                hr['dist_limit'] = float(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_FIT_ACCEPT:
                hr['accept'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_COORD_TRANS:
                hr['coord_trans'] = read_tag(fid, pos).data
        hrs.append(hr)
    info['hpi_results'] = hrs

    #   Locate HPI Measurement
    hpi_meass = dir_tree_find(meas_info, FIFF.FIFFB_HPI_MEAS)
    hms = list()
    for hpi_meas in hpi_meass:
        hm = dict()
        for k in range(hpi_meas['nent']):
            kind = hpi_meas['directory'][k].kind
            pos = hpi_meas['directory'][k].pos
            if kind == FIFF.FIFF_CREATOR:
                hm['creator'] = text_type(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_SFREQ:
                hm['sfreq'] = float(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_NCHAN:
                hm['nchan'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_NAVE:
                hm['nave'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_NCOIL:
                hm['ncoil'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_FIRST_SAMPLE:
                hm['first_samp'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                hm['last_samp'] = int(read_tag(fid, pos).data)
        hpi_coils = dir_tree_find(hpi_meas, FIFF.FIFFB_HPI_COIL)
        hcs = []
        for hpi_coil in hpi_coils:
            hc = dict()
            for k in range(hpi_coil['nent']):
                kind = hpi_coil['directory'][k].kind
                pos = hpi_coil['directory'][k].pos
                if kind == FIFF.FIFF_HPI_COIL_NO:
                    hc['number'] = int(read_tag(fid, pos).data)
                elif kind == FIFF.FIFF_EPOCH:
                    hc['epoch'] = read_tag(fid, pos).data
                elif kind == FIFF.FIFF_HPI_SLOPES:
                    hc['slopes'] = read_tag(fid, pos).data
                elif kind == FIFF.FIFF_HPI_CORR_COEFF:
                    hc['corr_coeff'] = read_tag(fid, pos).data
                elif kind == FIFF.FIFF_HPI_COIL_FREQ:
                    hc['coil_freq'] = read_tag(fid, pos).data
            hcs.append(hc)
        hm['hpi_coils'] = hcs
        hms.append(hm)
    info['hpi_meas'] = hms

    subject_info = dir_tree_find(meas_info, FIFF.FIFFB_SUBJECT)
    si = None
    if len(subject_info) == 1:
        subject_info = subject_info[0]
        si = dict()
        for k in range(subject_info['nent']):
            kind = subject_info['directory'][k].kind
            pos = subject_info['directory'][k].pos
            if kind == FIFF.FIFF_SUBJ_ID:
                tag = read_tag(fid, pos)
                si['id'] = int(tag.data)
            elif kind == FIFF.FIFF_SUBJ_HIS_ID:
                tag = read_tag(fid, pos)
                si['his_id'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_LAST_NAME:
                tag = read_tag(fid, pos)
                si['last_name'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_FIRST_NAME:
                tag = read_tag(fid, pos)
                si['first_name'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_MIDDLE_NAME:
                tag = read_tag(fid, pos)
                si['middle_name'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_BIRTH_DAY:
                tag = read_tag(fid, pos)
                si['birthday'] = tag.data
            elif kind == FIFF.FIFF_SUBJ_SEX:
                tag = read_tag(fid, pos)
                si['sex'] = int(tag.data)
            elif kind == FIFF.FIFF_SUBJ_HAND:
                tag = read_tag(fid, pos)
                si['hand'] = int(tag.data)
    info['subject_info'] = si

    hpi_subsystem = dir_tree_find(meas_info, FIFF.FIFFB_HPI_SUBSYSTEM)
    hs = None
    if len(hpi_subsystem) == 1:
        hpi_subsystem = hpi_subsystem[0]
        hs = dict()
        for k in range(hpi_subsystem['nent']):
            kind = hpi_subsystem['directory'][k].kind
            pos = hpi_subsystem['directory'][k].pos
            if kind == FIFF.FIFF_HPI_NCOIL:
                tag = read_tag(fid, pos)
                hs['ncoil'] = int(tag.data)
            elif kind == FIFF.FIFF_EVENT_CHANNEL:
                tag = read_tag(fid, pos)
                hs['event_channel'] = text_type(tag.data)
            hpi_coils = dir_tree_find(hpi_subsystem, FIFF.FIFFB_HPI_COIL)
            hc = []
            for coil in hpi_coils:
                this_coil = dict()
                for j in range(coil['nent']):
                    kind = coil['directory'][j].kind
                    pos = coil['directory'][j].pos
                    if kind == FIFF.FIFF_EVENT_BITS:
                        tag = read_tag(fid, pos)
                        this_coil['event_bits'] = np.array(tag.data)
                hc.append(this_coil)
            hs['hpi_coils'] = hc
    info['hpi_subsystem'] = hs

    #   Read processing history
    _read_proc_history(fid, tree, info)

    #  Make the most appropriate selection for the measurement id
    if meas_info['parent_id'] is None:
        if meas_info['id'] is None:
            if meas['id'] is None:
                if meas['parent_id'] is None:
                    info['meas_id'] = info['file_id']
                else:
                    info['meas_id'] = meas['parent_id']
            else:
                info['meas_id'] = meas['id']
        else:
            info['meas_id'] = meas_info['id']
    else:
        info['meas_id'] = meas_info['parent_id']

    info['experimenter'] = experimenter
    info['description'] = description
    info['proj_id'] = proj_id
    info['proj_name'] = proj_name

    if meas_date is None:
        meas_date = [info['meas_id']['secs'], info['meas_id']['usecs']]
    info['meas_date'] = meas_date

    info['sfreq'] = sfreq
    info['highpass'] = highpass if highpass is not None else 0.
    info['lowpass'] = lowpass if lowpass is not None else info['sfreq'] / 2.0
    info['line_freq'] = line_freq

    #   Add the channel information and make a list of channel names
    #   for convenience
    info['chs'] = chs

    #
    #  Add the coordinate transformations
    #
    info['dev_head_t'] = dev_head_t
    info['ctf_head_t'] = ctf_head_t
    info['dev_ctf_t'] = dev_ctf_t
    if dev_head_t is not None and ctf_head_t is not None and dev_ctf_t is None:
        from ..transforms import Transform
        head_ctf_trans = linalg.inv(ctf_head_t['trans'])
        dev_ctf_trans = np.dot(head_ctf_trans, info['dev_head_t']['trans'])
        info['dev_ctf_t'] = Transform('meg', 'ctf_head', dev_ctf_trans)

    #   All kinds of auxliary stuff
    info['dig'] = dig
    info['bads'] = bads
    info._update_redundant()
    if clean_bads:
        info['bads'] = [b for b in bads if b in info['ch_names']]
    info['projs'] = projs
    info['comps'] = comps
    info['acq_pars'] = acq_pars
    info['acq_stim'] = acq_stim
    info['custom_ref_applied'] = custom_ref_applied
    info['xplotter_layout'] = xplotter_layout
    info['kit_system_id'] = kit_system_id
    info._check_consistency()
    return info, meas


def write_meas_info(fid, info, data_type=None, reset_range=True):
    """Write measurement info into a file id (from a fif file).

    Parameters
    ----------
    fid : file
        Open file descriptor.
    info : instance of Info
        The measurement info structure.
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.

    Notes
    -----
    Tags are written in a particular order for compatibility with maxfilter.
    """
    info._check_consistency()

    # Measurement info
    start_block(fid, FIFF.FIFFB_MEAS_INFO)

    for event in info['events']:
        start_block(fid, FIFF.FIFFB_EVENTS)
        if event.get('channels') is not None:
            write_int(fid, FIFF.FIFF_EVENT_CHANNELS, event['channels'])
        if event.get('list') is not None:
            write_int(fid, FIFF.FIFF_EVENT_LIST, event['list'])
        end_block(fid, FIFF.FIFFB_EVENTS)

    #   HPI Result
    for hpi_result in info['hpi_results']:
        start_block(fid, FIFF.FIFFB_HPI_RESULT)
        write_dig_points(fid, hpi_result['dig_points'])
        if 'order' in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_DIGITIZATION_ORDER,
                      hpi_result['order'])
        if 'used' in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_COILS_USED, hpi_result['used'])
        if 'moments' in hpi_result:
            write_float_matrix(fid, FIFF.FIFF_HPI_COIL_MOMENTS,
                               hpi_result['moments'])
        if 'goodness' in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_GOODNESS,
                        hpi_result['goodness'])
        if 'good_limit' in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_GOOD_LIMIT,
                        hpi_result['good_limit'])
        if 'dist_limit' in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_DIST_LIMIT,
                        hpi_result['dist_limit'])
        if 'accept' in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_FIT_ACCEPT, hpi_result['accept'])
        if 'coord_trans' in hpi_result:
            write_coord_trans(fid, hpi_result['coord_trans'])
        end_block(fid, FIFF.FIFFB_HPI_RESULT)

    #   HPI Measurement
    for hpi_meas in info['hpi_meas']:
        start_block(fid, FIFF.FIFFB_HPI_MEAS)
        if hpi_meas.get('creator') is not None:
            write_string(fid, FIFF.FIFF_CREATOR, hpi_meas['creator'])
        if hpi_meas.get('sfreq') is not None:
            write_float(fid, FIFF.FIFF_SFREQ, hpi_meas['sfreq'])
        if hpi_meas.get('nchan') is not None:
            write_int(fid, FIFF.FIFF_NCHAN, hpi_meas['nchan'])
        if hpi_meas.get('nave') is not None:
            write_int(fid, FIFF.FIFF_NAVE, hpi_meas['nave'])
        if hpi_meas.get('ncoil') is not None:
            write_int(fid, FIFF.FIFF_HPI_NCOIL, hpi_meas['ncoil'])
        if hpi_meas.get('first_samp') is not None:
            write_int(fid, FIFF.FIFF_FIRST_SAMPLE, hpi_meas['first_samp'])
        if hpi_meas.get('last_samp') is not None:
            write_int(fid, FIFF.FIFF_LAST_SAMPLE, hpi_meas['last_samp'])
        for hpi_coil in hpi_meas['hpi_coils']:
            start_block(fid, FIFF.FIFFB_HPI_COIL)
            if hpi_coil.get('number') is not None:
                write_int(fid, FIFF.FIFF_HPI_COIL_NO, hpi_coil['number'])
            if hpi_coil.get('epoch') is not None:
                write_float_matrix(fid, FIFF.FIFF_EPOCH, hpi_coil['epoch'])
            if hpi_coil.get('slopes') is not None:
                write_float(fid, FIFF.FIFF_HPI_SLOPES, hpi_coil['slopes'])
            if hpi_coil.get('corr_coeff') is not None:
                write_float(fid, FIFF.FIFF_HPI_CORR_COEFF,
                            hpi_coil['corr_coeff'])
            if hpi_coil.get('coil_freq') is not None:
                write_float(fid, FIFF.FIFF_HPI_COIL_FREQ,
                            hpi_coil['coil_freq'])
            end_block(fid, FIFF.FIFFB_HPI_COIL)
        end_block(fid, FIFF.FIFFB_HPI_MEAS)

    #   Polhemus data
    write_dig_points(fid, info['dig'], block=True)

    #   megacq parameters
    if info['acq_pars'] is not None or info['acq_stim'] is not None:
        start_block(fid, FIFF.FIFFB_DACQ_PARS)
        if info['acq_pars'] is not None:
            write_string(fid, FIFF.FIFF_DACQ_PARS, info['acq_pars'])

        if info['acq_stim'] is not None:
            write_string(fid, FIFF.FIFF_DACQ_STIM, info['acq_stim'])

        end_block(fid, FIFF.FIFFB_DACQ_PARS)

    #   Coordinate transformations if the HPI result block was not there
    if info['dev_head_t'] is not None:
        write_coord_trans(fid, info['dev_head_t'])

    if info['ctf_head_t'] is not None:
        write_coord_trans(fid, info['ctf_head_t'])

    if info['dev_ctf_t'] is not None:
        write_coord_trans(fid, info['dev_ctf_t'])

    #   Projectors
    _write_proj(fid, info['projs'])

    #   Bad channels
    if len(info['bads']) > 0:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, info['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    #   General
    if info.get('experimenter') is not None:
        write_string(fid, FIFF.FIFF_EXPERIMENTER, info['experimenter'])
    if info.get('description') is not None:
        write_string(fid, FIFF.FIFF_DESCRIPTION, info['description'])
    if info.get('proj_id') is not None:
        write_int(fid, FIFF.FIFF_PROJ_ID, info['proj_id'])
    if info.get('proj_name') is not None:
        write_string(fid, FIFF.FIFF_PROJ_NAME, info['proj_name'])
    if info.get('meas_date') is not None:
        write_int(fid, FIFF.FIFF_MEAS_DATE, info['meas_date'])
    write_int(fid, FIFF.FIFF_NCHAN, info['nchan'])
    write_float(fid, FIFF.FIFF_SFREQ, info['sfreq'])
    if info['lowpass'] is not None:
        write_float(fid, FIFF.FIFF_LOWPASS, info['lowpass'])
    if info['highpass'] is not None:
        write_float(fid, FIFF.FIFF_HIGHPASS, info['highpass'])
    if info.get('line_freq') is not None:
        write_float(fid, FIFF.FIFF_LINE_FREQ, info['line_freq'])
    if data_type is not None:
        write_int(fid, FIFF.FIFF_DATA_PACK, data_type)
    if info.get('custom_ref_applied'):
        write_int(fid, FIFF.FIFF_MNE_CUSTOM_REF, info['custom_ref_applied'])
    if info.get('xplotter_layout'):
        write_string(fid, FIFF.FIFF_XPLOTTER_LAYOUT, info['xplotter_layout'])

    #  Channel information
    for k, c in enumerate(info['chs']):
        #   Scan numbers may have been messed up
        c = deepcopy(c)
        c['scanno'] = k + 1
        # for float/double, the "range" param is unnecessary
        if reset_range is True:
            c['range'] = 1.0
        write_ch_info(fid, c)

    # Subject information
    if info.get('subject_info') is not None:
        start_block(fid, FIFF.FIFFB_SUBJECT)
        si = info['subject_info']
        if si.get('id') is not None:
            write_int(fid, FIFF.FIFF_SUBJ_ID, si['id'])
        if si.get('his_id') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, si['his_id'])
        if si.get('last_name') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_LAST_NAME, si['last_name'])
        if si.get('first_name') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_FIRST_NAME, si['first_name'])
        if si.get('middle_name') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_MIDDLE_NAME, si['middle_name'])
        if si.get('birthday') is not None:
            write_julian(fid, FIFF.FIFF_SUBJ_BIRTH_DAY, si['birthday'])
        if si.get('sex') is not None:
            write_int(fid, FIFF.FIFF_SUBJ_SEX, si['sex'])
        if si.get('hand') is not None:
            write_int(fid, FIFF.FIFF_SUBJ_HAND, si['hand'])
        end_block(fid, FIFF.FIFFB_SUBJECT)

    if info.get('hpi_subsystem') is not None:
        hs = info['hpi_subsystem']
        start_block(fid, FIFF.FIFFB_HPI_SUBSYSTEM)
        if hs.get('ncoil') is not None:
            write_int(fid, FIFF.FIFF_HPI_NCOIL, hs['ncoil'])
        if hs.get('event_channel') is not None:
            write_string(fid, FIFF.FIFF_EVENT_CHANNEL, hs['event_channel'])
        if hs.get('hpi_coils') is not None:
            for coil in hs['hpi_coils']:
                start_block(fid, FIFF.FIFFB_HPI_COIL)
                if coil.get('event_bits') is not None:
                    write_int(fid, FIFF.FIFF_EVENT_BITS,
                              coil['event_bits'])
                end_block(fid, FIFF.FIFFB_HPI_COIL)
        end_block(fid, FIFF.FIFFB_HPI_SUBSYSTEM)

    #   CTF compensation info
    write_ctf_comp(fid, info['comps'])

    #   KIT system ID
    if info.get('kit_system_id') is not None:
        write_int(fid, FIFF.FIFF_MNE_KIT_SYSTEM_ID, info['kit_system_id'])

    end_block(fid, FIFF.FIFFB_MEAS_INFO)

    #   Processing history
    _write_proc_history(fid, info)


def write_info(fname, info, data_type=None, reset_range=True):
    """Write measurement info in fif file.

    Parameters
    ----------
    fname : str
        The name of the file. Should end by -info.fif.
    info : instance of Info
        The measurement info structure
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.
    """
    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MEAS)
    write_meas_info(fid, info, data_type, reset_range)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)


@verbose
def _merge_dict_values(dicts, key, verbose=None):
    """Merge things together.

    Fork for {'dict', 'list', 'array', 'other'}
    and consider cases where one or all are of the same type.
    """
    values = [d[key] for d in dicts]
    msg = ("Don't know how to merge '%s'. Make sure values are "
           "compatible." % key)

    def _flatten(lists):
        return [item for sublist in lists for item in sublist]

    def _check_isinstance(values, kind, func):
        return func([isinstance(v, kind) for v in values])

    def _where_isinstance(values, kind):
        """Get indices of instances."""
        return np.where([isinstance(v, type) for v in values])[0]

    # list
    if _check_isinstance(values, list, all):
        lists = (d[key] for d in dicts)
        return (_uniquify_projs(_flatten(lists)) if key == 'projs'
                else _flatten(lists))
    elif _check_isinstance(values, list, any):
        idx = _where_isinstance(values, list)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            lists = (d[key] for d in dicts if isinstance(d[key], list))
            return _flatten(lists)
    # dict
    elif _check_isinstance(values, dict, all):
        is_qual = all(object_diff(values[0], v) == '' for v in values[1:])
        if is_qual:
            return values[0]
        else:
            RuntimeError(msg)
    elif _check_isinstance(values, dict, any):
        idx = _where_isinstance(values, dict)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            raise RuntimeError(msg)
    # ndarray
    elif _check_isinstance(values, np.ndarray, all):
        is_qual = all(np.all(values[0] == x) for x in values[1:])
        if is_qual:
            return values[0]
        elif key == 'meas_date':
            logger.info('Found multiple entries for %s. '
                        'Setting value to `None`' % key)
            return None
        else:
            raise RuntimeError(msg)
    elif _check_isinstance(values, np.ndarray, any):
        idx = _where_isinstance(values, np.ndarray)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            raise RuntimeError(msg)
    # other
    else:
        unique_values = set(values)
        if len(unique_values) == 1:
            return list(values)[0]
        elif isinstance(list(unique_values)[0], BytesIO):
            logger.info('Found multiple StringIO instances. '
                        'Setting value to `None`')
            return None
        elif isinstance(list(unique_values)[0], string_types):
            logger.info('Found multiple filenames. '
                        'Setting value to `None`')
            return None
        else:
            raise RuntimeError(msg)


@verbose
def _merge_info(infos, force_update_to_first=False, verbose=None):
    """Merge multiple measurement info dictionaries.

     - Fields that are present in only one info object will be used in the
       merged info.
     - Fields that are present in multiple info objects and are the same
       will be used in the merged info.
     - Fields that are present in multiple info objects and are different
       will result in a None value in the merged info.
     - Channels will be concatenated. If multiple info objects contain
       channels with the same name, an exception is raised.

    Parameters
    ----------
    infos | list of instance of Info
        Info objects to merge into one info object.
    force_update_to_first : bool
        If True, force the fields for objects in `info` will be updated
        to match those in the first item. Use at your own risk, as this
        may overwrite important metadata.
    verbose : bool, str, int, or NonIe
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    info : instance of Info
        The merged info object.
    """
    for info in infos:
        info._check_consistency()
    if force_update_to_first is True:
        infos = deepcopy(infos)
        _force_update_info(infos[0], infos[1:])
    info = Info()
    info['chs'] = []
    for this_info in infos:
        info['chs'].extend(this_info['chs'])
    info._update_redundant()
    duplicates = set([ch for ch in info['ch_names']
                      if info['ch_names'].count(ch) > 1])
    if len(duplicates) > 0:
        msg = ("The following channels are present in more than one input "
               "measurement info objects: %s" % list(duplicates))
        raise ValueError(msg)

    transforms = ['ctf_head_t', 'dev_head_t', 'dev_ctf_t']
    for trans_name in transforms:
        trans = [i[trans_name] for i in infos if i[trans_name]]
        if len(trans) == 0:
            info[trans_name] = None
        elif len(trans) == 1:
            info[trans_name] = trans[0]
        elif all(np.all(trans[0]['trans'] == x['trans']) and
                 trans[0]['from'] == x['from'] and
                 trans[0]['to'] == x['to']
                 for x in trans[1:]):
            info[trans_name] = trans[0]
        else:
            msg = ("Measurement infos provide mutually inconsistent %s" %
                   trans_name)
            raise ValueError(msg)

    # KIT system-IDs
    kit_sys_ids = [i['kit_system_id'] for i in infos if i['kit_system_id']]
    if len(kit_sys_ids) == 0:
        info['kit_system_id'] = None
    elif len(set(kit_sys_ids)) == 1:
        info['kit_system_id'] = kit_sys_ids[0]
    else:
        raise ValueError("Trying to merge channels from different KIT systems")

    # other fields
    other_fields = ['acq_pars', 'acq_stim', 'bads', 'buffer_size_sec',
                    'comps', 'custom_ref_applied', 'description', 'dig',
                    'experimenter', 'file_id', 'highpass',
                    'hpi_results', 'hpi_meas', 'hpi_subsystem', 'events',
                    'line_freq', 'lowpass', 'meas_date', 'meas_id',
                    'proj_id', 'proj_name', 'projs', 'sfreq',
                    'subject_info', 'sfreq', 'xplotter_layout']
    for k in other_fields:
        info[k] = _merge_dict_values(infos, k)

    info._check_consistency()
    return info


def create_info(ch_names, sfreq, ch_types=None, montage=None):
    """Create a basic Info instance suitable for use with create_raw.

    Parameters
    ----------
    ch_names : list of str | int
        Channel names. If an int, a list of channel names will be created
        from :func:`range(ch_names) <range>`.
    sfreq : float
        Sample rate of the data.
    ch_types : list of str | str
        Channel types. If None, data are assumed to be misc.
        Currently supported fields are 'ecg', 'bio', 'stim', 'eog', 'misc',
        'seeg', 'ecog', 'mag', 'eeg', 'ref_meg', 'grad', 'hbr' or 'hbo'.
        If str, then all channels are assumed to be of the same type.
    montage : None | str | Montage | DigMontage | list
        A montage containing channel positions. If str or Montage is
        specified, the channel info will be updated with the channel
        positions. Default is None. If DigMontage is specified, the
        digitizer information will be updated. A list of unique montages,
        can be specifed and applied to the info. See also the documentation of
        :func:`mne.channels.read_montage` for more information.

    Returns
    -------
    info : instance of Info
        The measurement info.

    Notes
    -----
    The info dictionary will be sparsely populated to enable functionality
    within the rest of the package. Advanced functionality such as source
    localization can only be obtained through substantial, proper
    modifications of the info structure (not recommended).

    Note that the MEG device-to-head transform ``info['dev_head_t']`` will
    be initialized to the identity transform.

    Proper units of measure:
    * V: eeg, eog, seeg, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc
    """
    if isinstance(ch_names, int):
        ch_names = list(np.arange(ch_names).astype(str))
    if not isinstance(ch_names, (list, tuple)):
        raise TypeError('ch_names must be a list, tuple, or int')
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError('sfreq must be positive')
    nchan = len(ch_names)
    if ch_types is None:
        ch_types = ['misc'] * nchan
    if isinstance(ch_types, string_types):
        ch_types = [ch_types] * nchan
    if len(ch_types) != nchan:
        raise ValueError('ch_types and ch_names must be the same length '
                         '(%s != %s)' % (len(ch_types), nchan))
    info = _empty_info(sfreq)
    info['meas_date'] = np.array([0, 0], np.int32)
    loc = np.concatenate((np.zeros(3), np.eye(3).ravel())).astype(np.float32)
    for ci, (name, kind) in enumerate(zip(ch_names, ch_types)):
        if not isinstance(name, string_types):
            raise TypeError('each entry in ch_names must be a string')
        if not isinstance(kind, string_types):
            raise TypeError('each entry in ch_types must be a string')
        if kind not in _kind_dict:
            raise KeyError('kind must be one of %s, not %s'
                           % (list(_kind_dict.keys()), kind))
        kind = _kind_dict[kind]
        chan_info = dict(loc=loc.copy(), unit_mul=0, range=1., cal=1.,
                         kind=kind[0], coil_type=kind[1],
                         unit=kind[2], coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                         ch_name=name, scanno=ci + 1, logno=ci + 1)
        info['chs'].append(chan_info)
    info._update_redundant()
    if montage is not None:
        from ..channels.montage import (Montage, DigMontage, _set_montage,
                                        read_montage)
        if not isinstance(montage, list):
            montage = [montage]
        for montage_ in montage:
            if isinstance(montage_, (Montage, DigMontage)):
                _set_montage(info, montage_)
            elif isinstance(montage_, string_types):
                montage_ = read_montage(montage_)
                _set_montage(info, montage_)
            else:
                raise TypeError('Montage must be an instance of Montage, '
                                'DigMontage, a list of montages, or filepath, '
                                'not %s.' % type(montage))
    info._check_consistency()
    return info


RAW_INFO_FIELDS = (
    'acq_pars', 'acq_stim', 'bads', 'buffer_size_sec', 'ch_names', 'chs',
    'comps', 'ctf_head_t', 'custom_ref_applied', 'description', 'dev_ctf_t',
    'dev_head_t', 'dig', 'experimenter', 'events',
    'file_id', 'highpass', 'hpi_meas', 'hpi_results',
    'hpi_subsystem', 'kit_system_id', 'line_freq', 'lowpass', 'meas_date',
    'meas_id', 'nchan', 'proj_id', 'proj_name', 'projs', 'sfreq',
    'subject_info', 'xplotter_layout',
)


def _empty_info(sfreq):
    """Create an empty info dictionary."""
    from ..transforms import Transform
    _none_keys = (
        'acq_pars', 'acq_stim', 'buffer_size_sec', 'ctf_head_t', 'description',
        'dev_ctf_t', 'dig', 'experimenter',
        'file_id', 'highpass', 'hpi_subsystem', 'kit_system_id',
        'line_freq', 'lowpass', 'meas_date', 'meas_id', 'proj_id', 'proj_name',
        'subject_info', 'xplotter_layout',
    )
    _list_keys = ('bads', 'chs', 'comps', 'events', 'hpi_meas', 'hpi_results',
                  'projs')
    info = Info()
    for k in _none_keys:
        info[k] = None
    for k in _list_keys:
        info[k] = list()
    info['custom_ref_applied'] = False
    info['dev_head_t'] = Transform('meg', 'head')
    info['highpass'] = 0.
    info['sfreq'] = float(sfreq)
    info['lowpass'] = info['sfreq'] / 2.
    info._update_redundant()
    info._check_consistency()
    return info


def _force_update_info(info_base, info_target):
    """Update target info objects with values from info base.

    Note that values in info_target will be overwritten by those in info_base.
    This will overwrite all fields except for: 'chs', 'ch_names', 'nchan'.

    Parameters
    ----------
    info_base : mne.Info
        The Info object you want to use for overwriting values
        in target Info objects.
    info_target : mne.Info | list of mne.Info
        The Info object(s) you wish to overwrite using info_base. These objects
        will be modified in-place.
    """
    exclude_keys = ['chs', 'ch_names', 'nchan']
    info_target = np.atleast_1d(info_target).ravel()
    all_infos = np.hstack([info_base, info_target])
    for ii in all_infos:
        if not isinstance(ii, Info):
            raise ValueError('Inputs must be of type Info. '
                             'Found type %s' % type(ii))
    for key, val in info_base.items():
        if key in exclude_keys:
            continue
        for i_targ in info_target:
            i_targ[key] = val


def anonymize_info(info):
    """Anonymize measurement information in place.

    Reset 'subject_info', 'meas_date', 'file_id', and 'meas_id' keys if they
    exist in ``info``.

    Parameters
    ----------
    info : dict, instance of Info
        Measurement information for the dataset.

    Returns
    -------
    info : instance of Info
        Measurement information for the dataset.

    Notes
    -----
    Operates in place.
    """
    if not isinstance(info, Info):
        raise ValueError('self must be an Info instance.')
    if info.get('subject_info') is not None:
        del info['subject_info']
    info['meas_date'] = [0, 0]
    for key_1 in ('file_id', 'meas_id'):
        key = info.get(key_1)
        if key is None:
            continue
        for key_2 in ('secs', 'msecs', 'usecs'):
            if key_2 not in key:
                continue
            info[key_1][key_2] = 0
    return info
