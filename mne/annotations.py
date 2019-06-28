# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

from datetime import datetime, timedelta
import time
import os.path as op
import re
from copy import deepcopy
from itertools import takewhile
import collections

import numpy as np

from .utils import (_pl, check_fname, _validate_type, verbose, warn, logger,
                    _check_pandas_installed, _mask_to_onsets_offsets)
from .utils import _DefaultEventParser

from .io.write import (start_block, end_block, write_float, write_name_list,
                       write_double, start_file)
from .io.constants import FIFF
from .io.open import fiff_open
from .io.tree import dir_tree_find
from .io.tag import read_tag


def _check_o_d_s(onset, duration, description):
    onset = np.atleast_1d(np.array(onset, dtype=float))
    if onset.ndim != 1:
        raise ValueError('Onset must be a one dimensional array, got %s '
                         '(shape %s).'
                         % (onset.ndim, onset.shape))
    duration = np.array(duration, dtype=float)
    if duration.ndim == 0 or duration.shape == (1,):
        duration = np.repeat(duration, len(onset))
    if duration.ndim != 1:
        raise ValueError('Duration must be a one dimensional array, '
                         'got %d.' % (duration.ndim,))

    description = np.array(description, dtype=str)
    if description.ndim == 0 or description.shape == (1,):
        description = np.repeat(description, len(onset))
    if description.ndim != 1:
        raise ValueError('Description must be a one dimensional array, '
                         'got %d.' % (description.ndim,))
    if any([';' in desc for desc in description]):
        raise ValueError('Semicolons in descriptions not supported.')

    if not (len(onset) == len(duration) == len(description)):
        raise ValueError('Onset, duration and description must be '
                         'equal in sizes, got %s, %s, and %s.'
                         % (len(onset), len(duration), len(description)))
    return onset, duration, description


class Annotations(object):
    """Annotation object for annotating segments of raw data.

    Parameters
    ----------
    onset : array of float, shape (n_annotations,)
        The starting time of annotations in seconds after ``orig_time``.
    duration : array of float, shape (n_annotations,)
        Durations of the annotations in seconds.
    description : array of str, shape (n_annotations,) | str
        Array of strings containing description for each annotation. If a
        string, all the annotations are given the same description. To reject
        epochs, use description starting with keyword 'bad'. See example above.
    orig_time : float | int | instance of datetime.datetime | array of int | None | str
        A POSIX Timestamp, datetime or an array containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisiton is started at the
        same time. If it is a string, it should conform to the ISO8601 format.
        More precisely to this '%Y-%m-%d %H:%M:%S.%f' particular case of the
        ISO8601 format where the delimiter between date and time is ' '.

    Notes
    -----
    Annotations are added to instance of :class:`mne.io.Raw` as the attribute
    :attr:`raw.annotations <mne.io.Raw.annotations>`.

    To reject bad epochs using annotations, use
    annotation description starting with 'bad' keyword. The epochs with
    overlapping bad segments are then rejected automatically by default.

    To remove epochs with blinks you can do:

    >>> eog_events = mne.preprocessing.find_eog_events(raw)  # doctest: +SKIP
    >>> n_blinks = len(eog_events)  # doctest: +SKIP
    >>> onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  # doctest: +SKIP
    >>> duration = np.repeat(0.5, n_blinks)  # doctest: +SKIP
    >>> description = ['bad blink'] * n_blinks  # doctest: +SKIP
    >>> annotations = mne.Annotations(onset, duration, description)  # doctest: +SKIP
    >>> raw.set_annotations(annotations)  # doctest: +SKIP
    >>> epochs = mne.Epochs(raw, events, event_id, tmin, tmax)  # doctest: +SKIP

    **orig_time**

    If ``orig_time`` is None, the annotations are synced to the start of the
    data (0 seconds). Otherwise the annotations are synced to sample 0 and
    ``raw.first_samp`` is taken into account the same way as with events.

    When setting annotations, the following alignments
    between ``raw.info['meas_date']`` and ``annotation.orig_time`` take place:

    ::

        ----------- meas_date=XX, orig_time=YY -----------------------------

             |              +------------------+
             |______________|     RAW          |
             |              |                  |
             |              +------------------+
         meas_date      first_samp
             .
             .         |         +------+
             .         |_________| ANOT |
             .         |         |      |
             .         |         +------+
             .     orig_time   onset[0]
             .
             |                   +------+
             |___________________|      |
             |                   |      |
             |                   +------+
         orig_time            onset[0]'

        ----------- meas_date=XX, orig_time=None ---------------------------

             |              +------------------+
             |______________|     RAW          |
             |              |                  |
             |              +------------------+
             .              N         +------+
             .              o_________| ANOT |
             .              n         |      |
             .              e         +------+
             .
             |                        +------+
             |________________________|      |
             |                        |      |
             |                        +------+
         orig_time                 onset[0]'

        ----------- meas_date=None, orig_time=YY ---------------------------

             N              +------------------+
             o______________|     RAW          |
             n              |                  |
             e              +------------------+
                       |         +------+
                       |_________| ANOT |
                       |         |      |
                       |         +------+


                    [[[ CRASH ]]]

        ----------- meas_date=None, orig_time=None -------------------------

             N              +------------------+
             o______________|     RAW          |
             n              |                  |
             e              +------------------+
             .              N         +------+
             .              o_________| ANOT |
             .              n         |      |
             .              e         +------+
             .
             N                        +------+
             o________________________|      |
             n                        |      |
             e                        +------+
         orig_time                 onset[0]'

    """  # noqa: E501

    def __init__(self, onset, duration, description,
                 orig_time=None):  # noqa: D102
        if orig_time is not None:
            orig_time = _handle_meas_date(orig_time)
        self.orig_time = orig_time
        self.onset, self.duration, self.description = _check_o_d_s(
            onset, duration, description)
        self._sort()  # ensure we're sorted

    def __repr__(self):
        """Show the representation."""
        counter = collections.Counter(self.description)
        kinds = ['%s (%s)' % k for k in counter.items()]
        kinds = ', '.join(kinds[:3]) + ('' if len(kinds) <= 3 else '...')
        kinds = (': ' if len(kinds) > 0 else '') + kinds
        if self.orig_time is None:
            orig = 'orig_time : None'
        else:
            orig = 'orig_time : %s' % datetime.utcfromtimestamp(self.orig_time)
        return ('<Annotations  |  %s segment%s %s, %s>'
                % (len(self.onset), _pl(len(self.onset)), kinds, orig))

    def __len__(self):
        """Return the number of annotations."""
        return len(self.duration)

    def __add__(self, other):
        """Add (concatencate) two Annotation objects."""
        out = self.copy()
        out += other
        return out

    def __iadd__(self, other):
        """Add (concatencate) two Annotation objects in-place.

        Both annotations must have the same orig_time
        """
        if len(self) == 0:
            self.orig_time = other.orig_time
        if self.orig_time != other.orig_time:
            raise ValueError("orig_time should be the same to "
                             "add/concatenate 2 annotations "
                             "(got %s != %s)" % (self.orig_time,
                                                 other.orig_time))
        return self.append(other.onset, other.duration, other.description)

    def __iter__(self):
        """Iterate over the annotations."""
        for idx in range(len(self.onset)):
            yield self.__getitem__(idx)

    def __getitem__(self, key):
        """Propagate indexing and slicing to the underlying numpy structure."""
        if isinstance(key, int):
            out_keys = ('onset', 'duration', 'description', 'orig_time')
            out_vals = (self.onset[key], self.duration[key],
                        self.description[key], self.orig_time)
            return collections.OrderedDict(zip(out_keys, out_vals))
        else:
            key = list(key) if isinstance(key, tuple) else key
            return Annotations(onset=self.onset[key],
                               duration=self.duration[key],
                               description=self.description[key],
                               orig_time=self.orig_time)

    def append(self, onset, duration, description):
        """Add an annotated segment. Operates inplace.

        Parameters
        ----------
        onset : float | array-like
            Annotation time onset from the beginning of the recording in
            seconds.
        duration : float | array-like
            Duration of the annotation in seconds.
        description : str | array-like
            Description for the annotation. To reject epochs, use description
            starting with keyword 'bad'

        Returns
        -------
        self : mne.Annotations
            The modified Annotations object.

        Notes
        -----
        The array-like support for arguments allows this to be used similarly
        to not only ``list.append``, but also
        `list.extend <https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types>`__.
        """  # noqa: E501
        onset, duration, description = _check_o_d_s(
            onset, duration, description)
        self.onset = np.append(self.onset, onset)
        self.duration = np.append(self.duration, duration)
        self.description = np.append(self.description, description)
        self._sort()
        return self

    def copy(self):
        """Return a deep copy of self."""
        return deepcopy(self)

    def delete(self, idx):
        """Remove an annotation. Operates inplace.

        Parameters
        ----------
        idx : int | array-like of int
            Index of the annotation to remove. Can be array-like to
            remove multiple indices.
        """
        self.onset = np.delete(self.onset, idx)
        self.duration = np.delete(self.duration, idx)
        self.description = np.delete(self.description, idx)

    def save(self, fname):
        """Save annotations to FIF, CSV or TXT.

        Typically annotations get saved in the FIF file for raw data
        (e.g., as ``raw.annotations``), but this offers the possibility
        to also save them to disk separately in different file formats
        which are easier to share between packages.

        Parameters
        ----------
        fname : str
            The filename to use.
        """
        check_fname(fname, 'annotations', ('-annot.fif', '-annot.fif.gz',
                                           '_annot.fif', '_annot.fif.gz',
                                           '.txt', '.csv'))
        if fname.endswith(".txt"):
            _write_annotations_txt(fname, self)
        elif fname.endswith(".csv"):
            _write_annotations_csv(fname, self)
        else:
            with start_file(fname) as fid:
                _write_annotations(fid, self)

    def _sort(self):
        """Sort in place."""
        # instead of argsort here we use sorted so that it gives us
        # the onset-then-duration hierarchy
        vals = sorted(zip(self.onset, self.duration, range(len(self))))
        order = list(list(zip(*vals))[-1]) if len(vals) else []
        self.onset = self.onset[order]
        self.duration = self.duration[order]
        self.description = self.description[order]

    def crop(self, tmin=None, tmax=None, emit_warning=False):
        """Remove all annotation that are outside of [tmin, tmax].

        The method operates inplace.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        emit_warning : bool
            Whether to emit warnings when limiting or omitting annotations.
            Defaults to False.

        Returns
        -------
        self : instance of Annotations
            The cropped Annotations object.
        """
        if len(self) == 0:
            return  # no annotations, nothing to do

        offset = 0 if self.orig_time is None else self.orig_time
        absolute_onset = self.onset + offset
        absolute_offset = absolute_onset + self.duration

        tmin = tmin if tmin is not None else absolute_onset.min()
        tmax = tmax if tmax is not None else absolute_offset.max()

        if tmin > tmax:
            raise ValueError('tmax should be greater than tmin.')

        if tmin < 0:
            raise ValueError('tmin should be positive.')

        out_of_bounds = (absolute_onset > tmax) | (absolute_offset < tmin)

        # clip the left side
        clip_left_elem = (absolute_onset < tmin) & ~out_of_bounds
        self.onset[clip_left_elem] = tmin - offset
        diff = tmin - absolute_onset[clip_left_elem]
        self.duration[clip_left_elem] = self.duration[clip_left_elem] - diff

        # clip the right side
        clip_right_elem = (absolute_offset > tmax) & ~out_of_bounds
        diff = absolute_offset[clip_right_elem] - tmax
        self.duration[clip_right_elem] = self.duration[clip_right_elem] - diff

        # remove out of bounds
        self.onset = self.onset.compress(~out_of_bounds)
        self.duration = self.duration.compress(~out_of_bounds)
        self.description = self.description.compress(~out_of_bounds)

        if emit_warning:
            omitted = out_of_bounds.sum()
            if omitted > 0:
                warn('Omitted %s annotation(s) that were outside data'
                     ' range.' % omitted)
            limited = clip_left_elem.sum() + clip_right_elem.sum()
            if limited > 0:
                warn('Limited %s annotation(s) that were expanding outside the'
                     ' data range.' % limited)

        return self


def _combine_annotations(one, two, one_n_samples, one_first_samp,
                         two_first_samp, sfreq, meas_date):
    """Combine a tuple of annotations."""
    if one is None and two is None:
        return None
    elif two is None:
        return one
    elif one is None:
        one = Annotations([], [], [], None)

    # Compute the shift necessary for alignment:
    # 1. The shift (in time) due to concatenation
    shift = one_n_samples / sfreq
    meas_date = _handle_meas_date(meas_date)
    # 2. Shift by the difference in meas_date and one.orig_time
    if one.orig_time is not None:
        shift += one_first_samp / sfreq
        shift += meas_date - one.orig_time
    # 3. Shift by the difference in meas_date and two.orig_time
    if two.orig_time is not None:
        shift -= two_first_samp / sfreq
        shift -= meas_date - two.orig_time

    onset = np.concatenate([one.onset, two.onset + shift])
    duration = np.concatenate([one.duration, two.duration])
    description = np.concatenate([one.description, two.description])
    return Annotations(onset, duration, description, one.orig_time)


def _handle_meas_date(meas_date):
    """Convert meas_date to seconds.

    If `meas_date` is a string, it should conform to the ISO8601 format.
    More precisely to this '%Y-%m-%d %H:%M:%S.%f' particular case of the
    ISO8601 format where the delimiter between date and time is ' '.

    Otherwise, this function returns 0. Note that ISO8601 allows for ' ' or 'T'
    as delimiters between date and time.
    """
    if meas_date is None:
        meas_date = 0
    elif isinstance(meas_date, str):
        ACCEPTED_ISO8601 = '%Y-%m-%d %H:%M:%S.%f'
        try:
            meas_date = datetime.strptime(meas_date, ACCEPTED_ISO8601)
        except ValueError:
            meas_date = 0
        else:
            unix_ref_time = datetime.utcfromtimestamp(0)
            meas_date = (meas_date - unix_ref_time).total_seconds()
        meas_date = round(meas_date, 6)  # round that 6th decimal
    elif isinstance(meas_date, datetime):
        meas_date = float(time.mktime(meas_date.timetuple()))
    elif not np.isscalar(meas_date):
        if len(meas_date) > 1:
            meas_date = meas_date[0] + meas_date[1] / 1000000.
        else:
            meas_date = meas_date[0]
    return float(meas_date)


def _sync_onset(raw, onset, inverse=False):
    """Adjust onsets in relation to raw data."""
    meas_date = _handle_meas_date(raw.info['meas_date'])
    if raw.annotations.orig_time is None:
        annot_start = onset
    else:
        offset = -raw._first_time if inverse else raw._first_time
        annot_start = (raw.annotations.orig_time - meas_date) - offset + onset
    return annot_start


def _annotations_starts_stops(raw, kinds, name='unknown', invert=False):
    """Get starts and stops from given kinds.

    onsets and ends are inclusive.
    """
    _validate_type(kinds, (str, list, tuple), str(type(kinds)),
                   "str, list or tuple")
    if isinstance(kinds, str):
        kinds = [kinds]
    else:
        for kind in kinds:
            _validate_type(kind, 'str', "All entries")

    if len(raw.annotations) == 0:
        onsets, ends = np.array([], int), np.array([], int)
    else:
        idxs = [idx for idx, desc in enumerate(raw.annotations.description)
                if any(desc.upper().startswith(kind.upper())
                       for kind in kinds)]
        # onsets are already sorted
        onsets = raw.annotations.onset[idxs]
        onsets = _sync_onset(raw, onsets)
        ends = onsets + raw.annotations.duration[idxs]
        onsets = raw.time_as_index(onsets, use_rounding=True)
        ends = raw.time_as_index(ends, use_rounding=True)
    assert (onsets <= ends).all()  # all durations >= 0
    if invert:
        # We need to eliminate overlaps here, otherwise wacky things happen,
        # so we carefully invert the relationship
        mask = np.zeros(len(raw.times), bool)
        for onset, end in zip(onsets, ends):
            mask[onset:end] = True
        mask = ~mask
        extras = (onsets == ends)
        extra_onsets, extra_ends = onsets[extras], ends[extras]
        onsets, ends = _mask_to_onsets_offsets(mask)
        # Keep ones where things were exactly equal
        del extras
        # we could do this with a np.insert+np.searchsorted, but our
        # ordered-ness should get us it for free
        onsets = np.sort(np.concatenate([onsets, extra_onsets]))
        ends = np.sort(np.concatenate([ends, extra_ends]))
        assert (onsets <= ends).all()
    return onsets, ends


def _write_annotations(fid, annotations):
    """Write annotations."""
    start_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)
    write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, annotations.onset)
    write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX,
                annotations.duration + annotations.onset)
    # To allow : in description, they need to be replaced for serialization
    write_name_list(fid, FIFF.FIFF_COMMENT, [d.replace(':', ';') for d in
                                             annotations.description])
    if annotations.orig_time is not None:
        write_double(fid, FIFF.FIFF_MEAS_DATE, annotations.orig_time)
    end_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)


def _write_annotations_csv(fname, annot):
    pd = _check_pandas_installed(strict=True)
    meas_date = _handle_meas_date(annot.orig_time)
    dt = datetime.utcfromtimestamp(meas_date)
    onsets_dt = [dt + timedelta(seconds=o) for o in annot.onset]
    df = pd.DataFrame(dict(onset=onsets_dt, duration=annot.duration,
                           description=annot.description))
    df.to_csv(fname, index=False)


def _write_annotations_txt(fname, annot):
    content = "# MNE-Annotations\n"
    if annot.orig_time is not None:
        meas_date = _handle_meas_date(annot.orig_time)
        orig_dt = datetime.utcfromtimestamp(meas_date)
        content += "# orig_time : %s   \n" % orig_dt
    content += "# onset, duration, description\n"

    data = np.array([annot.onset, annot.duration, annot.description],
                    dtype=str).T
    with open(fname, 'wb') as fid:
        fid.write(content.encode())
        np.savetxt(fid, data, delimiter=',', fmt="%s")


def read_annotations(fname, sfreq='auto', uint16_codec=None):
    r"""Read annotations from a file.

    This function reads a .fif, .fif.gz, .vrmk, .edf, .txt, .csv or .set file
    and makes an :class:`mne.Annotations` object.

    Parameters
    ----------
    fname : str
        The filename.
    sfreq : float | 'auto'
        The sampling frequency in the file. This parameter is necessary for
        \*.vmrk files as Annotations are expressed in seconds and \*.vmrk files
        are in samples. For any other file format, ``sfreq`` is omitted.
        If set to 'auto' then the ``sfreq`` is taken from the \*.vhdr
        file that has the same name (without file extension). So data.vrmk
        looks for sfreq in data.vhdr.
    uint16_codec : str | None
        This parameter is only used in EEGLAB (\*.set) and omitted otherwise.
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.

    Notes
    -----
    The annotations stored in a .csv require the onset columns to be
    timestamps. If you have onsets as floats (in seconds), you should use the
    .txt extension.
    """
    from .io.brainvision.brainvision import _read_annotations_brainvision
    from .io.eeglab.eeglab import _read_annotations_eeglab
    from .io.edf.edf import _read_annotations_edf
    from .io.cnt.cnt import _read_annotations_cnt

    name = op.basename(fname)
    if name.endswith(('fif', 'fif.gz')):
        # Read FiF files
        ff, tree, _ = fiff_open(fname, preload=False)
        with ff as fid:
            annotations = _read_annotations_fif(fid, tree)
    elif name.endswith('txt'):
        orig_time = _read_annotations_txt_parse_header(fname)
        onset, duration, description = _read_annotations_txt(fname)
        annotations = Annotations(onset=onset, duration=duration,
                                  description=description,
                                  orig_time=orig_time)

    elif name.endswith('vmrk'):
        annotations = _read_annotations_brainvision(fname, sfreq=sfreq)

    elif name.endswith('csv'):
        annotations = _read_annotations_csv(fname)

    elif name.endswith('cnt'):
        annotations = _read_annotations_cnt(fname)

    elif name.endswith('set'):
        annotations = _read_annotations_eeglab(fname,
                                               uint16_codec=uint16_codec)

    elif name.endswith(('edf', 'bdf', 'gdf')):
        onset, duration, description = _read_annotations_edf(fname)
        onset = np.array(onset, dtype=float)
        duration = np.array(duration, dtype=float)
        annotations = Annotations(onset=onset, duration=duration,
                                  description=description,
                                  orig_time=None)

    elif name.startswith('events_') and fname.endswith('mat'):
        annotations = _read_brainstorm_annotations(fname)
    else:
        raise IOError('Unknown annotation file format "%s"' % fname)

    if annotations is None:
        raise IOError('No annotation data found in file "%s"' % fname)
    return annotations


def _read_annotations_csv(fname):
    """Read annotations from csv.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    annot : instance of Annotations
        The annotations.
    """
    pd = _check_pandas_installed(strict=True)
    df = pd.read_csv(fname)
    orig_time = df['onset'].values[0]
    try:
        float(orig_time)
        warn('It looks like you have provided annotation onsets as floats. '
             'These will be interpreted as MILLISECONDS. If that is not what '
             'you want, save your CSV as a TXT file; the TXT reader accepts '
             'onsets in seconds.')
    except ValueError:
        pass
    orig_time = _handle_meas_date(orig_time)
    onset_dt = pd.to_datetime(df['onset'])
    onset = (onset_dt - onset_dt[0]).dt.total_seconds()
    duration = df['duration'].values.astype(float)
    description = df['description'].values
    if orig_time == 0:
        orig_time = None

    return Annotations(onset, duration, description, orig_time)


def _read_brainstorm_annotations(fname, orig_time=None):
    """Read annotations from a Brainstorm events_ file.

    Parameters
    ----------
    fname : str
        The filename
    orig_time : float | int | instance of datetime | array of int | None
        A POSIX Timestamp, datetime or an array containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisiton is started at the
        same time.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    from scipy import io

    def get_duration_from_times(t):
        return t[1] - t[0] if t.shape[0] == 2 else np.zeros(len(t[0]))

    annot_data = io.loadmat(fname)
    onsets, durations, descriptions = (list(), list(), list())
    for label, _, _, _, times, _, _ in annot_data['events'][0]:
        onsets.append(times[0])
        durations.append(get_duration_from_times(times))
        n_annot = len(times[0])
        descriptions += [str(label[0])] * n_annot

    return Annotations(onset=np.concatenate(onsets),
                       duration=np.concatenate(durations),
                       description=descriptions,
                       orig_time=orig_time)


def _is_iso8601(candidate_str):
    ISO8601 = r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d{6}$'
    return re.compile(ISO8601).match(candidate_str) is not None


def _read_annotations_txt_parse_header(fname):
    def is_orig_time(x):
        return x.startswith('# orig_time :')

    with open(fname) as fid:
        header = list(takewhile(lambda x: x.startswith('#'), fid))

    orig_values = [h[13:].strip() for h in header if is_orig_time(h)]
    orig_values = [_handle_meas_date(orig) for orig in orig_values
                   if _is_iso8601(orig)]

    return None if not orig_values else orig_values[0]


def _read_annotations_txt(fname):
    onset, duration, desc = np.loadtxt(fname, delimiter=',',
                                       dtype=np.bytes_, unpack=True)
    onset = [float(o.decode()) for o in onset]
    duration = [float(d.decode()) for d in duration]
    desc = [str(d.decode()).strip() for d in desc]
    return onset, duration, desc


def _read_annotations_fif(fid, tree):
    """Read annotations."""
    annot_data = dir_tree_find(tree, FIFF.FIFFB_MNE_ANNOTATIONS)
    if len(annot_data) == 0:
        annotations = None
    else:
        annot_data = annot_data[0]
        orig_time = None
        onset, duration, description = list(), list(), list()
        for ent in annot_data['directory']:
            kind = ent.kind
            pos = ent.pos
            tag = read_tag(fid, pos)
            if kind == FIFF.FIFF_MNE_BASELINE_MIN:
                onset = tag.data
                onset = list() if onset is None else onset
            elif kind == FIFF.FIFF_MNE_BASELINE_MAX:
                duration = tag.data
                duration = list() if duration is None else duration - onset
            elif kind == FIFF.FIFF_COMMENT:
                description = tag.data.split(':')
                description = [d.replace(';', ':') for d in
                               description]
            elif kind == FIFF.FIFF_MEAS_DATE:
                orig_time = float(tag.data)
        assert len(onset) == len(duration) == len(description)
        annotations = Annotations(onset, duration, description,
                                  orig_time)
    return annotations


def _ensure_annotation_object(obj):
    """Check that the object is an Annotations instance.

    Raise error otherwise.
    """
    if not isinstance(obj, Annotations):
        raise ValueError('Annotations must be an instance of '
                         'mne.Annotations. Got %s.' % obj)


def _select_annotations_based_on_description(descriptions, event_id, regexp):
    """Get a collection of descriptions and returns index of selected."""
    regexp_comp = re.compile('.*' if regexp is None else regexp)

    event_id_ = dict()
    dropped = []
    # Iterate over the sorted descriptions so that the Counter mapping
    # is slightly less arbitrary
    for desc in sorted(descriptions):
        if desc in event_id_:
            continue

        if regexp_comp.match(desc) is None:
            continue

        if isinstance(event_id, dict):
            if desc in event_id:
                event_id_[desc] = event_id[desc]
            else:
                continue
        else:
            trigger = event_id(desc)
            if trigger is not None:
                event_id_[desc] = trigger
            else:
                dropped.append(desc)

    event_sel = [ii for ii, kk in enumerate(descriptions)
                 if kk in event_id_]

    if len(event_sel) == 0 and regexp is not None:
        raise ValueError('Could not find any of the events you specified.')

    return event_sel, event_id_


def _check_event_id(event_id, raw):
    from .io.brainvision.brainvision import _BVEventParser
    from .io.brainvision.brainvision import _check_bv_annot
    from .io.brainvision.brainvision import RawBrainVision
    from .io import RawFIF, RawArray

    if event_id is None:
        return _DefaultEventParser()
    elif event_id == 'auto':
        if isinstance(raw, RawBrainVision):
            return _BVEventParser()
        elif (isinstance(raw, (RawFIF, RawArray)) and
              _check_bv_annot(raw.annotations.description)):
            logger.info('Non-RawBrainVision raw using branvision markers')
            return _BVEventParser()
        else:
            return _DefaultEventParser()
    elif callable(event_id) or isinstance(event_id, dict):
        return event_id
    else:
        raise ValueError('Invalid input event_id')


@verbose
def events_from_annotations(raw, event_id="auto",
                            regexp=r'^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$',
                            use_rounding=True, chunk_duration=None,
                            verbose=None):
    """Get events and event_id from an Annotations object.

    Parameters
    ----------
    raw : instance of Raw
        The raw data for which Annotations are defined.
    event_id : dict | callable | None | 'auto'
        Can be:

        - **dict**: map descriptions (keys) to integer event codes (values).
          Only the descriptions present will be mapped, others will be ignored.
        - **callable**: must take a string input and returns an integer event
          code or None to ignore it.
        - **None**: Map descriptions to unique integer values based on their
          ``sorted`` order.
        - **'auto' (default)**: prefer a raw-format-specific parser:

          - Brainvision: map stimulus events to their integer part; response
            events to integer part + 1000; optic events to integer part + 2000;
            'SyncStatus/Sync On' to 99998; 'New Segment/' to 99999;
            all others like ``None`` with an offset of 10000.
          - Other raw formats: Behaves like None.

          .. versionadded:: 0.18
    regexp : str | None
        Regular expression used to filter the annotations whose
        descriptions is a match. The default ignores descriptions beginning
        ``'bad'`` or ``'edge'`` (case-insensitive).

        .. versionchanged:: 0.18
           Default ignores bad and edge descriptions.
    use_rounding : boolean
        If True, use rounding (instead of truncation) when converting
        times to indices. This can help avoid non-unique indices.
    chunk_duration: float | None
        Chunk duration in seconds. If ``chunk_duration`` is set to None
        (default), generated events correspond to the annotation onsets.
        If not, :func:`mne.events_from_annotations` returns as many events as
        they fit within the annotation duration spaced according to
        ``chunk_duration``. As a consequence annotations with duration shorter
        than ``chunk_duration`` will not contribute events.
    %(verbose)s

    Returns
    -------
    events : ndarray, shape (n_events, 3)
        The events.
    event_id : dict
        The event_id variable that can be passed to Epochs.
    """
    if len(raw.annotations) == 0:
        event_id = dict() if not isinstance(event_id, dict) else event_id
        return np.empty((0, 3), dtype=int), event_id

    annotations = raw.annotations

    event_id = _check_event_id(event_id, raw)

    event_sel, event_id_ = _select_annotations_based_on_description(
        annotations.description, event_id=event_id, regexp=regexp)

    if chunk_duration is None:
        inds = raw.time_as_index(annotations.onset, use_rounding=use_rounding,
                                 origin=annotations.orig_time) + raw.first_samp

        values = [event_id_[kk] for kk in annotations.description[event_sel]]
        inds = inds[event_sel]
    else:
        inds = values = np.array([]).astype(int)
        for annot in annotations[event_sel]:
            annot_offset = annot['onset'] + annot['duration']
            _onsets = np.arange(start=annot['onset'], stop=annot_offset,
                                step=chunk_duration)
            good_events = annot_offset - _onsets >= chunk_duration
            if good_events.any():
                _onsets = _onsets[good_events]
                _inds = raw.time_as_index(_onsets,
                                          use_rounding=use_rounding,
                                          origin=annotations.orig_time)
                _inds += raw.first_samp
                inds = np.append(inds, _inds)
                _values = np.full(shape=len(_inds),
                                  fill_value=event_id_[annot['description']],
                                  dtype=int)
                values = np.append(values, _values)

    events = np.c_[inds, np.zeros(len(inds)), values].astype(int)

    logger.info('Used Annotations descriptions: %s' %
                (list(event_id_.keys()),))

    return events, event_id_
