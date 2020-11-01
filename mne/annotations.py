# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import os.path as op
import re
from copy import deepcopy
from itertools import takewhile
from collections import Counter
from collections.abc import Iterable
import warnings
from textwrap import shorten
import numpy as np

from .utils import (_pl, check_fname, _validate_type, verbose, warn, logger,
                    _check_pandas_installed, _mask_to_onsets_offsets,
                    _DefaultEventParser, _check_dt, _stamp_to_dt, _dt_to_stamp,
                    _check_fname)

from .io.write import (start_block, end_block, write_float, write_name_list,
                       write_double, start_file)
from .io.constants import FIFF
from .io.open import fiff_open
from .io.tree import dir_tree_find
from .io.tag import read_tag

# For testing windows_like_datetime, we monkeypatch "datetime" in this module.
# Keep the true datetime object around for _validate_type use.
_datetime = datetime


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
    if any(['{COLON}' in desc for desc in description]):
        raise ValueError('The substring "{COLON}" '
                         'in descriptions not supported.')

    if not (len(onset) == len(duration) == len(description)):
        raise ValueError('Onset, duration and description must be '
                         'equal in sizes, got %s, %s, and %s.'
                         % (len(onset), len(duration), len(description)))
    return onset, duration, description


class Annotations(object):
    """Annotation object for annotating segments of raw data.

    .. note::
       To convert events to `~mne.Annotations`, use
       `~mne.annotations_from_events`. To convert existing `~mne.Annotations`
       to events, use  `~mne.events_from_annotations`.

    Parameters
    ----------
    onset : array of float, shape (n_annotations,)
        The starting time of annotations in seconds after ``orig_time``.
    duration : array of float, shape (n_annotations,) | float
        Durations of the annotations in seconds. If a float, all the
        annotations are given the same duration.
    description : array of str, shape (n_annotations,) | str
        Array of strings containing description for each annotation. If a
        string, all the annotations are given the same description. To reject
        epochs, use description starting with keyword 'bad'. See example above.
    orig_time : float | str | datetime | tuple of int | None
        A POSIX Timestamp, datetime or a tuple containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisiton is started at the
        same time. If it is a string, it should conform to the ISO8601 format.
        More precisely to this '%Y-%m-%d %H:%M:%S.%f' particular case of the
        ISO8601 format where the delimiter between date and time is ' '.

    See Also
    --------
    mne.annotations_from_events
    mne.events_from_annotations

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
        self._orig_time = _handle_meas_date(orig_time)
        self.onset, self.duration, self.description = _check_o_d_s(
            onset, duration, description)
        self._sort()  # ensure we're sorted

    @property
    def orig_time(self):
        """The time base of the Annotations."""
        return self._orig_time

    def __eq__(self, other):
        """Compare to another Annotations instance."""
        if not isinstance(other, Annotations):
            return False
        return (np.array_equal(self.onset, other.onset) and
                np.array_equal(self.duration, other.duration) and
                np.array_equal(self.description, other.description) and
                self.orig_time == other.orig_time)

    def __repr__(self):
        """Show the representation."""
        counter = Counter(self.description)
        kinds = ', '.join(['%s (%s)' % k for k in sorted(counter.items())])
        kinds = (': ' if len(kinds) > 0 else '') + kinds
        s = ('Annotations | %s segment%s%s' %
             (len(self.onset), _pl(len(self.onset)), kinds))
        return '<' + shorten(s, width=77, placeholder=' ...') + '>'

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
            self._orig_time = other.orig_time
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
            return OrderedDict(zip(out_keys, out_vals))
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
            starting with keyword 'bad'.

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
        """Return a copy of the Annotations.

        Returns
        -------
        inst : instance of Annotations
            A copy of the object.
        """
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

    @verbose
    def crop(self, tmin=None, tmax=None, emit_warning=False, verbose=None):
        """Remove all annotation that are outside of [tmin, tmax].

        The method operates inplace.

        Parameters
        ----------
        tmin : float | datetime | None
            Start time of selection in seconds.
        tmax : float | datetime | None
            End time of selection in seconds.
        emit_warning : bool
            Whether to emit warnings when limiting or omitting annotations.
            Defaults to False.
        %(verbose_meth)s

        Returns
        -------
        self : instance of Annotations
            The cropped Annotations object.
        """
        if len(self) == 0:
            return self  # no annotations, nothing to do
        if self.orig_time is None:
            offset = _handle_meas_date(0)
        else:
            offset = self.orig_time
        if tmin is None:
            tmin = timedelta(self.onset.min()) + offset
        if tmax is None:
            tmax = timedelta((self.onset + self.duration).max()) + offset
        for key, val in [('tmin', tmin), ('tmax', tmax)]:
            _validate_type(val, ('numeric', _datetime), key,
                           'numeric, datetime, or None')
        if tmin > tmax:
            raise ValueError('tmax should be greater than or equal to tmin '
                             '(%s < %s).' % (tmax, tmin))
        logger.debug('Cropping annotations %s - %s' % (tmin, tmax))
        absolute_tmin = _handle_meas_date(tmin)
        absolute_tmax = _handle_meas_date(tmax)
        del tmin, tmax

        onsets, durations, descriptions = [], [], []
        out_of_bounds, clip_left_elem, clip_right_elem = [], [], []
        for onset, duration, description in zip(
                self.onset, self.duration, self.description):
            # if duration is NaN behave like a zero
            if np.isnan(duration):
                duration = 0.
            # convert to absolute times
            absolute_onset = timedelta(0, onset) + offset
            absolute_offset = absolute_onset + timedelta(0, duration)
            out_of_bounds.append(
                absolute_onset > absolute_tmax or
                absolute_offset < absolute_tmin)
            if out_of_bounds[-1]:
                clip_left_elem.append(False)
                clip_right_elem.append(False)
            else:
                # clip the left side
                clip_left_elem.append(absolute_onset < absolute_tmin)
                if clip_left_elem[-1]:
                    absolute_onset = absolute_tmin
                clip_right_elem.append(absolute_offset > absolute_tmax)
                if clip_right_elem[-1]:
                    absolute_offset = absolute_tmax
                if clip_left_elem[-1] or clip_right_elem[-1]:
                    durations.append(
                        (absolute_offset - absolute_onset).total_seconds())
                else:
                    durations.append(duration)
                onsets.append(
                    (absolute_onset - offset).total_seconds())
                descriptions.append(description)
        self.onset = np.array(onsets, float)
        self.duration = np.array(durations, float)
        assert (self.duration >= 0).all()
        self.description = np.array(descriptions, dtype=str)

        if emit_warning:
            omitted = np.array(out_of_bounds).sum()
            if omitted > 0:
                warn('Omitted %s annotation(s) that were outside data'
                     ' range.' % omitted)
            limited = (np.array(clip_left_elem) |
                       np.array(clip_right_elem)).sum()
            if limited > 0:
                warn('Limited %s annotation(s) that were expanding outside the'
                     ' data range.' % limited)

        return self


def _combine_annotations(one, two, one_n_samples, one_first_samp,
                         two_first_samp, sfreq, meas_date):
    """Combine a tuple of annotations."""
    assert one is not None
    assert two is not None
    shift = one_n_samples / sfreq  # to the right by the number of samples
    shift += one_first_samp / sfreq  # to the right by the offset
    shift -= two_first_samp / sfreq  # undo its offset
    onset = np.concatenate([one.onset, two.onset + shift])
    duration = np.concatenate([one.duration, two.duration])
    description = np.concatenate([one.description, two.description])
    return Annotations(onset, duration, description, one.orig_time)


def _handle_meas_date(meas_date):
    """Convert meas_date to datetime or None.

    If `meas_date` is a string, it should conform to the ISO8601 format.
    More precisely to this '%Y-%m-%d %H:%M:%S.%f' particular case of the
    ISO8601 format where the delimiter between date and time is ' '.
    Note that ISO8601 allows for ' ' or 'T' as delimiters between date and
    time.
    """
    if isinstance(meas_date, str):
        ACCEPTED_ISO8601 = '%Y-%m-%d %H:%M:%S.%f'
        try:
            meas_date = datetime.strptime(meas_date, ACCEPTED_ISO8601)
        except ValueError:
            meas_date = None
        else:
            meas_date = meas_date.replace(tzinfo=timezone.utc)
    elif isinstance(meas_date, tuple):
        # old way
        meas_date = _stamp_to_dt(meas_date)
    if meas_date is not None:
        if np.isscalar(meas_date):
            # It would be nice just to do:
            #
            #     meas_date = datetime.fromtimestamp(meas_date, timezone.utc)
            #
            # But Windows does not like timestamps < 0. So we'll use
            # our specialized wrapper instead:
            meas_date = np.array(np.modf(meas_date)[::-1])
            meas_date *= [1, 1e6]
            meas_date = _stamp_to_dt(np.round(meas_date))
        _check_dt(meas_date)  # run checks
    return meas_date


def _sync_onset(raw, onset, inverse=False):
    """Adjust onsets in relation to raw data."""
    offset = (-1 if inverse else 1) * raw._first_time
    assert raw.info['meas_date'] == raw.annotations.orig_time
    annot_start = onset - offset
    return annot_start


def _annotations_starts_stops(raw, kinds, name='skip_by_annotation',
                              invert=False):
    """Get starts and stops from given kinds.

    onsets and ends are inclusive.
    """
    _validate_type(kinds, (str, list, tuple), name)
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
    # -> replace with "{COLON}". When read back in, replace it back with ":"
    write_name_list(fid, FIFF.FIFF_COMMENT, [d.replace(':', '{COLON}') for d in
                                             annotations.description])
    if annotations.orig_time is not None:
        write_double(fid, FIFF.FIFF_MEAS_DATE,
                     _dt_to_stamp(annotations.orig_time))
    end_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)


def _write_annotations_csv(fname, annot):
    pd = _check_pandas_installed(strict=True)
    dt = _handle_meas_date(annot.orig_time)
    if dt is None:
        dt = _handle_meas_date(0)
    dt = dt.replace(tzinfo=None)
    onsets_dt = [dt + timedelta(seconds=o) for o in annot.onset]
    df = pd.DataFrame(dict(onset=onsets_dt, duration=annot.duration,
                           description=annot.description))
    df.to_csv(fname, index=False)


def _write_annotations_txt(fname, annot):
    content = "# MNE-Annotations\n"
    if annot.orig_time is not None:
        # for backward compat, we do not write tzinfo (assumed UTC)
        content += ("# orig_time : %s   \n"
                    % annot.orig_time.replace(tzinfo=None))
    content += "# onset, duration, description\n"

    data = np.array([annot.onset, annot.duration, annot.description],
                    dtype=str).T
    with open(fname, 'wb') as fid:
        fid.write(content.encode())
        np.savetxt(fid, data, delimiter=',', fmt="%s")


def read_annotations(fname, sfreq='auto', uint16_codec=None):
    r"""Read annotations from a file.

    This function reads a .fif, .fif.gz, .vrmk, .edf, .txt, .csv .cnt, .cef,
    or .set file and makes an :class:`mne.Annotations` object.

    Parameters
    ----------
    fname : str
        The filename.
    sfreq : float | 'auto'
        The sampling frequency in the file. This parameter is necessary for
        \*.vmrk and \*.cef files as Annotations are expressed in seconds and
        \*.vmrk/\*.cef files are in samples. For any other file format,
        ``sfreq`` is omitted. If set to 'auto' then the ``sfreq`` is taken
        from the respective info file of the same name with according file
        extension (\*.vhdr for brainvision; \*.dap for Curry 7; \*.cdt.dpa for
        Curry 8). So data.vrmk looks for sfreq in data.vhdr, data.cef looks in
        data.dap and data.cdt.cef looks in data.cdt.dpa.
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
    from .io.curry.curry import _read_annotations_curry
    from .io.ctf.markers import _read_annotations_ctf
    _validate_type(fname, 'path-like', 'fname')
    fname = _check_fname(
        fname, overwrite='read', must_exist=True,
        allow_dir=str(fname).endswith('.ds'),  # allow_dir for CTF
        name='fname')
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

    elif name.endswith('ds'):
        annotations = _read_annotations_ctf(fname)

    elif name.endswith('cef'):
        annotations = _read_annotations_curry(fname, sfreq=sfreq)

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
    onset_dt = pd.to_datetime(df['onset'])
    onset = (onset_dt - onset_dt[0]).dt.total_seconds()
    duration = df['duration'].values.astype(float)
    description = df['description'].values
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
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        out = np.loadtxt(fname, delimiter=',',
                         dtype=np.bytes_, unpack=True)
    if len(out) == 0:
        onset, duration, desc = [], [], []
    else:
        onset, duration, desc = out

    onset = [float(o.decode()) for o in np.atleast_1d(onset)]
    duration = [float(d.decode()) for d in np.atleast_1d(duration)]
    desc = [str(d.decode()).strip() for d in np.atleast_1d(desc)]
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

                # replace all "{COLON}" in FIF files with necessary
                # : character
                description = [d.replace('{COLON}', ':') for d in
                               description]
            elif kind == FIFF.FIFF_MEAS_DATE:
                orig_time = tag.data
                try:
                    orig_time = float(orig_time)  # old way
                except TypeError:
                    orig_time = tuple(orig_time)  # new way
        assert len(onset) == len(duration) == len(description)
        annotations = Annotations(onset, duration, description,
                                  orig_time)
    return annotations


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


def _select_events_based_on_id(events, event_desc):
    """Get a collection of events and returns index of selected."""
    event_desc_ = dict()
    func = event_desc.get if isinstance(event_desc, dict) else event_desc
    event_ids = events[np.unique(events[:, 2], return_index=True)[1], 2]
    for e in event_ids:
        trigger = func(e)
        if trigger is not None:
            event_desc_[e] = trigger

    event_sel = [ii for ii, e in enumerate(events) if e[2] in event_desc_]

    if len(event_sel) == 0:
        raise ValueError('Could not find any of the events you specified.')

    return event_sel, event_desc_


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
        raise ValueError('Invalid type for event_id (should be None, str, '
                         'dict or callable). Got {}'.format(type(event_id)))


def _check_event_description(event_desc, events):
    """Check event_id and convert to default format."""
    if event_desc is None:  # convert to int to make typing-checks happy
        event_desc = list(np.unique(events[:, 2]))

    if isinstance(event_desc, dict):
        for val in event_desc.values():
            _validate_type(val, (str, None), 'Event names')
    elif isinstance(event_desc, Iterable):
        event_desc = np.asarray(event_desc)
        if event_desc.ndim != 1:
            raise ValueError('event_desc must be 1D, got shape {}'.format(
                             event_desc.shape))
        event_desc = dict(zip(event_desc, map(str, event_desc)))
    elif callable(event_desc):
        pass
    else:
        raise ValueError('Invalid type for event_desc (should be None, list, '
                         '1darray, dict or callable). Got {}'.format(
                             type(event_desc)))

    return event_desc


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
        - **callable**: must take a string input and return an integer event
          code, or return ``None`` to ignore the event.
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
    use_rounding : bool
        If True, use rounding (instead of truncation) when converting
        times to indices. This can help avoid non-unique indices.
    chunk_duration : float | None
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

    See Also
    --------
    mne.annotations_from_events

    Notes
    -----
    For data formats that store integer events as strings (e.g., NeuroScan
    ``.cnt`` files), passing the Python built-in function :class:`int` as the
    ``event_id`` parameter will do what most users probably want in those
    circumstances: return an ``event_id`` dictionary that maps event ``'1'`` to
    integer event code ``1``, ``'2'`` to ``2``, etc.
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
                                 origin=annotations.orig_time)
        if annotations.orig_time is not None:
            inds += raw.first_samp
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


@verbose
def annotations_from_events(events, sfreq, event_desc=None, first_samp=0,
                            orig_time=None, verbose=None):
    """Convert an event array to an Annotations object.

    Parameters
    ----------
    events : ndarray, shape (n_events, 3)
        The events.
    sfreq : float
        Sampling frequency.
    event_desc : dict | array-like | callable | None
        Events description. Can be:

        - **dict**: map integer event codes (keys) to descriptions (values).
          Only the descriptions present will be mapped, others will be ignored.
        - **array-like**: list, or 1d array of integers event codes to include.
          Only the event codes present will be mapped, others will be ignored.
          Event codes will be passed as string descriptions.
        - **callable**: must take a integer event code as input and return a
          string description or None to ignore it.
        - **None**: Use integer event codes as descriptions.
    first_samp : int
        The first data sample (default=0). See :attr:`mne.io.Raw.first_samp`
        docstring.
    orig_time : float | str | datetime | tuple of int | None
        Determines the starting time of annotation acquisition. If None
        (default), starting time is determined from beginning of raw data
        acquisition. For details, see :meth:`mne.Annotations` docstring.
    %(verbose)s

    Returns
    -------
    annot : instance of Annotations
        The annotations.

    See Also
    --------
    mne.events_from_annotations

    Notes
    -----
    Annotations returned by this function will all have zero (null) duration.

    Creating events from annotations via the function
    `mne.events_from_annotations` takes in event mappings with
    key→value pairs as description→ID, whereas `mne.annotations_from_events`
    takes in event mappings with key→value pairs as ID→description.
    If you need to use these together, you can invert the mapping by doing::

        event_desc = {v: k for k, v in event_id.items()}
    """
    event_desc = _check_event_description(event_desc, events)
    event_sel, event_desc_ = _select_events_based_on_id(events, event_desc)
    events_sel = events[event_sel]
    onsets = (events_sel[:, 0] - first_samp) / sfreq
    descriptions = [event_desc_[e[2]] for e in events_sel]
    durations = np.zeros(len(events_sel))  # dummy durations

    # Create annotations
    annots = Annotations(onset=onsets,
                         duration=durations,
                         description=descriptions,
                         orig_time=orig_time)

    return annots
