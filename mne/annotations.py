# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

from datetime import datetime
import time
from copy import deepcopy

import numpy as np

from .utils import _pl, check_fname
from .externals.six import string_types
from .io.write import (start_block, end_block, write_float, write_name_list,
                       write_double, start_file)
from .io.constants import FIFF
from .io.open import fiff_open
from .io.tree import dir_tree_find
from .io.tag import read_tag


class Annotations(object):
    """Annotation object for annotating segments of raw data.

    Annotations are added to instance of :class:`mne.io.Raw` as an attribute
    named ``annotations``. To reject bad epochs using annotations, use
    annotation description starting with 'bad' keyword. The epochs with
    overlapping bad segments are then rejected automatically by default.

    To remove epochs with blinks you can do::

        >>> eog_events = mne.preprocessing.find_eog_events(raw)  # doctest: +SKIP
        >>> n_blinks = len(eog_events)  # doctest: +SKIP
        >>> onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  # doctest: +SKIP
        >>> duration = np.repeat(0.5, n_blinks)  # doctest: +SKIP
        >>> description = ['bad blink'] * n_blinks  # doctest: +SKIP
        >>> annotations = mne.Annotations(onset, duration, description)  # doctest: +SKIP
        >>> raw.annotations = annotations  # doctest: +SKIP
        >>> epochs = mne.Epochs(raw, events, event_id, tmin, tmax)  # doctest: +SKIP

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
    orig_time : float | int | instance of datetime | array of int | None
        A POSIX Timestamp, datetime or an array containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisiton is started at the
        same time.

    Notes
    -----
    If ``orig_time`` is None, the annotations are synced to the start of the
    data (0 seconds). Otherwise the annotations are synced to sample 0 and
    ``raw.first_samp`` is taken into account the same way as with events.
    """  # noqa: E501

    def __init__(self, onset, duration, description,
                 orig_time=None):  # noqa: D102
        if orig_time is not None:
            if isinstance(orig_time, datetime):
                orig_time = float(time.mktime(orig_time.timetuple()))
            elif not np.isscalar(orig_time):
                orig_time = orig_time[0] + orig_time[1] / 1000000.
            else:  # isscalar
                orig_time = float(orig_time)  # np.int not serializable
        self.orig_time = orig_time

        onset = np.array(onset, dtype=float)
        if onset.ndim != 1:
            raise ValueError('Onset must be a one dimensional array.')
        duration = np.array(duration, dtype=float)
        if isinstance(description, string_types):
            description = np.repeat(description, len(onset))
        if duration.ndim != 1:
            raise ValueError('Duration must be a one dimensional array.')
        if not (len(onset) == len(duration) == len(description)):
            raise ValueError('Onset, duration and description must be '
                             'equal in sizes.')
        if any([';' in desc for desc in description]):
            raise ValueError('Semicolons in descriptions not supported.')

        self.onset = onset
        self.duration = duration
        self.description = np.array(description, dtype=str)

    def __repr__(self):
        """Show the representation."""
        kinds = sorted(set('%s' % d.split(' ')[0].lower()
                           for d in self.description))
        kinds = ['%s (%s)' % (kind, sum(d.lower().startswith(kind)
                                        for d in self.description))
                 for kind in kinds]
        kinds = ', '.join(kinds[:3]) + ('' if len(kinds) <= 3 else '...')
        kinds = (': ' if len(kinds) > 0 else '') + kinds
        return ('<Annotations  |  %s segment%s %s >'
                % (len(self.onset), _pl(len(self.onset)), kinds))

    def __len__(self):
        """Return the number of annotations."""
        return len(self.duration)

    def __add__(self, other):
        """Add (concatencate) two Annotation objects."""
        return self.copy().append(other.onset, other.duration,
                                  other.description)

    def __iadd__(self, other):
        """Add (concatencate) two Annotation objects in-place."""
        return self.append(other.onset, other.duration, other.description)

    def append(self, onset, duration, description):
        """Add an annotated segment. Operates inplace.

        Parameters
        ----------
        onset : float
            Annotation time onset from the beginning of the recording in
            seconds.
        duration : float
            Duration of the annotation in seconds.
        description : str
            Description for the annotation. To reject epochs, use description
            starting with keyword 'bad'

        Returns
        -------
        self : mne.Annotations
            The modified Annotations object.
        """
        self.onset = np.append(self.onset, onset)
        self.duration = np.append(self.duration, duration)
        self.description = np.append(self.description, description)
        return self

    def copy(self):
        """Return a deep copy of self."""
        return deepcopy(self)

    def delete(self, idx):
        """Remove an annotation. Operates inplace.

        Parameters
        ----------
        idx : int | list of int
            Index of the annotation to remove.
        """
        self.onset = np.delete(self.onset, idx)
        self.duration = np.delete(self.duration, idx)
        self.description = np.delete(self.description, idx)

    def save(self, fname):
        """Save annotations to FIF.

        Typically annotations get saved in the FIF file for raw data
        (e.g., as ``raw.annotations``), but this offers the possibility
        to also save them to disk separately.

        Parameters
        ----------
        fname : str
            The filename to use.
        """
        check_fname(fname, 'annotations', ('-annot.fif', '-annot.fif.gz',
                                           '_annot.fif', '_annot.fif.gz'))
        with start_file(fname) as fid:
            _write_annotations(fid, self)


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
    """Convert meas_date to seconds."""
    if meas_date is None:
        meas_date = 0
    elif not np.isscalar(meas_date):
        if len(meas_date) > 1:
            meas_date = meas_date[0] + meas_date[1] / 1000000.
        else:
            meas_date = meas_date[0]
    return meas_date


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
    if not isinstance(kinds, (string_types, list, tuple)):
        raise TypeError('%s must be str, list, or tuple, got %s'
                        % (type(kinds), name))
    elif isinstance(kinds, string_types):
        kinds = [kinds]
    elif not all(isinstance(kind, string_types) for kind in kinds):
        raise TypeError('All entries in %s must be str' % (name,))
    if raw.annotations is None:
        onsets, ends = np.array([], int), np.array([], int)
    else:
        idxs = [idx for idx, desc in enumerate(raw.annotations.description)
                if any(desc.upper().startswith(kind.upper())
                       for kind in kinds)]
        onsets = raw.annotations.onset[idxs]
        onsets = _sync_onset(raw, onsets)
        ends = onsets + raw.annotations.duration[idxs]
        order = np.argsort(onsets)
        onsets = raw.time_as_index(onsets[order], use_rounding=True)
        ends = raw.time_as_index(ends[order], use_rounding=True)
    if invert:
        # We invert the relationship (i.e., get segments that do not satisfy)
        if len(onsets) == 0 or onsets[0] != 0:
            onsets = np.concatenate([[0], onsets])
            ends = np.concatenate([[0], ends])
        if len(ends) == 1 or ends[-1] != len(raw.times):
            onsets = np.concatenate([onsets, [len(raw.times)]])
            ends = np.concatenate([ends, [len(raw.times)]])
        onsets, ends = ends[:-1], onsets[1:]
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


def read_annotations(fname):
    """Read annotations from a FIF file.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    ff, tree, _ = fiff_open(fname, preload=False)
    with ff as fid:
        annotations = _read_annotations(fid, tree)
    if annotations is None:
        raise IOError('No annotation data found in file "%s"' % fname)
    return annotations


def read_brainstorm_annotations(fname, orig_time=None):
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
        if t.shape[0] == 2:
            return t[1] - t[0]
        else:
            return np.zeros(len(t[0]))

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


def _read_annotations(fid, tree):
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
