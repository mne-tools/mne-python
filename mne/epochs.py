"""Tools for working with epoched data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import operator
import os.path as op
from collections import Counter
from copy import deepcopy
from functools import partial
from inspect import getfullargspec
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from ._fiff.constants import FIFF
from ._fiff.meas_info import (
    ContainsMixin,
    SetChannelsMixin,
    _ensure_infos_match,
    read_meas_info,
    write_meas_info,
)
from ._fiff.open import _get_next_fname, fiff_open
from ._fiff.pick import (
    _DATA_CH_TYPES_SPLIT,
    _pick_data_channels,
    _picks_to_idx,
    channel_indices_by_type,
    channel_type,
    pick_channels,
    pick_info,
)
from ._fiff.proj import ProjMixin, setup_proj
from ._fiff.tag import _read_tag_header, read_tag
from ._fiff.tree import dir_tree_find
from ._fiff.utils import _make_split_fnames
from ._fiff.write import (
    _NEXT_FILE_BUFFER,
    INT32_MAX,
    _get_split_size,
    end_block,
    start_and_end_file,
    start_block,
    write_complex_double_matrix,
    write_complex_float_matrix,
    write_double_matrix,
    write_float,
    write_float_matrix,
    write_id,
    write_int,
    write_string,
)
from .annotations import (
    EpochAnnotationsMixin,
    _read_annotations_fif,
    _write_annotations,
    events_from_annotations,
)
from .baseline import _check_baseline, _log_rescale, rescale
from .bem import _check_origin
from .channels.channels import InterpolationMixin, ReferenceMixin, UpdateChannelsMixin
from .event import _read_events_fif, make_fixed_length_events, match_event_names
from .evoked import EvokedArray
from .filter import FilterMixin, _check_fun, detrend
from .fixes import rng_uniform
from .html_templates import _get_html_template
from .parallel import parallel_func
from .time_frequency.spectrum import EpochsSpectrum, SpectrumMixin, _validate_method
from .time_frequency.tfr import AverageTFR, EpochsTFR
from .utils import (
    ExtendedTimeMixin,
    GetEpochsMixin,
    SizeMixin,
    _build_data_frame,
    _check_combine,
    _check_event_id,
    _check_fname,
    _check_option,
    _check_pandas_index_arguments,
    _check_pandas_installed,
    _check_preload,
    _check_time_format,
    _convert_times,
    _ensure_events,
    _gen_events,
    _on_missing,
    _path_like,
    _pl,
    _prepare_read_metadata,
    _prepare_write_metadata,
    _scale_dataframe_data,
    _validate_type,
    check_fname,
    check_random_state,
    copy_function_doc_to_method_doc,
    logger,
    object_size,
    repr_html,
    sizeof_fmt,
    verbose,
    warn,
)
from .utils.docs import fill_doc
from .viz import plot_drop_log, plot_epochs, plot_epochs_image, plot_topo_image_epochs


def _pack_reject_params(epochs):
    reject_params = dict()
    for key in ("reject", "flat", "reject_tmin", "reject_tmax"):
        val = getattr(epochs, key, None)
        if val is not None:
            reject_params[key] = val
    return reject_params


def _save_split(epochs, split_fnames, part_idx, n_parts, fmt, overwrite):
    """Split epochs.

    Anything new added to this function also needs to be added to
    BaseEpochs.save to account for new file sizes.
    """
    # insert index in filename
    this_fname = split_fnames[part_idx]
    _check_fname(this_fname, overwrite=overwrite)

    next_fname, next_idx = None, None
    if part_idx < n_parts - 1:
        next_idx = part_idx + 1
        next_fname = split_fnames[next_idx]

    with start_and_end_file(this_fname) as fid:
        _save_part(fid, epochs, fmt, n_parts, next_fname, next_idx)


def _save_part(fid, epochs, fmt, n_parts, next_fname, next_idx):
    info = epochs.info
    meas_id = info["meas_id"]

    start_block(fid, FIFF.FIFFB_MEAS)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    if info["meas_id"] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info["meas_id"])

    # Write measurement info
    write_meas_info(fid, info)

    # One or more evoked data sets
    start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
    start_block(fid, FIFF.FIFFB_MNE_EPOCHS)

    # write events out after getting data to ensure bad events are dropped
    data = epochs.get_data(copy=False)

    _check_option("fmt", fmt, ["single", "double"])

    if np.iscomplexobj(data):
        if fmt == "single":
            write_function = write_complex_float_matrix
        elif fmt == "double":
            write_function = write_complex_double_matrix
    else:
        if fmt == "single":
            write_function = write_float_matrix
        elif fmt == "double":
            write_function = write_double_matrix

    # Epoch annotations are written if there are any
    annotations = getattr(epochs, "annotations", [])
    if annotations is not None and len(annotations):
        _write_annotations(fid, annotations)

    # write Epoch event windows
    start_block(fid, FIFF.FIFFB_MNE_EVENTS)
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, epochs.events.T)
    write_string(fid, FIFF.FIFF_DESCRIPTION, _event_id_string(epochs.event_id))
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)

    # Metadata
    if epochs.metadata is not None:
        start_block(fid, FIFF.FIFFB_MNE_METADATA)
        metadata = _prepare_write_metadata(epochs.metadata)
        write_string(fid, FIFF.FIFF_DESCRIPTION, metadata)
        end_block(fid, FIFF.FIFFB_MNE_METADATA)

    # First and last sample
    first = int(round(epochs.tmin * info["sfreq"]))  # round just to be safe
    last = first + len(epochs.times) - 1
    write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first)
    write_int(fid, FIFF.FIFF_LAST_SAMPLE, last)

    # write raw original sampling rate
    write_float(fid, FIFF.FIFF_MNE_EPOCHS_RAW_SFREQ, epochs._raw_sfreq)

    # save baseline
    if epochs.baseline is not None:
        bmin, bmax = epochs.baseline
        write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, bmin)
        write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX, bmax)

    # The epochs itself
    decal = np.empty(info["nchan"])
    for k in range(info["nchan"]):
        decal[k] = 1.0 / (info["chs"][k]["cal"] * info["chs"][k].get("scale", 1.0))

    data *= decal[np.newaxis, :, np.newaxis]

    write_function(fid, FIFF.FIFF_EPOCH, data)

    # undo modifications to data
    data /= decal[np.newaxis, :, np.newaxis]

    write_string(fid, FIFF.FIFF_MNE_EPOCHS_DROP_LOG, json.dumps(epochs.drop_log))

    reject_params = _pack_reject_params(epochs)
    if reject_params:
        write_string(fid, FIFF.FIFF_MNE_EPOCHS_REJECT_FLAT, json.dumps(reject_params))

    write_int(fid, FIFF.FIFF_MNE_EPOCHS_SELECTION, epochs.selection)

    # And now write the next file info in case epochs are split on disk
    if next_fname is not None and n_parts > 1:
        start_block(fid, FIFF.FIFFB_REF)
        write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_NEXT_FILE)
        write_string(fid, FIFF.FIFF_REF_FILE_NAME, op.basename(next_fname))
        if meas_id is not None:
            write_id(fid, FIFF.FIFF_REF_FILE_ID, meas_id)
        write_int(fid, FIFF.FIFF_REF_FILE_NUM, next_idx)
        end_block(fid, FIFF.FIFFB_REF)

    end_block(fid, FIFF.FIFFB_MNE_EPOCHS)
    end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)


def _event_id_string(event_id):
    return ";".join([k + ":" + str(v) for k, v in event_id.items()])


def _merge_events(events, event_id, selection):
    """Merge repeated events."""
    event_id = event_id.copy()
    new_events = events.copy()
    event_idxs_to_delete = list()
    unique_events, counts = np.unique(events[:, 0], return_counts=True)
    for ev in unique_events[counts > 1]:
        # indices at which the non-unique events happened
        idxs = (events[:, 0] == ev).nonzero()[0]

        # Figure out new value for events[:, 1]. Set to 0, if mixed vals exist
        unique_priors = np.unique(events[idxs, 1])
        new_prior = unique_priors[0] if len(unique_priors) == 1 else 0

        # If duplicate time samples have same event val, "merge" == "drop"
        # and no new event_id key will be created
        ev_vals = np.unique(events[idxs, 2])
        if len(ev_vals) <= 1:
            new_event_val = ev_vals[0]

        # Else, make a new event_id for the merged event
        else:
            # Find all event_id keys involved in duplicated events. These
            # keys will be merged to become a new entry in "event_id"
            event_id_keys = list(event_id.keys())
            event_id_vals = list(event_id.values())
            new_key_comps = [
                event_id_keys[event_id_vals.index(value)] for value in ev_vals
            ]

            # Check if we already have an entry for merged keys of duplicate
            # events ... if yes, reuse it
            for key in event_id:
                if set(key.split("/")) == set(new_key_comps):
                    new_event_val = event_id[key]
                    break

            # Else, find an unused value for the new key and make an entry into
            # the event_id dict
            else:
                ev_vals = np.unique(
                    np.concatenate(
                        (list(event_id.values()), events[:, 1:].flatten()), axis=0
                    )
                )
                if ev_vals[0] > 1:
                    new_event_val = 1
                else:
                    diffs = np.diff(ev_vals)
                    idx = np.where(diffs > 1)[0]
                    idx = -1 if len(idx) == 0 else idx[0]
                    new_event_val = ev_vals[idx] + 1

                new_event_id_key = "/".join(sorted(new_key_comps))
                event_id[new_event_id_key] = int(new_event_val)

        # Replace duplicate event times with merged event and remember which
        # duplicate indices to delete later
        new_events[idxs[0], 1] = new_prior
        new_events[idxs[0], 2] = new_event_val
        event_idxs_to_delete.extend(idxs[1:])

    # Delete duplicate event idxs
    new_events = np.delete(new_events, event_idxs_to_delete, 0)
    new_selection = np.delete(selection, event_idxs_to_delete, 0)

    return new_events, event_id, new_selection


def _handle_event_repeated(events, event_id, event_repeated, selection, drop_log):
    """Handle repeated events.

    Note that drop_log will be modified inplace
    """
    assert len(events) == len(selection)
    selection = np.asarray(selection)

    unique_events, u_ev_idxs = np.unique(events[:, 0], return_index=True)

    # Return early if no duplicates
    if len(unique_events) == len(events):
        return events, event_id, selection, drop_log

    # Else, we have duplicates. Triage ...
    _check_option("event_repeated", event_repeated, ["error", "drop", "merge"])
    drop_log = list(drop_log)
    if event_repeated == "error":
        raise RuntimeError(
            "Event time samples were not unique. Consider "
            'setting the `event_repeated` parameter."'
        )

    elif event_repeated == "drop":
        logger.info(
            "Multiple event values for single event times found. "
            "Keeping the first occurrence and dropping all others."
        )
        new_events = events[u_ev_idxs]
        new_selection = selection[u_ev_idxs]
        drop_ev_idxs = np.setdiff1d(selection, new_selection)
        for idx in drop_ev_idxs:
            drop_log[idx] = drop_log[idx] + ("DROP DUPLICATE",)
        selection = new_selection
    elif event_repeated == "merge":
        logger.info(
            "Multiple event values for single event times found. "
            "Creating new event value to reflect simultaneous events."
        )
        new_events, event_id, new_selection = _merge_events(events, event_id, selection)
        drop_ev_idxs = np.setdiff1d(selection, new_selection)
        for idx in drop_ev_idxs:
            drop_log[idx] = drop_log[idx] + ("MERGE DUPLICATE",)
        selection = new_selection
    drop_log = tuple(drop_log)

    # Remove obsolete kv-pairs from event_id after handling
    keys = new_events[:, 1:].flatten()
    event_id = {k: v for k, v in event_id.items() if v in keys}

    return new_events, event_id, selection, drop_log


@fill_doc
class BaseEpochs(
    ProjMixin,
    ContainsMixin,
    UpdateChannelsMixin,
    ReferenceMixin,
    SetChannelsMixin,
    InterpolationMixin,
    FilterMixin,
    ExtendedTimeMixin,
    SizeMixin,
    GetEpochsMixin,
    EpochAnnotationsMixin,
    SpectrumMixin,
):
    """Abstract base class for `~mne.Epochs`-type classes.

    .. note::
        This class should not be instantiated directly via
        ``mne.BaseEpochs(...)``. Instead, use one of the functions listed in
        the See Also section below.

    Parameters
    ----------
    %(info_not_none)s
    data : ndarray | None
        If ``None``, data will be read from the Raw object. If ndarray, must be
        of shape (n_epochs, n_channels, n_times).
    %(events_epochs)s
    %(event_id)s
    %(epochs_tmin_tmax)s
    %(baseline_epochs)s
        Defaults to ``(None, 0)``, i.e. beginning of the the data until
        time point zero.
    %(raw_epochs)s
    %(picks_all)s
    %(reject_epochs)s
    %(flat)s
    %(decim)s
    %(epochs_reject_tmin_tmax)s
    %(detrend_epochs)s
    %(proj_epochs)s
    %(on_missing_epochs)s
    preload_at_end : bool
        %(epochs_preload)s
    %(selection)s

        .. versionadded:: 0.16
    %(drop_log)s
    filename : Path | None
        The filename (if the epochs are read from disk).
    %(metadata_epochs)s

        .. versionadded:: 0.16
    %(event_repeated_epochs)s
    %(raw_sfreq)s
    annotations : instance of mne.Annotations | None
        Annotations to set.
    %(verbose)s

    See Also
    --------
    Epochs
    EpochsArray
    make_fixed_length_epochs

    Notes
    -----
    The ``BaseEpochs`` class is public to allow for stable type-checking in
    user code (i.e., ``isinstance(my_epochs, BaseEpochs)``) but should not be
    used as a constructor for Epochs objects (use instead :class:`mne.Epochs`).
    """

    @verbose
    def __init__(
        self,
        info,
        data,
        events,
        event_id=None,
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        raw=None,
        picks=None,
        reject=None,
        flat=None,
        decim=1,
        reject_tmin=None,
        reject_tmax=None,
        detrend=None,
        proj=True,
        on_missing="raise",
        preload_at_end=False,
        selection=None,
        drop_log=None,
        filename=None,
        metadata=None,
        event_repeated="error",
        *,
        raw_sfreq=None,
        annotations=None,
        verbose=None,
    ):
        if events is not None:  # RtEpochs can have events=None
            events = _ensure_events(events)
            # Allow reading empty epochs (ToDo: Maybe not anymore in the future)
            if len(events) == 0:
                self._allow_empty = True
                selection = None
            else:
                self._allow_empty = False
                events_max = events.max()
                if events_max > INT32_MAX:
                    raise ValueError(
                        f"events array values must not exceed {INT32_MAX}, "
                        f"got {events_max}"
                    )
        event_id = _check_event_id(event_id, events)
        self.event_id = event_id
        del event_id

        if events is not None:  # RtEpochs can have events=None
            for key, val in self.event_id.items():
                if val not in events[:, 2]:
                    msg = f"No matching events found for {key} (event id {val})"
                    _on_missing(on_missing, msg)

            # ensure metadata matches original events size
            self.selection = np.arange(len(events))
            self.events = events

            # same as self.metadata = metadata, but suppress log in favor
            # of logging below (after setting self.selection)
            GetEpochsMixin.metadata.fset(self, metadata, verbose=False)
            del events

            values = list(self.event_id.values())
            selected = np.where(np.isin(self.events[:, 2], values))[0]
            if selection is None:
                selection = selected
            else:
                selection = np.array(selection, int)
            if selection.shape != (len(selected),):
                raise ValueError(
                    f"selection must be shape {selected.shape} got shape "
                    f"{selection.shape}"
                )
            self.selection = selection
            if drop_log is None:
                self.drop_log = tuple(
                    () if k in self.selection else ("IGNORED",)
                    for k in range(max(len(self.events), max(self.selection) + 1))
                )
            else:
                self.drop_log = drop_log

            self.events = self.events[selected]

            (
                self.events,
                self.event_id,
                self.selection,
                self.drop_log,
            ) = _handle_event_repeated(
                self.events,
                self.event_id,
                event_repeated,
                self.selection,
                self.drop_log,
            )

            # then subselect
            sub = np.where(np.isin(selection, self.selection))[0]
            if isinstance(metadata, list):
                metadata = [metadata[s] for s in sub]
            elif metadata is not None:
                metadata = metadata.iloc[sub]

            # Remove temporarily set metadata from above, and set
            # again to get the correct log ("adding metadata", instead of
            # "replacing existing metadata")
            GetEpochsMixin.metadata.fset(self, None, verbose=False)
            self.metadata = metadata
            del metadata

            n_events = len(self.events)
            if n_events > 1:
                if np.diff(self.events.astype(np.int64)[:, 0]).min() <= 0:
                    warn(
                        "The events passed to the Epochs constructor are not "
                        "chronologically ordered.",
                        RuntimeWarning,
                    )

            if n_events > 0:
                logger.info(f"{n_events} matching events found")
            else:
                # Allow reading empty epochs (ToDo: Maybe not anymore in the future)
                if not self._allow_empty:
                    raise ValueError("No desired events found.")
        else:
            self.drop_log = tuple()
            self.selection = np.array([], int)
            self.metadata = metadata
            # do not set self.events here, let subclass do it

        if (detrend not in [None, 0, 1]) or isinstance(detrend, bool):
            raise ValueError("detrend must be None, 0, or 1")
        self.detrend = detrend

        self._raw = raw
        info._check_consistency()
        self.picks = _picks_to_idx(
            info, picks, none="all", exclude=(), allow_empty=False
        )
        self.info = pick_info(info, self.picks)
        del info
        self._current = 0

        if data is None:
            self.preload = False
            self._data = None
            self._do_baseline = True
        else:
            assert decim == 1
            if (
                data.ndim != 3
                or data.shape[2] != round((tmax - tmin) * self.info["sfreq"]) + 1
            ):
                raise RuntimeError("bad data shape")
            if data.shape[0] != len(self.events):
                raise ValueError(
                    "The number of epochs and the number of events must match"
                )
            self.preload = True
            self._data = data
            self._do_baseline = False
        self._offset = None

        if tmin > tmax:
            raise ValueError("tmin has to be less than or equal to tmax")

        # Handle times
        sfreq = float(self.info["sfreq"])
        start_idx = int(round(tmin * sfreq))
        self._raw_times = np.arange(start_idx, int(round(tmax * sfreq)) + 1) / sfreq
        self._set_times(self._raw_times)

        # check reject_tmin and reject_tmax
        if reject_tmin is not None:
            if np.isclose(reject_tmin, tmin):
                # adjust for potential small deviations due to sampling freq
                reject_tmin = self.tmin
            elif reject_tmin < tmin:
                raise ValueError(
                    f"reject_tmin needs to be None or >= tmin (got {reject_tmin})"
                )

        if reject_tmax is not None:
            if np.isclose(reject_tmax, tmax):
                # adjust for potential small deviations due to sampling freq
                reject_tmax = self.tmax
            elif reject_tmax > tmax:
                raise ValueError(
                    f"reject_tmax needs to be None or <= tmax (got {reject_tmax})"
                )

        if (reject_tmin is not None) and (reject_tmax is not None):
            if reject_tmin >= reject_tmax:
                raise ValueError(
                    f"reject_tmin ({reject_tmin}) needs to be "
                    f" < reject_tmax ({reject_tmax})"
                )

        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax

        # decimation
        self._decim = 1
        self.decimate(decim)

        # baseline correction: replace `None` tuple elements  with actual times
        self.baseline = _check_baseline(
            baseline, times=self.times, sfreq=self.info["sfreq"]
        )
        if self.baseline is not None and self.baseline != baseline:
            logger.info(
                f"Setting baseline interval to "
                f"[{self.baseline[0]}, {self.baseline[1]}] s"
            )

        logger.info(_log_rescale(self.baseline))

        # setup epoch rejection
        self.reject = None
        self.flat = None
        self._reject_setup(reject, flat)

        # do the rest
        valid_proj = [True, "delayed", False]
        if proj not in valid_proj:
            raise ValueError(f'"proj" must be one of {valid_proj}, not {proj}')
        if proj == "delayed":
            self._do_delayed_proj = True
            logger.info("Entering delayed SSP mode.")
        else:
            self._do_delayed_proj = False
        activate = False if self._do_delayed_proj else proj
        self._projector, self.info = setup_proj(self.info, False, activate=activate)
        if preload_at_end:
            assert self._data is None
            assert self.preload is False
            self.load_data()  # this will do the projection
        elif proj is True and self._projector is not None and data is not None:
            # let's make sure we project if data was provided and proj
            # requested
            # we could do this with np.einsum, but iteration should be
            # more memory safe in most instances
            for ii, epoch in enumerate(self._data):
                self._data[ii] = np.dot(self._projector, epoch)
        self.filename = filename if filename is not None else filename
        if raw_sfreq is None:
            raw_sfreq = self.info["sfreq"]
        self._raw_sfreq = raw_sfreq
        self._check_consistency()
        self.set_annotations(annotations, on_missing="ignore")

    def _check_consistency(self):
        """Check invariants of epochs object."""
        if hasattr(self, "events"):
            assert len(self.selection) == len(self.events)
            assert len(self.drop_log) >= len(self.events)
        assert len(self.selection) == sum(len(dl) == 0 for dl in self.drop_log)
        assert hasattr(self, "_times_readonly")
        assert not self.times.flags["WRITEABLE"]
        assert isinstance(self.drop_log, tuple)
        assert all(isinstance(log, tuple) for log in self.drop_log)
        assert all(isinstance(s, str) for log in self.drop_log for s in log)

    def reset_drop_log_selection(self):
        """Reset the drop_log and selection entries.

        This method will simplify ``self.drop_log`` and ``self.selection``
        so that they are meaningless (tuple of empty tuples and increasing
        integers, respectively). This can be useful when concatenating
        many Epochs instances, as ``drop_log`` can accumulate many entries
        which can become problematic when saving.
        """
        self.selection = np.arange(len(self.events))
        self.drop_log = (tuple(),) * len(self.events)
        self._check_consistency()

    def load_data(self):
        """Load the data if not already preloaded.

        Returns
        -------
        epochs : instance of Epochs
            The epochs object.

        Notes
        -----
        This function operates in-place.

        .. versionadded:: 0.10.0
        """
        if self.preload:
            return self
        self._data = self._get_data()
        self.preload = True
        self._do_baseline = False
        self._decim_slice = slice(None, None, None)
        self._decim = 1
        self._raw_times = self.times
        assert self._data.shape[-1] == len(self.times)
        self._raw = None  # shouldn't need it anymore
        return self

    @verbose
    def apply_baseline(self, baseline=(None, 0), *, verbose=None):
        """Baseline correct epochs.

        Parameters
        ----------
        %(baseline_epochs)s
            Defaults to ``(None, 0)``, i.e. beginning of the the data until
            time point zero.
        %(verbose)s

        Returns
        -------
        epochs : instance of Epochs
            The baseline-corrected Epochs object.

        Notes
        -----
        Baseline correction can be done multiple times, but can never be
        reverted once the data has been loaded.

        .. versionadded:: 0.10.0
        """
        baseline = _check_baseline(baseline, times=self.times, sfreq=self.info["sfreq"])

        if self.preload:
            if self.baseline is not None and baseline is None:
                raise RuntimeError(
                    "You cannot remove baseline correction "
                    "from preloaded data once it has been "
                    "applied."
                )
            self._do_baseline = True
            picks = self._detrend_picks
            rescale(self._data, self.times, baseline, copy=False, picks=picks)
            self._do_baseline = False
        else:  # logging happens in "rescale" in "if" branch
            logger.info(_log_rescale(baseline))
            # For EpochsArray and Epochs, this is already True:
            # assert self._do_baseline is True
            # ... but for EpochsFIF it's not, so let's set it explicitly
            self._do_baseline = True
        self.baseline = baseline
        return self

    def _reject_setup(self, reject, flat, *, allow_callable=False):
        """Set self._reject_time and self._channel_type_idx."""
        idx = channel_indices_by_type(self.info)
        reject = deepcopy(reject) if reject is not None else dict()
        flat = deepcopy(flat) if flat is not None else dict()
        for rej, kind in zip((reject, flat), ("reject", "flat")):
            _validate_type(rej, dict, kind)
            bads = set(rej.keys()) - set(idx.keys())
            if len(bads) > 0:
                raise KeyError(f"Unknown channel types found in {kind}: {bads}")

        for key in idx.keys():
            # don't throw an error if rejection/flat would do nothing
            if len(idx[key]) == 0 and (
                np.isfinite(reject.get(key, np.inf)) or flat.get(key, -1) >= 0
            ):
                # This is where we could eventually add e.g.
                # self.allow_missing_reject_keys check to allow users to
                # provide keys that don't exist in data
                raise ValueError(
                    f"No {key.upper()} channel found. Cannot reject based on "
                    f"{key.upper()}."
                )

            # check for invalid values
            for rej, kind in zip((reject, flat), ("Rejection", "Flat")):
                for key, val in rej.items():
                    name = f"{kind} dict value for {key}"
                    if callable(val) and allow_callable:
                        continue
                    extra_str = ""
                    if allow_callable:
                        extra_str = "or callable"
                    _validate_type(val, "numeric", name, extra=extra_str)
                    if val is None or val < 0:
                        raise ValueError(
                            f"If using numerical {name} criteria, the value "
                            f"must be >= 0, not {repr(val)}"
                        )

        # now check to see if our rejection and flat are getting more
        # restrictive
        old_reject = self.reject if self.reject is not None else dict()
        old_flat = self.flat if self.flat is not None else dict()
        bad_msg = (
            '{kind}["{key}"] == {new} {op} {old} (old value), new '
            "{kind} values must be at least as stringent as "
            "previous ones"
        )

        # copy thresholds for channel types that were used previously, but not
        # passed this time
        for key in set(old_reject) - set(reject):
            reject[key] = old_reject[key]
        # make sure new thresholds are at least as stringent as the old ones
        for key in reject:
            # Skip this check if old_reject and reject are callables
            if callable(reject[key]) and allow_callable:
                continue
            if key in old_reject and reject[key] > old_reject[key]:
                raise ValueError(
                    bad_msg.format(
                        kind="reject",
                        key=key,
                        new=reject[key],
                        old=old_reject[key],
                        op=">",
                    )
                )

        # same for flat thresholds
        for key in set(old_flat) - set(flat):
            flat[key] = old_flat[key]
        for key in flat:
            if callable(flat[key]) and allow_callable:
                continue
            if key in old_flat and flat[key] < old_flat[key]:
                raise ValueError(
                    bad_msg.format(
                        kind="flat", key=key, new=flat[key], old=old_flat[key], op="<"
                    )
                )

        # after validation, set parameters
        self._bad_dropped = False
        self._channel_type_idx = idx
        self.reject = reject if len(reject) > 0 else None
        self.flat = flat if len(flat) > 0 else None

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

    @verbose  # verbose is used by mne-realtime
    def _is_good_epoch(self, data, verbose=None):
        """Determine if epoch is good."""
        if isinstance(data, str):
            return False, (data,)
        if data is None:
            return False, ("NO_DATA",)
        n_times = len(self.times)
        if data.shape[1] < n_times:
            # epoch is too short ie at the end of the data
            return False, ("TOO_SHORT",)
        if self.reject is None and self.flat is None:
            return True, None
        else:
            if self._reject_time is not None:
                data = data[:, self._reject_time]

            return _is_good(
                data,
                self.ch_names,
                self._channel_type_idx,
                self.reject,
                self.flat,
                full_report=True,
                ignore_chs=self.info["bads"],
            )

    @verbose
    def _detrend_offset_decim(self, epoch, picks, verbose=None):
        """Aux Function: detrend, baseline correct, offset, decim.

        Note: operates inplace
        """
        if (epoch is None) or isinstance(epoch, str):
            return epoch

        # Detrend
        if self.detrend is not None:
            # We explicitly detrend just data channels (not EMG, ECG, EOG which
            # are processed by baseline correction)
            use_picks = _pick_data_channels(self.info, exclude=())
            epoch[use_picks] = detrend(epoch[use_picks], self.detrend, axis=1)

        # Baseline correct
        if self._do_baseline:
            rescale(
                epoch,
                self._raw_times,
                self.baseline,
                picks=picks,
                copy=False,
                verbose=False,
            )

        # Decimate if necessary (i.e., epoch not preloaded)
        epoch = epoch[:, self._decim_slice]

        # handle offset
        if self._offset is not None:
            epoch += self._offset

        return epoch

    def iter_evoked(self, copy=False):
        """Iterate over epochs as a sequence of Evoked objects.

        The Evoked objects yielded will each contain a single epoch (i.e., no
        averaging is performed).

        This method resets the object iteration state to the first epoch.

        Parameters
        ----------
        copy : bool
            If False copies of data and measurement info will be omitted
            to save time.
        """
        self.__iter__()

        while True:
            try:
                out = self.__next__(True)
            except StopIteration:
                break
            data, event_id = out
            tmin = self.times[0]
            info = self.info
            if copy:
                info = deepcopy(self.info)
                data = data.copy()

            yield EvokedArray(data, info, tmin, comment=str(event_id))

    def subtract_evoked(self, evoked=None):
        """Subtract an evoked response from each epoch.

        Can be used to exclude the evoked response when analyzing induced
        activity, see e.g. [1]_.

        Parameters
        ----------
        evoked : instance of Evoked | None
            The evoked response to subtract. If None, the evoked response
            is computed from Epochs itself.

        Returns
        -------
        self : instance of Epochs
            The modified instance (instance is also modified inplace).

        References
        ----------
        .. [1] David et al. "Mechanisms of evoked and induced responses in
               MEG/EEG", NeuroImage, vol. 31, no. 4, pp. 1580-1591, July 2006.
        """
        logger.info("Subtracting Evoked from Epochs")
        if evoked is None:
            picks = _pick_data_channels(self.info, exclude=[])
            evoked = self.average(picks)

        # find the indices of the channels to use
        picks = pick_channels(evoked.ch_names, include=self.ch_names, ordered=False)

        # make sure the omitted channels are not data channels
        if len(picks) < len(self.ch_names):
            sel_ch = [evoked.ch_names[ii] for ii in picks]
            diff_ch = list(set(self.ch_names).difference(sel_ch))
            diff_idx = [self.ch_names.index(ch) for ch in diff_ch]
            diff_types = [channel_type(self.info, idx) for idx in diff_idx]
            bad_idx = [
                diff_types.index(t) for t in diff_types if t in _DATA_CH_TYPES_SPLIT
            ]
            if len(bad_idx) > 0:
                bad_str = ", ".join([diff_ch[ii] for ii in bad_idx])
                raise ValueError(
                    "The following data channels are missing "
                    f"in the evoked response: {bad_str}"
                )
            logger.info(
                "    The following channels are not included in the subtraction: "
                + ", ".join(diff_ch)
            )

        # make sure the times match
        if (
            len(self.times) != len(evoked.times)
            or np.max(np.abs(self.times - evoked.times)) >= 1e-7
        ):
            raise ValueError(
                "Epochs and Evoked object do not contain the same time points."
            )

        # handle SSPs
        if not self.proj and evoked.proj:
            warn("Evoked has SSP applied while Epochs has not.")
        if self.proj and not evoked.proj:
            evoked = evoked.copy().apply_proj()

        # find the indices of the channels to use in Epochs
        ep_picks = [self.ch_names.index(evoked.ch_names[ii]) for ii in picks]

        # do the subtraction
        if self.preload:
            self._data[:, ep_picks, :] -= evoked.data[picks][None, :, :]
        else:
            if self._offset is None:
                self._offset = np.zeros(
                    (len(self.ch_names), len(self.times)), dtype=np.float64
                )
            self._offset[ep_picks] -= evoked.data[picks]
        logger.info("[done]")

        return self

    @fill_doc
    def average(self, picks=None, method="mean", by_event_type=False):
        """Compute an average over epochs.

        Parameters
        ----------
        %(picks_all_data)s
        method : str | callable
            How to combine the data. If "mean"/"median", the mean/median
            are returned.
            Otherwise, must be a callable which, when passed an array of shape
            (n_epochs, n_channels, n_time) returns an array of shape
            (n_channels, n_time).
            Note that due to file type limitations, the kind for all
            these will be "average".
        %(by_event_type)s

        Returns
        -------
        %(evoked_by_event_type_returns)s

        Notes
        -----
        Computes an average of all epochs in the instance, even if
        they correspond to different conditions. To average by condition,
        do ``epochs[condition].average()`` for each condition separately.

        When picks is None and epochs contain only ICA channels, no channels
        are selected, resulting in an error. This is because ICA channels
        are not considered data channels (they are of misc type) and only data
        channels are selected when picks is None.

        The ``method`` parameter allows e.g. robust averaging.
        For example, one could do:

            >>> from scipy.stats import trim_mean  # doctest:+SKIP
            >>> trim = lambda x: trim_mean(x, 0.1, axis=0)  # doctest:+SKIP
            >>> epochs.average(method=trim)  # doctest:+SKIP

        This would compute the trimmed mean.
        """
        self._handle_empty("raise", "average")
        if by_event_type:
            evokeds = list()
            for event_type in self.event_id.keys():
                ev = self[event_type]._compute_aggregate(picks=picks, mode=method)
                ev.comment = event_type
                evokeds.append(ev)
        else:
            evokeds = self._compute_aggregate(picks=picks, mode=method)
        return evokeds

    @fill_doc
    def standard_error(self, picks=None, by_event_type=False):
        """Compute standard error over epochs.

        Parameters
        ----------
        %(picks_all_data)s
        %(by_event_type)s

        Returns
        -------
        %(std_err_by_event_type_returns)s
        """
        return self.average(picks=picks, method="std", by_event_type=by_event_type)

    def _compute_aggregate(self, picks, mode="mean"):
        """Compute the mean, median, or std over epochs and return Evoked."""
        # if instance contains ICA channels they won't be included unless picks
        # is specified
        if picks is None:
            check_ICA = [x.startswith("ICA") for x in self.ch_names]
            if np.all(check_ICA):
                raise TypeError(
                    "picks must be specified (i.e. not None) for ICA channel data"
                )
            elif np.any(check_ICA):
                warn(
                    "ICA channels will not be included unless explicitly "
                    "selected in picks"
                )

        n_channels = len(self.ch_names)
        n_times = len(self.times)

        if self.preload:
            n_events = len(self.events)
            fun = _check_combine(mode, valid=("mean", "median", "std"))
            data = fun(self._data)
            assert len(self.events) == len(self._data)
            if data.shape != self._data.shape[1:]:
                raise RuntimeError(
                    f"You passed a function that resulted n data of shape "
                    f"{data.shape}, but it should be {self._data.shape[1:]}."
                )
        else:
            if mode not in {"mean", "std"}:
                raise ValueError(
                    "If data are not preloaded, can only compute "
                    "mean or standard deviation."
                )
            data = np.zeros((n_channels, n_times))
            n_events = 0
            for e in self:
                if np.iscomplexobj(e):
                    data = data.astype(np.complex128)
                data += e
                n_events += 1

            if n_events > 0:
                data /= n_events
            else:
                data.fill(np.nan)

            # convert to stderr if requested, could do in one pass but do in
            # two (slower) in case there are large numbers
            if mode == "std":
                data_mean = data.copy()
                data.fill(0.0)
                for e in self:
                    data += (e - data_mean) ** 2
                data = np.sqrt(data / n_events)

        if mode == "std":
            kind = "standard_error"
            data /= np.sqrt(n_events)
        else:
            kind = "average"

        return self._evoked_from_epoch_data(
            data, self.info, picks, n_events, kind, self._name
        )

    @property
    def _name(self):
        """Give a nice string representation based on event ids."""
        return self._get_name()

    def _get_name(self, count="frac", ms="×", sep="+"):
        """Generate human-readable name for epochs and evokeds from event_id.

        Parameters
        ----------
        count : 'frac' | 'total'
            Whether to include the fraction or total number of epochs that each
            event type contributes to the number of all epochs.
            Ignored if only one event type is present.
        ms : str | None
            The multiplication sign to use. Pass ``None`` to omit the sign.
            Ignored if only one event type is present.
        sep : str
            How to separate the different events names. Ignored if only one
            event type is present.
        """
        _check_option("count", value=count, allowed_values=["frac", "total"])

        if len(self.event_id) == 1:
            comment = next(iter(self.event_id.keys()))
        else:
            counter = Counter(self.events[:, 2])
            comments = list()

            # Take care of padding
            if ms is None:
                ms = " "
            else:
                ms = f" {ms} "

            for event_name, event_code in self.event_id.items():
                if count == "frac":
                    frac = float(counter[event_code]) / len(self.events)
                    comment = f"{frac:.2f}{ms}{event_name}"
                else:  # 'total'
                    comment = f"{counter[event_code]}{ms}{event_name}"
                comments.append(comment)

            comment = f" {sep} ".join(comments)
        return comment

    def _evoked_from_epoch_data(self, data, info, picks, n_events, kind, comment):
        """Create an evoked object from epoch data."""
        info = deepcopy(info)
        # don't apply baseline correction; we'll set evoked.baseline manually
        evoked = EvokedArray(
            data,
            info,
            tmin=self.times[0],
            comment=comment,
            nave=n_events,
            kind=kind,
            baseline=None,
        )
        evoked.baseline = self.baseline

        # the above constructor doesn't recreate the times object precisely
        # due to numerical precision issues
        evoked._set_times(self.times.copy())

        # pick channels
        picks = _picks_to_idx(self.info, picks, "data_or_ica", ())
        ch_names = [evoked.ch_names[p] for p in picks]
        evoked.pick(ch_names)

        if len(evoked.info["ch_names"]) == 0:
            raise ValueError("No data channel found when averaging.")

        if evoked.nave < 1:
            warn("evoked object is empty (based on less than 1 epoch)")

        return evoked

    @property
    def ch_names(self):
        """Channel names."""
        return self.info["ch_names"]

    @copy_function_doc_to_method_doc(plot_epochs)
    def plot(
        self,
        picks=None,
        scalings=None,
        n_epochs=20,
        n_channels=20,
        title=None,
        events=False,
        event_color=None,
        order=None,
        show=True,
        block=False,
        decim="auto",
        noise_cov=None,
        butterfly=False,
        show_scrollbars=True,
        show_scalebars=True,
        epoch_colors=None,
        event_id=None,
        group_by="type",
        precompute=None,
        use_opengl=None,
        *,
        theme=None,
        overview_mode=None,
        splash=True,
    ):
        return plot_epochs(
            self,
            picks=picks,
            scalings=scalings,
            n_epochs=n_epochs,
            n_channels=n_channels,
            title=title,
            events=events,
            event_color=event_color,
            order=order,
            show=show,
            block=block,
            decim=decim,
            noise_cov=noise_cov,
            butterfly=butterfly,
            show_scrollbars=show_scrollbars,
            show_scalebars=show_scalebars,
            epoch_colors=epoch_colors,
            event_id=event_id,
            group_by=group_by,
            precompute=precompute,
            use_opengl=use_opengl,
            theme=theme,
            overview_mode=overview_mode,
            splash=splash,
        )

    @copy_function_doc_to_method_doc(plot_topo_image_epochs)
    def plot_topo_image(
        self,
        layout=None,
        sigma=0.0,
        vmin=None,
        vmax=None,
        colorbar=None,
        order=None,
        cmap="RdBu_r",
        layout_scale=0.95,
        title=None,
        scalings=None,
        border="none",
        fig_facecolor="k",
        fig_background=None,
        font_color="w",
        select=False,
        show=True,
    ):
        return plot_topo_image_epochs(
            self,
            layout=layout,
            sigma=sigma,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            order=order,
            cmap=cmap,
            layout_scale=layout_scale,
            title=title,
            scalings=scalings,
            border=border,
            fig_facecolor=fig_facecolor,
            fig_background=fig_background,
            font_color=font_color,
            select=select,
            show=show,
        )

    @verbose
    def drop_bad(self, reject="existing", flat="existing", verbose=None):
        """Drop bad epochs without retaining the epochs data.

        Should be used before slicing operations.

        .. warning:: This operation is slow since all epochs have to be read
                     from disk. To avoid reading epochs from disk multiple
                     times, use :meth:`mne.Epochs.load_data()`.

        .. note:: To constrain the time period used for estimation of signal
                  quality, set ``epochs.reject_tmin`` and
                  ``epochs.reject_tmax``, respectively.

        Parameters
        ----------
        %(reject_drop_bad)s
        %(flat_drop_bad)s
        %(verbose)s

        Returns
        -------
        epochs : instance of Epochs
            The epochs with bad epochs dropped. Operates in-place.

        Notes
        -----
        Dropping bad epochs can be done multiple times with different
        ``reject`` and ``flat`` parameters. However, once an epoch is
        dropped, it is dropped forever, so if more lenient thresholds may
        subsequently be applied, :meth:`epochs.copy <mne.Epochs.copy>` should be
        used.
        """
        if reject == "existing":
            if flat == "existing" and self._bad_dropped:
                return
            reject = self.reject
        if flat == "existing":
            flat = self.flat
        if any(isinstance(rej, str) and rej != "existing" for rej in (reject, flat)):
            raise ValueError('reject and flat, if strings, must be "existing"')
        self._reject_setup(reject, flat, allow_callable=True)
        self._get_data(out=False, verbose=verbose)
        return self

    def drop_log_stats(self, ignore=("IGNORED",)):
        """Compute the channel stats based on a drop_log from Epochs.

        Parameters
        ----------
        ignore : list
            The drop reasons to ignore.

        Returns
        -------
        perc : float
            Total percentage of epochs dropped.

        See Also
        --------
        plot_drop_log
        """
        return _drop_log_stats(self.drop_log, ignore)

    @copy_function_doc_to_method_doc(plot_drop_log)
    def plot_drop_log(
        self,
        threshold=0,
        n_max_plot=20,
        subject=None,
        color=(0.9, 0.9, 0.9),
        width=0.8,
        ignore=("IGNORED",),
        show=True,
    ):
        if not self._bad_dropped:
            raise ValueError(
                "You cannot use plot_drop_log since bad "
                "epochs have not yet been dropped. "
                "Use epochs.drop_bad()."
            )
        return plot_drop_log(
            self.drop_log,
            threshold,
            n_max_plot,
            subject,
            color=color,
            width=width,
            ignore=ignore,
            show=show,
        )

    @copy_function_doc_to_method_doc(plot_epochs_image)
    def plot_image(
        self,
        picks=None,
        sigma=0.0,
        vmin=None,
        vmax=None,
        colorbar=True,
        order=None,
        show=True,
        units=None,
        scalings=None,
        cmap=None,
        fig=None,
        axes=None,
        overlay_times=None,
        combine=None,
        group_by=None,
        evoked=True,
        ts_args=None,
        title=None,
        clear=False,
    ):
        return plot_epochs_image(
            self,
            picks=picks,
            sigma=sigma,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            order=order,
            show=show,
            units=units,
            scalings=scalings,
            cmap=cmap,
            fig=fig,
            axes=axes,
            overlay_times=overlay_times,
            combine=combine,
            group_by=group_by,
            evoked=evoked,
            ts_args=ts_args,
            title=title,
            clear=clear,
        )

    @verbose
    def drop(self, indices, reason="USER", verbose=None):
        """Drop epochs based on indices or boolean mask.

        .. note:: The indices refer to the current set of undropped epochs
                  rather than the complete set of dropped and undropped epochs.
                  They are therefore not necessarily consistent with any
                  external indices (e.g., behavioral logs). To drop epochs
                  based on external criteria, do not use the ``preload=True``
                  flag when constructing an Epochs object, and call this
                  method before calling the :meth:`mne.Epochs.drop_bad` or
                  :meth:`mne.Epochs.load_data` methods.

        Parameters
        ----------
        indices : array of int or bool
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are
            correspondingly modified.
        reason : list | tuple | str
            Reason(s) for dropping the epochs ('ECG', 'timeout', 'blink' etc).
            Reason(s) are applied to all indices specified.
            Default: 'USER'.
        %(verbose)s

        Returns
        -------
        epochs : instance of Epochs
            The epochs with indices dropped. Operates in-place.
        """
        indices = np.atleast_1d(indices)

        if indices.ndim > 1:
            raise TypeError("indices must be a scalar or a 1-d array")
        # Check if indices and reasons are of the same length
        # if using collection to drop epochs

        if indices.dtype == np.dtype(bool):
            indices = np.where(indices)[0]
        try_idx = np.where(indices < 0, indices + len(self.events), indices)

        out_of_bounds = (try_idx < 0) | (try_idx >= len(self.events))
        if out_of_bounds.any():
            first = indices[out_of_bounds][0]
            raise IndexError(f"Epoch index {first} is out of bounds")
        keep = np.setdiff1d(np.arange(len(self.events)), try_idx)
        self._getitem(keep, reason, copy=False, drop_event_id=False)
        count = len(try_idx)
        logger.info(
            "Dropped %d epoch%s: %s",
            count,
            _pl(count),
            ", ".join(map(str, np.sort(try_idx))),
        )

        return self

    def _get_epoch_from_raw(self, idx, verbose=None):
        """Get a given epoch from disk."""
        raise NotImplementedError

    def _project_epoch(self, epoch):
        """Process a raw epoch based on the delayed param."""
        # whenever requested, the first epoch is being projected.
        if (epoch is None) or isinstance(epoch, str):
            # can happen if t < 0 or reject based on annotations
            return epoch
        proj = self._do_delayed_proj or self.proj
        if self._projector is not None and proj is True:
            epoch = np.dot(self._projector, epoch)
        return epoch

    def _handle_empty(self, on_empty, meth):
        if len(self.events) == 0:
            msg = (
                f"epochs.{meth}() can't run because this Epochs-object is empty. "
                f"You might want to check Epochs.drop_log or Epochs.plot_drop_log()"
                f" to see why epochs were dropped."
            )
            _on_missing(on_empty, msg, error_klass=RuntimeError)

    @verbose
    def _get_data(
        self,
        out=True,
        picks=None,
        item=None,
        *,
        units=None,
        tmin=None,
        tmax=None,
        copy=False,
        on_empty="warn",
        verbose=None,
    ):
        """Load all data, dropping bad epochs along the way.

        Parameters
        ----------
        out : bool
            Return the data. Setting this to False is used to reject bad
            epochs without caching all the data, which saves memory.
        %(picks_all)s
        item : slice | array-like | str | list | None
            See docstring of get_data method.
        %(units)s
        tmin : int | float | None
            Start time of data to get in seconds.
        tmax : int | float | None
            End time of data to get in seconds.
        %(verbose)s
        """
        from .io.base import _get_ch_factors

        if copy is not None:
            _validate_type(copy, bool, "copy")

        # Handle empty epochs
        self._handle_empty(on_empty, "_get_data")
        # if called with 'out=False', the call came from 'drop_bad()'
        # if no reasons to drop, just declare epochs as good and return
        if not out:
            # make sure first and last epoch not out of bounds of raw
            in_bounds = self.preload or (
                self._get_epoch_from_raw(idx=0) is not None
                and self._get_epoch_from_raw(idx=-1) is not None
            )
            # might be BaseEpochs or Epochs, only the latter has the attribute
            reject_by_annotation = getattr(self, "reject_by_annotation", False)
            if (
                self.reject is None
                and self.flat is None
                and in_bounds
                and self._reject_time is None
                and not reject_by_annotation
            ):
                logger.debug("_get_data is a noop, returning")
                self._bad_dropped = True
                return None
        start, stop = self._handle_tmin_tmax(tmin, tmax)

        if item is None:
            item = slice(None)
        elif not self._bad_dropped:
            raise ValueError(
                "item must be None in epochs.get_data() unless bads have been "
                "dropped. Consider using epochs.drop_bad()."
            )
        select = self._item_to_select(item)  # indices or slice
        use_idx = np.arange(len(self.events))[select]
        n_events = len(use_idx)
        # in case there are no good events
        if self.preload:
            # we will store our result in our existing array
            data = self._data
        else:
            # we start out with an empty array, allocate only if necessary
            data = np.empty((0, len(self.info["ch_names"]), len(self.times)))
            msg = (
                f"for {n_events} events and {len(self._raw_times)} original time points"
            )
            if self._decim > 1:
                msg += " (prior to decimation)"
            if getattr(self._raw, "preload", False):
                logger.info(f"Using data from preloaded Raw {msg} ...")
            else:
                logger.info(f"Loading data {msg} ...")

        orig_picks = picks
        if orig_picks is None:
            picks = _picks_to_idx(self.info, picks, "all", exclude=())
        else:
            picks = _picks_to_idx(self.info, picks)

        # handle units param only if we are going to return data (out==True)
        if (units is not None) and out:
            ch_factors = _get_ch_factors(self, units, picks)
        else:
            ch_factors = None

        if self._bad_dropped:
            if not out:
                return
            if self.preload:
                return self._data_sel_copy_scale(
                    data,
                    select=select,
                    orig_picks=orig_picks,
                    picks=picks,
                    ch_factors=ch_factors,
                    start=start,
                    stop=stop,
                    copy=copy,
                )

            # we need to load from disk, drop, and return data
            detrend_picks = self._detrend_picks
            for ii, idx in enumerate(use_idx):
                # faster to pre-allocate memory here
                epoch_noproj = self._get_epoch_from_raw(idx)
                epoch_noproj = self._detrend_offset_decim(epoch_noproj, detrend_picks)
                if self._do_delayed_proj:
                    epoch_out = epoch_noproj
                else:
                    epoch_out = self._project_epoch(epoch_noproj)
                if ii == 0:
                    data = np.empty(
                        (n_events, len(self.ch_names), len(self.times)),
                        dtype=epoch_out.dtype,
                    )
                data[ii] = epoch_out
        else:
            # bads need to be dropped, this might occur after a preload
            # e.g., when calling drop_bad w/new params
            good_idx = []
            n_out = 0
            drop_log = list(self.drop_log)
            assert n_events == len(self.selection)
            if not self.preload:
                detrend_picks = self._detrend_picks
            for idx, sel in enumerate(self.selection):
                if self.preload:  # from memory
                    if self._do_delayed_proj:
                        epoch_noproj = self._data[idx]
                        epoch = self._project_epoch(epoch_noproj)
                    else:
                        epoch_noproj = None
                        epoch = self._data[idx]
                else:  # from disk
                    epoch_noproj = self._get_epoch_from_raw(idx)
                    epoch_noproj = self._detrend_offset_decim(
                        epoch_noproj, detrend_picks
                    )
                    epoch = self._project_epoch(epoch_noproj)

                epoch_out = epoch_noproj if self._do_delayed_proj else epoch
                is_good, bad_tuple = self._is_good_epoch(epoch, verbose=verbose)
                if not is_good:
                    assert isinstance(bad_tuple, tuple)
                    assert all(isinstance(x, str) for x in bad_tuple)
                    drop_log[sel] = drop_log[sel] + bad_tuple
                    continue
                good_idx.append(idx)

                # store the epoch if there is a reason to (output or update)
                if out or self.preload:
                    # faster to pre-allocate, then trim as necessary
                    if n_out == 0 and not self.preload:
                        data = np.empty(
                            (n_events, epoch_out.shape[0], epoch_out.shape[1]),
                            dtype=epoch_out.dtype,
                            order="C",
                        )
                    data[n_out] = epoch_out
                    n_out += 1
            self.drop_log = tuple(drop_log)
            del drop_log

            self._bad_dropped = True
            n_bads_dropped = n_events - len(good_idx)
            logger.info(f"{n_bads_dropped} bad epochs dropped")

            if n_bads_dropped == n_events:
                warn(
                    "All epochs were dropped!\n"
                    "You might need to alter reject/flat-criteria "
                    "or drop bad channels to avoid this. "
                    "You can use Epochs.plot_drop_log() to see which "
                    "channels are responsible for the dropping of epochs."
                )

            # adjust the data size if there is a reason to (output or update)
            if out or self.preload:
                if data.flags["OWNDATA"] and data.flags["C_CONTIGUOUS"]:
                    data.resize((n_out,) + data.shape[1:], refcheck=False)
                else:
                    data = data[:n_out]
                    if self.preload:
                        self._data = data

            # Now update our properties (excepd data, which is already fixed)
            self._getitem(
                good_idx, None, copy=False, drop_event_id=False, select_data=False
            )

        if not out:
            return
        return self._data_sel_copy_scale(
            data,
            select=slice(None),
            orig_picks=orig_picks,
            picks=picks,
            ch_factors=ch_factors,
            start=start,
            stop=stop,
            copy=copy,
        )

    def _data_sel_copy_scale(
        self, data, *, select, orig_picks, picks, ch_factors, start, stop, copy
    ):
        # data arg starts out as self._data when data is preloaded
        data_is_self_data = bool(self.preload)
        logger.debug(f"Data is self data: {data_is_self_data}")
        # only two types of epoch subselection allowed
        assert isinstance(select, slice | np.ndarray), type(select)
        if not isinstance(select, slice):
            logger.debug("  Copying, fancy indexed epochs")
            data_is_self_data = False  # copy (fancy indexing)
        elif select != slice(None):
            logger.debug("  Slicing epochs")
        if orig_picks is not None:
            logger.debug("  Copying, fancy indexed picks")
            assert isinstance(picks, np.ndarray), type(picks)
            data_is_self_data = False  # copy (fancy indexing)
        else:
            picks = slice(None)
        if not all(isinstance(x, slice) and x == slice(None) for x in (select, picks)):
            data = data[select][:, picks]
        del picks
        if start != 0 or stop != self.times.size:
            logger.debug("  Slicing time")
            data = data[..., start:stop]  # view (slice)
        if ch_factors is not None:
            if data_is_self_data:
                logger.debug("  Copying, scale factors applied")
                data = data.copy()
                data_is_self_data = False
            data *= ch_factors[:, np.newaxis]
        if not data_is_self_data:
            return data
        if copy:
            logger.debug("  Copying, copy=True")
            data = data.copy()
        return data

    @property
    def _detrend_picks(self):
        if self._do_baseline:
            return _pick_data_channels(
                self.info, with_ref_meg=True, with_aux=True, exclude=()
            )
        else:
            return []

    @verbose
    def get_data(
        self,
        picks=None,
        item=None,
        units=None,
        tmin=None,
        tmax=None,
        *,
        copy=True,
        verbose=None,
    ):
        """Get all epochs as a 3D array.

        Parameters
        ----------
        %(picks_all)s
        item : slice | array-like | str | list | None
            The items to get. See :meth:`mne.Epochs.__getitem__` for
            a description of valid options. This can be substantially faster
            for obtaining an ndarray than :meth:`~mne.Epochs.__getitem__`
            for repeated access on large Epochs objects.
            None (default) is an alias for ``slice(None)``.

            .. versionadded:: 0.20
        %(units)s

            .. versionadded:: 0.24
        tmin : int | float | None
            Start time of data to get in seconds.

            .. versionadded:: 0.24.0
        tmax : int | float | None
            End time of data to get in seconds.

            .. versionadded:: 0.24.0
        copy : bool
            Whether to return a copy of the object's data, or (if possible) a view.
            See :ref:`the NumPy docs <numpy:basics.copies-and-views>` for an
            explanation. Default is ``False`` in 1.6 but will change to ``True`` in 1.7,
            set it explicitly to avoid a warning in some cases. A view is only possible
            when ``item is None``, ``picks is None``, ``units is None``, and data are
            preloaded.

            .. warning::
               Using ``copy=False`` and then modifying the returned ``data`` will in
               turn modify the Epochs object. Use with caution!

            .. versionchanged:: 1.7
               The default changed from ``False`` to ``True``.

            .. versionadded:: 1.6
        %(verbose)s

        Returns
        -------
        data : array of shape (n_epochs, n_channels, n_times)
            The epochs data. Will be a copy when ``copy=True`` and will be a view
            when possible when ``copy=False``.
        """
        return self._get_data(
            picks=picks, item=item, units=units, tmin=tmin, tmax=tmax, copy=copy
        )

    @verbose
    def apply_function(
        self,
        fun,
        picks=None,
        dtype=None,
        n_jobs=None,
        channel_wise=True,
        verbose=None,
        **kwargs,
    ):
        """Apply a function to a subset of channels.

        %(applyfun_summary_epochs)s

        Parameters
        ----------
        %(fun_applyfun)s
        %(picks_all_data_noref)s
        %(dtype_applyfun)s
        %(n_jobs)s Ignored if ``channel_wise=False`` as the workload
            is split across channels.
        %(channel_wise_applyfun_epo)s
        %(verbose)s
        %(kwargs_fun)s

        Returns
        -------
        self : instance of Epochs
            The epochs object with transformed data.
        """
        _check_preload(self, "epochs.apply_function")
        picks = _picks_to_idx(self.info, picks, exclude=(), with_ref_meg=False)

        if not callable(fun):
            raise ValueError("fun needs to be a function")

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        args = getfullargspec(fun).args + getfullargspec(fun).kwonlyargs
        if channel_wise is False:
            if ("ch_idx" in args) or ("ch_name" in args):
                raise ValueError(
                    "apply_function cannot access ch_idx or ch_name "
                    "when channel_wise=False"
                )
        if "ch_idx" in args:
            logger.info("apply_function requested to access ch_idx")
        if "ch_name" in args:
            logger.info("apply_function requested to access ch_name")

        if channel_wise:
            parallel, p_fun, n_jobs = parallel_func(_check_fun, n_jobs)
            if n_jobs == 1:
                _fun = partial(_check_fun, fun)
                # modify data inplace to save memory
                for ch_idx in picks:
                    if "ch_idx" in args:
                        kwargs.update(ch_idx=ch_idx)
                    if "ch_name" in args:
                        kwargs.update(ch_name=self.info["ch_names"][ch_idx])
                    self._data[:, ch_idx, :] = np.apply_along_axis(
                        _fun, -1, data_in[:, ch_idx, :], **kwargs
                    )
            else:
                # use parallel function
                _fun = partial(np.apply_along_axis, fun, -1)
                data_picks_new = parallel(
                    p_fun(
                        _fun,
                        data_in[:, ch_idx, :],
                        **kwargs,
                        **{
                            k: v
                            for k, v in [
                                ("ch_name", self.info["ch_names"][ch_idx]),
                                ("ch_idx", ch_idx),
                            ]
                            if k in args
                        },
                    )
                    for ch_idx in picks
                )
                for run_idx, ch_idx in enumerate(picks):
                    self._data[:, ch_idx, :] = data_picks_new[run_idx]
        else:
            self._data = _check_fun(fun, data_in, **kwargs)

        return self

    @property
    def filename(self) -> Path | None:
        """The filename if the epochs are loaded from disk.

        :type: :class:`pathlib.Path` | ``None``
        """
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is not None:
            value = _check_fname(value, overwrite="read", must_exist=True)
        self._filename = value

    def __repr__(self):
        """Build string representation."""
        s = f"{len(self.events)} events "
        s += "(all good)" if self._bad_dropped else "(good & bad)"
        s += f", {self.tmin:.3f}".rstrip("0").rstrip(".")
        s += f" – {self.tmax:.3f}".rstrip("0").rstrip(".")
        s += " s (baseline "
        if self.baseline is None:
            s += "off"
        else:
            s += f"{self.baseline[0]:.3f}".rstrip("0").rstrip(".")
            s += f" – {self.baseline[1]:.3f}".rstrip("0").rstrip(".")
            s += " s"
            if self.baseline != _check_baseline(
                self.baseline,
                times=self.times,
                sfreq=self.info["sfreq"],
                on_baseline_outside_data="adjust",
            ):
                s += " (baseline period was cropped after baseline correction)"

        s += f"), ~{sizeof_fmt(self._size)}"
        s += f", data{'' if self.preload else ' not'} loaded"
        s += ", with metadata" if self.metadata is not None else ""
        max_events = 10
        counts = [
            f"{k!r}: {sum(self.events[:, 2] == v)}"
            for k, v in list(self.event_id.items())[:max_events]
        ]
        if len(self.event_id) > 0:
            s += "," + "\n ".join([""] + counts)
        if len(self.event_id) > max_events:
            not_shown_events = len(self.event_id) - max_events
            s += f"\n and {not_shown_events} more events ..."
        class_name = self.__class__.__name__
        class_name = "Epochs" if class_name == "BaseEpochs" else class_name
        return f"<{class_name} | {s}>"

    @repr_html
    def _repr_html_(self):
        if isinstance(self.event_id, dict):
            event_strings = []
            for k, v in sorted(self.event_id.items()):
                n_events = sum(self.events[:, 2] == v)
                event_strings.append(f"{k}: {n_events}")
        elif isinstance(self.event_id, list):
            event_strings = []
            for k in self.event_id:
                n_events = sum(self.events[:, 2] == k)
                event_strings.append(f"{k}: {n_events}")
        elif isinstance(self.event_id, int):
            n_events = len(self.events[:, 2])
            event_strings = [f"{self.event_id}: {n_events}"]
        else:
            event_strings = None

        t = _get_html_template("repr", "epochs.html.jinja")
        t = t.render(
            inst=self,
            filenames=(
                [Path(self.filename).name]
                if getattr(self, "filename", None) is not None
                else None
            ),
            event_counts=event_strings,
        )
        return t

    @verbose
    def crop(self, tmin=None, tmax=None, include_tmax=True, verbose=None):
        """Crop a time interval from the epochs.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        %(include_tmax)s
        %(verbose)s

        Returns
        -------
        epochs : instance of Epochs
            The cropped epochs object, modified in-place.

        Notes
        -----
        %(notes_tmax_included_by_default)s
        """
        # XXX this could be made to work on non-preloaded data...
        _check_preload(self, "Modifying data of epochs")

        super().crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)

        # Adjust rejection period
        if self.reject_tmin is not None and self.reject_tmin < self.tmin:
            logger.info(
                f"reject_tmin is not in epochs time interval. "
                f"Setting reject_tmin to epochs.tmin ({self.tmin} s)"
            )
            self.reject_tmin = self.tmin
        if self.reject_tmax is not None and self.reject_tmax > self.tmax:
            logger.info(
                f"reject_tmax is not in epochs time interval. "
                f"Setting reject_tmax to epochs.tmax ({self.tmax} s)"
            )
            self.reject_tmax = self.tmax
        return self

    def copy(self):
        """Return copy of Epochs instance.

        Returns
        -------
        epochs : instance of Epochs
            A copy of the object.
        """
        return deepcopy(self)

    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            # drop_log is immutable and _raw is private (and problematic to
            # deepcopy)
            if k in ("drop_log", "_raw", "_times_readonly"):
                memodict[id(v)] = v
            else:
                v = deepcopy(v, memodict)
            result.__dict__[k] = v
        return result

    @verbose
    def save(
        self,
        fname,
        split_size="2GB",
        fmt="single",
        overwrite=False,
        split_naming="neuromag",
        verbose=None,
    ):
        """Save epochs in a fif file.

        Parameters
        ----------
        fname : path-like
            The name of the file, which should end with ``-epo.fif`` or
            ``-epo.fif.gz``.
        split_size : str | int
            Large raw files are automatically split into multiple pieces. This
            parameter specifies the maximum size of each piece. If the
            parameter is an integer, it specifies the size in Bytes. It is
            also possible to pass a human-readable string, e.g., 100MB.
            Note: Due to FIFF file limitations, the maximum split size is 2GB.

            .. versionadded:: 0.10.0
        fmt : str
            Format to save data. Valid options are 'double' or
            'single' for 64- or 32-bit float, or for 128- or
            64-bit complex numbers respectively. Note: Data are processed with
            double precision. Choosing single-precision, the saved data
            will slightly differ due to the reduction in precision.

            .. versionadded:: 0.17
        %(overwrite)s
            To overwrite original file (the same one that was loaded),
            data must be preloaded upon reading. This defaults to True in 0.18
            but will change to False in 0.19.

            .. versionadded:: 0.18
        %(split_naming)s

            .. versionadded:: 0.24
        %(verbose)s

        Returns
        -------
        fnames : List of path-like
            List of path-like objects containing the path to each file split.
            .. versionadded:: 1.9

        Notes
        -----
        Bad epochs will be dropped before saving the epochs to disk.
        """
        check_fname(
            fname, "epochs", ("-epo.fif", "-epo.fif.gz", "_epo.fif", "_epo.fif.gz")
        )

        # check for file existence and expand `~` if present
        fname = str(
            _check_fname(
                fname=fname,
                overwrite=overwrite,
                check_bids_split=True,
                name="fname",
            )
        )

        split_size_bytes = _get_split_size(split_size)

        _check_option("fmt", fmt, ["single", "double"])

        # to know the length accurately. The get_data() call would drop
        # bad epochs anyway
        self.drop_bad()
        # total_size tracks sizes that get split
        # over_size tracks overhead (tags, things that get written to each)
        if len(self) == 0:
            warn("Saving epochs with no data")
            total_size = 0
        else:
            d = self[0].get_data(copy=False)
            # this should be guaranteed by subclasses
            assert d.dtype in (">f8", "<f8", ">c16", "<c16")
            total_size = d.nbytes * len(self)
        self._check_consistency()
        over_size = 0
        if fmt == "single":
            total_size //= 2  # 64bit data converted to 32bit before writing.
        over_size += 32  # FIF tags
        # Account for all the other things we write, too
        # 1. meas_id block plus main epochs block
        over_size += 132
        # 2. measurement info (likely slight overestimate, but okay)
        over_size += object_size(self.info) + 16 * len(self.info)
        # 3. events and event_id in its own block
        total_size += self.events.size * 4
        over_size += len(_event_id_string(self.event_id)) + 72
        # 4. Metadata in a block of its own
        if self.metadata is not None:
            total_size += len(_prepare_write_metadata(self.metadata))
        over_size += 56
        # 5. first sample, last sample, baseline
        over_size += 40 * (self.baseline is not None) + 40
        # 6. drop log: gets written to each, with IGNORE for ones that are
        #    not part of it. So make a fake one with all having entries.
        drop_size = len(json.dumps(self.drop_log)) + 16
        drop_size += 8 * (len(self.selection) - 1)  # worst case: all but one
        over_size += drop_size
        # 7. reject params
        reject_params = _pack_reject_params(self)
        if reject_params:
            over_size += len(json.dumps(reject_params)) + 16
        # 8. selection
        total_size += self.selection.size * 4
        over_size += 16
        # 9. end of file tags
        over_size += _NEXT_FILE_BUFFER
        logger.debug(f"    Overhead size:   {str(over_size).rjust(15)}")
        logger.debug(f"    Splittable size: {str(total_size).rjust(15)}")
        logger.debug(f"    Split size:      {str(split_size_bytes).rjust(15)}")
        # need at least one per
        n_epochs = len(self)
        n_per = total_size // n_epochs if n_epochs else 0
        min_size = n_per + over_size
        if split_size_bytes < min_size:
            raise ValueError(
                f"The split size {split_size} is too small to safely write "
                "the epochs contents, minimum split size is "
                f"{sizeof_fmt(min_size)} ({min_size} bytes)"
            )

        # This is like max(int(ceil(total_size / split_size)), 1) but cleaner
        n_parts = max((total_size - 1) // (split_size_bytes - over_size) + 1, 1)
        assert n_parts >= 1, n_parts
        if n_parts > 1:
            logger.info(f"Splitting into {n_parts} parts")
            if n_parts > 100:  # This must be an error
                raise ValueError(
                    f"Split size {split_size} would result in writing {n_parts} files"
                )

        if len(self.drop_log) > 100000:
            warn(
                f"epochs.drop_log contains {len(self.drop_log)} entries "
                f"which will incur up to a {sizeof_fmt(drop_size)} writing "
                f"overhead (per split file), consider using "
                f"epochs.reset_drop_log_selection() prior to writing"
            )

        epoch_idxs = np.array_split(np.arange(n_epochs), n_parts)

        _check_option("split_naming", split_naming, ("neuromag", "bids"))
        split_fnames = _make_split_fnames(fname, n_parts, split_naming)
        for part_idx, epoch_idx in enumerate(epoch_idxs):
            this_epochs = self[epoch_idx] if n_parts > 1 else self
            # avoid missing event_ids in splits
            this_epochs.event_id = self.event_id

            _save_split(this_epochs, split_fnames, part_idx, n_parts, fmt, overwrite)
        return split_fnames

    @verbose
    def export(self, fname, fmt="auto", *, overwrite=False, verbose=None):
        """Export Epochs to external formats.

        %(export_fmt_support_epochs)s

        %(export_warning)s

        Parameters
        ----------
        %(fname_export_params)s
        %(export_fmt_params_epochs)s
        %(overwrite)s

            .. versionadded:: 0.24.1
        %(verbose)s

        Notes
        -----
        .. versionadded:: 0.24

        %(export_warning_note_epochs)s
        %(export_eeglab_note)s
        """
        from .export import export_epochs

        export_epochs(fname, self, fmt, overwrite=overwrite, verbose=verbose)

    @fill_doc
    def equalize_event_counts(
        self, event_ids=None, method="mintime", *, random_state=None
    ):
        """Equalize the number of trials in each condition.

        It tries to make the remaining epochs occurring as close as possible in
        time. This method works based on the idea that if there happened to be
        some time-varying (like on the scale of minutes) noise characteristics
        during a recording, they could be compensated for (to some extent) in
        the equalization process. This method thus seeks to reduce any of
        those effects by minimizing the differences in the times of the events
        within a `~mne.Epochs` instance. For example, if one event type
        occurred at time points ``[1, 2, 3, 4, 120, 121]`` and the another one
        at ``[3.5, 4.5, 120.5, 121.5]``, this method would remove the events at
        times ``[1, 2]`` for the first event type – and not the events at times
        ``[120, 121]``.

        Parameters
        ----------
        event_ids : None | list | dict
            The event types to equalize.

            If ``None`` (default), equalize the counts of **all** event types
            present in the `~mne.Epochs` instance.

            If a list, each element can either be a string (event name) or a
            list of strings. In the case where one of the entries is a list of
            strings, event types in that list will be grouped together before
            equalizing trial counts across conditions.

            If a dictionary, the keys are considered as the event names whose
            counts to equalize, i.e., passing ``dict(A=1, B=2)`` will have the
            same effect as passing ``['A', 'B']``. This is useful if you intend
            to pass an ``event_id`` dictionary that was used when creating
            `~mne.Epochs`.

            In the case where partial matching is used (using ``/`` in
            the event names), the event types will be matched according to the
            provided tags, that is, processing works as if the ``event_ids``
            matched by the provided tags had been supplied instead.
            The ``event_ids`` must identify non-overlapping subsets of the
            epochs.
        %(equalize_events_method)s
        %(random_state)s Used only if ``method='random'``.

        Returns
        -------
        epochs : instance of Epochs
            The modified instance. It is modified in-place.
        indices : array of int
            Indices from the original events list that were dropped.

        Notes
        -----
        For example (if ``epochs.event_id`` was ``{'Left': 1, 'Right': 2,
        'Nonspatial':3}``:

            epochs.equalize_event_counts([['Left', 'Right'], 'Nonspatial'])

        would equalize the number of trials in the ``'Nonspatial'`` condition
        with the total number of trials in the ``'Left'`` and ``'Right'``
        conditions combined.

        If multiple indices are provided (e.g. ``'Left'`` and ``'Right'`` in
        the example above), it is not guaranteed that after equalization the
        conditions will contribute equally. E.g., it is possible to end up
        with 70 ``'Nonspatial'`` epochs, 69 ``'Left'`` and 1 ``'Right'``.

        .. versionchanged:: 0.23
            Default to equalizing all events in the passed instance if no
            event names were specified explicitly.
        """
        from collections.abc import Iterable

        _validate_type(
            event_ids,
            types=(Iterable, None),
            item_name="event_ids",
            type_name="list-like or None",
        )
        if isinstance(event_ids, str):
            raise TypeError(
                f"event_ids must be list-like or None, but "
                f"received a string: {event_ids}"
            )

        if event_ids is None:
            event_ids = list(self.event_id)
        elif not event_ids:
            raise ValueError("event_ids must have at least one element")

        if not self._bad_dropped:
            self.drop_bad()
        # figure out how to equalize
        eq_inds = list()

        # deal with hierarchical tags
        ids = self.event_id
        orig_ids = list(event_ids)
        tagging = False
        if "/" in "".join(ids):
            # make string inputs a list of length 1
            event_ids = [[x] if isinstance(x, str) else x for x in event_ids]
            for ids_ in event_ids:  # check if tagging is attempted
                if any([id_ not in ids for id_ in ids_]):
                    tagging = True
            # 1. treat everything that's not in event_id as a tag
            # 2a. for tags, find all the event_ids matched by the tags
            # 2b. for non-tag ids, just pass them directly
            # 3. do this for every input
            event_ids = [
                (
                    [
                        k for k in ids if all(tag in k.split("/") for tag in id_)
                    ]  # ids matching all tags
                    if all(id__ not in ids for id__ in id_)
                    else id_
                )  # straight pass for non-tag inputs
                for id_ in event_ids
            ]
            for ii, id_ in enumerate(event_ids):
                if len(id_) == 0:
                    raise KeyError(
                        f"{orig_ids[ii]} not found in the epoch object's event_id."
                    )
                elif len({sub_id in ids for sub_id in id_}) != 1:
                    err = (
                        "Don't mix hierarchical and regular event_ids"
                        f" like in '{', '.join(id_)}'."
                    )
                    raise ValueError(err)

            # raise for non-orthogonal tags
            if tagging is True:
                events_ = [set(self[x].events[:, 0]) for x in event_ids]
                doubles = events_[0].intersection(events_[1])
                if len(doubles):
                    raise ValueError(
                        "The two sets of epochs are "
                        "overlapping. Provide an "
                        "orthogonal selection."
                    )

        for eq in event_ids:
            eq_inds.append(self._keys_to_idx(eq))

        sample_nums = [self.events[e, 0] for e in eq_inds]
        indices = _get_drop_indices(sample_nums, method, random_state)
        # need to re-index indices
        indices = np.concatenate([e[idx] for e, idx in zip(eq_inds, indices)])
        self.drop(indices, reason="EQUALIZED_COUNT")
        # actually remove the indices
        return self, indices

    @verbose
    def compute_psd(
        self,
        method="multitaper",
        fmin=0,
        fmax=np.inf,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        remove_dc=True,
        exclude=(),
        *,
        n_jobs=1,
        verbose=None,
        **method_kw,
    ):
        """Perform spectral analysis on sensor data.

        Parameters
        ----------
        %(method_psd)s
            Default is ``'multitaper'``.
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(remove_dc)s
        %(exclude_psd)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s

        Returns
        -------
        spectrum : instance of EpochsSpectrum
            The spectral representation of each epoch.

        Notes
        -----
        .. versionadded:: 1.2

        References
        ----------
        .. footbibliography::
        """
        method = _validate_method(method, type(self).__name__)
        self._set_legacy_nfft_default(tmin, tmax, method, method_kw)

        return EpochsSpectrum(
            self,
            method=method,
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            exclude=exclude,
            proj=proj,
            remove_dc=remove_dc,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @verbose
    def compute_tfr(
        self,
        method,
        freqs,
        *,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        output="power",
        average=False,
        return_itc=False,
        decim=1,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        """Compute a time-frequency representation of epoched data.

        Parameters
        ----------
        %(method_tfr_epochs)s
        %(freqs_tfr_epochs)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(output_compute_tfr)s
        average : bool
            Whether to return average power across epochs (instead of single-trial
            power). ``average=True`` is not compatible with ``output="complex"`` or
            ``output="phase"``. Ignored if ``method="stockwell"`` (Stockwell method
            *requires* averaging). Default is ``False``.
        return_itc : bool
            Whether to return inter-trial coherence (ITC) as well as power estimates.
            If ``True`` then must specify ``average=True`` (or ``method="stockwell",
            average="auto"``). Default is ``False``.
        %(decim_tfr)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_epochs_tfr)s

        Returns
        -------
        tfr : instance of EpochsTFR or AverageTFR
            The time-frequency-resolved power estimates.
        itc : instance of AverageTFR
            The inter-trial coherence (ITC). Only returned if ``return_itc=True``.

        Notes
        -----
        If ``average=True`` (or ``method="stockwell", average="auto"``) the result will
        be an :class:`~mne.time_frequency.AverageTFR` instead of an
        :class:`~mne.time_frequency.EpochsTFR`.

        .. versionadded:: 1.7

        References
        ----------
        .. footbibliography::
        """
        if method == "stockwell" and not average:  # stockwell method *must* average
            logger.info(
                'Requested `method="stockwell"` so ignoring parameter `average=False`.'
            )
            average = True
        if average:
            # augment `output` value for use by tfr_array_* functions
            _check_option("output", output, ("power",), extra=" when average=True")
            method_kw["output"] = "avg_power_itc" if return_itc else "avg_power"
        else:
            msg = (
                "compute_tfr() got incompatible parameters `average=False` and `{}` "
                "({} requires averaging over epochs)."
            )
            if return_itc:
                raise ValueError(msg.format("return_itc=True", "computing ITC"))
            if method == "stockwell":
                raise ValueError(msg.format('method="stockwell"', "Stockwell method"))
            # `average` and `return_itc` both False, so "phase" and "complex" are OK
            _check_option("output", output, ("power", "phase", "complex"))
            method_kw["output"] = output

        if method == "stockwell":
            method_kw["return_itc"] = return_itc
            method_kw.pop("output")
            if isinstance(freqs, str):
                _check_option("freqs", freqs, "auto")
            else:
                _validate_type(freqs, "array-like")
                _check_option(
                    "freqs", np.array(freqs).shape, ((2,),), extra=" (wrong shape)."
                )
        if average:
            out = AverageTFR(
                inst=self,
                method=method,
                freqs=freqs,
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                proj=proj,
                decim=decim,
                n_jobs=n_jobs,
                verbose=verbose,
                **method_kw,
            )
            # tfr_array_stockwell always returns ITC (but sometimes it's None)
            if hasattr(out, "_itc"):
                if out._itc is not None:
                    state = out.__getstate__()
                    state["data"] = out._itc
                    state["data_type"] = "Inter-trial coherence"
                    itc = AverageTFR(inst=state)
                    del out._itc
                    return out, itc
                del out._itc
            return out
        # now handle average=False
        return EpochsTFR(
            inst=self,
            method=method,
            freqs=freqs,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            decim=decim,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @verbose
    def plot_psd(
        self,
        fmin=0,
        fmax=np.inf,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        *,
        method="auto",
        average=False,
        dB=True,
        estimate="power",
        xscale="linear",
        area_mode="std",
        area_alpha=0.33,
        color="black",
        line_alpha=None,
        spatial_colors=True,
        sphere=None,
        exclude="bads",
        ax=None,
        show=True,
        n_jobs=1,
        verbose=None,
        **method_kw,
    ):
        """%(plot_psd_doc)s.

        Parameters
        ----------
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(method_plot_psd_auto)s
        %(average_plot_psd)s
        %(dB_plot_psd)s
        %(estimate_plot_psd)s
        %(xscale_plot_psd)s
        %(area_mode_plot_psd)s
        %(area_alpha_plot_psd)s
        %(color_plot_psd)s
        %(line_alpha_plot_psd)s
        %(spatial_colors_psd)s
        %(sphere_topomap_auto)s

            .. versionadded:: 0.22.0
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the bad
            channels are excluded. Pass an empty list to plot all channels
            (including channels marked "bad", if any).

            .. versionadded:: 0.24.0
        %(ax_plot_psd)s
        %(show)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s

        Returns
        -------
        fig : instance of Figure
            Figure with frequency spectra of the data channels.

        Notes
        -----
        %(notes_plot_psd_meth)s
        """
        return super().plot_psd(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            reject_by_annotation=False,
            method=method,
            average=average,
            dB=dB,
            estimate=estimate,
            xscale=xscale,
            area_mode=area_mode,
            area_alpha=area_alpha,
            color=color,
            line_alpha=line_alpha,
            spatial_colors=spatial_colors,
            sphere=sphere,
            exclude=exclude,
            ax=ax,
            show=show,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @verbose
    def to_data_frame(
        self,
        picks=None,
        index=None,
        scalings=None,
        copy=True,
        long_format=False,
        time_format=None,
        *,
        verbose=None,
    ):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        additional columns "time", "epoch" (epoch number), and "condition"
        (epoch event description) are added, unless ``index`` is not ``None``
        (in which case the columns specified in ``index`` will be used to form
        the DataFrame's index instead).

        Parameters
        ----------
        %(picks_all)s
        %(index_df_epo)s
            Valid string values are 'time', 'epoch', and 'condition'.
            Defaults to ``None``.
        %(scalings_df)s
        %(copy_df)s
        %(long_format_df_epo)s
        %(time_format_df)s

            .. versionadded:: 0.20
        %(verbose)s

        Returns
        -------
        %(df_return)s
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ["time", "epoch", "condition"]
        valid_time_formats = ["ms", "timedelta"]
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        picks = _picks_to_idx(self.info, picks, "all", exclude=())
        data = self._get_data(on_empty="raise")[:, picks, :]
        times = self.times
        n_epochs, n_picks, n_times = data.shape
        data = np.hstack(data).T  # (time*epochs) x signals
        if copy:
            data = data.copy()
        data = _scale_dataframe_data(self, data, picks, scalings)
        # prepare extra columns / multiindex
        mindex = list()
        times = np.tile(times, n_epochs)
        times = _convert_times(times, time_format, self.info["meas_date"])
        mindex.append(("time", times))
        rev_event_id = {v: k for k, v in self.event_id.items()}
        conditions = [rev_event_id[k] for k in self.events[:, 2]]
        mindex.append(("condition", np.repeat(conditions, n_times)))
        mindex.append(("epoch", np.repeat(self.selection, n_times)))
        assert all(len(mdx) == len(mindex[0]) for mdx in mindex)
        # build DataFrame
        df = _build_data_frame(
            self,
            data,
            picks,
            long_format,
            mindex,
            index,
            default_index=["condition", "epoch", "time"],
        )
        return df

    def as_type(self, ch_type="grad", mode="fast"):
        """Compute virtual epochs using interpolated fields.

        .. Warning:: Using virtual epochs to compute inverse can yield
            unexpected results. The virtual channels have ``'_v'`` appended
            at the end of the names to emphasize that the data contained in
            them are interpolated.

        Parameters
        ----------
        ch_type : str
            The destination channel type. It can be 'mag' or 'grad'.
        mode : str
            Either ``'accurate'`` or ``'fast'``, determines the quality of the
            Legendre polynomial expansion used. ``'fast'`` should be sufficient
            for most applications.

        Returns
        -------
        epochs : instance of mne.EpochsArray
            The transformed epochs object containing only virtual channels.

        Notes
        -----
        This method returns a copy and does not modify the data it
        operates on. It also returns an EpochsArray instance.

        .. versionadded:: 0.20.0
        """
        from .forward import _as_meg_type_inst

        self._handle_empty("raise", "as_type")
        return _as_meg_type_inst(self, ch_type=ch_type, mode=mode)


def _drop_log_stats(drop_log, ignore=("IGNORED",)):
    """Compute drop log stats.

    Parameters
    ----------
    drop_log : list of list
        Epoch drop log from Epochs.drop_log.
    ignore : list
        The drop reasons to ignore.

    Returns
    -------
    perc : float
        Total percentage of epochs dropped.
    """
    if (
        not isinstance(drop_log, tuple)
        or not all(isinstance(d, tuple) for d in drop_log)
        or not all(isinstance(s, str) for d in drop_log for s in d)
    ):
        raise TypeError("drop_log must be a tuple of tuple of str")
    perc = 100 * np.mean(
        [len(d) > 0 for d in drop_log if not any(r in ignore for r in d)]
    )
    return perc


def make_metadata(
    events,
    event_id,
    tmin,
    tmax,
    sfreq,
    row_events=None,
    keep_first=None,
    keep_last=None,
):
    """Automatically generate metadata for use with `mne.Epochs` from events.

    This function mimics the epoching process (it constructs time windows
    around time-locked "events of interest") and collates information about
    any other events that occurred within those time windows. The information
    is returned as a :class:`pandas.DataFrame`, suitable for use as
    `~mne.Epochs` metadata: one row per time-locked event, and columns
    indicating presence or absence and latency of each ancillary event type.

    The function will also return a new ``events`` array and ``event_id``
    dictionary that correspond to the generated metadata, which together can then be
    readily fed into `~mne.Epochs`.

    Parameters
    ----------
    events : array, shape (m, 3)
        The :term:`events array <events>`. By default, the returned metadata
        :class:`~pandas.DataFrame` will have as many rows as the events array.
        To create rows for only a subset of events, pass the ``row_events``
        parameter.
    event_id : dict
        A mapping from event names (keys) to event IDs (values). The event
        names will be incorporated as columns of the returned metadata
        :class:`~pandas.DataFrame`.
    tmin, tmax : float | str | list of str | None
        If float, start and end of the time interval for metadata generation in seconds,
        relative to the time-locked event of the respective time window (the "row
        events").

        .. note::
           If you are planning to attach the generated metadata to
           `~mne.Epochs` and intend to include only events that fall inside
           your epoch's time interval, pass the same ``tmin`` and ``tmax``
           values here as you use for your epochs.

        If ``None``, the time window used for metadata generation is bounded by the
        ``row_events``. This is can be particularly practical if trial duration varies
        greatly, but each trial starts with a known event (e.g., a visual cue or
        fixation).

        .. note::
           If ``tmin=None``, the first time window for metadata generation starts with
           the first row event. If ``tmax=None``, the last time window for metadata
           generation ends with the last event in ``events``.

        If a string or a list of strings, the events bounding the metadata around each
        "row event". For ``tmin``, the events are assumed to occur **before** the row
        event, and for ``tmax``, the events are assumed to occur **after** – unless
        ``tmin`` or ``tmax`` are equal to a row event, in which case the row event
        serves as the bound.

        .. versionchanged:: 1.6.0
           Added support for ``None``.

        .. versionadded:: 1.7.0
           Added support for strings.
    sfreq : float
        The sampling frequency of the data from which the events array was
        extracted.
    row_events : list of str | str | None
        Event types around which to create the time windows. For each of these
        time-locked events, we will create a **row** in the returned metadata
        :class:`pandas.DataFrame`. If provided, the string(s) must be keys of
        ``event_id``. If ``None`` (default), rows are created for **all** event types
        present in ``event_id``.
    keep_first : str | list of str | None
        Specify subsets of :term:`hierarchical event descriptors` (HEDs,
        inspired by :footcite:`BigdelyShamloEtAl2013`) matching events of which
        the **first occurrence** within each time window shall be stored in
        addition to the original events.

        .. note::
           There is currently no way to retain **all** occurrences of a
           repeated event. The ``keep_first`` parameter can be used to specify
           subsets of HEDs, effectively creating a new event type that is the
           union of all events types described by the matching HED pattern.
           Only the very first event of this set will be kept.

        For example, you might have two response events types,
        ``response/left`` and ``response/right``; and in trials with both
        responses occurring, you want to keep only the first response. In this
        case, you can pass ``keep_first='response'``. This will add two new
        columns to the metadata: ``response``, indicating at what **time** the
        event  occurred, relative to the time-locked event; and
        ``first_response``, stating which **type** (``'left'`` or ``'right'``)
        of event occurred.
        To match specific subsets of HEDs describing different sets of events,
        pass a list of these subsets, e.g.
        ``keep_first=['response', 'stimulus']``. If ``None`` (default), no
        event aggregation will take place and no new columns will be created.

        .. note::
           By default, this function will always retain  the first instance
           of any event in each time window. For example, if a time window
           contains two ``'response'`` events, the generated ``response``
           column will automatically refer to the first of the two events. In
           this specific case, it is therefore **not** necessary to make use of
           the ``keep_first`` parameter – unless you need to differentiate
           between two types of responses, like in the example above.

    keep_last : list of str | None
        Same as ``keep_first``, but for keeping only the **last**  occurrence
        of matching events. The column indicating the **type** of an event
        ``myevent`` will be named ``last_myevent``.

    Returns
    -------
    metadata : pandas.DataFrame
        Metadata for each row event, with the following columns:

        - ``event_name``, with strings indicating the name of the time-locked
          event ("row event") for that specific time window

        - one column per event type in ``event_id``, with the same name; floats
          indicating the latency of the event in seconds, relative to the
          time-locked event

        - if applicable, additional columns named after the ``keep_first`` and
          ``keep_last`` event types; floats indicating the latency  of the
          event in seconds, relative to the time-locked event

        - if applicable, additional columns ``first_{event_type}`` and
          ``last_{event_type}`` for ``keep_first`` and ``keep_last`` event
          types, respetively; the values will be strings indicating which event
          types were matched by the provided HED patterns

    events : array, shape (n, 3)
        The events corresponding to the generated metadata, i.e. one
        time-locked event per row.
    event_id : dict
        The event dictionary corresponding to the new events array. This will
        be identical to the input dictionary unless ``row_events`` is supplied,
        in which case it will only contain the events provided there.

    Notes
    -----
    The time window used for metadata generation need not correspond to the
    time window used to create the `~mne.Epochs`, to which the metadata will
    be attached; it may well be much shorter or longer, or not overlap at all,
    if desired. This can be useful, for example, to include events that
    occurred before or after an epoch, e.g. during the inter-trial interval.
    If either ``tmin``, ``tmax``, or both are ``None``, or a string referring e.g. to a
    response event, the time window will typically vary, too.

    .. versionadded:: 0.23

    References
    ----------
    .. footbibliography::
    """
    pd = _check_pandas_installed()

    _validate_type(events, types=("array-like",), item_name="events")
    _validate_type(event_id, types=(dict,), item_name="event_id")
    _validate_type(sfreq, types=("numeric",), item_name="sfreq")
    _validate_type(tmin, types=("numeric", str, "array-like", None), item_name="tmin")
    _validate_type(tmax, types=("numeric", str, "array-like", None), item_name="tmax")
    _validate_type(row_events, types=(None, str, "array-like"), item_name="row_events")
    _validate_type(keep_first, types=(None, str, "array-like"), item_name="keep_first")
    _validate_type(keep_last, types=(None, str, "array-like"), item_name="keep_last")

    if not event_id:
        raise ValueError("event_id dictionary must contain at least one entry")

    def _ensure_list(x):
        if x is None:
            return []
        elif isinstance(x, str):
            return [x]
        else:
            return list(x)

    row_events = _ensure_list(row_events)
    keep_first = _ensure_list(keep_first)
    keep_last = _ensure_list(keep_last)

    # Turn tmin, tmax into a list if they're strings or arrays of strings
    try:
        _validate_type(tmin, types=(str, "array-like"), item_name="tmin")
        tmin = _ensure_list(tmin)
    except TypeError:
        pass

    try:
        _validate_type(tmax, types=(str, "array-like"), item_name="tmax")
        tmax = _ensure_list(tmax)
    except TypeError:
        pass

    keep_first_and_last = set(keep_first) & set(keep_last)
    if keep_first_and_last:
        raise ValueError(
            f"The event names in keep_first and keep_last must "
            f"be mutually exclusive. Specified in both: "
            f"{', '.join(sorted(keep_first_and_last))}"
        )
    del keep_first_and_last

    for param_name, values in dict(keep_first=keep_first, keep_last=keep_last).items():
        for first_last_event_name in values:
            try:
                match_event_names(event_id, [first_last_event_name])
            except KeyError:
                raise ValueError(
                    f'Event "{first_last_event_name}", specified in '
                    f"{param_name}, cannot be found in event_id dictionary"
                )

    # If tmin, tmax are strings, ensure these event names are present in event_id
    def _diff_input_strings_vs_event_id(input_strings, input_name, event_id):
        event_name_diff = sorted(set(input_strings) - set(event_id.keys()))
        if event_name_diff:
            raise ValueError(
                f"Present in {input_name}, but missing from event_id: "
                f"{', '.join(event_name_diff)}"
            )

    _diff_input_strings_vs_event_id(
        input_strings=row_events, input_name="row_events", event_id=event_id
    )
    if isinstance(tmin, list):
        _diff_input_strings_vs_event_id(
            input_strings=tmin, input_name="tmin", event_id=event_id
        )
    if isinstance(tmax, list):
        _diff_input_strings_vs_event_id(
            input_strings=tmax, input_name="tmax", event_id=event_id
        )

    # First and last sample of each epoch, relative to the time-locked event
    # This follows the approach taken in mne.Epochs
    # For strings and None, we don't know the start and stop samples in advance as the
    # time window can vary.
    if isinstance(tmin, type(None) | list):
        start_sample = None
    else:
        start_sample = int(round(tmin * sfreq))

    if isinstance(tmax, type(None) | list):
        stop_sample = None
    else:
        stop_sample = int(round(tmax * sfreq)) + 1

    # Make indexing easier
    # We create the DataFrame before subsetting the events so we end up with
    # indices corresponding to the original event indices. Not used for now,
    # but might come in handy sometime later
    events_df = pd.DataFrame(events, columns=("sample", "prev_id", "id"))
    id_to_name_map = {v: k for k, v in event_id.items()}

    # Only keep events that are of interest
    events = events[np.isin(events[:, 2], list(event_id.values()))]
    events_df = events_df.loc[events_df["id"].isin(event_id.values()), :]

    # Prepare & condition the metadata DataFrame

    # Avoid column name duplications if the exact same event name appears in
    # event_id.keys() and keep_first / keep_last simultaneously
    keep_first_cols = [col for col in keep_first if col not in event_id]
    keep_last_cols = [col for col in keep_last if col not in event_id]
    first_cols = [f"first_{col}" for col in keep_first_cols]
    last_cols = [f"last_{col}" for col in keep_last_cols]

    columns = [
        "event_name",
        *event_id.keys(),
        *keep_first_cols,
        *keep_last_cols,
        *first_cols,
        *last_cols,
    ]

    data = np.empty((len(events_df), len(columns)), float)
    metadata = pd.DataFrame(data=data, columns=columns, index=events_df.index)

    # Event names
    metadata["event_name"] = ""

    # Event times
    start_idx = 1
    stop_idx = start_idx + len(event_id.keys()) + len(keep_first_cols + keep_last_cols)
    metadata.iloc[:, start_idx:stop_idx] = np.nan

    # keep_first and keep_last names
    start_idx = stop_idx
    metadata[columns[start_idx:]] = None

    # We're all set, let's iterate over all events and fill in in the
    # respective cells in the metadata. We will subset this to include only
    # `row_events` later
    for row_event in events_df.itertuples(name="RowEvent"):
        row_idx = row_event.Index
        metadata.loc[row_idx, "event_name"] = id_to_name_map[row_event.id]

        # Determine which events fall into the current time window
        if start_sample is None and isinstance(tmin, list):
            # Lower bound is the the current or the closest previpus event with a name
            # in "tmin"; if there is no such event (e.g., beginning of the recording is
            # being approached), the upper lower becomes the last event in the
            # recording.
            prev_matching_events = events_df.loc[
                (events_df["sample"] <= row_event.sample)
                & (events_df["id"].isin([event_id[name] for name in tmin])),
                :,
            ]
            if prev_matching_events.size == 0:
                # No earlier matching event. Use the current one as the beginning of the
                # time window. This may occur at the beginning of a recording.
                window_start_sample = row_event.sample
            else:
                # At least one earlier matching event. Use the closest one.
                window_start_sample = prev_matching_events.iloc[-1]["sample"]
        elif start_sample is None:
            # Lower bound is the current event.
            window_start_sample = row_event.sample
        else:
            # Lower bound is determined by tmin.
            window_start_sample = row_event.sample + start_sample

        if stop_sample is None and isinstance(tmax, list):
            # Upper bound is the the current or the closest following event with a name
            # in "tmax"; if there is no such event (e.g., end of the recording is being
            # approached), the upper bound becomes the last event in the recording.
            next_matching_events = events_df.loc[
                (events_df["sample"] >= row_event.sample)
                & (events_df["id"].isin([event_id[name] for name in tmax])),
                :,
            ]
            if next_matching_events.size == 0:
                # No matching event after the current one; use the end of the recording
                # as upper bound. This may occur at the end of a recording.
                window_stop_sample = events_df["sample"].iloc[-1]
            else:
                # At least one matching later event. Use the closest one..
                window_stop_sample = next_matching_events.iloc[0]["sample"]
        elif stop_sample is None:
            # Upper bound: next event of the same type, or the last event (of
            # any type) if no later event of the same type can be found.
            next_events = events_df.loc[
                (events_df["sample"] > row_event.sample),
                :,
            ]
            if next_events.size == 0:
                # We've reached the last event in the recording.
                window_stop_sample = row_event.sample
            elif next_events.loc[next_events["id"] == row_event.id, :].size > 0:
                # There's still an event of the same type appearing after the
                # current event. Stop one sample short, we don't want to include that
                # last event here, but in the next iteration.
                window_stop_sample = (
                    next_events.loc[next_events["id"] == row_event.id, :].iloc[0][
                        "sample"
                    ]
                    - 1
                )
            else:
                # There are still events after the current one, but not of the
                # same type.
                window_stop_sample = next_events.iloc[-1]["sample"]
        else:
            # Upper bound is determined by tmax.
            window_stop_sample = row_event.sample + stop_sample

        events_in_window = events_df.loc[
            (events_df["sample"] >= window_start_sample)
            & (events_df["sample"] <= window_stop_sample),
            :,
        ]

        assert not events_in_window.empty

        # Store the metadata
        for event in events_in_window.itertuples(name="Event"):
            event_sample = event.sample - row_event.sample
            event_time = event_sample / sfreq
            event_time = 0 if np.isclose(event_time, 0) else event_time
            event_name = id_to_name_map[event.id]

            if not np.isnan(metadata.loc[row_idx, event_name]):
                # Event already exists in current time window!
                assert metadata.loc[row_idx, event_name] <= event_time

                if event_name not in keep_last:
                    continue

            metadata.loc[row_idx, event_name] = event_time

            # Handle keep_first and keep_last event aggregation
            for event_group_name in keep_first + keep_last:
                if event_name not in match_event_names(event_id, [event_group_name]):
                    continue

                if event_group_name in keep_first:
                    first_last_col = f"first_{event_group_name}"
                else:
                    first_last_col = f"last_{event_group_name}"

                old_time = metadata.loc[row_idx, event_group_name]
                if not np.isnan(old_time):
                    if (event_group_name in keep_first and old_time <= event_time) or (
                        event_group_name in keep_last and old_time >= event_time
                    ):
                        continue

                if event_group_name not in event_id:
                    # This is an HED. Strip redundant information from the
                    # event name
                    name = (
                        event_name.replace(event_group_name, "")
                        .replace("//", "/")
                        .strip("/")
                    )
                    metadata.loc[row_idx, first_last_col] = name
                    del name

                metadata.loc[row_idx, event_group_name] = event_time

    # Only keep rows of interest
    if row_events:
        event_id_timelocked = {
            name: val for name, val in event_id.items() if name in row_events
        }
        events = events[np.isin(events[:, 2], list(event_id_timelocked.values()))]
        metadata = metadata.loc[metadata["event_name"].isin(event_id_timelocked)]
        assert len(events) == len(metadata)
        event_id = event_id_timelocked

    return metadata, events, event_id


def _events_from_annotations(raw, events, event_id, annotations, on_missing):
    """Generate events and event_ids from annotations."""
    events, event_id_tmp = events_from_annotations(raw)
    if events.size == 0:
        raise RuntimeError(
            "No usable annotations found in the raw object. "
            "Either `events` must be provided or the raw "
            "object must have annotations to construct epochs"
        )
    if any(raw.annotations.duration > 0):
        logger.info(
            "Ignoring annotation durations and creating fixed-duration epochs "
            "around annotation onsets."
        )
    if event_id is None:
        event_id = event_id_tmp
    # if event_id is the names of events, map to events integers
    if isinstance(event_id, str):
        event_id = [event_id]
    if isinstance(event_id, list | tuple | set):
        if not set(event_id).issubset(set(event_id_tmp)):
            msg = (
                "No matching annotations found for event_id(s) "
                f"{set(event_id) - set(event_id_tmp)}"
            )
            _on_missing(on_missing, msg)
        # remove extras if on_missing not error
        event_id = set(event_id) & set(event_id_tmp)
        event_id = {my_id: event_id_tmp[my_id] for my_id in event_id}
        # remove any non-selected annotations
        annotations.delete(~np.isin(raw.annotations.description, list(event_id)))
    return events, event_id, annotations


@fill_doc
class Epochs(BaseEpochs):
    """Epochs extracted from a Raw instance.

    Parameters
    ----------
    %(raw_epochs)s

        .. note::
            If ``raw`` contains annotations, ``Epochs`` can be constructed around
            ``raw.annotations.onset``, but note that the durations of the annotations
            are ignored in this case.
    %(events_epochs)s

        .. versionchanged:: 1.7
            Allow ``events=None`` to use ``raw.annotations.onset`` as the source of
            epoch times.
    %(event_id)s
    %(epochs_tmin_tmax)s
    %(baseline_epochs)s
        Defaults to ``(None, 0)``, i.e. beginning of the the data until
        time point zero.
    %(picks_all)s
    preload : bool
        %(epochs_preload)s
    %(reject_epochs)s
    %(flat)s
    %(proj_epochs)s
    %(decim)s
    %(epochs_reject_tmin_tmax)s
    %(detrend_epochs)s
    %(on_missing_epochs)s
    %(reject_by_annotation_epochs)s
    %(metadata_epochs)s

        .. versionadded:: 0.16
    %(event_repeated_epochs)s
    %(verbose)s

    Attributes
    ----------
    %(info_not_none)s
    %(event_id_attr)s
    ch_names : list of string
        List of channel names.
    %(selection_attr)s
    preload : bool
        Indicates whether epochs are in memory.
    drop_log : tuple of tuple
        A tuple of the same length as the event array used to initialize the
        Epochs object. If the i-th original event is still part of the
        selection, drop_log[i] will be an empty tuple; otherwise it will be
        a tuple of the reasons the event is not longer in the selection, e.g.:

        - 'IGNORED'
            If it isn't part of the current subset defined by the user
        - 'NO_DATA' or 'TOO_SHORT'
            If epoch didn't contain enough data names of channels that exceeded
            the amplitude threshold
        - 'EQUALIZED_COUNTS'
            See :meth:`~mne.Epochs.equalize_event_counts`
        - 'USER'
            For user-defined reasons (see :meth:`~mne.Epochs.drop`).

        When dropping based on flat or reject parameters the tuple of
        reasons contains a tuple of channels that satisfied the rejection
        criteria.
    filename : str
        The filename of the object.
    times :  ndarray
        Time vector in seconds. Goes from ``tmin`` to ``tmax``. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.

    See Also
    --------
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts

    Notes
    -----
    When accessing data, Epochs are detrended, baseline-corrected, and
    decimated, then projectors are (optionally) applied.

    For indexing and slicing using ``epochs[...]``, see
    :meth:`mne.Epochs.__getitem__`.

    All methods for iteration over objects (using :meth:`mne.Epochs.__iter__`,
    :meth:`mne.Epochs.iter_evoked` or :meth:`mne.Epochs.next`) use the same
    internal state.

    If ``event_repeated`` is set to ``'merge'``, the coinciding events
    (duplicates) will be merged into a single event_id and assigned a new
    id_number as::

        event_id['{event_id_1}/{event_id_2}/...'] = new_id_number

    For example with the event_id ``{'aud': 1, 'vis': 2}`` and the events
    ``[[0, 0, 1], [0, 0, 2]]``, the "merge" behavior will update both event_id
    and events to be: ``{'aud/vis': 3}`` and ``[[0, 0, 3]]`` respectively.

    There is limited support for :class:`~mne.Annotations` in the
    :class:`~mne.Epochs` class. Currently annotations that are present in the
    :class:`~mne.io.Raw` object will be preserved in the resulting
    :class:`~mne.Epochs` object, but:

    1. It is not yet possible to add annotations
       to the Epochs object programmatically (via code) or interactively
       (through the plot window)
    2. Concatenating :class:`~mne.Epochs` objects
       that contain annotations is not supported, and any annotations will
       be dropped when concatenating.
    3. Annotations will be lost on save.
    """

    @verbose
    def __init__(
        self,
        raw,
        events=None,
        event_id=None,
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        picks=None,
        preload=False,
        reject=None,
        flat=None,
        proj=True,
        decim=1,
        reject_tmin=None,
        reject_tmax=None,
        detrend=None,
        on_missing="raise",
        reject_by_annotation=True,
        metadata=None,
        event_repeated="error",
        verbose=None,
    ):
        from .io import BaseRaw

        if not isinstance(raw, BaseRaw):
            raise ValueError(
                "The first argument to `Epochs` must be an instance of mne.io.BaseRaw"
            )
        info = deepcopy(raw.info)
        annotations = raw.annotations.copy()

        # proj is on when applied in Raw
        proj = proj or raw.proj

        self.reject_by_annotation = reject_by_annotation

        # keep track of original sfreq (needed for annotations)
        raw_sfreq = raw.info["sfreq"]

        # get events from annotations if no events given
        if events is None:
            events, event_id, annotations = _events_from_annotations(
                raw, events, event_id, annotations, on_missing
            )

            # add the annotations.extras to the metadata
            if not all(len(d) == 0 for d in annotations.extras):
                pd = _check_pandas_installed(strict=True)
                extras_df = pd.DataFrame(annotations.extras)
                if metadata is None:
                    metadata = extras_df
                else:
                    extras_df.set_index(metadata.index, inplace=True)
                    metadata = pd.concat(
                        [metadata, extras_df], axis=1, ignore_index=False
                    )

        # call BaseEpochs constructor
        super().__init__(
            info,
            None,
            events,
            event_id,
            tmin,
            tmax,
            metadata=metadata,
            baseline=baseline,
            raw=raw,
            picks=picks,
            reject=reject,
            flat=flat,
            decim=decim,
            reject_tmin=reject_tmin,
            reject_tmax=reject_tmax,
            detrend=detrend,
            proj=proj,
            on_missing=on_missing,
            preload_at_end=preload,
            event_repeated=event_repeated,
            verbose=verbose,
            raw_sfreq=raw_sfreq,
            annotations=annotations,
        )

    @verbose
    def _get_epoch_from_raw(self, idx, verbose=None):
        """Load one epoch from disk.

        Returns
        -------
        data : array | str | None
            If string, it's details on rejection reason.
            If array, it's the data in the desired range (good segment)
            If None, it means no data is available.
        """
        if self._raw is None:
            # This should never happen, as raw=None only if preload=True
            raise ValueError(
                "An error has occurred, no valid raw file found. "
                "Please report this to the mne-python "
                "developers."
            )
        sfreq = self._raw.info["sfreq"]
        event_samp = self.events[idx, 0]
        # Read a data segment from "start" to "stop" in samples
        first_samp = self._raw.first_samp
        start = int(round(event_samp + self._raw_times[0] * sfreq))
        start -= first_samp
        stop = start + len(self._raw_times)

        # reject_tmin, and reject_tmax need to be converted to samples to
        # check the reject_by_annotation boundaries: reject_start, reject_stop
        reject_tmin = self.reject_tmin
        if reject_tmin is None:
            reject_tmin = self._raw_times[0]
        reject_start = int(round(event_samp + reject_tmin * sfreq))
        reject_start -= first_samp

        reject_tmax = self.reject_tmax
        if reject_tmax is None:
            reject_tmax = self._raw_times[-1]
        diff = int(round((self._raw_times[-1] - reject_tmax) * sfreq))
        reject_stop = stop - diff

        logger.debug(f"    Getting epoch for {start}-{stop}")
        data = self._raw._check_bad_segment(
            start,
            stop,
            self.picks,
            reject_start,
            reject_stop,
            self.reject_by_annotation,
        )
        return data


@fill_doc
class EpochsArray(BaseEpochs):
    """Epochs object from numpy array.

    Parameters
    ----------
    data : array, shape (n_epochs, n_channels, n_times)
        The channels' time series for each epoch. See notes for proper units of
        measure.
    %(info_not_none)s Consider using :func:`mne.create_info` to populate this
        structure.
    %(events_epochs)s
    %(tmin_epochs)s
    %(event_id)s
    %(reject_epochs)s
    %(flat)s
    %(epochs_reject_tmin_tmax)s
    %(baseline_epochs)s
        Defaults to ``None``, i.e. no baseline correction.
    %(proj_epochs)s
    %(on_missing_epochs)s
    %(metadata_epochs)s

        .. versionadded:: 0.16
    %(selection)s
    %(drop_log)s

        .. versionadded:: 1.3
    %(raw_sfreq)s

        .. versionadded:: 1.3
    %(verbose)s

    See Also
    --------
    create_info
    EvokedArray
    io.RawArray

    Notes
    -----
    Proper units of measure:

    * V: eeg, eog, seeg, dbs, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc

    EpochsArray does not set `Annotations`. If you would like to create
    simulated data with Annotations that are then preserved in the Epochs
    object, you would use `mne.io.RawArray` first and then create an
    `mne.Epochs` object.
    """

    @verbose
    def __init__(
        self,
        data,
        info,
        events=None,
        tmin=0.0,
        event_id=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        baseline=None,
        proj=True,
        on_missing="raise",
        metadata=None,
        selection=None,
        *,
        drop_log=None,
        raw_sfreq=None,
        verbose=None,
    ):
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)
        if data.ndim != 3:
            raise ValueError(
                "Data must be a 3D array of shape (n_epochs, n_channels, n_samples)"
            )

        if len(info["ch_names"]) != data.shape[1]:
            raise ValueError("Info and data must have same number of channels.")
        if events is None:
            n_epochs = len(data)
            events = _gen_events(n_epochs)
        info = info.copy()  # do not modify original info
        tmax = (data.shape[2] - 1) / info["sfreq"] + tmin

        super().__init__(
            info,
            data,
            events,
            event_id,
            tmin,
            tmax,
            baseline,
            reject=reject,
            flat=flat,
            reject_tmin=reject_tmin,
            reject_tmax=reject_tmax,
            decim=1,
            metadata=metadata,
            selection=selection,
            proj=proj,
            on_missing=on_missing,
            drop_log=drop_log,
            raw_sfreq=raw_sfreq,
            verbose=verbose,
        )
        if self.baseline is not None:
            self._do_baseline = True
        if (
            len(events)
            != np.isin(self.events[:, 2], list(self.event_id.values())).sum()
        ):
            raise ValueError("The events must only contain event numbers from event_id")
        detrend_picks = self._detrend_picks
        for e in self._data:
            # This is safe without assignment b/c there is no decim
            self._detrend_offset_decim(e, detrend_picks)
        self.drop_bad()


def combine_event_ids(epochs, old_event_ids, new_event_id, copy=True):
    """Collapse event_ids from an epochs instance into a new event_id.

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
        Whether to return a new instance or modify in place.

    Returns
    -------
    epochs : instance of Epochs
        The modified epochs.

    Notes
    -----
    This For example (if epochs.event_id was ``{'Left': 1, 'Right': 2}``::

        combine_event_ids(epochs, ['Left', 'Right'], {'Directional': 12})

    would create a 'Directional' entry in epochs.event_id replacing
    'Left' and 'Right' (combining their trials).
    """
    epochs = epochs.copy() if copy else epochs
    old_event_ids = np.asanyarray(old_event_ids)
    if isinstance(new_event_id, int):
        new_event_id = {str(new_event_id): new_event_id}
    else:
        if not isinstance(new_event_id, dict):
            raise ValueError("new_event_id must be a dict or int")
        if not len(list(new_event_id.keys())) == 1:
            raise ValueError("new_event_id dict must have one entry")
    new_event_num = list(new_event_id.values())[0]
    new_event_num = operator.index(new_event_num)
    if new_event_num in epochs.event_id.values():
        raise ValueError("new_event_id value must not already exist")
    # could use .pop() here, but if a latter one doesn't exist, we're
    # in trouble, so run them all here and pop() later
    old_event_nums = np.array([epochs.event_id[key] for key in old_event_ids])
    # find the ones to replace
    inds = np.any(
        epochs.events[:, 2][:, np.newaxis] == old_event_nums[np.newaxis, :], axis=1
    )
    # replace the event numbers in the events list
    epochs.events[inds, 2] = new_event_num
    # delete old entries
    for key in old_event_ids:
        epochs.event_id.pop(key)
    # add the new entry
    epochs.event_id.update(new_event_id)
    return epochs


@fill_doc
def equalize_epoch_counts(epochs_list, method="mintime", *, random_state=None):
    """Equalize the number of trials in multiple Epochs or EpochsTFR instances.

    Parameters
    ----------
    epochs_list : list of Epochs instances
        The Epochs instances to equalize trial counts for.
    %(equalize_events_method)s
    %(random_state)s Used only if ``method='random'``.

    Notes
    -----
    The method ``'mintime'`` tries to make the remaining epochs occurring as close as
    possible in time. This method is motivated by the possibility that if there happened
    to be some time-varying (like on the scale of minutes) noise characteristics during
    a recording, they could be compensated for (to some extent) in the
    equalization process. This method thus seeks to reduce any of those effects
    by minimizing the differences in the times of the events in the two sets of
    epochs. For example, if one had event times [1, 2, 3, 4, 120, 121] and the
    other one had [3.5, 4.5, 120.5, 121.5], it would remove events at times
    [1, 2] in the first epochs and not [120, 121].

    Examples
    --------
    >>> equalize_epoch_counts([epochs1, epochs2])  # doctest: +SKIP
    """
    if not all(isinstance(epoch, BaseEpochs | EpochsTFR) for epoch in epochs_list):
        raise ValueError("All inputs must be Epochs instances")
    # make sure bad epochs are dropped
    for epoch in epochs_list:
        if not epoch._bad_dropped:
            epoch.drop_bad()
    sample_nums = [epoch.events[:, 0] for epoch in epochs_list]
    indices = _get_drop_indices(sample_nums, method, random_state)
    for epoch, inds in zip(epochs_list, indices):
        epoch.drop(inds, reason="EQUALIZED_COUNT")


def _get_drop_indices(sample_nums, method, random_state):
    """Get indices to drop from multiple event timing lists."""
    small_idx = np.argmin([e.size for e in sample_nums])
    small_epoch_indices = sample_nums[small_idx]
    _check_option("method", method, ["mintime", "truncate", "random"])
    indices = list()
    for event in sample_nums:
        if method == "mintime":
            mask = _minimize_time_diff(small_epoch_indices, event)
        elif method == "truncate":
            mask = np.ones(event.size, dtype=bool)
            mask[small_epoch_indices.size :] = False
        elif method == "random":
            rng = check_random_state(random_state)
            mask = np.zeros(event.size, dtype=bool)
            idx = rng.choice(
                np.arange(event.size), size=small_epoch_indices.size, replace=False
            )
            mask[idx] = True
        indices.append(np.where(np.logical_not(mask))[0])
    return indices


def _minimize_time_diff(t_shorter, t_longer):
    """Find a boolean mask to minimize timing differences."""
    keep = np.ones((len(t_longer)), dtype=bool)
    # special case: length zero or one
    if len(t_shorter) < 2:  # interp1d won't work
        keep.fill(False)
        if len(t_shorter) == 1:
            idx = np.argmin(np.abs(t_longer - t_shorter))
            keep[idx] = True
        return keep
    scores = np.ones(len(t_longer))
    x1 = np.arange(len(t_shorter))
    # The first set of keep masks to test
    kwargs = dict(copy=False, bounds_error=False, assume_sorted=True)
    shorter_interp = interp1d(x1, t_shorter, fill_value=t_shorter[-1], **kwargs)
    for ii in range(len(t_longer) - len(t_shorter)):
        scores.fill(np.inf)
        # set up the keep masks to test, eliminating any rows that are already
        # gone
        keep_mask = ~np.eye(len(t_longer), dtype=bool)[keep]
        keep_mask[:, ~keep] = False
        # Check every possible removal to see if it minimizes
        x2 = np.arange(len(t_longer) - ii - 1)
        t_keeps = np.array([t_longer[km] for km in keep_mask])
        longer_interp = interp1d(
            x2, t_keeps, axis=1, fill_value=t_keeps[:, -1], **kwargs
        )
        d1 = longer_interp(x1) - t_shorter
        d2 = shorter_interp(x2) - t_keeps
        scores[keep] = np.abs(d1, d1).sum(axis=1) + np.abs(d2, d2).sum(axis=1)
        keep[np.argmin(scores)] = False
    return keep


@verbose
def _is_good(
    e,
    ch_names,
    channel_type_idx,
    reject,
    flat,
    full_report=False,
    ignore_chs=(),
    verbose=None,
):
    """Test if data segment e is good according to reject and flat.

    The reject and flat parameters can accept functions as values.

    If full_report=True, it will give True/False as well as a list of all
    offending channels.
    """
    bad_tuple = tuple()
    has_printed = False
    checkable = np.ones(len(ch_names), dtype=bool)
    checkable[np.array([c in ignore_chs for c in ch_names], dtype=bool)] = False

    for refl, f, t in zip([reject, flat], [np.greater, np.less], ["", "flat"]):
        if refl is not None:
            for key, refl in refl.items():
                criterion = refl
                idx = channel_type_idx[key]
                name = key.upper()
                if len(idx) > 0:
                    e_idx = e[idx]
                    checkable_idx = checkable[idx]
                    # Check if criterion is a function and apply it
                    if callable(criterion):
                        result = criterion(e_idx)
                        _validate_type(result, tuple, "reject/flat output")
                        if len(result) != 2:
                            raise TypeError(
                                "Function criterion must return a tuple of length 2"
                            )
                        cri_truth, reasons = result
                        _validate_type(cri_truth, (bool, np.bool_), cri_truth, "bool")
                        _validate_type(
                            reasons, (str, list, tuple), reasons, "str, list, or tuple"
                        )
                        idx_deltas = np.where(np.logical_and(cri_truth, checkable_idx))[
                            0
                        ]
                    else:
                        deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)
                        idx_deltas = np.where(
                            np.logical_and(f(deltas, criterion), checkable_idx)
                        )[0]

                    if len(idx_deltas) > 0:
                        # Check to verify that refl is a callable that returns
                        # (bool, reason). Reason must be a str/list/tuple.
                        # If using tuple
                        if callable(refl):
                            if isinstance(reasons, str):
                                reasons = (reasons,)
                            for idx, reason in enumerate(reasons):
                                _validate_type(reason, str, reason)
                            bad_tuple += tuple(reasons)
                        else:
                            bad_names = [ch_names[idx[i]] for i in idx_deltas]
                            if not has_printed:
                                logger.info(
                                    f"    Rejecting {t} epoch based on {name} : "
                                    f"{bad_names}"
                                )
                                has_printed = True
                            if not full_report:
                                return False
                            else:
                                bad_tuple += tuple(bad_names)

    if not full_report:
        return True
    else:
        if bad_tuple == ():
            return True, None
        else:
            return False, bad_tuple


def _read_one_epoch_file(f, tree, preload):
    """Read a single FIF file."""
    with f as fid:
        #   Read the measurement info
        info, meas = read_meas_info(fid, tree, clean_bads=True)

        # read in the Annotations if they exist
        annotations = _read_annotations_fif(fid, tree)
        try:
            events, mappings = _read_events_fif(fid, tree)
        except ValueError as e:
            # Allow reading empty epochs (ToDo: Maybe not anymore in the future)
            if str(e) == "Could not find any events":
                events = np.empty((0, 3), dtype=np.int32)
                mappings = dict()
            else:
                raise
        #   Metadata
        metadata = None
        metadata_tree = dir_tree_find(tree, FIFF.FIFFB_MNE_METADATA)
        if len(metadata_tree) > 0:
            for dd in metadata_tree[0]["directory"]:
                kind = dd.kind
                pos = dd.pos
                if kind == FIFF.FIFF_DESCRIPTION:
                    metadata = read_tag(fid, pos).data
                    metadata = _prepare_read_metadata(metadata)
                    break

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        del meas
        if len(processed) == 0:
            raise ValueError("Could not find processed data")

        epochs_node = dir_tree_find(tree, FIFF.FIFFB_MNE_EPOCHS)
        if len(epochs_node) == 0:
            # before version 0.11 we errantly saved with this tag instead of
            # an MNE tag
            epochs_node = dir_tree_find(tree, FIFF.FIFFB_MNE_EPOCHS)
            if len(epochs_node) == 0:
                epochs_node = dir_tree_find(tree, 122)  # 122 used before v0.11
                if len(epochs_node) == 0:
                    raise ValueError("Could not find epochs data")

        my_epochs = epochs_node[0]

        # Now find the data in the block
        data = None
        data_tag = None
        bmin, bmax = None, None
        baseline = None
        selection = None
        drop_log = None
        raw_sfreq = None
        reject_params = {}
        for k in range(my_epochs["nent"]):
            kind = my_epochs["directory"][k].kind
            pos = my_epochs["directory"][k].pos
            if kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, pos)
                first = int(tag.data.item())
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                tag = read_tag(fid, pos)
                last = int(tag.data.item())
            elif kind == FIFF.FIFF_EPOCH:
                # delay reading until later
                fid.seek(pos, 0)
                data_tag = _read_tag_header(fid, pos)
                data_tag.type = data_tag.type ^ (1 << 30)
            elif kind in [FIFF.FIFF_MNE_BASELINE_MIN, 304]:
                # Constant 304 was used before v0.11
                tag = read_tag(fid, pos)
                bmin = float(tag.data.item())
            elif kind in [FIFF.FIFF_MNE_BASELINE_MAX, 305]:
                # Constant 305 was used before v0.11
                tag = read_tag(fid, pos)
                bmax = float(tag.data.item())
            elif kind == FIFF.FIFF_MNE_EPOCHS_SELECTION:
                tag = read_tag(fid, pos)
                selection = np.array(tag.data)
            elif kind == FIFF.FIFF_MNE_EPOCHS_DROP_LOG:
                tag = read_tag(fid, pos)
                drop_log = tag.data
                drop_log = json.loads(drop_log)
                drop_log = tuple(tuple(x) for x in drop_log)
            elif kind == FIFF.FIFF_MNE_EPOCHS_REJECT_FLAT:
                tag = read_tag(fid, pos)
                reject_params = json.loads(tag.data)
            elif kind == FIFF.FIFF_MNE_EPOCHS_RAW_SFREQ:
                tag = read_tag(fid, pos)
                raw_sfreq = tag.data

        if bmin is not None or bmax is not None:
            baseline = (bmin, bmax)

        n_samp = last - first + 1
        logger.info("    Found the data of interest:")
        logger.info(
            f"        t = {1000 * first / info['sfreq']:10.2f} ... "
            f"{1000 * last / info['sfreq']:10.2f} ms"
        )
        if info["comps"] is not None:
            logger.info(
                f"        {len(info['comps'])} CTF compensation matrices available"
            )

        # Inspect the data
        if data_tag is None:
            raise ValueError("Epochs data not found")
        epoch_shape = (len(info["ch_names"]), n_samp)
        size_expected = len(events) * np.prod(epoch_shape)
        # on read double-precision is always used
        if data_tag.type == FIFF.FIFFT_FLOAT:
            datatype = np.float64
            fmt = ">f4"
        elif data_tag.type == FIFF.FIFFT_DOUBLE:
            datatype = np.float64
            fmt = ">f8"
        elif data_tag.type == FIFF.FIFFT_COMPLEX_FLOAT:
            datatype = np.complex128
            fmt = ">c8"
        elif data_tag.type == FIFF.FIFFT_COMPLEX_DOUBLE:
            datatype = np.complex128
            fmt = ">c16"
        fmt_itemsize = np.dtype(fmt).itemsize
        assert fmt_itemsize in (4, 8, 16)
        size_actual = data_tag.size // fmt_itemsize - 16 // fmt_itemsize

        if not size_actual == size_expected:
            raise ValueError(
                f"Incorrect number of samples ({size_actual} instead of "
                f"{size_expected})."
            )

        # Calibration factors
        cals = np.array(
            [
                [info["chs"][k]["cal"] * info["chs"][k].get("scale", 1.0)]
                for k in range(info["nchan"])
            ],
            np.float64,
        )

        # Read the data
        if preload:
            data = read_tag(fid, data_tag.pos).data.astype(datatype)
            data *= cals

        # Put it all together
        tmin = first / info["sfreq"]
        tmax = last / info["sfreq"]
        event_id = (
            {str(e): e for e in np.unique(events[:, 2])}
            if mappings is None
            else mappings
        )
        # In case epochs didn't have a FIFF.FIFF_MNE_EPOCHS_SELECTION tag
        # (version < 0.8):
        if selection is None:
            selection = np.arange(len(events))
        if drop_log is None:
            drop_log = ((),) * len(events)

    return (
        info,
        data,
        data_tag,
        events,
        event_id,
        metadata,
        tmin,
        tmax,
        baseline,
        selection,
        drop_log,
        epoch_shape,
        cals,
        reject_params,
        fmt,
        annotations,
        raw_sfreq,
    )


@verbose
def read_epochs(fname, proj=True, preload=True, verbose=None) -> "EpochsFIF":
    """Read epochs from a fif file.

    Parameters
    ----------
    %(fname_epochs)s
    %(proj_epochs)s
    preload : bool
        If True, read all epochs from disk immediately. If ``False``, epochs
        will be read on demand.
    %(verbose)s

    Returns
    -------
    epochs : instance of Epochs
        The epochs.
    """
    return EpochsFIF(fname, proj, preload, verbose)


class _RawContainer:
    """Helper for a raw data container."""

    def __init__(self, fid, data_tag, event_samps, epoch_shape, cals, fmt):
        self.fid = fid
        self.data_tag = data_tag
        self.event_samps = event_samps
        self.epoch_shape = epoch_shape
        self.cals = cals
        self.proj = False
        self.fmt = fmt

    def __del__(self):  # noqa: D105
        self.fid.close()


@fill_doc
class EpochsFIF(BaseEpochs):
    """Epochs read from disk.

    Parameters
    ----------
    %(fname_epochs)s
    %(proj_epochs)s
    preload : bool
        If True, read all epochs from disk immediately. If False, epochs will
        be read on demand.
    %(verbose)s

    See Also
    --------
    mne.Epochs
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts
    """

    @verbose
    def __init__(self, fname, proj=True, preload=True, verbose=None):
        from .io.base import _get_fname_rep

        if _path_like(fname):
            check_fname(
                fname=fname,
                filetype="epochs",
                endings=("-epo.fif", "-epo.fif.gz", "_epo.fif", "_epo.fif.gz"),
            )
            fname = _check_fname(fname=fname, must_exist=True, overwrite="read")
        elif not preload:
            raise ValueError("preload must be used with file-like objects")

        fnames = [fname]
        fname_rep = _get_fname_rep(fname)
        ep_list = list()
        raw = list()
        for fname in fnames:
            logger.info(f"Reading {fname_rep} ...")
            fid, tree, _ = fiff_open(fname, preload=preload)
            next_fname = _get_next_fname(fid, fname, tree)
            (
                info,
                data,
                data_tag,
                events,
                event_id,
                metadata,
                tmin,
                tmax,
                baseline,
                selection,
                drop_log,
                epoch_shape,
                cals,
                reject_params,
                fmt,
                annotations,
                raw_sfreq,
            ) = _read_one_epoch_file(fid, tree, preload)

            if (events[:, 0] < 0).any():
                events = events.copy()
                warn(
                    "Incorrect events detected on disk, setting event "
                    "numbers to consecutive increasing integers"
                )
                events[:, 0] = np.arange(1, len(events) + 1)
            # here we ignore missing events, since users should already be
            # aware of missing events if they have saved data that way
            # we also retain original baseline without re-applying baseline
            # correction (data is being baseline-corrected when written to
            # disk)
            epoch = BaseEpochs(
                info,
                data,
                events,
                event_id,
                tmin,
                tmax,
                baseline=None,
                metadata=metadata,
                on_missing="ignore",
                selection=selection,
                drop_log=drop_log,
                proj=False,
                verbose=False,
                raw_sfreq=raw_sfreq,
            )
            epoch.baseline = baseline
            epoch._do_baseline = False  # might be superfluous but won't hurt
            ep_list.append(epoch)

            if not preload:
                # store everything we need to index back to the original data
                raw.append(
                    _RawContainer(
                        fiff_open(fname)[0],
                        data_tag,
                        events[:, 0].copy(),
                        epoch_shape,
                        cals,
                        fmt,
                    )
                )

            if next_fname is not None:
                fnames.append(next_fname)

        unsafe_annot_add = raw_sfreq is None
        (
            info,
            data,
            raw_sfreq,
            events,
            event_id,
            tmin,
            tmax,
            metadata,
            baseline,
            selection,
            drop_log,
        ) = _concatenate_epochs(
            ep_list,
            with_data=preload,
            add_offset=False,
            on_mismatch="raise",
        )
        # we need this uniqueness for non-preloaded data to work properly
        if len(np.unique(events[:, 0])) != len(events):
            raise RuntimeError("Event time samples were not unique")

        # correct the drop log
        assert len(drop_log) % len(fnames) == 0
        step = len(drop_log) // len(fnames)
        offsets = np.arange(step, len(drop_log) + 1, step)
        drop_log = list(drop_log)
        for i1, i2 in zip(offsets[:-1], offsets[1:]):
            other_log = drop_log[i1:i2]
            for k, (a, b) in enumerate(zip(drop_log, other_log)):
                if a == ("IGNORED",) and b != ("IGNORED",):
                    drop_log[k] = b
        drop_log = tuple(drop_log[:step])

        # call BaseEpochs constructor
        # again, ensure we're retaining the baseline period originally loaded
        # from disk without trying to re-apply baseline correction
        super().__init__(
            info,
            data,
            events,
            event_id,
            tmin,
            tmax,
            baseline=None,
            raw=raw,
            proj=proj,
            preload_at_end=False,
            on_missing="ignore",
            selection=selection,
            drop_log=drop_log,
            filename=fname_rep,
            metadata=metadata,
            verbose=verbose,
            raw_sfreq=raw_sfreq,
            annotations=annotations,
            **reject_params,
        )
        self.baseline = baseline
        self._do_baseline = False
        # use the private property instead of drop_bad so that epochs
        # are not all read from disk for preload=False
        self._bad_dropped = True
        # private property to suggest that people re-save epochs if they add
        # annotations
        self._unsafe_annot_add = unsafe_annot_add

    @verbose
    def _get_epoch_from_raw(self, idx, verbose=None):
        """Load one epoch from disk."""
        # Find the right file and offset to use
        event_samp = self.events[idx, 0]
        for raw in self._raw:
            idx = np.where(raw.event_samps == event_samp)[0]
            if len(idx) == 1:
                fmt = raw.fmt
                idx = idx[0]
                size = np.prod(raw.epoch_shape) * np.dtype(fmt).itemsize
                offset = idx * size + 16  # 16 = Tag header
                break
        else:
            # read the correct subset of the data
            raise RuntimeError(
                "Correct epoch could not be found, please contact mne-python developers"
            )
        # the following is equivalent to this, but faster:
        #
        # >>> data = read_tag(raw.fid, raw.data_tag.pos).data.astype(float)
        # >>> data *= raw.cals[np.newaxis, :, :]
        # >>> data = data[idx]
        #
        # Eventually this could be refactored in io/tag.py if other functions
        # could make use of it
        raw.fid.seek(raw.data_tag.pos + offset, 0)
        if fmt == ">c8":
            read_fmt = ">f4"
        elif fmt == ">c16":
            read_fmt = ">f8"
        else:
            read_fmt = fmt
        data = np.frombuffer(raw.fid.read(size), read_fmt)
        if read_fmt != fmt:
            data = data.view(fmt)
            data = data.astype(np.complex128)
        else:
            data = data.astype(np.float64)

        data.shape = raw.epoch_shape
        data *= raw.cals
        return data


@fill_doc
def bootstrap(epochs, random_state=None):
    """Compute epochs selected by bootstrapping.

    Parameters
    ----------
    epochs : Epochs instance
        epochs data to be bootstrapped
    %(random_state)s

    Returns
    -------
    epochs : Epochs instance
        The bootstrap samples
    """
    if not epochs.preload:
        raise RuntimeError(
            "Modifying data of epochs is only supported "
            "when preloading is used. Use preload=True "
            "in the constructor."
        )

    rng = check_random_state(random_state)
    epochs_bootstrap = epochs.copy()
    n_events = len(epochs_bootstrap.events)
    idx = rng_uniform(rng)(0, n_events, n_events)
    epochs_bootstrap = epochs_bootstrap[idx]
    return epochs_bootstrap


def _concatenate_epochs(
    epochs_list, *, with_data=True, add_offset=True, on_mismatch="raise"
):
    """Auxiliary function for concatenating epochs."""
    if not isinstance(epochs_list, list | tuple):
        raise TypeError(f"epochs_list must be a list or tuple, got {type(epochs_list)}")

    # to make warning messages only occur once during concatenation
    warned = False

    for ei, epochs in enumerate(epochs_list):
        if not isinstance(epochs, BaseEpochs):
            raise TypeError(
                f"epochs_list[{ei}] must be an instance of Epochs, got {type(epochs)}"
            )

        if (
            getattr(epochs, "annotations", None) is not None
            and len(epochs.annotations) > 0
            and not warned
        ):
            warned = True
            warn(
                "Concatenation of Annotations within Epochs is not supported yet. All "
                "annotations will be dropped."
            )

            # create a copy, so that the Annotations are not modified in place
            # from the original object
            epochs = epochs.copy()
            epochs.set_annotations(None)
    out = epochs_list[0]
    offsets = [0]
    if with_data:
        out.drop_bad()
        offsets.append(len(out))
    events = [out.events]
    metadata = [out.metadata]
    baseline, tmin, tmax = out.baseline, out.tmin, out.tmax
    raw_sfreq = out._raw_sfreq
    info = deepcopy(out.info)
    drop_log = out.drop_log
    event_id = deepcopy(out.event_id)
    selection = out.selection
    # offset is the last epoch + tmax + 10 second
    shift = np.int64((10 + tmax) * out.info["sfreq"])
    # Allow reading empty epochs (ToDo: Maybe not anymore in the future)
    if out._allow_empty:
        events_offset = 0
    else:
        events_offset = int(np.max(events[0][:, 0])) + shift
    events_offset = np.int64(events_offset)
    events_overflow = False
    warned = False
    for ii, epochs in enumerate(epochs_list[1:], 1):
        _ensure_infos_match(epochs.info, info, f"epochs[{ii}]", on_mismatch=on_mismatch)
        if not np.allclose(epochs.times, epochs_list[0].times):
            raise ValueError("Epochs must have same times")

        if epochs.baseline != baseline:
            raise ValueError("Baseline must be same for all epochs")

        if epochs._raw_sfreq != raw_sfreq and not warned:
            warned = True
            warn(
                "The original raw sampling rate of the Epochs does not "
                "match for all Epochs. Please proceed cautiously."
            )

        # compare event_id
        common_keys = list(set(event_id).intersection(set(epochs.event_id)))
        for key in common_keys:
            if not event_id[key] == epochs.event_id[key]:
                msg = (
                    "event_id values must be the same for identical keys "
                    'for all concatenated epochs. Key "{}" maps to {} in '
                    "some epochs and to {} in others."
                )
                raise ValueError(msg.format(key, event_id[key], epochs.event_id[key]))

        if with_data:
            epochs.drop_bad()
            offsets.append(len(epochs))
        evs = epochs.events.copy()
        if len(epochs.events) == 0:
            warn("One of the Epochs objects to concatenate was empty.")
        elif add_offset:
            # We need to cast to a native Python int here to detect an
            # overflow of a numpy int32 (which is the default on windows)
            max_timestamp = int(np.max(evs[:, 0]))
            evs[:, 0] += events_offset
            events_offset += max_timestamp + shift
            if events_offset > INT32_MAX:
                warn(
                    f"Event number greater than {INT32_MAX} created, "
                    "events[:, 0] will be assigned consecutive increasing "
                    "integer values"
                )
                events_overflow = True
                add_offset = False  # we no longer need to add offset
        events.append(evs)
        selection = np.concatenate((selection, epochs.selection))
        drop_log = drop_log + epochs.drop_log
        event_id.update(epochs.event_id)
        metadata.append(epochs.metadata)
    events = np.concatenate(events, axis=0)
    # check to see if we exceeded our maximum event offset
    if events_overflow:
        events[:, 0] = np.arange(1, len(events) + 1)

    # Create metadata object (or make it None)
    n_have = sum(this_meta is not None for this_meta in metadata)
    if n_have == 0:
        metadata = None
    elif n_have != len(metadata):
        raise ValueError(
            f"{n_have} of {len(metadata)} epochs instances have metadata, either "
            "all or none must have metadata"
        )
    else:
        pd = _check_pandas_installed(strict=False)
        if pd is not False:
            metadata = pd.concat(metadata)
        else:  # dict of dicts
            metadata = sum(metadata, list())
    assert len(offsets) == (len(epochs_list) if with_data else 0) + 1
    data = None
    if with_data:
        offsets = np.cumsum(offsets)
        for start, stop, epochs in zip(offsets[:-1], offsets[1:], epochs_list):
            this_data = epochs.get_data(copy=False)
            if data is None:
                data = np.empty(
                    (offsets[-1], len(out.ch_names), len(out.times)),
                    dtype=this_data.dtype,
                )
            data[start:stop] = this_data
    return (
        info,
        data,
        raw_sfreq,
        events,
        event_id,
        tmin,
        tmax,
        metadata,
        baseline,
        selection,
        drop_log,
    )


@verbose
def concatenate_epochs(
    epochs_list, add_offset=True, *, on_mismatch="raise", verbose=None
):
    """Concatenate a list of `~mne.Epochs` into one `~mne.Epochs` object.

    .. note:: Unlike `~mne.concatenate_raws`, this function does **not**
              modify any of the input data.

    Parameters
    ----------
    epochs_list : list
        List of `~mne.Epochs` instances to concatenate (in that order).
    add_offset : bool
        If True, a fixed offset is added to the event times from different
        Epochs sets, such that they are easy to distinguish after the
        concatenation.
        If False, the event times are unaltered during the concatenation.
    %(on_mismatch_info)s
    %(verbose)s

        .. versionadded:: 0.24

    Returns
    -------
    epochs : instance of EpochsArray
        The result of the concatenation. All data will be loaded into memory.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    (
        info,
        data,
        raw_sfreq,
        events,
        event_id,
        tmin,
        tmax,
        metadata,
        baseline,
        selection,
        drop_log,
    ) = _concatenate_epochs(
        epochs_list,
        with_data=True,
        add_offset=add_offset,
        on_mismatch=on_mismatch,
    )
    selection = np.where([len(d) == 0 for d in drop_log])[0]
    out = EpochsArray(
        data=data,
        info=info,
        events=events,
        event_id=event_id,
        tmin=tmin,
        baseline=baseline,
        selection=selection,
        drop_log=drop_log,
        proj=False,
        on_missing="ignore",
        metadata=metadata,
        raw_sfreq=raw_sfreq,
    )
    out.drop_bad()
    return out


@verbose
def average_movements(
    epochs,
    head_pos=None,
    orig_sfreq=None,
    picks=None,
    origin="auto",
    weight_all=True,
    int_order=8,
    ext_order=3,
    destination=None,
    ignore_ref=False,
    return_mapping=False,
    mag_scale=100.0,
    verbose=None,
):
    """Average data using Maxwell filtering, transforming using head positions.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to operate on.
    %(head_pos_maxwell)s
    orig_sfreq : float | None
        The original sample frequency of the data (that matches the
        event sample numbers in ``epochs.events``). Can be ``None``
        if data have not been decimated or resampled.
    %(picks_all_data)s
    %(origin_maxwell)s
    weight_all : bool
        If True, all channels are weighted by the SSS basis weights.
        If False, only MEG channels are weighted, other channels
        receive uniform weight per epoch.
    %(int_order_maxwell)s
    %(ext_order_maxwell)s
    %(destination_maxwell_dest)s
    %(ignore_ref_maxwell)s
    return_mapping : bool
        If True, return the mapping matrix.
    %(mag_scale_maxwell)s

        .. versionadded:: 0.13
    %(verbose)s

    Returns
    -------
    evoked : instance of Evoked
        The averaged epochs.

    See Also
    --------
    mne.preprocessing.maxwell_filter
    mne.chpi.read_head_pos

    Notes
    -----
    The Maxwell filtering version of this algorithm is described in [1]_,
    in section V.B "Virtual signals and movement correction", equations
    40-44. For additional validation, see [2]_.

    Regularization has not been added because in testing it appears to
    decrease dipole localization accuracy relative to using all components.
    Fine calibration and cross-talk cancellation, however, could be added
    to this algorithm based on user demand.

    .. versionadded:: 0.11

    References
    ----------
    .. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
           multichannel data: The signal space separation method,"
           Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.
    .. [2] Wehner DT, Hämäläinen MS, Mody M, Ahlfors SP. "Head movements
           of children in MEG: Quantification, effects on source
           estimation, and compensation. NeuroImage 40:541–550, 2008.
    """  # noqa: E501
    from .preprocessing.maxwell import (
        _check_destination,
        _check_usable,
        _col_norm_pinv,
        _get_coil_scale,
        _get_mf_picks_fix_mags,
        _get_n_moments,
        _get_sensor_operator,
        _prep_mf_coils,
        _remove_meg_projs_comps,
        _reset_meg_bads,
        _trans_sss_basis,
    )

    if head_pos is None:
        raise TypeError("head_pos must be provided and cannot be None")
    from .chpi import head_pos_to_trans_rot_t

    if not isinstance(epochs, BaseEpochs):
        raise TypeError(f"epochs must be an instance of Epochs, not {type(epochs)}")
    orig_sfreq = epochs.info["sfreq"] if orig_sfreq is None else orig_sfreq
    orig_sfreq = float(orig_sfreq)
    if isinstance(head_pos, np.ndarray):
        head_pos = head_pos_to_trans_rot_t(head_pos)
    trn, rot, t = head_pos
    del head_pos
    _check_usable(epochs, ignore_ref)
    origin = _check_origin(origin, epochs.info, "head")
    recon_trans = _check_destination(destination, epochs.info, "head")

    logger.info(f"Aligning and averaging up to {len(epochs.events)} epochs")
    if not np.array_equal(epochs.events[:, 0], np.unique(epochs.events[:, 0])):
        raise RuntimeError("Epochs must have monotonically increasing events")
    info_to = epochs.info.copy()
    meg_picks, mag_picks, grad_picks, good_mask, _ = _get_mf_picks_fix_mags(
        info_to, int_order, ext_order, ignore_ref
    )
    coil_scale, mag_scale = _get_coil_scale(
        meg_picks, mag_picks, grad_picks, mag_scale, info_to
    )
    mult = _get_sensor_operator(epochs, meg_picks)
    n_channels, n_times = len(epochs.ch_names), len(epochs.times)
    other_picks = np.setdiff1d(np.arange(n_channels), meg_picks)
    data = np.zeros((n_channels, n_times))
    count = 0
    # keep only MEG w/bad channels marked in "info_from"
    info_from = pick_info(info_to, meg_picks[good_mask], copy=True)
    all_coils_recon = _prep_mf_coils(info_to, ignore_ref=ignore_ref)
    all_coils = _prep_mf_coils(info_from, ignore_ref=ignore_ref)
    # remove MEG bads in "to" info
    _reset_meg_bads(info_to)
    # set up variables
    w_sum = 0.0
    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_decomp = 0.0  # this will end up being a weighted average
    last_trans = None
    decomp_coil_scale = coil_scale[good_mask]
    exp = dict(int_order=int_order, ext_order=ext_order, head_frame=True, origin=origin)
    n_in = _get_n_moments(int_order)
    for ei, epoch in enumerate(epochs):
        event_time = epochs.events[epochs._current - 1, 0] / orig_sfreq
        use_idx = np.where(t <= event_time)[0]
        if len(use_idx) == 0:
            trans = info_to["dev_head_t"]["trans"]
        else:
            use_idx = use_idx[-1]
            trans = np.vstack(
                [np.hstack([rot[use_idx], trn[[use_idx]].T]), [[0.0, 0.0, 0.0, 1.0]]]
            )
        loc_str = ", ".join(f"{tr:0.1f}" for tr in (trans[:3, 3] * 1000))
        if last_trans is None or not np.allclose(last_trans, trans):
            logger.info(
                f"    Processing epoch {ei + 1} (device location: {loc_str} mm)"
            )
            reuse = False
            last_trans = trans
        else:
            logger.info(f"    Processing epoch {ei + 1} (device location: same)")
            reuse = True
        epoch = epoch.copy()  # because we operate inplace
        if not reuse:
            S = _trans_sss_basis(exp, all_coils, trans, coil_scale=decomp_coil_scale)
            # Get the weight from the un-regularized version (eq. 44)
            weight = np.linalg.norm(S[:, :n_in])
            # XXX Eventually we could do cross-talk and fine-cal here
            S *= weight
        S_decomp += S  # eq. 41
        epoch[slice(None) if weight_all else meg_picks] *= weight
        data += epoch  # eq. 42
        w_sum += weight
        count += 1
    del info_from
    mapping = None
    if count == 0:
        data.fill(np.nan)
    else:
        data[meg_picks] /= w_sum
        data[other_picks] /= w_sum if weight_all else count
        # Finalize weighted average decomp matrix
        S_decomp /= w_sum
        # Get recon matrix
        # (We would need to include external here for regularization to work)
        exp["ext_order"] = 0
        S_recon = _trans_sss_basis(exp, all_coils_recon, recon_trans)
        if mult is not None:
            S_decomp = mult @ S_decomp
            S_recon = mult @ S_recon
        exp["ext_order"] = ext_order
        # We could determine regularization on basis of destination basis
        # matrix, restricted to good channels, as regularizing individual
        # matrices within the loop above does not seem to work. But in
        # testing this seemed to decrease localization quality in most cases,
        # so we do not provide the option here.
        S_recon /= coil_scale
        # Invert
        pS_ave = _col_norm_pinv(S_decomp)[0][:n_in]
        pS_ave *= decomp_coil_scale.T
        # Get mapping matrix
        mapping = np.dot(S_recon, pS_ave)
        # Apply mapping
        data[meg_picks] = np.dot(mapping, data[meg_picks[good_mask]])
    info_to["dev_head_t"] = recon_trans  # set the reconstruction transform
    evoked = epochs._evoked_from_epoch_data(
        data, info_to, picks, n_events=count, kind="average", comment=epochs._name
    )
    _remove_meg_projs_comps(evoked, ignore_ref)
    logger.info(f"Created Evoked dataset from {count} epochs")
    return (evoked, mapping) if return_mapping else evoked


@verbose
def make_fixed_length_epochs(
    raw,
    duration=1.0,
    preload=False,
    reject_by_annotation=True,
    proj=True,
    overlap=0.0,
    id=1,  # noqa: A002
    verbose=None,
):
    """Divide continuous raw data into equal-sized consecutive epochs.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to divide into segments.
    duration : float
        Duration of each epoch in seconds. Defaults to 1.
    %(preload)s
    %(reject_by_annotation_epochs)s

        .. versionadded:: 0.21.0
    %(proj_epochs)s

        .. versionadded:: 0.22.0
    overlap : float
        The overlap between epochs, in seconds. Must be
        ``0 <= overlap < duration``. Default is 0, i.e., no overlap.

        .. versionadded:: 0.23.0
    id : int
        The id to use (default 1).

        .. versionadded:: 0.24.0
    %(verbose)s

    Returns
    -------
    epochs : instance of Epochs
        Segmented data.

    Notes
    -----
    .. versionadded:: 0.20
    """
    events = make_fixed_length_events(raw, id=id, duration=duration, overlap=overlap)
    delta = 1.0 / raw.info["sfreq"]
    return Epochs(
        raw,
        events,
        event_id=[id],
        tmin=0,
        tmax=duration - delta,
        baseline=None,
        preload=preload,
        reject_by_annotation=reject_by_annotation,
        proj=proj,
        verbose=verbose,
    )
