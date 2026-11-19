"""SR Research Eyelink Load Function."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ...utils import (
    _check_fname,
    _verbose_control,
    fill_doc_static,
    logger,
)
from ..base import BaseRaw
from ._utils import _make_eyelink_annots, _make_gap_annots, _parse_eyelink_ascii


@fill_doc_static(
    "eyelink_fname",
    "eyelink_create_annotations",
    "eyelink_apply_offsets",
    "eyelink_find_overlaps",
    "eyelink_overlap_threshold",
    "verbose",
)
def read_raw_eyelink(
    fname: Path | str,
    *,
    create_annotations: bool | list = True,
    apply_offsets: bool = False,
    find_overlaps: bool = False,
    overlap_threshold: float = 0.05,
    verbose: bool | str | int | None = None,
) -> "RawEyelink":
    """Reader for an Eyelink ``.asc`` file.

    Parameters
    ----------
    fname : path-like
        Path to the eyelink file (``.asc``).
    create_annotations : bool | list
        Whether to create :class:`~mne.Annotations` from ocular events
        (blinks, fixations, saccades) and experiment messages. If a list, must
        contain one or more of ``['fixations', 'saccades',' blinks', messages']``.
        If True, creates :class:`~mne.Annotations` for both ocular events and
        experiment messages.
    apply_offsets : bool
        Adjusts the onset time of the :class:`~mne.Annotations` created from Eyelink
        experiment messages, if offset values exist in the ASCII file. If False, any
        offset-like values will be prepended to the annotation description.
    find_overlaps : bool
        Combine left and right eye :class:`mne.Annotations` (blinks, fixations,
        saccades) if their start times and their stop times are both not
        separated by more than overlap_threshold.
    overlap_threshold : float
        Time in seconds. Threshold of allowable time-gap between both the start and
        stop times of the left and right eyes. If the gap is larger than the threshold,
        the :class:`mne.Annotations` will be kept separate (i.e. ``"blink_L"``,
        ``"blink_R"``). If the gap is smaller than the threshold, the
        :class:`mne.Annotations` will be merged and labeled as ``"blink_both"``.
        Defaults to ``0.05`` seconds (50 ms), meaning that if the blink start times of
        the left and right eyes are separated by less than 50 ms, and the blink stop
        times of the left and right eyes are separated by less than 50 ms, then the
        blink will be merged into a single :class:`mne.Annotations`.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    raw : instance of RawEyelink
        A Raw object containing eyetracker data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    It is common for SR Research Eyelink eye trackers to only record data during trials.
    To avoid frequent data discontinuities and to ensure that the data is continuous
    so that it can be aligned with EEG and MEG data (if applicable), this reader will
    preserve the times between recording trials and annotate them with
    ``'BAD_ACQ_SKIP'``.
    """
    fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")

    raw_eyelink = RawEyelink(
        fname,
        create_annotations=create_annotations,
        apply_offsets=apply_offsets,
        find_overlaps=find_overlaps,
        overlap_threshold=overlap_threshold,
        verbose=verbose,
    )
    return raw_eyelink


@fill_doc_static(
    "eyelink_fname",
    "eyelink_create_annotations",
    "eyelink_apply_offsets",
    "eyelink_find_overlaps",
    "eyelink_overlap_threshold",
    "verbose",
)
class RawEyelink(BaseRaw):
    """Raw object from an Eyelink file.

    Parameters
    ----------
    fname : path-like
        Path to the eyelink file (``.asc``).
    create_annotations : bool | list
        Whether to create :class:`~mne.Annotations` from ocular events
        (blinks, fixations, saccades) and experiment messages. If a list, must
        contain one or more of ``['fixations', 'saccades',' blinks', messages']``.
        If True, creates :class:`~mne.Annotations` for both ocular events and
        experiment messages.
    apply_offsets : bool
        Adjusts the onset time of the :class:`~mne.Annotations` created from Eyelink
        experiment messages, if offset values exist in the ASCII file. If False, any
        offset-like values will be prepended to the annotation description.
    find_overlaps : bool
        Combine left and right eye :class:`mne.Annotations` (blinks, fixations,
        saccades) if their start times and their stop times are both not
        separated by more than overlap_threshold.
    overlap_threshold : float
        Time in seconds. Threshold of allowable time-gap between both the start and
        stop times of the left and right eyes. If the gap is larger than the threshold,
        the :class:`mne.Annotations` will be kept separate (i.e. ``"blink_L"``,
        ``"blink_R"``). If the gap is smaller than the threshold, the
        :class:`mne.Annotations` will be merged and labeled as ``"blink_both"``.
        Defaults to ``0.05`` seconds (50 ms), meaning that if the blink start times of
        the left and right eyes are separated by less than 50 ms, and the blink stop
        times of the left and right eyes are separated by less than 50 ms, then the
        blink will be merged into a single :class:`mne.Annotations`.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @_verbose_control
    def __init__(
        self,
        fname,
        *,
        create_annotations=True,
        apply_offsets=False,
        find_overlaps=False,
        overlap_threshold=0.05,
        verbose=None,
    ):
        logger.info(f"Loading {fname}")

        fname = Path(fname)

        # ======================== Parse ASCII file ==========================
        eye_ch_data, info, raw_extras = _parse_eyelink_ascii(
            fname, find_overlaps, overlap_threshold, apply_offsets
        )
        # ======================== Create Raw Object =========================
        super().__init__(
            info,
            preload=eye_ch_data,
            filenames=[fname],
            verbose=verbose,
            raw_extras=[raw_extras],
        )
        self.set_meas_date(self._raw_extras[0]["dt"])

        # ======================== Make Annotations =========================
        gap_annots = None
        if self._raw_extras[0]["n_blocks"] > 1:
            gap_annots = _make_gap_annots(self._raw_extras[0])
        eye_annots = None
        if create_annotations:
            eye_annots = _make_eyelink_annots(
                self._raw_extras[0]["dfs"], create_annotations, apply_offsets
            )
        if gap_annots and eye_annots:  # set both
            self.set_annotations(gap_annots + eye_annots)
        elif gap_annots:
            self.set_annotations(gap_annots)
        elif eye_annots:
            self.set_annotations(eye_annots)
        else:
            logger.info("Not creating any annotations")
