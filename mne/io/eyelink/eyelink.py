"""SR Research Eyelink Load Function."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ...utils import (
    _check_fname,
    fill_doc,
    logger,
    verbose,
)
from ..base import BaseRaw
from ._utils import _make_eyelink_annots, _make_gap_annots, _parse_eyelink_ascii


@fill_doc
def read_raw_eyelink(
    fname,
    *,
    create_annotations=True,
    apply_offsets=False,
    find_overlaps=False,
    overlap_threshold=0.05,
    verbose=None,
) -> "RawEyelink":
    """Reader for an Eyelink ``.asc`` file.

    Parameters
    ----------
    %(eyelink_fname)s
    %(eyelink_create_annotations)s
    %(eyelink_apply_offsets)s
    %(eyelink_find_overlaps)s
    %(eyelink_overlap_threshold)s
    %(verbose)s

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


@fill_doc
class RawEyelink(BaseRaw):
    """Raw object from an XXX file.

    Parameters
    ----------
    %(eyelink_fname)s
    %(eyelink_create_annotations)s
    %(eyelink_apply_offsets)s
    %(eyelink_find_overlaps)s
    %(eyelink_overlap_threshold)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
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
