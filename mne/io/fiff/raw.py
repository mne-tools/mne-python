# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
import os.path as op
from pathlib import Path

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import read_meas_info
from ..._fiff.open import _fiff_get_fid, _get_next_fname, fiff_open
from ..._fiff.tag import _call_dict, read_tag
from ..._fiff.tree import dir_tree_find
from ..._fiff.utils import _mult_cal_one
from ...annotations import Annotations, _read_annotations_fif
from ...channels import fix_mag_coil_types
from ...event import AcqParserFIF
from ...utils import (
    _check_fname,
    _file_like,
    _on_missing,
    check_fname,
    fill_doc,
    logger,
    verbose,
    warn,
)
from ..base import (
    BaseRaw,
    _check_maxshield,
    _check_raw_compatibility,
    _get_fname_rep,
    _RawShell,
)


@fill_doc
class Raw(BaseRaw):
    """Raw data in FIF format.

    Parameters
    ----------
    fname : path-like | file-like
        The raw filename to load. For files that have automatically been split,
        the split part will be automatically loaded. Filenames not ending with
        ``raw.fif``, ``raw_sss.fif``, ``raw_tsss.fif``, ``_meg.fif``,
        ``_eeg.fif``,  or ``_ieeg.fif`` (with or without an optional additional
        ``.gz`` extension) will generate a warning. If a file-like object is
        provided, preloading must be used.

        .. versionchanged:: 0.18
           Support for file-like objects.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    %(preload)s
    %(on_split_missing)s
    %(verbose)s

    Attributes
    ----------
    %(info_not_none)s
    ch_names : list of string
        List of channels' names.
    n_times : int
        Total number of time points in the raw file.
    times :  ndarray
        Time vector in seconds. Starts from 0, independently of `first_samp`
        value. Time interval between consecutive time samples is equal to the
        inverse of the sampling frequency.
    duration : float
        The duration of the raw file in seconds.

        .. versionadded:: 1.9
    preload : bool
        Indicates whether raw data are in memory.
    """

    _extra_attributes = (
        "fix_mag_coil_types",
        "acqparser",
        "_read_raw_file",  # this would be ugly to move, but maybe we should
    )

    @verbose
    def __init__(
        self,
        fname,
        allow_maxshield=False,
        preload=False,
        on_split_missing="raise",
        verbose=None,
    ):
        raws = []
        do_check_ext = not _file_like(fname)
        next_fname = fname
        while next_fname is not None:
            raw, next_fname, buffer_size_sec = self._read_raw_file(
                next_fname, allow_maxshield, preload, do_check_ext
            )
            do_check_ext = False
            raws.append(raw)
            if next_fname is not None:
                if not op.exists(next_fname):
                    msg = (
                        f"Split raw file detected but next file {next_fname} "
                        "does not exist. Ensure all files were transferred "
                        "properly and that split and original files were not "
                        "manually renamed on disk (split files should be "
                        "renamed by loading and re-saving with MNE-Python to "
                        "preserve proper filename linkage)."
                    )
                    _on_missing(on_split_missing, msg, name="on_split_missing")
                    break
        # If using a file-like object, we need to be careful about serialization and
        # types.
        #
        # 1. We must change both the variable named "fname" here so that _get_argvalues
        #    (magic) does not store the file-like object.
        # 2. We need to ensure "filenames" passed to the constructor below gets a list
        #    of Path or None.
        # 3. We need to remove the file-like objects from _raw_extras. This must
        #    be done *after* the super().__init__ call, because the constructor
        #    needs the file-like objects to read the data (which it will do because we
        #    force preloading for file-like objects).

        # Avoid file-like in _get_argvalues (1)
        fname = _path_from_fname(fname)

        _check_raw_compatibility(raws)
        super().__init__(
            copy.deepcopy(raws[0].info),
            preload=False,
            first_samps=[r.first_samp for r in raws],
            last_samps=[r.last_samp for r in raws],
            # Avoid file-like objects in raw.filenames (2)
            filenames=[_path_from_fname(r._raw_extras["filename"]) for r in raws],
            raw_extras=[r._raw_extras for r in raws],
            orig_format=raws[0].orig_format,
            dtype=None,
            buffer_size_sec=buffer_size_sec,
            verbose=verbose,
        )

        # combine annotations
        self.set_annotations(raws[0].annotations, emit_warning=False)

        # Add annotations for in-data skips
        for extra in self._raw_extras:
            mask = [ent is None for ent in extra["ent"]]
            start = extra["bounds"][:-1][mask]
            stop = extra["bounds"][1:][mask] - 1
            duration = (stop - start + 1.0) / self.info["sfreq"]
            annot = Annotations(
                onset=(start / self.info["sfreq"]),
                duration=duration,
                description="BAD_ACQ_SKIP",
                orig_time=self.info["meas_date"],
            )

            self._annotations += annot

        if preload:
            self._preload_data(preload)
        else:
            self.preload = False
        # Avoid file-like objects in _raw_extras (3)
        for extra in self._raw_extras:
            if not isinstance(extra["filename"], Path):
                extra["filename"] = None

    @verbose
    def _read_raw_file(
        self, fname, allow_maxshield, preload, do_check_ext=True, verbose=None
    ):
        """Read in header information from a raw file."""
        logger.info(f"Opening raw data file {fname}...")

        #   Read in the whole file if preload is on and .fif.gz (saves time)
        if not _file_like(fname):
            if do_check_ext:
                endings = (
                    "raw.fif",
                    "raw_sss.fif",
                    "raw_tsss.fif",
                    "_meg.fif",
                    "_eeg.fif",
                    "_ieeg.fif",
                )
                endings += tuple([f"{e}.gz" for e in endings])
                check_fname(fname, "raw", endings)
            # filename
            fname = _check_fname(fname, "read", True, "fname")
            whole_file = preload if fname.suffix == ".gz" else False
        else:
            # file-like
            if not preload:
                raise ValueError("preload must be used with file-like objects")
            whole_file = True
        ff, tree, _ = fiff_open(fname, preload=whole_file)
        with ff as fid:
            #   Read the measurement info

            info, meas = read_meas_info(fid, tree, clean_bads=True)
            annotations = _read_annotations_fif(fid, tree)

            #   Locate the data of interest
            raw_node = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
            if len(raw_node) == 0:
                raw_node = dir_tree_find(meas, FIFF.FIFFB_CONTINUOUS_DATA)
                if len(raw_node) == 0:
                    raw_node = dir_tree_find(meas, FIFF.FIFFB_IAS_RAW_DATA)
                    if len(raw_node) == 0:
                        raise ValueError(f"No raw data in {_get_fname_rep(fname)}")
                    _check_maxshield(allow_maxshield)
                    with info._unlock():
                        info["maxshield"] = True
            del meas

            if len(raw_node) == 1:
                raw_node = raw_node[0]

            #   Process the directory
            directory = raw_node["directory"]
            nent = raw_node["nent"]
            nchan = int(info["nchan"])
            first = 0
            first_samp = 0
            first_skip = 0

            #   Get first sample tag if it is there
            if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, directory[first].pos)
                first_samp = int(tag.data.item())
                first += 1
                _check_entry(first, nent)

            #   Omit initial skip
            if directory[first].kind == FIFF.FIFF_DATA_SKIP:
                # This first skip can be applied only after we know the bufsize
                tag = read_tag(fid, directory[first].pos)
                first_skip = int(tag.data.item())
                first += 1
                _check_entry(first, nent)

            raw = _RawShell()
            raw.first_samp = first_samp
            if info["meas_date"] is None and annotations is not None:
                # we need to adjust annotations.onset as when there is no meas
                # date set_annotations considers that the origin of time is the
                # first available sample (ignores first_samp)
                annotations.onset -= first_samp / info["sfreq"]
            raw.set_annotations(annotations)

            #   Go through the remaining tags in the directory
            raw_extras = list()
            nskip = 0
            orig_format = None

            _byte_dict = {
                FIFF.FIFFT_DAU_PACK16: 2,
                FIFF.FIFFT_SHORT: 2,
                FIFF.FIFFT_FLOAT: 4,
                FIFF.FIFFT_DOUBLE: 8,
                FIFF.FIFFT_INT: 4,
                FIFF.FIFFT_COMPLEX_FLOAT: 8,
                FIFF.FIFFT_COMPLEX_DOUBLE: 16,
            }
            _orig_format_dict = {
                FIFF.FIFFT_DAU_PACK16: "short",
                FIFF.FIFFT_SHORT: "short",
                FIFF.FIFFT_FLOAT: "single",
                FIFF.FIFFT_DOUBLE: "double",
                FIFF.FIFFT_INT: "int",
                FIFF.FIFFT_COMPLEX_FLOAT: "single",
                FIFF.FIFFT_COMPLEX_DOUBLE: "double",
            }

            for k in range(first, nent):
                ent = directory[k]
                # There can be skips in the data (e.g., if the user unclicked)
                # an re-clicked the button
                if ent.kind == FIFF.FIFF_DATA_BUFFER:
                    #   Figure out the number of samples in this buffer
                    try:
                        div = _byte_dict[ent.type]
                    except KeyError:
                        raise RuntimeError(
                            f"Cannot handle data buffers of type {ent.type}"
                        ) from None
                    nsamp = ent.size // (div * nchan)
                    if orig_format is None:
                        orig_format = _orig_format_dict[ent.type]

                    #  Do we have an initial skip pending?
                    if first_skip > 0:
                        first_samp += nsamp * first_skip
                        raw.first_samp = first_samp
                        first_skip = 0

                    #  Do we have a skip pending?
                    if nskip > 0:
                        raw_extras.append(
                            dict(
                                ent=None,
                                first=first_samp,
                                nsamp=nskip * nsamp,
                                last=first_samp + nskip * nsamp - 1,
                            )
                        )
                        first_samp += nskip * nsamp
                        nskip = 0

                    #  Add a data buffer
                    raw_extras.append(
                        dict(
                            ent=ent,
                            first=first_samp,
                            last=first_samp + nsamp - 1,
                            nsamp=nsamp,
                        )
                    )
                    first_samp += nsamp
                elif ent.kind == FIFF.FIFF_DATA_SKIP:
                    tag = read_tag(fid, ent.pos)
                    nskip = int(tag.data.item())

            next_fname = _get_next_fname(fid, _path_from_fname(fname), tree)

        # reformat raw_extras to be a dict of list/ndarray rather than
        # list of dict (faster access)
        raw_extras = {key: [r[key] for r in raw_extras] for key in raw_extras[0]}
        for key in raw_extras:
            if key != "ent":  # dict or None
                raw_extras[key] = np.array(raw_extras[key], int)
        if not np.array_equal(raw_extras["last"][:-1], raw_extras["first"][1:] - 1):
            raise RuntimeError("FIF file appears to be broken")
        bounds = np.cumsum(
            np.concatenate([raw_extras["first"][:1], raw_extras["nsamp"]])
        )
        raw_extras["bounds"] = bounds
        assert len(raw_extras["bounds"]) == len(raw_extras["ent"]) + 1
        # store the original buffer size
        buffer_size_sec = np.median(raw_extras["nsamp"]) / info["sfreq"]
        del raw_extras["first"]
        del raw_extras["last"]
        del raw_extras["nsamp"]
        raw_extras["filename"] = fname

        raw.last_samp = first_samp - 1
        raw.orig_format = orig_format

        #   Add the calibration factors
        cals = np.zeros(info["nchan"])
        for k in range(info["nchan"]):
            cals[k] = info["chs"][k]["range"] * info["chs"][k]["cal"]

        raw._cals = cals
        raw._raw_extras = raw_extras
        logger.info(
            "    Range : %d ... %d =  %9.3f ... %9.3f secs",
            raw.first_samp,
            raw.last_samp,
            float(raw.first_samp) / info["sfreq"],
            float(raw.last_samp) / info["sfreq"],
        )

        raw.info = info

        logger.info("Ready.")

        return raw, next_fname, buffer_size_sec

    @property
    def _dtype(self):
        """Get the dtype to use to store data from disk."""
        if self._dtype_ is not None:
            return self._dtype_
        dtype = None
        for raw_extra in self._raw_extras:
            for ent in raw_extra["ent"]:
                if ent is not None:
                    if ent.type in (
                        FIFF.FIFFT_COMPLEX_FLOAT,
                        FIFF.FIFFT_COMPLEX_DOUBLE,
                    ):
                        dtype = np.complex128
                    else:
                        dtype = np.float64
                    break
            if dtype is not None:
                break
        if dtype is None:
            raise RuntimeError("bug in reading")
        self._dtype_ = dtype
        return dtype

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        n_bad = 0
        with _fiff_get_fid(self._raw_extras[fi]["filename"]) as fid:
            bounds = self._raw_extras[fi]["bounds"]
            ents = self._raw_extras[fi]["ent"]
            nchan = self._raw_extras[fi]["orig_nchan"]
            use = (stop > bounds[:-1]) & (start < bounds[1:])
            offset = 0
            for ei in np.where(use)[0]:
                first = bounds[ei]
                last = bounds[ei + 1]
                nsamp = last - first
                ent = ents[ei]
                first_pick = max(start - first, 0)
                last_pick = min(nsamp, stop - first)
                picksamp = last_pick - first_pick
                this_start = offset
                offset += picksamp
                this_stop = offset
                # only read data if it exists
                if ent is None:
                    continue  # just use zeros for gaps
                # faster to always read full tag, taking advantage of knowing the header
                # already (cutting out some of read_tag) ...
                fid.seek(ent.pos + 16, 0)
                one = _call_dict[ent.type](fid, ent, shape=None, rlims=None)
                try:
                    one.shape = (nsamp, nchan)
                except AttributeError:  # one is None
                    n_bad += picksamp
                else:
                    # ... then pick samples we want
                    if first_pick != 0 or last_pick != nsamp:
                        one = one[first_pick:last_pick]
                    _mult_cal_one(
                        data[:, this_start:this_stop],
                        one.T,
                        idx,
                        cals,
                        mult,
                    )
            if n_bad:
                warn(
                    f"FIF raw buffer could not be read, acquisition error "
                    f"likely: {n_bad} samples set to zero"
                )
            assert offset == stop - start

    def fix_mag_coil_types(self):
        """Fix Elekta magnetometer coil types.

        Returns
        -------
        raw : instance of Raw
            The raw object. Operates in place.

        Notes
        -----
        This function changes magnetometer coil types 3022 (T1: SQ20483N) and
        3023 (T2: SQ20483-A) to 3024 (T3: SQ20950N) in the channel definition
        records in the info structure.

        Neuromag Vectorview systems can contain magnetometers with two
        different coil sizes (3022 and 3023 vs. 3024). The systems
        incorporating coils of type 3024 were introduced last and are used at
        the majority of MEG sites. At some sites with 3024 magnetometers,
        the data files have still defined the magnetometers to be of type
        3022 to ensure compatibility with older versions of Neuromag software.
        In the MNE software as well as in the present version of Neuromag
        software coil type 3024 is fully supported. Therefore, it is now safe
        to upgrade the data files to use the true coil type.

        .. note:: The effect of the difference between the coil sizes on the
                  current estimates computed by the MNE software is very small.
                  Therefore the use of mne_fix_mag_coil_types is not mandatory.
        """
        fix_mag_coil_types(self.info)
        return self

    @property
    def acqparser(self):
        """The AcqParserFIF for the measurement info.

        See Also
        --------
        mne.AcqParserFIF
        """
        if getattr(self, "_acqparser", None) is None:
            self._acqparser = AcqParserFIF(self.info)
        return self._acqparser


def _check_entry(first, nent):
    """Sanity check entries."""
    if first >= nent:
        raise OSError("Could not read data, perhaps this is a corrupt file")


@fill_doc
def read_raw_fif(
    fname, allow_maxshield=False, preload=False, on_split_missing="raise", verbose=None
) -> Raw:
    """Reader function for Raw FIF data.

    Parameters
    ----------
    fname : path-like | file-like
        The raw filename to load. For files that have automatically been split,
        the split part will be automatically loaded. Filenames should end
        with raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz, raw_tsss.fif,
        raw_tsss.fif.gz, or _meg.fif. If a file-like object is provided,
        preloading must be used.

        .. versionchanged:: 0.18
           Support for file-like objects.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    %(preload)s
    %(on_split_missing)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        A Raw object containing FIF data.

    Notes
    -----
    .. versionadded:: 0.9.0

    When reading a FIF file, note that the first N seconds annotated
    ``BAD_ACQ_SKIP`` are **skipped**. They are removed from ``raw.times`` and
    ``raw.n_times`` parameters but ``raw.first_samp`` and ``raw.first_time``
    are updated accordingly.
    """
    return Raw(
        fname=fname,
        allow_maxshield=allow_maxshield,
        preload=preload,
        verbose=verbose,
        on_split_missing=on_split_missing,
    )


def _path_from_fname(fname) -> Path | None:
    if not isinstance(fname, Path):
        if isinstance(fname, str):
            fname = Path(fname)
        else:
            # Try to get a filename from the file-like object
            try:
                fname = Path(fname.name)
            except Exception:
                fname = None
    return fname
