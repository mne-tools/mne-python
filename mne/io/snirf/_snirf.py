# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
import re

import numpy as np

from ..._fiff._digitization import _make_dig_points
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _format_dig_points, create_info
from ..._fiff.utils import _mult_cal_one
from ..._freesurfer import get_mni_fiducials
from ...annotations import Annotations
from ...transforms import _frame_to_str, apply_trans
from ...utils import (
    NamedInt,
    _check_fname,
    _check_option,
    _import_h5py,
    _validate_type,
    fill_doc,
    logger,
    verbose,
    warn,
)
from ..base import BaseRaw
from ..nirx.nirx import _convert_fnirs_to_head

SNIRF_CW_AMPLITUDE = NamedInt("SNIRF_CW_AMPLITUDE", 1)
SNIRF_TD_GATED_AMPLITUDE = NamedInt("SNIRF_TD_GATED_AMPLITUDE", 201)
SNIRF_TD_MOMENTS_AMPLITUDE = NamedInt("SNIRF_TD_MOMENTS_AMPLITUDE", 301)
SNIRF_PROCESSED = NamedInt("SNIRF_PROCESSED", 99999)
_AVAILABLE_SNIRF_DATA_TYPES = (
    SNIRF_CW_AMPLITUDE,
    SNIRF_TD_GATED_AMPLITUDE,
    SNIRF_TD_MOMENTS_AMPLITUDE,
    SNIRF_PROCESSED,
)


# SNIRF: Supported measurementList(k).dataTypeLabel values in dataTimeSeries
FNIRS_SNIRF_DATATYPELABELS = {
    # These types are specified here:
    # https://github.com/fNIRS/snirf/blob/master/snirf_specification.md#supported-measurementlistkdatatypelabel-values-in-datatimeseries  # noqa: E501
    "HbO": 1,  # Oxygenated hemoglobin (oxyhemoglobin) concentration
    "HbR": 2,  # Deoxygenated hemoglobin (deoxyhemoglobin) concentration
    "HbT": 3,  # Total hemoglobin concentration
    "dOD": 4,  # Change in optical density
    "mua": 5,  # Absorption coefficient
    "musp": 6,  # Scattering coefficient
    "H2O": 7,  # Water content
    "Lipid": 8,  # Lipid concentration
    "BFi": 9,  # Blood flow index
    "HRF dOD": 10,  # HRF for change in optical density
    "HRF HbO": 11,  # HRF for oxyhemoglobin concentration
    "HRF HbR": 12,  # HRF for deoxyhemoglobin concentration
    "HRF HbT": 13,  # HRF for total hemoglobin concentration
    "HRF BFi": 14,  # HRF for blood flow index
}

# In each file, the TD moment order maps to these values
_TD_MOMENT_ORDER_MAP = {
    0: "intensity",
    1: "mean",
    2: "variance",
}


@fill_doc
def read_raw_snirf(
    fname, optode_frame="unknown", *, sfreq=None, preload=False, verbose=None
) -> "RawSNIRF":
    """Reader for a continuous wave SNIRF data.

    .. note:: This reader supports the .snirf file type only,
              not the .jnirs version.
              Files with either 3D or 2D locations can be read.
              However, we strongly recommend using 3D positions.
              If 2D positions are used the behaviour of MNE functions
              can not be guaranteed.

    Parameters
    ----------
    fname : path-like
        Path to the SNIRF data file.
    optode_frame : str
        Coordinate frame used for the optode positions. The default is unknown,
        in which case the positions are not modified. If a known coordinate
        frame is provided (head, meg, mri), then the positions are transformed
        in to the Neuromag head coordinate frame (head).
    sfreq : float | None
        The nominal sampling frequency at which the data were acquired. If ``None``,
        will be estimated from the time data in the file.

        .. versionadded:: 1.10
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawSNIRF
        A Raw object containing fNIRS data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawSNIRF.
    """
    return RawSNIRF(fname, optode_frame, sfreq=sfreq, preload=preload, verbose=verbose)


def _open(fname):
    return open(fname, encoding="latin-1")


@fill_doc
class RawSNIRF(BaseRaw):
    """Raw object from a continuous wave SNIRF file.

    Parameters
    ----------
    fname : path-like
        Path to the SNIRF data file.
    optode_frame : str
        Coordinate frame used for the optode positions. The default is unknown,
        in which case the positions are not modified. If a known coordinate
        frame is provided (head, meg, mri), then the positions are transformed
        in to the Neuromag head coordinate frame (head).
    sfreq : float | None
        The nominal sampling frequency at which the data were acquired. If ``None``,
        will be estimated from the time data in the file.

        .. versionadded:: 1.10
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(
        self, fname, optode_frame="unknown", *, sfreq=None, preload=False, verbose=None
    ):
        # Must be here due to circular import error
        from ...preprocessing.nirs import _validate_nirs_info

        h5py = _import_h5py()

        fname = str(_check_fname(fname, "read", True, "fname"))
        logger.info(f"Loading {fname}")

        with h5py.File(fname, "r") as dat:
            if "data2" in dat["nirs"]:
                warn(
                    "File contains multiple recordings. "
                    "MNE does not support this feature. "
                    "Only the first dataset will be processed."
                )

            manufacturer = _get_metadata_str(dat, "ManufacturerName")
            if (optode_frame == "unknown") & (manufacturer == "Gowerlabs"):
                optode_frame = "head"

            snirf_data_type = _correct_shape(
                np.array(dat.get("nirs/data1/measurementList1/dataType"))
            )[0]
            _check_option(
                "SNIRF data type",
                snirf_data_type,
                list(_AVAILABLE_SNIRF_DATA_TYPES),
            )

            last_samps = dat.get("/nirs/data1/dataTimeSeries").shape[0] - 1

            sampling_rate = _extract_sampling_rate(dat, sfreq)

            if sampling_rate == 0:
                warn("Unable to extract sample rate from SNIRF file.")

            # Extract wavelengths
            fnirs_wavelengths = np.array(dat.get("nirs/probe/wavelengths"))
            fnirs_wavelengths = [int(w) for w in fnirs_wavelengths]
            if len(fnirs_wavelengths) != 2:
                raise RuntimeError(
                    f"The data contains "
                    f"{len(fnirs_wavelengths)}"
                    f" wavelengths: {fnirs_wavelengths}. "
                    f"MNE only supports reading continuous"
                    " wave amplitude SNIRF files "
                    "with two wavelengths."
                )

            # Get data type specific probe information
            if snirf_data_type == SNIRF_TD_GATED_AMPLITUDE:
                fnirs_time_delays = np.array(dat.get("nirs/probe/timeDelays"), float)
                fnirs_time_delay_widths = np.array(
                    dat.get("nirs/probe/timeDelayWidths"), float
                )
            elif snirf_data_type == SNIRF_TD_MOMENTS_AMPLITUDE:
                fnirs_moment_orders = np.array(dat.get("nirs/probe/momentOrders"), int)

            # Extract channels
            def atoi(text):
                return int(text) if text.isdigit() else text

            def natural_keys(text):
                return [atoi(c) for c in re.split(r"(\d+)", text)]

            channels = np.array([name for name in dat["nirs"]["data1"].keys()])
            channels_idx = np.array(["measurementList" in n for n in channels])
            channels = channels[channels_idx]
            channels = sorted(channels, key=natural_keys)

            # Source and detector labels are optional fields.
            # Use S1, S2, S3, etc if not specified.
            if "sourceLabels_disabled" in dat["nirs/probe"]:
                # This is disabled as
                # MNE-Python does not currently support custom source names.
                # Instead, sources must be integer values.
                sources = np.array(dat.get("nirs/probe/sourceLabels"))
                sources = [s.decode("UTF-8") for s in sources]
            else:
                sources = np.unique(
                    [
                        _correct_shape(
                            np.array(dat.get(f"nirs/data1/{c}/sourceIndex"))
                        )[0]
                        for c in channels
                    ]
                )
                sources = {int(s): f"S{int(s)}" for s in sources}

            if "detectorLabels_disabled" in dat["nirs/probe"]:
                # This is disabled as
                # MNE-Python does not currently support custom detector names.
                # Instead, detector must be integer values.
                detectors = np.array(dat.get("nirs/probe/detectorLabels"))
                detectors = [d.decode("UTF-8") for d in detectors]
            else:
                detectors = np.unique(
                    [
                        _correct_shape(
                            np.array(dat.get(f"nirs/data1/{c}/detectorIndex"))
                        )[0]
                        for c in channels
                    ]
                )
                detectors = {int(d): f"D{int(d)}" for d in detectors}

            # Extract source and detector locations
            # 3D positions are optional in SNIRF,
            # but highly recommended in MNE.
            if ("detectorPos3D" in dat["nirs/probe"]) & (
                "sourcePos3D" in dat["nirs/probe"]
            ):
                # If 3D positions are available they are used even if 2D exists
                detPos3D = np.array(dat.get("nirs/probe/detectorPos3D"))
                srcPos3D = np.array(dat.get("nirs/probe/sourcePos3D"))
            elif ("detectorPos2D" in dat["nirs/probe"]) & (
                "sourcePos2D" in dat["nirs/probe"]
            ):
                warn(
                    "The data only contains 2D location information for the "
                    "optode positions. "
                    "It is highly recommended that data is used "
                    "which contains 3D location information for the "
                    "optode positions. With only 2D locations it can not be "
                    "guaranteed that MNE functions will behave correctly "
                    "and produce accurate results. If it is not possible to "
                    "include 3D positions in your data, please consider "
                    "using the set_montage() function."
                )

                detPos2D = np.array(dat.get("nirs/probe/detectorPos2D"))
                srcPos2D = np.array(dat.get("nirs/probe/sourcePos2D"))
                # Set the third dimension to zero. See gh#9308
                detPos3D = np.append(detPos2D, np.zeros((detPos2D.shape[0], 1)), axis=1)
                srcPos3D = np.append(srcPos2D, np.zeros((srcPos2D.shape[0], 1)), axis=1)

            else:
                raise RuntimeError(
                    "No optode location information is "
                    "provided. MNE requires at least 2D "
                    "location information"
                )

            # Uniform scale factor assumed here!
            snirf_data_unit = np.array(
                dat.get("nirs/data1/measurementList1/dataUnit", b"M")
            )
            snirf_data_unit = snirf_data_unit.item().decode("utf-8")
            scale = _get_dataunit_scaling(snirf_data_unit)

            chnames = []
            ch_types = []
            ch_cals = []
            for chan in channels:
                ch_root = f"nirs/data1/{chan}"
                src_idx = int(
                    _correct_shape(np.array(dat.get(f"{ch_root}/sourceIndex")))[0]
                )
                det_idx = int(
                    _correct_shape(np.array(dat.get(f"{ch_root}/detectorIndex")))[0]
                )
                ch_name = f"{sources[src_idx]}_{detectors[det_idx]}"
                ch_cal = scale

                if snirf_data_type in (
                    SNIRF_CW_AMPLITUDE,
                    SNIRF_TD_GATED_AMPLITUDE,
                    SNIRF_TD_MOMENTS_AMPLITUDE,
                ):
                    wve_idx = int(
                        _correct_shape(np.array(dat.get(f"{ch_root}/wavelengthIndex")))[
                            0
                        ]
                    )
                    # append wavelength
                    ch_name = f"{ch_name} {fnirs_wavelengths[wve_idx - 1]}"
                    if snirf_data_type == SNIRF_CW_AMPLITUDE:
                        ch_type = "fnirs_cw_amplitude"
                    elif snirf_data_type == SNIRF_TD_GATED_AMPLITUDE:
                        bin_idx = int(
                            _correct_shape(
                                np.array(dat.get(f"{ch_root}/dataTypeIndex"))
                            )[0]
                        )
                        # append time delay
                        ch_name = f"{ch_name} bin{fnirs_time_delays[bin_idx - 1]}"
                        ch_type = "fnirs_td_gated_amplitude"
                    else:
                        assert snirf_data_type == SNIRF_TD_MOMENTS_AMPLITUDE
                        moment_idx = int(
                            _correct_shape(
                                np.array(dat.get(f"{ch_root}/dataTypeIndex"))
                            )[0]
                        )
                        # append moment order
                        order = fnirs_moment_orders[moment_idx - 1]
                        _check_option(
                            f"SNIRF channel {chan} moment order",
                            order,
                            _TD_MOMENT_ORDER_MAP,
                        )
                        ch_name = f"{ch_name} moment{order}"
                        kind = _TD_MOMENT_ORDER_MAP[order]
                        ch_type = f"fnirs_td_moments_{kind}"
                        if kind == "mean":
                            # Stored in picoseconds
                            ch_cal = 1e-12
                        elif kind == "variance":
                            ch_cal = 1e-24
                elif snirf_data_type == SNIRF_PROCESSED:
                    dt_id = _correct_shape(
                        np.array(dat.get(f"{ch_root}/dataTypeLabel"))
                    )[0].decode("UTF-8")

                    # Convert between SNIRF processed names and MNE type names
                    dt_id = dt_id.lower().replace("dod", "fnirs_od")

                    if dt_id == "fnirs_od":
                        wve_idx = int(
                            _correct_shape(
                                np.array(dat.get(f"{ch_root}/wavelengthIndex"))
                            )[0]
                        )
                        suffix = str(fnirs_wavelengths[wve_idx - 1])
                    else:
                        if dt_id not in ("hbo", "hbr"):
                            raise RuntimeError(
                                "read_raw_snirf can only handle processed "
                                "data in the form of optical density or "
                                f"HbO/HbR, but got type f{dt_id}"
                            )
                        suffix = dt_id.lower()
                    ch_name = f"{ch_name} {suffix}"
                    ch_type = dt_id
                chnames.append(ch_name)
                ch_types.append(ch_type)
                ch_cals.append(ch_cal)
                del ch_root, ch_name, ch_type, ch_cal
            del scale

            # Create mne structure
            info = create_info(chnames, sampling_rate, ch_types=ch_types)
            for ch, ch_cal in zip(info["chs"], ch_cals):
                ch["cal"] = ch_cal

            subject_info = {}
            names = np.array(dat.get("nirs/metaDataTags/SubjectID"))
            names = _correct_shape(names)[0].decode("UTF-8")
            subject_info["his_id"] = names
            # Read non standard (but allowed) custom metadata tags
            if "lastName" in dat.get("nirs/metaDataTags/"):
                ln = dat.get("/nirs/metaDataTags/lastName")[0].decode("UTF-8")
                subject_info["last_name"] = ln
            if "middleName" in dat.get("nirs/metaDataTags/"):
                m = dat.get("/nirs/metaDataTags/middleName")[0].decode("UTF-8")
                subject_info["middle_name"] = m
            if "firstName" in dat.get("nirs/metaDataTags/"):
                fn = dat.get("/nirs/metaDataTags/firstName")[0].decode("UTF-8")
                subject_info["first_name"] = fn
            else:
                # MNE < 1.7 used to not write the firstName tag, so pull it from names
                subject_info["first_name"] = names.split("_")[0]
            if "sex" in dat.get("nirs/metaDataTags/"):
                s = dat.get("/nirs/metaDataTags/sex")[0].decode("UTF-8")
                if s in {"M", "Male", "1", "m"}:
                    subject_info["sex"] = FIFF.FIFFV_SUBJ_SEX_MALE
                elif s in {"F", "Female", "2", "f"}:
                    subject_info["sex"] = FIFF.FIFFV_SUBJ_SEX_FEMALE
                elif s in {"0", "u"}:
                    subject_info["sex"] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
            # End non standard name reading
            # Update info
            info.update(subject_info=subject_info)

            length_unit = _get_metadata_str(dat, "LengthUnit")
            length_scaling = _get_lengthunit_scaling(length_unit)

            srcPos3D *= length_scaling
            detPos3D *= length_scaling

            if optode_frame in ["mri", "meg"]:
                # These are all in MNI or MEG coordinates, so let's transform
                # them to the Neuromag head coordinate frame
                srcPos3D, detPos3D, _, head_t = _convert_fnirs_to_head(
                    "fsaverage", optode_frame, "head", srcPos3D, detPos3D, []
                )
            else:
                head_t = np.eye(4)

            if optode_frame in ["head", "mri", "meg"]:
                # Then the transformation to head was performed above
                coord_frame = FIFF.FIFFV_COORD_HEAD
            elif "MNE_coordFrame" in dat.get("nirs/metaDataTags/"):
                coord_frame = int(dat.get("/nirs/metaDataTags/MNE_coordFrame")[0])
            else:
                coord_frame = FIFF.FIFFV_COORD_UNKNOWN

            for idx, chan in enumerate(channels):
                ch_root = f"nirs/data1/{chan}"
                src_idx = int(
                    _correct_shape(np.array(dat.get(f"{ch_root}/sourceIndex")))[0]
                )
                det_idx = int(
                    _correct_shape(np.array(dat.get(f"{ch_root}/detectorIndex")))[0]
                )

                info["chs"][idx]["loc"][3:6] = srcPos3D[src_idx - 1, :]
                info["chs"][idx]["loc"][6:9] = detPos3D[det_idx - 1, :]
                # Store channel as mid point
                midpoint = (
                    info["chs"][idx]["loc"][3:6] + info["chs"][idx]["loc"][6:9]
                ) / 2
                info["chs"][idx]["loc"][0:3] = midpoint
                info["chs"][idx]["coord_frame"] = coord_frame

                # get data type specific info:
                wve_idx = int(
                    _correct_shape(
                        np.array(dat.get(f"{ch_root}/wavelengthIndex", [1]))
                    )[0]
                )
                if snirf_data_type == SNIRF_CW_AMPLITUDE or (
                    snirf_data_type == SNIRF_PROCESSED and ch_types[idx] == "fnirs_od"
                ):
                    info["chs"][idx]["loc"][9] = fnirs_wavelengths[wve_idx - 1]
                elif snirf_data_type in (
                    SNIRF_TD_GATED_AMPLITUDE,
                    SNIRF_TD_MOMENTS_AMPLITUDE,
                ):
                    info["chs"][idx]["loc"][9] = fnirs_wavelengths[wve_idx - 1]
                    if snirf_data_type == SNIRF_TD_GATED_AMPLITUDE:
                        bin_idx = int(
                            _correct_shape(
                                np.array(dat.get(f"{ch_root}/dataTypeIndex"))
                            )[0]
                        )
                        info["chs"][idx]["loc"][10] = (
                            fnirs_time_delays[bin_idx - 1]
                            * fnirs_time_delay_widths[bin_idx - 1]
                        )
                    else:
                        assert snirf_data_type == SNIRF_TD_MOMENTS_AMPLITUDE
                        moment_idx = int(
                            _correct_shape(
                                np.array(dat.get(f"{ch_root}/dataTypeIndex"))
                            )[0]
                        )
                        info["chs"][idx]["loc"][10] = fnirs_moment_orders[
                            moment_idx - 1
                        ]
                elif snirf_data_type == SNIRF_PROCESSED:
                    hb_id = (
                        np.array(dat.get(f"{ch_root}/dataTypeLabel"))
                        .item()
                        .decode("UTF-8")
                    )
                    info["chs"][idx]["loc"][9] = FNIRS_SNIRF_DATATYPELABELS[hb_id]

            if "landmarkPos3D" in dat.get("nirs/probe/"):
                diglocs = np.array(dat.get("/nirs/probe/landmarkPos3D"))
                diglocs /= length_scaling
                digname = np.array(dat.get("/nirs/probe/landmarkLabels"))
                nasion, lpa, rpa, hpi = None, None, None, None
                extra_ps = dict()
                for idx, dign in enumerate(digname):
                    dign = dign.lower()
                    if dign in [b"lpa", b"al"]:
                        lpa = diglocs[idx, :3]
                    elif dign in [b"nasion"]:
                        nasion = diglocs[idx, :3]
                    elif dign in [b"rpa", b"ar"]:
                        rpa = diglocs[idx, :3]
                    else:
                        extra_ps[f"EEG{len(extra_ps) + 1:03d}"] = diglocs[idx, :3]
                add_missing_fiducials = (
                    coord_frame == FIFF.FIFFV_COORD_HEAD
                    and lpa is None
                    and rpa is None
                    and nasion is None
                )
                dig = _make_dig_points(
                    nasion=nasion,
                    lpa=lpa,
                    rpa=rpa,
                    hpi=hpi,
                    dig_ch_pos=extra_ps,
                    coord_frame=_frame_to_str[coord_frame],
                    add_missing_fiducials=add_missing_fiducials,
                )
            else:
                ch_locs = [info["chs"][idx]["loc"][0:3] for idx in range(len(channels))]
                # Set up digitization
                dig = get_mni_fiducials("fsaverage", verbose=False)
                for fid in dig:
                    fid["r"] = apply_trans(head_t, fid["r"])
                    fid["coord_frame"] = FIFF.FIFFV_COORD_HEAD
                for ii, ch_loc in enumerate(ch_locs, 1):
                    dig.append(
                        dict(
                            kind=FIFF.FIFFV_POINT_EEG,  # misnomer prob okay
                            r=ch_loc,
                            ident=ii,
                            coord_frame=FIFF.FIFFV_COORD_HEAD,
                        )
                    )
                dig = _format_dig_points(dig)
                del head_t
            with info._unlock():
                info["dig"] = dig

            str_date = _correct_shape(
                np.array(dat.get("/nirs/metaDataTags/MeasurementDate"))
            )[0].decode("UTF-8")
            str_time = _correct_shape(
                np.array(dat.get("/nirs/metaDataTags/MeasurementTime"))
            )[0].decode("UTF-8")
            str_datetime = str_date + str_time

            # Several formats have been observed so we try each in turn
            for dt_code in [
                "%Y-%m-%d%H:%M:%SZ",
                "%Y-%m-%d%H:%M:%S",
                "%Y-%m-%d%H:%M:%S.%f",
                "%Y-%m-%d%H:%M:%S.%f%z",
            ]:
                try:
                    meas_date = datetime.datetime.strptime(str_datetime, dt_code)
                except ValueError:
                    pass
                else:
                    break
            else:
                warn(
                    "Extraction of measurement date from SNIRF file failed. "
                    "The date is being set to January 1st, 2000, "
                    f"instead of {str_datetime}"
                )
                meas_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
            meas_date = meas_date.replace(tzinfo=datetime.timezone.utc)
            with info._unlock():
                info["meas_date"] = meas_date

            if "DateOfBirth" in dat.get("nirs/metaDataTags/"):
                str_birth = (
                    np.array(dat.get("/nirs/metaDataTags/DateOfBirth")).item().decode()
                )
                birth_matched = re.fullmatch(r"(\d+)-(\d+)-(\d+)", str_birth)
                if birth_matched is not None:
                    birthday = datetime.date(
                        int(birth_matched.groups()[0]),
                        int(birth_matched.groups()[1]),
                        int(birth_matched.groups()[2]),
                    )
                    with info._unlock():
                        info["subject_info"]["birthday"] = birthday

            super().__init__(
                info,
                preload,
                filenames=[fname],
                last_samps=[last_samps],
                verbose=verbose,
            )

            # Extract annotations
            # As described at https://github.com/fNIRS/snirf/
            # blob/master/snirf_specification.md#nirsistimjdata
            annot = Annotations([], [], [])
            for key in dat["nirs"]:
                if "stim" in key:
                    data = np.atleast_2d(np.array(dat.get(f"/nirs/{key}/data")))
                    if data.shape[1] >= 3:
                        desc = _correct_shape(np.array(dat.get(f"/nirs/{key}/name")))[0]
                        annot.append(data[:, 0], data[:, 1], desc.decode("UTF-8"))
            self.set_annotations(annot, emit_warning=False)

        # Validate that the fNIRS info is correctly formatted
        _validate_nirs_info(self.info)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        import h5py

        with h5py.File(self.filenames[0], "r") as dat:
            one = dat["/nirs/data1/dataTimeSeries"][start:stop].T

        _mult_cal_one(data, one, idx, cals, mult)


# Helper function for when the numpy array has shape (), i.e. just one element.
def _correct_shape(arr):
    if arr.shape == ():
        arr = arr[np.newaxis]
    return arr


def _get_timeunit_scaling(time_unit):
    """MNE expects time in seconds, return required scaling."""
    scalings = {"ms": 1000, "s": 1, "unknown": 1}
    if time_unit in scalings:
        return scalings[time_unit]
    else:
        raise RuntimeError(
            f"The time unit {time_unit} is not supported by "
            "MNE. Please report this error as a GitHub "
            "issue to inform the developers."
        )


def _get_lengthunit_scaling(length_unit):
    """MNE expects distance in m, return required scaling."""
    scalings = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    if length_unit in scalings:
        return scalings[length_unit]
    else:
        raise RuntimeError(
            f"The length unit {length_unit} is not supported "
            "by MNE. Please report this error as a GitHub "
            "issue to inform the developers."
        )


def _get_dataunit_scaling(hbx_unit):
    """MNE expects hbo/hbr in M, return required scaling."""
    scalings = {"M": 1.0, "uM": 1e-6, "": 1.0}
    try:
        return scalings[hbx_unit]
    except KeyError:
        raise RuntimeError(
            f"The Hb unit {repr(hbx_unit)} is not supported "
            "by MNE. Please report this error as a GitHub "
            "issue to inform the developers."
        ) from None


def _extract_sampling_rate(dat, user_sfreq):
    """Extract the sample rate from the time field."""
    # This is a workaround to provide support for Artinis data.
    # It allows for a 1% variation in the sampling times relative
    # to the average sampling rate of the file.
    MAXIMUM_ALLOWED_SAMPLING_JITTER_PERCENTAGE = 1.0

    _validate_type(user_sfreq, ("numeric", None), "sfreq")
    time_data = np.array(dat.get("nirs/data1/time"))
    time_unit = _get_metadata_str(dat, "TimeUnit")
    time_unit_scaling = _get_timeunit_scaling(time_unit)  # always 1 (s) or 1000 (ms)
    if len(time_data) == 2:  # special-cased in the snirf standard as (onset, period)
        onset, period = time_data
        file_sfreq = time_unit_scaling / period
    else:
        onset = time_data[0]
        periods = np.diff(time_data)
        sfreqs = time_unit_scaling / periods
        file_sfreq = sfreqs.mean()  # our best estimate, likely including some jitter
    if user_sfreq is not None:
        logger.info(f"Setting sampling frequency to user-supplied value: {user_sfreq}")
        if not np.allclose(file_sfreq, user_sfreq, rtol=0.01, atol=0):
            warn(
                f"User-supplied sampling frequency ({user_sfreq} Hz) differs by "
                f"{(user_sfreq - file_sfreq) / file_sfreq:.1%} from the frequency "
                f"estimated from data in the file ({file_sfreq} Hz)."
            )
    sfreq = user_sfreq or file_sfreq  # user-passed value overrides value from file
    # estimate jitter
    if len(time_data) > 2:
        ideal_times = onset + np.arange(len(time_data)) / sfreq
        max_jitter = np.max(np.abs(time_data - ideal_times))
        percent_jitter = 100.0 * max_jitter / periods.mean()
        msg = f"Found jitter of {percent_jitter:3f}% in sample times."
        if percent_jitter > MAXIMUM_ALLOWED_SAMPLING_JITTER_PERCENTAGE:
            warn(
                f"{msg} Note that MNE-Python does not currently support SNIRF "
                "files with non-uniformly-sampled data."
            )
        else:
            logger.info(msg)
    return sfreq


def _get_metadata_str(dat, field):
    if field not in np.array(dat.get("nirs/metaDataTags")):
        return None
    data = dat.get(f"/nirs/metaDataTags/{field}")
    data = _correct_shape(np.array(data))
    data = str(data[0], "utf-8")
    return data
