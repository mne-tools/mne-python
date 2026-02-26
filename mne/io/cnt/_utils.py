# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import namedtuple
from datetime import datetime
from math import modf
from os import SEEK_END
from struct import Struct

import numpy as np

from ...utils import _check_option, logger, warn

# Offsets from SETUP structure in http://paulbourke.net/dataformats/eeg/
_NCHANNELS_OFFSET = 370
_NSAMPLES_OFFSET = 864
_RATE_OFFSET = 376
_EVENTTABLEPOS_OFFSET = 886
_DATA_OFFSET = 900  # Size of the 'SETUP' header.
_CH_SIZE = 75  # Size of each channel in bytes


def _read_teeg(f, teeg_offset):
    """
    Read TEEG structure from an open CNT file.

    # from TEEG structure in http://paulbourke.net/dataformats/eeg/
    typedef struct {
       char Teeg;       /* Either 1 or 2                    */
       long Size;       /* Total length of all the events   */
       long Offset;     /* Hopefully always 0               */
    } TEEG;
    """
    # we use a more descriptive names based on TEEG doc comments
    Teeg = namedtuple("Teeg", "event_type total_length offset")
    teeg_parser = Struct("<Bll")

    f.seek(teeg_offset)
    return Teeg(*teeg_parser.unpack(f.read(teeg_parser.size)))


CNTEventType1 = namedtuple("CNTEventType1", ("StimType KeyBoard KeyPad_Accept Offset"))
# typedef struct {
#    unsigned short StimType;     /* range 0-65535                           */
#    unsigned char  KeyBoard;     /* range 0-11 corresponding to fcn keys +1 */
#    char KeyPad_Accept;          /* 0->3 range 0-15 bit coded response pad  */
#                                 /* 4->7 values 0xd=Accept 0xc=Reject       */
#    long Offset;                 /* file offset of event                    */
# } EVENT1;


CNTEventType2 = namedtuple(
    "CNTEventType2",
    (
        "StimType KeyBoard KeyPad_Accept Offset Type "
        "Code Latency EpochEvent Accept2 Accuracy"
    ),
)
# unsigned short StimType; /* range 0-65535                           */
# unsigned char  KeyBoard; /* range 0-11 corresponding to fcn keys +1 */
# char KeyPad_Accept;      /* 0->3 range 0-15 bit coded response pad  */
#                          /* 4->7 values 0xd=Accept 0xc=Reject       */
# long Offset;             /* file offset of event                    */
# short Type;
# short Code;
# float Latency;
# char EpochEvent;
# char Accept2;
# char Accuracy;


# needed for backward compat: EVENT type 3 has the same structure as type 2
CNTEventType3 = namedtuple(
    "CNTEventType3",
    (
        "StimType KeyBoard KeyPad_Accept Offset Type "
        "Code Latency EpochEvent Accept2 Accuracy"
    ),
)


def _get_event_parser(event_type):
    if event_type == 1:
        event_maker = CNTEventType1
        struct_pattern = "<HBcl"
    elif event_type == 2:
        event_maker = CNTEventType2
        struct_pattern = "<HBclhhfccc"
    elif event_type == 3:
        event_maker = CNTEventType3
        struct_pattern = "<HBclhhfccc"  # Same as event type 2
    else:
        raise ValueError(f"unknown CNT even type {event_type}")

    def parser(buffer):
        struct = Struct(struct_pattern)
        for chunk in struct.iter_unpack(buffer):
            yield event_maker(*chunk)

    return parser


def _session_date_2_meas_date(session_date, date_format):
    try:
        frac_part, int_part = modf(
            datetime.strptime(session_date, date_format).timestamp()
        )
    except ValueError:
        warn("  Could not parse meas date from the header. Setting to None.")
        return None
    else:
        return (int_part, frac_part)


def _compute_robust_sizes(*, fid, data_format):
    """Compute n_channels, n_samples, n_bytes, and event_table_position.

    When recording event_table_position is computed (as accomulation). If the
    file recording is large then this value overflows and ends up pointing
    somewhere else. (SEE #gh-6535)

    If the file is smaller than 2G the value in the SETUP is returned.
    Otherwise, the address of the table position is computed from:
    n_samples, n_channels, and the bytes size.

    Reference: https://paulbourke.net/dataformats/eeg/
    Header has a field for number of samples, but it does not seem to be
    too reliable.
    """
    _check_option("data_format", data_format, ["auto", "int16", "int32"])
    # Read the number of channels and samples from the header
    fid.seek(_NCHANNELS_OFFSET)
    n_channels = int(np.fromfile(fid, dtype="<u2", count=1).item())
    logger.debug("Number of channels: %d", n_channels)
    fid.seek(_NSAMPLES_OFFSET)
    n_samples = int(np.frombuffer(fid.read(4), dtype="<i4").item())  # may be unreliable
    logger.debug("Header number of samples: %d", n_samples)
    file_size = fid.seek(0, SEEK_END)
    workaround = "pass data_format='int16' or 'int32' explicitly"
    samples_offset = _DATA_OFFSET + _CH_SIZE * n_channels
    if file_size < 2e9:
        # Our most reliable way to get the number of samples is to compute it
        logger.debug("File size < 2GB, using header values")
        fid.seek(_EVENTTABLEPOS_OFFSET)
        event_offset = int(np.frombuffer(fid.read(4), dtype="<i4").item())
        logger.debug("Event table offset from header: %d", event_offset)
        if event_offset > file_size:
            problem = (
                f"Event table offset from header ({event_offset}) is larger than file "
                f"size ({file_size})"
            )
            if data_format == "auto":
                raise RuntimeError(
                    f"{problem}, cannot automatically compute data format, {workaround}"
                )
            warn(
                f"Event table offset from header ({event_offset}) is larger than file "
                f"size ({file_size}), recomputing event table offset."
            )
            n_bytes = 2 if data_format == "int16" else 4
            event_offset = samples_offset + n_samples * n_channels * n_bytes
        n_data_bytes = event_offset - samples_offset
        if data_format == "auto":
            n_bytes_per_samp, rem = divmod(n_data_bytes, n_channels)
            why = ""
            n_bytes = 2
            if rem != 0:
                why = (
                    f"number of data bytes {n_data_bytes} is not evenly divisible by "
                    f"{n_channels=}"
                )
            elif n_samples == 0:
                why = "number of read samples is 0"
            else:
                n_bytes, rem = divmod(n_bytes_per_samp, n_samples)
                if rem != 0 or n_bytes not in [2, 4]:
                    why = (
                        f"number of bytes per sample {n_bytes_per_samp} is not evenly "
                        f"divisible by {n_samples=} or does not result in 2 or 4 bytes "
                        f"per sample ({n_bytes=})"
                    )
                logger.debug("Inferred data format with %d bytes per sample", n_bytes)
            if why:
                raise RuntimeError(
                    "Could not automatically compute number of bytes per sample as the "
                    f"{why}.  set data_format manually."
                )
        else:
            n_bytes = 2 if data_format == "int16" else 4
        logger.debug(
            "Using %d bytes per sample from data_format=%s", n_bytes, data_format
        )
        n_samples, rem = divmod(n_data_bytes, (n_channels * n_bytes))
        logger.debug("Computed number of samples: %d", n_samples)
        if rem != 0:
            warn(
                "Inconsistent file information detected, number of data bytes "
                f"({n_data_bytes}) not evenly divisible by number of channels "
                f"({n_channels}) times number of bytes ({n_bytes})"
            )
    else:
        logger.debug("File size >= 2GB, computing event table offset")
        if data_format == "auto":
            raise RuntimeError(
                "Using `data_format='auto' for a CNT file larger"
                " than 2Gb is not supported, explicitly pass data_format as "
                "'int16' or 'int32'"
            )
        n_bytes = 2 if data_format == "int16" else 4
        event_offset = (
            _DATA_OFFSET + _CH_SIZE * n_channels + n_bytes * n_channels * n_samples
        )
        logger.debug("Computed event table offset: %d", event_offset)

    return n_channels, n_samples, n_bytes, event_offset
