# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

from mne.export._egimff import export_evokeds_mff
from mne.utils import (
    _check_fname,
    _validate_type,
    logger,
    verbose_static,
    warn,
)


@verbose_static(
    "export_fmt_support_raw",
    "export_warning",
    "fname_export_params",
    "export_fmt_params_raw",
    "physical_range_export_params",
    "digital_range_export_params",
    "add_ch_type_export_params",
    "overwrite",
    "export_warning_note_raw",
    "export_eeglab_note",
    "export_edf_note",
)
def export_raw(
    fname,
    raw,
    fmt="auto",
    *,
    physical_range="auto",
    digital_range="auto",
    add_ch_type=False,
    overwrite=False,
    verbose=None,
):
    """Export Raw to external formats.

    Supported formats:

    - BrainVision (``.vhdr``, ``.vmrk``, ``.eeg``, uses `pybv
      <https://github.com/bids-standard/pybv>`_)
    - EEGLAB (``.set``, uses :mod:`eeglabio`)
    - EDF (``.edf``, uses `edfio <https://github.com/the-siesta-group/edfio>`_)

    .. warning::
        Since we are exporting to external formats, there's no guarantee that all
        the info will be preserved in the external format. See Notes for details.

    .. warning::
        When exporting ``Raw`` with annotations, ``raw.info["meas_date"]`` must be the
        same as ``raw.annotations.orig_time``. This guarantees that the annotations are
        in the same reference frame as the samples. When
        :attr:`Raw.first_time <mne.io.Raw.first_time>` is not zero (e.g., after
        cropping), the onsets are automatically corrected so that onsets are always
        relative to the first sample.

    Parameters
    ----------
    fname : str
        Name of the output file.
    raw : instance of Raw
        The raw instance to export.
    fmt : 'auto' | 'brainvision' | 'edf' | 'eeglab'
        Format of the export. Defaults to ``'auto'``, which will infer the format
        from the filename extension. See supported formats above for more
        information.
    physical_range : str | tuple
        The physical range of the data. If 'auto' (default), the physical range is
        inferred from the data: if the data came from and EDF/BDF/GDF file, the
        physical range of the original file will be preserved (with a warning if
        clipping will occur). If the data did not originate from an EDF/BDF/GDF
        file, ``"auto"`` will set the physical range as the minimum and maximum
        values *per channel type*. If ``'channelwise'``, the range will be defined
        *per channel*. If a tuple of minimum and maximum, that manually-specified
        physical range will be used for all channels. Only used for exporting EDF
        files.
    digital_range : "auto" | "orig"
        For EDF/BDF files, this controls the amplitude resolution of the
        signals. "auto" uses the maximum available (16-bit for EDF, 24-bit for BDF).
        If the :class:`~mne.io.Raw` object was originally read from and EDF or BDF
        file, "orig" will use the digital range that was present in that file. For
        :class:`~mne.io.Raw` objects that did not originate from EDF/BDF files,
        "orig" falls back to the behavior of "auto".

        .. versionadded:: 1.13.1
    add_ch_type : bool
        Whether to incorporate the channel type into the signal label (e.g. whether
        to store channel "Fz" as "EEG Fz"). Only used for EDF format. Default is
        ``False``.
    overwrite : bool
        If True (default False), overwrite the destination file if it
        exists.

        .. versionadded:: 0.24.1
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Notes
    -----
    .. versionadded:: 0.24

    Export to external format may not preserve all the information from the
    instance. To save in native MNE format (``.fif``) without information loss,
    use :meth:`mne.io.Raw.save` instead.
    Export does not apply projector(s). Unapplied projector(s) will be lost.
    Consider applying projector(s) before exporting with
    :meth:`mne.io.Raw.apply_proj`.

    For EEGLAB exports, channel locations are expanded to full EEGLAB format.
    For more details see :func:`eeglabio.utils.cart_to_eeglab`.

    Although this function supports storing channel types in the signal label (e.g.
    ``EEG Fz`` or ``MISC E``), other software may not support this (optional)
    feature of the EDF standard.

    If ``add_ch_type`` is True, then channel types are written based on what they
    are currently set in MNE-Python. One should double check that all their channels
    are set correctly. You can call :meth:`mne.io.Raw.set_channel_types` to set
    channel types.

    In addition, EDF does not support storing a montage. You will need to store the
    montage separately and call :meth:`mne.io.Raw.set_montage`.

    The physical range of the signals is determined by signal type by default
    (``physical_range="auto"``). However, if individual channel ranges vary
    significantly due to the presence of e.g. drifts/offsets/biases, setting
    ``physical_range="channelwise"`` might be more appropriate. This will ensure a
    maximum resolution for each individual channel, but some tools might not be able
    to handle this appropriately (even though channel-wise ranges are covered by the
    EDF standard).
    """
    fname = str(_check_fname(fname, overwrite=overwrite))
    supported_export_formats = {  # format : (extensions,)
        "bdf": ("bdf",),
        "brainvision": (
            "eeg",
            "vmrk",
            "vhdr",
        ),
        "edf": ("edf",),
        "eeglab": ("set",),
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    # check for unapplied projectors
    if any(not proj["active"] for proj in raw.info["projs"]):
        warn(
            "Raw instance has unapplied projectors. Consider applying "
            "them before exporting with raw.apply_proj()."
        )

    match fmt:
        case "bdf":
            from mne.export._edf_bdf import _export_raw_bdf

            _export_raw_bdf(fname, raw, physical_range, digital_range, add_ch_type)
        case "brainvision":
            from mne.export._brainvision import _export_raw

            _export_raw(fname, raw, overwrite)
        case "edf":
            from mne.export._edf_bdf import _export_raw_edf

            _export_raw_edf(fname, raw, physical_range, digital_range, add_ch_type)
        case "eeglab":
            from mne.export._eeglab import _export_raw

            _export_raw(fname, raw)


@verbose_static(
    "export_fmt_support_epochs",
    "export_warning",
    "fname_export_params",
    "export_fmt_params_epochs",
    "overwrite",
    "export_warning_note_epochs",
    "export_eeglab_note",
)
def export_epochs(fname, epochs, fmt="auto", *, overwrite=False, verbose=None):
    """Export Epochs to external formats.

    Supported formats:

    - EEGLAB (``.set``, uses :mod:`eeglabio`)

    .. warning::
        Since we are exporting to external formats, there's no guarantee that all
        the info will be preserved in the external format. See Notes for details.

    Parameters
    ----------
    fname : str
        Name of the output file.
    epochs : instance of Epochs
        The epochs to export.
    fmt : 'auto' | 'eeglab'
        Format of the export. Defaults to ``'auto'``, which will infer the format
        from the filename extension. See supported formats above for more
        information.
    overwrite : bool
        If True (default False), overwrite the destination file if it
        exists.

        .. versionadded:: 0.24.1
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Notes
    -----
    .. versionadded:: 0.24

    Export to external format may not preserve all the information from the
    instance. To save in native MNE format (``.fif``) without information loss,
    use :meth:`mne.Epochs.save` instead.
    Export does not apply projector(s). Unapplied projector(s) will be lost.
    Consider applying projector(s) before exporting with
    :meth:`mne.Epochs.apply_proj`.

    For EEGLAB exports, channel locations are expanded to full EEGLAB format.
    For more details see :func:`eeglabio.utils.cart_to_eeglab`.
    """
    fname = str(_check_fname(fname, overwrite=overwrite))
    supported_export_formats = {
        "eeglab": ("set",),
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    # check for unapplied projectors
    if any(not proj["active"] for proj in epochs.info["projs"]):
        warn(
            "Epochs instance has unapplied projectors. Consider applying "
            "them before exporting with epochs.apply_proj()."
        )

    if fmt == "eeglab":
        from mne.export._eeglab import _export_epochs

        _export_epochs(fname, epochs)


@verbose_static(
    "export_fmt_support_evoked",
    "export_warning",
    "fname_export_params",
    "export_fmt_params_evoked",
    "overwrite",
    "export_warning_note_evoked",
)
def export_evokeds(fname, evoked, fmt="auto", *, overwrite=False, verbose=None):
    """Export evoked dataset to external formats.

    This function is a wrapper for format-specific export functions. The export
    function is selected based on the inferred file format. For additional
    options, use the format-specific functions.

    Supported formats:

    - MFF (``.mff``, uses :func:`mne.export.export_evokeds_mff`)

    .. warning::
        Since we are exporting to external formats, there's no guarantee that all
        the info will be preserved in the external format. See Notes for details.

    Parameters
    ----------
    fname : str
        Name of the output file.
    evoked : Evoked instance, or list of Evoked instances
        The evoked dataset, or list of evoked datasets, to export to one file.
        Note that the measurement info from the first evoked instance is used,
        so be sure that information matches.
    fmt : 'auto' | 'mff'
        Format of the export. Defaults to ``'auto'``, which will infer the format
        from the filename extension. See supported formats above for more
        information.
    overwrite : bool
        If True (default False), overwrite the destination file if it
        exists.

        .. versionadded:: 0.24.1
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    mne.write_evokeds
    mne.export.export_evokeds_mff

    Notes
    -----
    .. versionadded:: 0.24

    Export to external format may not preserve all the information from the
    instance. To save in native MNE format (``.fif``) without information loss,
    use :meth:`mne.Evoked.save` instead.
    Export does not apply projector(s). Unapplied projector(s) will be lost.
    Consider applying projector(s) before exporting with
    :meth:`mne.Evoked.apply_proj`.
    """
    fname = str(_check_fname(fname, overwrite=overwrite))
    supported_export_formats = {
        "mff": ("mff",),
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    if not isinstance(evoked, list):
        evoked = [evoked]

    logger.info(f"Exporting evoked dataset to {fname}...")

    if fmt == "mff":
        export_evokeds_mff(fname, evoked, overwrite=overwrite)


def _infer_check_export_fmt(fmt, fname, supported_formats):
    """Infer export format from filename extension if auto.

    Raises error if fmt is auto and no file extension found,
    then checks format against supported formats, raises error if format is not
    supported.

    Parameters
    ----------
    fmt : str
        Format of the export, will only infer the format from filename if fmt
        is auto.
    fname : str
        Name of the target export file, only used when fmt is auto.
    supported_formats : dict of str : tuple/list
        Dictionary containing supported formats (as keys) and each format's
        corresponding file extensions in a tuple (e.g., {'eeglab': ('set',)})
    """
    _validate_type(fmt, str, "fmt")
    fmt = fmt.lower()
    if fmt == "auto":
        fmt = os.path.splitext(fname)[1]
        if fmt:
            fmt = fmt[1:].lower()
            # find fmt in supported formats dict's tuples
            fmt = next(
                (k for k, v in supported_formats.items() if fmt in v), fmt
            )  # default to original fmt for raising error later
        else:
            raise ValueError(
                f"Couldn't infer format from filename {fname} (no extension found)"
            )

    if fmt not in supported_formats:
        supported = []
        for supp_format, extensions in supported_formats.items():
            ext_str = ", ".join(f"*.{ext}" for ext in extensions)
            supported.append(f"{supp_format} ({ext_str})")

        supported_str = ", ".join(supported)
        raise ValueError(
            f"Format '{fmt}' is not supported. Supported formats are {supported_str}."
        )
    return fmt
