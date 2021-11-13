# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import os.path as op

from ._egimff import export_evokeds_mff
from ..utils import logger, verbose, warn, _check_fname, _validate_type


@verbose
def export_raw(fname, raw, fmt='auto', physical_range='auto',
               add_ch_type=False, *, overwrite=False, verbose=None):
    """Export Raw to external formats.

    Supported formats:
        - EEGLAB (.set, uses :mod:`eeglabio`)
        - EDF (.edf, uses ``EDFlib-Python``)

    %(export_warning)s

    Parameters
    ----------
    %(export_params_fname)s
    raw : instance of Raw
        The raw instance to export.
    %(export_params_fmt)s
    %(export_params_physical_range)s
    %(export_params_add_ch_type)s
    %(overwrite)s

        .. versionadded:: 0.24.1
    %(verbose)s

    Notes
    -----
    .. versionadded:: 0.24

    %(export_warning_note_raw)s
    %(export_eeglab_note)s
    %(export_edf_note)s
    """
    fname = _check_fname(fname, overwrite=overwrite)
    supported_export_formats = {  # format : extensions
        'eeglab': ('set',),
        'edf': ('edf',),
        'brainvision': ('eeg', 'vmrk', 'vhdr',)
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    # check for unapplied projectors
    if any(not proj['active'] for proj in raw.info['projs']):
        warn('Raw instance has unapplied projectors. Consider applying '
             'them before exporting with raw.apply_proj().')

    if fmt == 'eeglab':
        from ._eeglab import _export_raw
        _export_raw(fname, raw)
    elif fmt == 'edf':
        from ._edf import _export_raw
        _export_raw(fname, raw, physical_range, add_ch_type)
    elif fmt == 'brainvision':
        raise NotImplementedError('Export to BrainVision not implemented.')


@verbose
def export_epochs(fname, epochs, fmt='auto', *, overwrite=False, verbose=None):
    """Export Epochs to external formats.

    Supported formats: EEGLAB (set, uses :mod:`eeglabio`)

    %(export_warning)s

    Parameters
    ----------
    %(export_params_fname)s
    epochs : instance of Epochs
        The epochs to export.
    %(export_params_fmt)s
    %(overwrite)s

        .. versionadded:: 0.24.1
    %(verbose)s

    Notes
    -----
    .. versionadded:: 0.24

    %(export_warning_note_epochs)s
    %(export_eeglab_note)s
    """
    fname = _check_fname(fname, overwrite=overwrite)
    supported_export_formats = {
        'eeglab': ('set',),
        'edf': ('edf',),
        'brainvision': ('eeg', 'vmrk', 'vhdr',)
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    # check for unapplied projectors
    if any(not proj['active'] for proj in epochs.info['projs']):
        warn('Epochs instance has unapplied projectors. Consider applying '
             'them before exporting with epochs.apply_proj().')

    if fmt == 'eeglab':
        from ._eeglab import _export_epochs
        _export_epochs(fname, epochs)
    elif fmt == 'edf':
        raise NotImplementedError('Export to EDF format not implemented.')
    elif fmt == 'brainvision':
        raise NotImplementedError('Export to BrainVision not implemented.')


@verbose
def export_evokeds(fname, evoked, fmt='auto', *, overwrite=False,
                   verbose=None):
    """Export evoked dataset to external formats.

    This function is a wrapper for format-specific export functions. The export
    function is selected based on the inferred file format. For additional
    options, use the format-specific functions.

    Supported formats
        MFF (mff, uses :func:`mne.export.export_evokeds_mff`)

    %(export_warning)s

    Parameters
    ----------
    %(export_params_fname)s
    evoked : Evoked instance, or list of Evoked instances
        The evoked dataset, or list of evoked datasets, to export to one file.
        Note that the measurement info from the first evoked instance is used,
        so be sure that information matches.
    fmt : 'auto' | 'mff'
        Format of the export. Defaults to ``'auto'``, which will infer the
        format from the filename extension. See supported formats above for
        more information.
    %(overwrite)s

        .. versionadded:: 0.24.1
    %(verbose)s

    See Also
    --------
    mne.write_evokeds
    mne.export.export_evokeds_mff

    Notes
    -----
    .. versionadded:: 0.24

    %(export_warning_note_evoked)s
    """
    fname = _check_fname(fname, overwrite=overwrite)
    supported_export_formats = {
        'mff': ('mff',),
        'eeglab': ('set',),
        'edf': ('edf',),
        'brainvision': ('eeg', 'vmrk', 'vhdr',)
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    if not isinstance(evoked, list):
        evoked = [evoked]

    logger.info(f'Exporting evoked dataset to {fname}...')

    if fmt == 'mff':
        export_evokeds_mff(fname, evoked, overwrite=overwrite)
    elif fmt == 'eeglab':
        raise NotImplementedError('Export to EEGLAB not implemented.')
    elif fmt == 'edf':
        raise NotImplementedError('Export to EDF not implemented.')
    elif fmt == 'brainvision':
        raise NotImplementedError('Export to BrainVision not implemented.')


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
        corresponding file extensions in a tuple/list (e.g. 'eeglab': ('set',))
    """
    _validate_type(fmt, str, 'fmt')
    fmt = fmt.lower()
    if fmt == "auto":
        fmt = op.splitext(fname)[1]
        if fmt:
            fmt = fmt[1:].lower()
            # find fmt in supported formats dict's tuples
            fmt = next((k for k, v in supported_formats.items() if fmt in v),
                       fmt)  # default to original fmt for raising error later
        else:
            raise ValueError(f"Couldn't infer format from filename {fname}"
                             " (no extension found)")

    if fmt not in supported_formats:
        supported = []
        for format, extensions in supported_formats.items():
            ext_str = ', '.join(f'*.{ext}' for ext in extensions)
            supported.append(f'{format} ({ext_str})')

        supported_str = ', '.join(supported)
        raise ValueError(f"Format '{fmt}' is not supported. "
                         f"Supported formats are {supported_str}.")
    return fmt
