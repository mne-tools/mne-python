# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD (3-clause)

import os.path as op

from ..utils import verbose, _validate_type


@verbose
def export_raw(fname, raw, fmt='auto', verbose=None):
    """Export Raw to external formats.

    Supported formats: EEGLAB (set, uses :mod:`eeglabio`)
    %(export_warning)s

    Parameters
    ----------
    %(export_params_fname)s
    raw : instance of Raw
        The raw instance to export.
    %(export_params_fmt)s
    %(verbose)s

    Notes
    -----
    %(export_eeglab_note)s
    """
    supported_export_formats = {  # format : extensions
        'eeglab': ('set',),
        'edf': ('edf',),
        'brainvision': ('eeg', 'vmrk', 'vhdr',)
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    if fmt == 'eeglab':
        from ._eeglab import _export_raw
        _export_raw(fname, raw)
    elif fmt == 'edf':
        raise NotImplementedError('Export to EDF format not implemented.')
    elif fmt == 'brainvision':
        raise NotImplementedError('Export to BrainVision not implemented.')


@verbose
def export_epochs(fname, epochs, fmt='auto', verbose=None):
    """Export Epochs to external formats.

    Supported formats: EEGLAB (set, uses :mod:`eeglabio`)
    %(export_warning)s

    Parameters
    ----------
    %(export_params_fname)s
    epochs : instance of Epochs
        The epochs to export.
    %(export_params_fmt)s
    %(verbose)s

    Notes
    -----
    %(export_eeglab_note)s
    """
    supported_export_formats = {
        'eeglab': ('set',),
        'edf': ('edf',),
        'brainvision': ('eeg', 'vmrk', 'vhdr',)
    }
    fmt = _infer_check_export_fmt(fmt, fname, supported_export_formats)

    if fmt == 'eeglab':
        from ._eeglab import _export_epochs
        _export_epochs(fname, epochs)
    elif fmt == 'edf':
        raise NotImplementedError('Export to EDF format not implemented.')
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
