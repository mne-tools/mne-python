#!/usr/bin/env python
"""Anonymize .fif file.

You can do for example:

$ mne anonymize sample_audvis_raw.fif
"""

# Authors : Dominik Krzeminski

import sys
import mne


def mne_anonymize(fif_fname):
    """ Call *anonymize_info* on fif file and save.
    Parameters
    ----------
    fif_fname : str
        Raw fif File
    """
    raw = mne.io.read_raw_fif(fif_fname, preload=True)
    mne.io.anonymize_info(raw.info)
    raw.save(fif_fname, overwrite=True)


def run():
    """Run *mne_anonymize* command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    fname = args[0]

    if not fname.endswith('.fif'):
        raise ValueError('%s does not seem to be a .fif file.' % fname)

    mne_anonymize(fname)


is_main = (__name__ == '__main__')
if is_main:
    run()
