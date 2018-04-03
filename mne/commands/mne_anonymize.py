#!/usr/bin/env python
"""Anonymize .fif file.

You can do for example:

$ mne anonymize sample_audvis_raw.fif
"""

# Authors : Dominik Krzeminski

import sys
import mne
import os

ANONYMIZE_FILE_PREFIX = 'anon'


def mne_anonymize(fif_fname, out_fname, overwrite=False):
    """Call *anonymize_info* on fif file and save.

    Parameters
    ----------
    fif_fname : str
        Raw fif File
    out_fname : str
        Output file name
    overwrite : bool
        Overwrite output file if it already exists
    """
    dir_name = os.path.split(fif_fname)[0]
    fif_fname = os.path.basename(fif_fname)

    raw = mne.io.read_raw_fif(os.path.join(dir_name, fif_fname), preload=True)
    mne.io.anonymize_info(raw.info)

    if out_fname is not None:
        out = out_fname
    else:
        out = "{}-{}".format(ANONYMIZE_FILE_PREFIX, fif_fname)
    raw.save(os.path.join(dir_name, out), overwrite=overwrite)


def run():
    """Run *mne_anonymize* command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-f", "--file", type="string", dest="file",
                      help="the file to modify.", metavar="FILE",
                      default=None)
    parser.add_option("-o", "--output", type="string", dest="output",
                      help="name of anonymized output file. "
                      "`anon-` prefix is added to FILE if not given",
                      metavar="OUTFILE", default=None)
    parser.add_option("--overwrite", dest="overwrite", action="store_true",
                      help="overwrite input file", default=False)

    options, args = parser.parse_args()
    if options.file is None:
        parser.print_help()
        sys.exit(1)

    fname = options.file
    out_fname = options.output
    overwrite = options.overwrite

    if not fname.endswith('.fif'):
        raise ValueError('%s does not seem to be a .fif file.' % fname)

    mne_anonymize(fname, out_fname, overwrite)


is_main = (__name__ == '__main__')
if is_main:
    run()
