#!/usr/bin/env python
# Authors : Dominik Krzeminski
#           Luke Bloy <luke.bloy@gmail.com>

"""Anonymize raw fif file.

To anonymize other file types call :func:`mne.io.anonymize_info` on their
`info` objects and resave to disk.

Examples
--------
.. code-block:: console

    $ mne anonymize -f sample_audvis_raw.fif

"""

import sys
import mne
import os.path as op

ANONYMIZE_FILE_PREFIX = 'anon'


def mne_anonymize(fif_fname, out_fname, keep_his, daysback, overwrite):
    """Call *anonymize_info* on fif file and save.

    Parameters
    ----------
    fif_fname : str
        Raw fif File
    out_fname : str | None
        Output file name
        relative paths are saved relative to parent dir of fif_fname
        None will save to parent dir of fif_fname with default prefix
    daysback : int | None
        Number of days to subtract from all dates.
        If None will default to move date of service to Jan 1 2000
    keep_his : bool
        If True his_id of subject_info will NOT be overwritten.
        defaults to False
    overwrite : bool
        Overwrite output file if it already exists
    """
    raw = mne.io.read_raw_fif(fif_fname, allow_maxshield=True)
    raw.anonymize(daysback=daysback, keep_his=keep_his)

    # determine out_fname
    dir_name = op.split(fif_fname)[0]
    if out_fname is None:
        fif_bname = op.basename(fif_fname)
        out_fname = op.join(dir_name,
                            "{}-{}".format(ANONYMIZE_FILE_PREFIX, fif_bname))
    elif not op.isabs(out_fname):
        out_fname = op.join(dir_name, out_fname)

    raw.save(out_fname, overwrite=overwrite)


def run():
    """Run *mne_anonymize* command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-f", "--file", type="string", dest="file",
                      help="Name of file to modify.", metavar="FILE",
                      default=None)
    parser.add_option("-o", "--output", type="string", dest="output",
                      help="Name of anonymized output file."
                      "`anon-` prefix is added to FILE if not given",
                      metavar="OUTFILE", default=None)
    parser.add_option("--keep_his", dest="keep_his", action="store_true",
                      help="Keep the HIS tag (not advised)", default=False)
    parser.add_option("-d", "--daysback", type="int", dest="daysback",
                      help="Move dates in file backwards by this many days.",
                      metavar="N_DAYS", default=None)
    parser.add_option("--overwrite", dest="overwrite", action="store_true",
                      help="Overwrite input file.", default=False)

    options, args = parser.parse_args()
    if options.file is None:
        parser.print_help()
        sys.exit(1)

    fname = options.file
    out_fname = options.output
    keep_his = options.keep_his
    daysback = options.daysback
    overwrite = options.overwrite
    if not fname.endswith('.fif'):
        raise ValueError('%s does not seem to be a .fif file.' % fname)

    mne_anonymize(fname, out_fname, keep_his, daysback, overwrite)


is_main = (__name__ == '__main__')
if is_main:
    run()
