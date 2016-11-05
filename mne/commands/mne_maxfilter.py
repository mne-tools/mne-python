#!/usr/bin/env python
"""Apply MaxFilter.

Example usage:

$ mne maxfilter -i sample_audvis_raw.fif --st

This will apply MaxFilter with the MaxSt extension. The origin used
by MaxFilter is computed by mne-python by fitting a sphere to the
headshape points.
"""

# Authors : Martin Luessi <mluessi@nmr.mgh.harvard.edu>

import sys
import os
import mne


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-i", "--in", dest="in_fname",
                      help="Input raw FIF file", metavar="FILE")
    parser.add_option("-o", dest="out_fname",
                      help="Output FIF file (if not set, suffix  '_sss' will "
                      "be used)", metavar="FILE", default=None)
    parser.add_option("--origin", dest="origin",
                      help="Head origin in mm, or a filename to read the "
                      "origin from. If not set it will be estimated from "
                      "headshape points", default=None)
    parser.add_option("--origin-out", dest="origin_out",
                      help="Filename to use for computed origin", default=None)
    parser.add_option("--frame", dest="frame", type="string",
                      help="Coordinate frame for head center ('device' or "
                      "'head')", default="device")
    parser.add_option("--bad", dest="bad", type="string",
                      help="List of static bad channels",
                      default=None)
    parser.add_option("--autobad", dest="autobad", type="string",
                      help="Set automated bad channel detection ('on', 'off', "
                      "'n')", default="off")
    parser.add_option("--skip", dest="skip",
                      help="Skips raw data sequences, time intervals pairs in "
                      "sec, e.g.: 0 30 120 150", default=None)
    parser.add_option("--force", dest="force", action="store_true",
                      help="Ignore program warnings",
                      default=False)
    parser.add_option("--st", dest="st", action="store_true",
                      help="Apply the time-domain MaxST extension",
                      default=False)
    parser.add_option("--buflen", dest="st_buflen", type="float",
                      help="MaxSt buffer length in sec",
                      default=16.0)
    parser.add_option("--corr", dest="st_corr", type="float",
                      help="MaxSt subspace correlation",
                      default=0.96)
    parser.add_option("--trans", dest="mv_trans",
                      help="Transforms the data into the coil definitions of "
                      "in_fname, or into the default frame", default=None)
    parser.add_option("--movecomp", dest="mv_comp", action="store_true",
                      help="Estimates and compensates head movements in "
                      "continuous raw data", default=False)
    parser.add_option("--headpos", dest="mv_headpos", action="store_true",
                      help="Estimates and stores head position parameters, "
                      "but does not compensate movements", default=False)
    parser.add_option("--hp", dest="mv_hp", type="string",
                      help="Stores head position data in an ascii file",
                      default=None)
    parser.add_option("--hpistep", dest="mv_hpistep", type="float",
                      help="Sets head position update interval in ms",
                      default=None)
    parser.add_option("--hpisubt", dest="mv_hpisubt", type="string",
                      help="Subtracts hpi signals: sine amplitudes, amp + "
                      "baseline, or switch off", default=None)
    parser.add_option("--nohpicons", dest="mv_hpicons", action="store_false",
                      help="Do not check initial consistency isotrak vs "
                      "hpifit", default=True)
    parser.add_option("--linefreq", dest="linefreq", type="float",
                      help="Sets the basic line interference frequency (50 or "
                      "60 Hz)", default=None)
    parser.add_option("--nooverwrite", dest="overwrite", action="store_false",
                      help="Do not overwrite output file if it already exists",
                      default=True)
    parser.add_option("--args", dest="mx_args", type="string",
                      help="Additional command line arguments to pass to "
                      "MaxFilter", default="")

    options, args = parser.parse_args()

    in_fname = options.in_fname

    if in_fname is None:
        parser.print_help()
        sys.exit(1)

    out_fname = options.out_fname
    origin = options.origin
    origin_out = options.origin_out
    frame = options.frame
    bad = options.bad
    autobad = options.autobad
    skip = options.skip
    force = options.force
    st = options.st
    st_buflen = options.st_buflen
    st_corr = options.st_corr
    mv_trans = options.mv_trans
    mv_comp = options.mv_comp
    mv_headpos = options.mv_headpos
    mv_hp = options.mv_hp
    mv_hpistep = options.mv_hpistep
    mv_hpisubt = options.mv_hpisubt
    mv_hpicons = options.mv_hpicons
    linefreq = options.linefreq
    overwrite = options.overwrite
    mx_args = options.mx_args

    if in_fname.endswith('_raw.fif') or in_fname.endswith('-raw.fif'):
        prefix = in_fname[:-8]
    else:
        prefix = in_fname[:-4]

    if out_fname is None:
        if st:
            out_fname = prefix + '_tsss.fif'
        else:
            out_fname = prefix + '_sss.fif'

    if origin is not None and os.path.exists(origin):
        with open(origin, 'r') as fid:
            origin = fid.readlines()[0].strip()

    origin = mne.preprocessing.apply_maxfilter(
        in_fname, out_fname, origin, frame,
        bad, autobad, skip, force, st, st_buflen, st_corr, mv_trans,
        mv_comp, mv_headpos, mv_hp, mv_hpistep, mv_hpisubt, mv_hpicons,
        linefreq, mx_args, overwrite)

    if origin_out is not None:
        with open(origin_out, 'w') as fid:
            fid.write(origin + '\n')

is_main = (__name__ == '__main__')
if is_main:
    run()
