#!/usr/bin/env python
# Authors: Teon Brooks  <teon@nyu.edu>

""" Import KIT / NYU data to fif file.

example usage: mne_kit2fiff.py --input input.sqd --output output.fif

"""

import os
import sys
from mne.fiff.kit import read_raw_kit

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('--input', dest='input_fname',
                      help='Input data file name', metavar='filename')
    parser.add_option('--mrk', dest='mrk_fname',
                      help='MEG Marker file name', metavar='filename')
    parser.add_option('--elp', dest='elp_fname',
                      help='Headshape points file name', metavar='filename')
    parser.add_option('--hsp', dest='hsp_fname',
                      help='Headshape file name', metavar='filename')
    parser.add_option('--stim', dest='stim',
                      help='Colon Separated Stimulus Trigger Channels',
                      metavar='chs')
    parser.add_option('--slope', dest='slope', help='Slope direction',
                      metavar='slope')
    parser.add_option('--stimthresh', dest='stimthresh', default=1,
                      help='Threshold value for trigger channels',
                      metavar='value')
    parser.add_option('--output', dest='out_fname',
                      help='Name of the resulting fiff file',
                      metavar='filename')
    parser.add_option("--version", dest="version", action="store_true",
                      help="Return script version",
                      default=False)

    options, args = parser.parse_args()

    if options.version:
        print "%s %s" % (os.path.basename(__file__), mne.__version__)
        sys.exit(0)

    input_fname = options.input_fname
    if input_fname is None:
        parser.print_help()
        sys.exit(-1)

    hsp_fname = options.hsp_fname
    elp_fname = options.elp_fname
    mrk_fname = options.mrk_fname
    stim = options.stim
    slope = options.slope
    stimthresh = options.stimthresh
    out_fname = options.out_fname

    if isinstance(stim, str):
        stim = stim.split(':')

    raw = read_raw_kit(input_fname=input_fname, mrk=mrk_fname, elp=elp_fname,
                       hsp=hsp_fname, stim=stim, slope=slope,
                       stimthresh=stimthresh)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
