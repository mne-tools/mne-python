#!/usr/bin/env python
# Authors: Teon Brooks  <teon@nyu.edu>

""" Import KIT / NYU data to fif file.

example usage: mne_kit2fiff.py -i input.sqd -o output.fif

"""

from mne.fiff.kit import read_raw_kit
import sys

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input_fname',
                    help='Input data file name', metavar='FILE')
    parser.add_option('--sns', dest='sns_fname',
                    help='Sensor info file name', metavar='FILE')
    parser.add_option('--hsp', dest='hsp_fname',
                    help='Headshape file name', metavar='FILE')
    parser.add_option('--elp', dest='elp_fname',
                    help='Headshape file name', metavar='FILE')
    parser.add_option('--mrk', dest='mrk_fname',
                    help='MEG Marker file name', metavar='FILE')
    parser.add_option('--stimthresh', dest='stimthresh', default=3.5,
                      help='Threshold value for trigger channels',
                      metavar='INT')
    parser.add_option('--stim', dest='stim',
                      default='167:166:165:164:163:162:161:160',
                      help='Stimulus Trigger Channels', metavar='LIST')
    parser.add_option('-o', '--output', dest='out_fname',
                      help='Name of the resulting fiff file', metavar='FILE')

    options, args = parser.parse_args()

    input_fname = options.input_fname
    if input_fname is None:
        parser.print_help()
        sys.exit(-1)

    sns_fname = options.sns_fname
    hsp_fname = options.hsp_fname
    elp_fname = options.elp_fname
    mrk_fname = options.mrk_fname
    stim = options.stim
    stimthresh = options.stimthresh
    out_fname = options.out_fname

    if isinstance(stim, str):
        stim = stim.split(':')

    raw = read_raw_kit(input_fname=input_fname, sns_fname=sns_fname,
                       hsp_fname=hsp_fname, elp_fname=elp_fname,
                       mrk_fname=mrk_fname, stim=stim, stimthresh=stimthresh)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
