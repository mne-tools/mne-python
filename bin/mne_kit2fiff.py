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
    parser.add_option('--mrk', dest='marker_fname',
                    help='MEG Marker file name', metavar='FILE')
    parser.add_option('--stimthresh', dest='stimtresh',
                      help='Threshold value for trigger channels')
    parser.add_option('--stim', dest='stim',
                      help='Stimulus Trigger Channels') 
    parser.add_option('-a', '--add', dest='add_chs',
                      help='Add additional channels')                                            
    parser.add_option('-o', '--output', dest='out_fname',
                      help='Name of the resulting fiff file')
                     

    options, args = parser.parse_args()

    sqd_fname = options.sqd_fname
    if sqd_fname is None:
        parser.print_help()
        sys.exit(-1)

    sns_fname = options.sns_fname
    hsp_fname = options.hsp_fname
    elp_fname = options.elp_fname
    mrk_fname = options.mrk_fname
    stim = options.stim
    stimthresh = options.stimtresh
    add_chs = options.add_chs
    out_fname = options.out_fname


#     if out_fname == 'as_data_fname':
#         out_fname = sqd_fname + '_raw.fif'

    raw = read_raw_kit(input_fname=input_fname, sns_fname=sns_fname,
                       hsp_fname=hsp_fname, elp_fname=elp_fname,
                       mrk_fname=mrk_fname, stim=stim, stimthresh=stimtresh,
                       add_chs=add_chs)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
