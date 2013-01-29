#!/usr/bin/env python

# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Yuval Harpaz <yuvharpaz@gmail.com>
#
#          simplified bsd-3 license

"""

Import BTi / 4D MagnesWH3600 data to fif file.

example usage: mne_bti2fiff.py -pdf C,rfDC -o my_raw.fif

Note.
1) Currently direct inclusion of reference channel weights
is not supported. Please use \'mne_create_comp_data\' to include
the weights or use the low level functions from this module to
include them by yourself.
2) The informed guess for the 4D name is E31 for the ECG channel and
E63, E63 for the EOG channels. Pleas check and adjust if those channels
are present in your dataset but 'ECG 01' and 'EOG 01', 'EOG 02' don't
appear in the channel names of the raw object.
"""

from mne.fiff.bti import read_raw_bti
# from mne import verbose
import sys

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-p', '--pdf', dest='pdf_fname',
                    help='Input data file name', metavar='FILE')
    parser.add_option('-c', '--config', dest='config_fname',
                    help='Input config file name', metavar='FILE', default='config')
    parser.add_option('--head_shape', dest='head_shape_fname',
                    help='Headshape file name', metavar='FILE',
                    default='hs_file')
    parser.add_option('-o', '--out_fname', dest='out_fname',
                      help='Name of the resulting fiff file',
                      default='as_data_fname')
    parser.add_option('-r', '--rotation_x', dest='rotation_x', type='float',
                    help='Compensatory rotation about Neuromag x axis, deg',
                    default=2.0)
    parser.add_option('-T', '--translation', dest='translation', type='str',
                    help='Default translation, meter',
                    default=(0.00, 0.02, 0.11))
    parser.add_option('--ecg_ch', dest='ecg_ch', type='str',
                    help='4D ECG channel name',
                    default='E31')
    parser.add_option('--eog_ch', dest='eog_ch', type='str',
                    help='4D EOG channel names',
                    default='E63,E64')

    options, args = parser.parse_args()

    pdf_fname = options.pdf_fname
    if pdf_fname is None:
        parser.print_help()
        sys.exit(-1)

    config_fname = options.config_fname
    head_shape_fname = options.head_shape_fname
    out_fname = options.out_fname
    rotation_x = options.rotation_x
    translation = options.translation
    ecg_ch = options.ecg_ch
    eog_ch = options.ecg_ch.split(',')

    if out_fname == 'as_data_fname':
        out_fname = pdf_fname + '_raw.fif'

    raw = read_raw_bti(pdf_fname=pdf_fname, config_fname=config_fname,
                       head_shape_fname=head_shape_fname,
                       rotation_x=rotation_x, translation=translation,
                       ecg_ch=ecg_ch, eog_ch=eog_ch)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
