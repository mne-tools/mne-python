#!/usr/bin/env python

# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Yuval Harpaz <yuvharpaz@gmail.com>
#
#          simplified bsd-3 license

""" Import BTi / 4-D MagnesWH3600 data to fif file. """

from mne.fiff.bti import read_raw_bti
from mne import verbose
import sys

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-p', '--pdf', dest='pdf_fname',
                    help='Input data file ', metavar='FILE')
    parser.add_option('-c', '--config', dest='config_fname',
                    help='Input data file ', metavar='FILE', default='config')
    parser.add_option('--head_shape', dest='head_shape_fname',
                    help='Headshape file name', metavar='FILE',
                    default='hs_file')
    parser.add_option('-o', '--out_fname', dest='out_fname',
                      default='as_data_fname')
    parser.add_option('-r', '--rotation_x', dest='rotation_x', type='float',
                    help='Compensatory rotation about Neuromag x axis, deg',
                    default=2.0)
    parser.add_option('-T', '--translation', dest='translation', type='str',
                    help='Default translation, meter',
                    default=(0.00, 0.02, 0.11))
    parser.add_option('-v', '--verbose', dest='verbose',
                    help='Print processing steps to command line',
                    default=True)

    options, args = parser.parse_args()

    pdf_fname = options.pdf_fname
    config_fname = options.config_fname
    head_shape_fname = options.head_shape_fname
    out_fname = options.out_fname
    rotation_x = options.rotation_x
    translation = options.translation
    level = options.verbose

    if out_fname == 'as_data_fname' and pdf_fname is not None:
        out_fname = pdf_fname + '_raw.fif'

    raw = read_raw_bti(pdf_fname=pdf_fname, config_fname=config_fname,
                       head_shape_fname=head_shape_fname,
                       rotation_x=rotation_x, translation=translation,
                       verbose=level)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
