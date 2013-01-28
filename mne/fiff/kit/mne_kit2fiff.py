# Authors: Teon Brooks  <teon@nyu.edu>

""" Import BTi / 4D MagnesWH3600 data to fif file.

example usage: mne_bti2fiff.py -pdf C,rfDC -o my_raw.fif

"""

from mne.fiff.kit import read_raw_kit
import sys

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-s', '--sqd', dest='sqd_fname',
                    help='Input data file name', metavar='FILE')
    parser.add_option('-c', '--config', dest='config_fname',
                    help='Input config file name', metavar='FILE', default='config')
    parser.add_option('--hsp', dest='head_shape_fname',
                    help='Headshape file name', metavar='FILE',
                    default='hs_file')
    parser.add_option('--elp', dest='elp_fname',
                    help='Headshape file name', metavar='FILE',
                    default='hs_file')
    parser.add_option('--mrk', dest='marker_fname',
                    help='Headshape file name', metavar='FILE',
                    default='hs_file')
    parser.add_option('-o', '--out_fname', dest='out_fname',
                      help='Name of the resulting fiff file',
                      default='as_data_fname')

    options, args = parser.parse_args()

    sqd_fname = options.sqd_fname
    if sqd_fname is None:
        parser.print_help()
        sys.exit(-1)

    config_fname = options.config_fname
    head_shape_fname = options.head_shape_fname
    out_fname = options.out_fname
    rotation_x = options.rotation_x
    translation = options.translation

    if out_fname == 'as_data_fname':
        out_fname = sqd_fname + '_raw.fif'

    raw = read_raw_bti(sqd_fname=sqd_fname, config_fname=config_fname,
                       head_shape_fname=head_shape_fname,
                       rotation_x=rotation_x, translation=translation)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
