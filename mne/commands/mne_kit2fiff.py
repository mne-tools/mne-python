#!/usr/bin/env python
# Authors: Teon Brooks  <teon.brooks@gmail.com>

"""Import KIT / NYU data to fif file.

Examples
--------
.. code-block:: console

    $ mne kit2fiff --input input.sqd --output output.fif

Use without arguments to invoke GUI:

.. code-block:: console

    $ mne kt2fiff

"""

import sys

import mne
from mne.io import read_raw_kit


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

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
    parser.add_option('--debug', dest='debug', action='store_true',
                      default=False,
                      help='Set logging level for terminal output to debug')

    options, args = parser.parse_args()

    if options.debug:
        mne.set_log_level('debug')

    input_fname = options.input_fname
    if input_fname is None:
        try:
            from mne_kit_gui import kit2fiff  # noqa
        except ImportError:
            raise ImportError(
                'The mne-kit-gui package is required, install it using '
                'conda or pip') from None
        kit2fiff()
        sys.exit(0)

    hsp_fname = options.hsp_fname
    elp_fname = options.elp_fname
    mrk_fname = options.mrk_fname
    stim = options.stim
    slope = options.slope
    stimthresh = options.stimthresh
    out_fname = options.out_fname

    if isinstance(stim, str):
        stim = map(int, stim.split(':'))

    raw = read_raw_kit(input_fname=input_fname, mrk=mrk_fname, elp=elp_fname,
                       hsp=hsp_fname, stim=stim, slope=slope,
                       stimthresh=stimthresh)

    raw.save(out_fname)
    raw.close()


mne.utils.run_command_if_main()
