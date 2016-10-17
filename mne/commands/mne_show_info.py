#!/usr/bin/env python
"""Show measurement info from .fif file.

You can do for example:

$ mne show_info sample_audvis_raw.fif
"""

# Authors : Alexandre Gramfort, Ph.D.

import sys
import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(
        __file__, usage='mne show_info <file>')
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    fname = args[0]

    if not fname.endswith('.fif'):
        raise ValueError('%s does not seem to be a .fif file.' % fname)

    info = mne.io.read_info(fname)
    print("File : %s" % fname)
    print(info)

is_main = (__name__ == '__main__')
if is_main:
    run()
