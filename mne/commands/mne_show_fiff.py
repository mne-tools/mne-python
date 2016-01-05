#!/usr/bin/env python
"""Show the contents of a FIFF file

You can do for example:

$ mne show_fiff test_raw.fif
"""

# Authors : Eric Larson, PhD

import sys
import mne


def run():
    parser = mne.commands.utils.get_optparser(
        __file__, usage='mne show_fiff <file>')
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)
    print(mne.io.show_fiff(args[0]))


is_main = (__name__ == '__main__')
if is_main:
    run()
