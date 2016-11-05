#!/usr/bin/env python
"""Show the contents of a FIFF file.

You can do for example:

$ mne show_fiff test_raw.fif
"""

# Authors : Eric Larson, PhD

import codecs
import sys
import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(
        __file__, usage='mne show_fiff <file>')
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)
    # This works around an annoying bug on Windows for show_fiff, see:
    # https://pythonhosted.org/kitchen/unicode-frustrations.html
    if int(sys.version[0]) < 3:
        UTF8Writer = codecs.getwriter('utf8')
        sys.stdout = UTF8Writer(sys.stdout)
    print(mne.io.show_fiff(args[0]))


is_main = (__name__ == '__main__')
if is_main:
    run()
