#!/usr/bin/env python
"""Show the contents of a FIFF file.

Examples
--------
.. code-block:: console

    $ mne show_fiff test_raw.fif


To see only tag 102:

.. code-block:: console

    $ mne show_fiff test_raw.fif --tag=102

"""

# Authors : Eric Larson, PhD

import codecs
import sys
import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(
        __file__, usage='mne show_fiff <file>')
    parser.add_option("-t", "--tag", dest="tag",
                      help="provide information about this tag", metavar="TAG")
    options, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(1)
    if sys.platform == "win32" and int(sys.version[0]) < 3:
        # This works around an annoying bug on Windows for show_fiff, see:
        # https://pythonhosted.org/kitchen/unicode-frustrations.html
        UTF8Writer = codecs.getwriter('utf8')
        sys.stdout = UTF8Writer(sys.stdout)
    msg = mne.io.show_fiff(args[0], tag=options.tag).strip()
    print(msg)


is_main = (__name__ == '__main__')
if is_main:
    run()
