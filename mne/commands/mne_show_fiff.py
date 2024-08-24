"""Show the contents of a FIFF file.

Examples
--------
.. code-block:: console

    $ mne show_fiff test_raw.fif


To see only tag 102:

.. code-block:: console

    $ mne show_fiff test_raw.fif --tag=102

"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys

import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(__file__, usage="mne show_fiff <file>")
    parser.add_option(
        "-t",
        "--tag",
        dest="tag",
        help="provide information about this tag",
        metavar="TAG",
    )
    parser.add_option(
        "-b",
        "--bytes",
        dest="show_bytes",
        help="show the byte offset of each tag",
        action="store_true",
    )
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)
    msg = mne.io.show_fiff(
        args[0], tag=options.tag, show_bytes=options.show_bytes
    ).strip()
    print(msg)


mne.utils.run_command_if_main()
