#!/usr/bin/env python
"""Show system information.

Examples
--------
.. code-block:: console

    $ mne sys_info

"""

# Authors : Eric Larson <larson.eric.d@gmail.com>

import sys
import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(__file__, usage='mne sys_info')
    parser.add_option('-p', '--show-paths', dest='show_paths',
                      help='Show module paths', action='store_true')
    options, args = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        sys.exit(1)

    mne.sys_info(show_paths=options.show_paths)


mne.utils.run_command_if_main()
