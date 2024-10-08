"""Show system information.

Examples
--------
.. code-block:: console

    $ mne sys_info

"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys

import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(__file__, usage="mne sys_info")
    parser.add_option(
        "-p",
        "--show-paths",
        dest="show_paths",
        help="Show module paths",
        action="store_true",
    )
    parser.add_option(
        "-d",
        "--developer",
        dest="developer",
        help="Show additional developer module information",
        action="store_true",
    )
    parser.add_option(
        "-a",
        "--ascii",
        dest="unicode",
        help="Use ASCII instead of unicode symbols",
        action="store_false",
        default=True,
    )
    parser.add_option(
        "--no-check-version",
        dest="check_version",
        help="Disable MNE-Python remote version checking.",
        action="store_false",
        default=True,
    )
    options, args = parser.parse_args()
    dependencies = "developer" if options.developer else "user"
    if len(args) != 0:
        parser.print_help()
        sys.exit(1)

    mne.sys_info(
        show_paths=options.show_paths,
        dependencies=dependencies,
        unicode=options.unicode,
        check_version=options.check_version,
    )


mne.utils.run_command_if_main()
