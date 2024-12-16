# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Create a BEM solution using the linear collocation approach.

Examples
--------
.. code-block:: console

    $ mne prepare_bem_model --bem sample-5120-5120-5120-bem.fif

"""

import os
import sys

import mne


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "--bem",
        dest="bem_fname",
        help="The name of the file containing the "
        "triangulations of the BEM surfaces and the "
        "conductivities of the compartments. The standard "
        "ending for this file is -bem.fif.",
        metavar="FILE",
    )
    parser.add_option(
        "--sol",
        dest="bem_sol_fname",
        help="The name of the resulting file containing BEM "
        "solution (geometry matrix). It uses the linear "
        "collocation approach. The file should end with "
        "-bem-sof.fif.",
        metavar="FILE",
        default=None,
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()
    bem_fname = options.bem_fname
    bem_sol_fname = options.bem_sol_fname
    verbose = True if options.verbose is not None else False

    if bem_fname is None:
        parser.print_help()
        sys.exit(1)

    if bem_sol_fname is None:
        base, _ = os.path.splitext(bem_fname)
        bem_sol_fname = base + "-sol.fif"

    bem_model = mne.read_bem_surfaces(bem_fname, patch_stats=False, verbose=verbose)
    bem_solution = mne.make_bem_solution(bem_model, verbose=verbose)
    mne.write_bem_solution(bem_sol_fname, bem_solution)


mne.utils.run_command_if_main()
