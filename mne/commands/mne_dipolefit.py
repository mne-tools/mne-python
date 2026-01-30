# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Open the dipole fitting GUI on the given evoked file ("-ave.fif").

Examples
--------
.. code-block:: console

    $ mne dipolefit

"""

import os.path as op
import sys

import mne


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__, usage="mne dipolefit EVOKED_FILE")

    parser.add_option(
        "--condition",
        default=0,
        help="The condition to use.",
    )
    parser.add_option(
        "--baseline",
        default=None,
        metavar="TIME-START,TIME-END",
        help=(
            "The time period to use as baseline, written as two numbers (in seconds, "
            "relative to the stimulus onset) separated by a comma. "
            "For example: --baseline=-0.2,0.1"
        ),
    )
    parser.add_option(
        "-c",
        "--cov",
        default=None,
        metavar="COV_FILE",
        help='The noise covariance ("-cov.fif") to use.',
    )
    parser.add_option(
        "-b",
        "--bem",
        default=None,
        metavar="BEM_FILE",
        help='The BEM model ("-bem-sol.fif") to use.',
    )
    parser.add_option(
        "-t",
        "--initial-time",
        default=None,
        type=float,
        metavar="TIME",
        help="The initial time to show",
    )
    parser.add_option(
        "--trans",
        default=None,
        metavar="TRANS_FILE",
        help='Head<->MRI transform FIF file ("-trans.fif")',
    )
    parser.add_option(
        "--stc",
        default=None,
        metavar="STC_FILE",
        help="An optional distributed source estimate to show during dipole fitting.",
    )
    parser.add_option(
        "-s", "--subject", dest="subject", default=None, help="Subject name"
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        default=None,
        help="Subjects directory",
    )
    parser.add_option(
        "--hide-density",
        action="store_true",
        default=False,
        help="Prevent showing the magnetic field density as blobs of color.",
    )
    parser.add_option(
        "--channel-type",
        default=None,
        help=(
            'Restrict channel types to either "meg" or "eeg". By default both are used '
            "if present."
        ),
    )
    parser.add_option(
        "-j", "--n-jobs", default=-1, type=int, help="Number of CPUs to use."
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    # expanduser allows ~ for paths
    subjects_dir = options.subjects_dir
    if subjects_dir is not None:
        subjects_dir = op.expanduser(subjects_dir)
    bem = options.bem
    if bem is not None:
        bem = op.expanduser(bem)
    trans = options.trans
    if trans is not None:
        trans = op.expanduser(trans)
    stc = options.stc
    if stc is not None:
        stc = op.expanduser(stc)

    # Condition can be specified as integer index or string comment.
    if options.condition is not None:
        try:
            condition = int(options.condition)
        except ValueError:
            condition = options.condition
    else:
        condition = None

    # Parse the baseline time period
    baseline = None
    if options.baseline:
        try:
            baseline = [float(x) for x in options.baseline.split(",")]
            if len(baseline) != 2:
                raise ValueError()
        except ValueError:
            raise ValueError(
                "The 'baseline' parameter should be written as two numbers (in seconds,"
                " relative to the stimulus onset) separated by a comma. "
                "For example: --baseline=-0.2,0.1"
            )

    mne.gui.dipolefit(
        evoked=args[0],
        condition=condition,
        baseline=baseline,
        cov=options.cov,
        bem=bem,
        subject=options.subject,
        subjects_dir=subjects_dir,
        stc=stc,
        ch_type=options.channel_type,
        initial_time=options.initial_time,
        trans=trans,
        n_jobs=options.n_jobs,
        show_density=not options.hide_density,
        show=True,
        block=True,
        verbose=options.verbose,
    )


mne.utils.run_command_if_main()
