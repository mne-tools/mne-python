# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Create high-resolution head surfaces for coordinate alignment.

Examples
--------
.. code-block:: console

    $ mne make_scalp_surfaces --overwrite --subject sample

"""

import os
import sys

import mne
from mne.bem import make_scalp_surfaces


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config("SUBJECTS_DIR")

    parser.add_option(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite previously computed surface",
    )
    parser.add_option(
        "-s", "--subject", dest="subject", help="The name of the subject", type="str"
    )
    parser.add_option(
        "-m",
        "--mri",
        dest="mri",
        type="str",
        default="T1.mgz",
        help="The MRI file to process using mkheadsurf.",
    )
    parser.add_option(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help="Force creation of the surface even if it has some topological defects.",
    )
    parser.add_option(
        "-t",
        "--threshold",
        dest="threshold",
        type="int",
        default=20,
        help="Threshold value to use with the MRI.",
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        dest="subjects_dir",
        help="Subjects directory",
        default=subjects_dir,
    )
    parser.add_option(
        "-n",
        "--no-decimate",
        dest="no_decimate",
        help="Disable medium and sparse decimations (dense only)",
        action="store_true",
    )
    _add_verbose_flag(parser)
    options, args = parser.parse_args()

    subject = vars(options).get("subject", os.getenv("SUBJECT"))
    subjects_dir = options.subjects_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    make_scalp_surfaces(
        subject=subject,
        subjects_dir=subjects_dir,
        force=options.force,
        overwrite=options.overwrite,
        no_decimate=options.no_decimate,
        threshold=options.threshold,
        mri=options.mri,
        verbose=options.verbose,
    )


mne.utils.run_command_if_main()
