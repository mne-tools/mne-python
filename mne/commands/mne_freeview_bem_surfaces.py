"""View the 3-Layers BEM model using Freeview.

Examples
--------
.. code-block:: console

    $ mne freeview_bem_surfaces -s sample

"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import os.path as op
import sys

import mne
from mne.utils import get_subjects_dir, run_subprocess


def freeview_bem_surfaces(subject, subjects_dir, method=None):
    """View 3-Layers BEM model with Freeview.

    Parameters
    ----------
    subject : str
        Subject name
    subjects_dir : path-like
        Directory containing subjects data (Freesurfer SUBJECTS_DIR)
    method : str | None
        Can be ``'flash'`` or ``'watershed'``, or None to use the ``bem/`` directory
        files.
    """
    subjects_dir = str(get_subjects_dir(subjects_dir, raise_error=True))

    if subject is None:
        raise ValueError("subject argument is None.")

    subject_dir = op.join(subjects_dir, subject)

    if not op.isdir(subject_dir):
        raise ValueError(
            f"Wrong path: '{subject_dir}'. Check subjects-dir or subject argument."
        )

    env = os.environ.copy()
    env["SUBJECT"] = subject
    env["SUBJECTS_DIR"] = subjects_dir

    if "FREESURFER_HOME" not in env:
        raise RuntimeError("The FreeSurfer environment needs to be set up.")

    mri_dir = op.join(subject_dir, "mri")
    bem_dir = op.join(subject_dir, "bem")
    mri = op.join(mri_dir, "T1.mgz")

    if method == "watershed":
        bem_dir = op.join(bem_dir, "watershed")
        outer_skin = op.join(bem_dir, f"{subject}_outer_skin_surface")
        outer_skull = op.join(bem_dir, f"{subject}_outer_skull_surface")
        inner_skull = op.join(bem_dir, f"{subject}_inner_skull_surface")
    else:
        if method == "flash":
            bem_dir = op.join(bem_dir, "flash")
        outer_skin = op.join(bem_dir, "outer_skin.surf")
        outer_skull = op.join(bem_dir, "outer_skull.surf")
        inner_skull = op.join(bem_dir, "inner_skull.surf")

    # put together the command
    cmd = ["freeview"]
    cmd += ["--volume", mri]
    cmd += ["--surface", f"{inner_skull}:color=red:edgecolor=red"]
    cmd += ["--surface", f"{outer_skull}:color=yellow:edgecolor=yellow"]
    cmd += ["--surface", f"{outer_skin}:color=255,170,127:edgecolor=255,170,127"]

    run_subprocess(cmd, env=env, stdout=sys.stdout)
    print("[done]")


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    subject = os.environ.get("SUBJECT")
    parser.add_option(
        "-s", "--subject", dest="subject", help="Subject name", default=subject
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        dest="subjects_dir",
        help="Subjects directory",
    )
    parser.add_option(
        "-m",
        "--method",
        dest="method",
        help="Method used to generate the BEM model. Can be flash or watershed.",
    )

    options, args = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir
    method = options.method

    freeview_bem_surfaces(subject, subjects_dir, method)


mne.utils.run_command_if_main()
