#!/usr/bin/env python
"""View the 3-Layers BEM model using Freeview.

Examples
--------
.. code-block:: console

    $ mne freeview_bem_surfaces -s sample

"""
# Authors:  Alexandre Gramfort <alexandre.gramfort@inria.fr>

import sys
import os
import os.path as op

import mne
from mne.utils import run_subprocess, get_subjects_dir, _check_freesurfer_home


def freeview_bem_surfaces(subject, subjects_dir, method):
    """View 3-Layers BEM model with Freeview.

    Parameters
    ----------
    subject : string
        Subject name
    subjects_dir : string
        Directory containing subjects data (Freesurfer SUBJECTS_DIR)
    method : string
        Can be 'flash' or 'watershed'.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    if subject is None:
        raise ValueError("subject argument is None.")

    subject_dir = op.join(subjects_dir, subject)

    if not op.isdir(subject_dir):
        raise ValueError("Wrong path: '{}'. Check subjects-dir or"
                         "subject argument.".format(subject_dir))

    fs_home = _check_freesurfer_home()
    env = os.environ.copy()
    env.update(SUBJECT=subject, SUBJECTS_DIR=subjects_dir,
               PATH=f'{fs_home}/bin:' + env.get('PATH', ''))

    mri_dir = op.join(subject_dir, 'mri')
    bem_dir = op.join(subject_dir, 'bem')
    mri = op.join(mri_dir, 'T1.mgz')
    cmd = ['freeview', '-v', mri]
    colors = dict(
        outer_skin='255,170,127', outer_skull='yellow', inner_skull='red',
        brain='white')
    for key in ('brain', 'inner_skull', 'outer_skull', 'outer_skin'):
        if method == 'watershed':
            bem_dir = op.join(bem_dir, 'watershed')
            fname = op.join(bem_dir, f'{subject}_{key}_surface')
        else:
            if method == 'flash':
                bem_dir = op.join(bem_dir, 'flash')
            fname = op.join(bem_dir, f'{key}.surf')
        color = colors[key]
        this_cmd = ['--surface',
                    f"{fname}"
                    f":color={color}:edgecolor={color}"]
        cmd += this_cmd if key != 'brain' or op.isfile(fname) else []

    run_subprocess(cmd, env=env, stdout=sys.stdout)


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    subject = os.environ.get('SUBJECT')
    subjects_dir = get_subjects_dir()

    parser.add_option("-s", "--subject", dest="subject",
                      help="Subject name", default=subject)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=subjects_dir)
    parser.add_option("-m", "--method", dest="method",
                      help=("Method used to generate the BEM model. "
                            "Can be flash or watershed."))

    options, args = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir
    method = options.method

    freeview_bem_surfaces(subject, subjects_dir, method)


mne.utils.run_command_if_main()
