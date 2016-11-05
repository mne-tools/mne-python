#!/usr/bin/env python
"""View the 3-Layers BEM model using Freeview.

You can do for example:

$ mne freeview_bem_surfaces -s sample
"""
from __future__ import print_function

# Authors:  Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import sys
import os
import os.path as op

from mne.utils import run_subprocess, get_subjects_dir


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

    env = os.environ.copy()
    env['SUBJECT'] = subject
    env['SUBJECTS_DIR'] = subjects_dir

    if 'FREESURFER_HOME' not in env:
        raise RuntimeError('The FreeSurfer environment needs to be set up.')

    mri_dir = op.join(subjects_dir, subject, 'mri')
    bem_dir = op.join(subjects_dir, subject, 'bem')
    mri = op.join(mri_dir, 'T1.mgz')

    if method == 'watershed':
        bem_dir = op.join(bem_dir, 'watershed')
        outer_skin = op.join(bem_dir, '%s_outer_skin_surface' % subject)
        outer_skull = op.join(bem_dir, '%s_outer_skull_surface' % subject)
        inner_skull = op.join(bem_dir, '%s_inner_skull_surface' % subject)
    else:
        if method == 'flash':
            bem_dir = op.join(bem_dir, 'flash')
        outer_skin = op.join(bem_dir, 'outer_skin.surf')
        outer_skull = op.join(bem_dir, 'outer_skull.surf')
        inner_skull = op.join(bem_dir, 'inner_skull.surf')

    # put together the command
    cmd = ['freeview']
    cmd += ["--volume", mri]
    cmd += ["--surface", "%s:color=red:edgecolor=red" % inner_skull]
    cmd += ["--surface", "%s:color=yellow:edgecolor=yellow" % outer_skull]
    cmd += ["--surface",
            "%s:color=255,170,127:edgecolor=255,170,127" % outer_skin]

    run_subprocess(cmd, env=env, stdout=sys.stdout)
    print("[done]")


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


is_main = (__name__ == '__main__')
if is_main:
    run()
