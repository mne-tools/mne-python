#!/usr/bin/env python

# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
#          simplified bsd-3 license

"""
Create high-resolution head surfaces for coordinate alignment.

example usage: mne make_scalp_surfaces --overwrite --subject sample
"""
from __future__ import print_function

import os
import copy
import os.path as op
import sys
import mne


def run():
    from mne.commands.utils import get_optparser
    from mne.utils import run_subprocess

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')

    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite previously computed surface')
    parser.add_option('-s', '--subject', dest='subject',
                      help='The name of the subject', type='str')
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force transformation of surface into bem.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true',
                      help='Print the debug messages.')
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=subjects_dir)

    options, args = parser.parse_args()

    env = os.environ
    subject = vars(options).get('subject', env.get('SUBJECT'))
    subjects_dir = options.subjects_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    this_env = copy.copy(os.environ)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject

    # overwrite = options.overwrite  # XXX Not actually used anywhere
    # verbose = options.verbose  # XXX Not actually used anywhere
    force = '--force' if options.force else '--check'

    if not 'SUBJECTS_DIR' in env:
        raise RuntimeError('The environment variable SUBJECTS_DIR should '
                           'be set')

    if not op.isdir(subjects_dir):
        raise RuntimeError('subjects directory %s not found, specify using '
                           'the environment variable SUBJECTS_DIR or '
                           'the command line option --subjects-dir')

    if not 'MNE_ROOT' in env:
        raise RuntimeError('MNE_ROOT environment variable is not set')

    if not 'FREESURFER_HOME' in env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')

    subj_path = op.join(subjects_dir, subject)
    if not op.exists(subj_path):
        raise RuntimeError('%s does not exits. Please check your subject '
                           'directory path.' % subj_path)

    if op.exists(op.join(subj_path, 'mri', 'T1.mgz')):
        mri = 'T1.mgz'
    else:
        mri = 'T1'

    print('1. Creating a dense scalp tessellation with mkheadsurf...')

    def check_seghead(surf_path=op.join(subj_path, 'surf')):
        for k in ['/lh.seghead', '/lh.smseghead']:
            surf = surf_path + k if op.exists(surf_path + k) else None
            if surf is not None:
                break
        return surf

    my_seghead = check_seghead()
    if my_seghead is None:
        cmd = ['mkheadsurf', '-subjid', subject, '-srcvol', mri]
        run_subprocess(cmd, env=this_env)
    else:
        print('%s already there' % my_seghead)

    surf = check_seghead()
    if surf is None:
        raise RuntimeError('mkheadsurf did not produce the standard output '
                           'file.')

    fif = '{0}/{1}/bem/{1}-head-dense.fif'.format(subjects_dir, subject)
    print('2. Creating %s ...' % fif)
    cmd = ['mne_surf2bem', '--surf', surf, '--id', '4', force,
           '--fif', fif]
    run_subprocess(cmd, env=this_env)
    levels = 'medium', 'sparse'
    my_surf = mne.read_bem_surfaces(fif)[0]
    tris = [30000, 2500]
    if os.getenv('_MNE_TESTING_SCALP', 'false') == 'true':
        tris = [len(my_surf['tris'])]  # don't actually decimate
    for ii, (n_tri, level) in enumerate(zip(tris, levels), 3):
        print('%i. Creating medium grade tessellation...' % ii)
        print('%i.1 Decimating the dense tessellation...' % ii)
        points, tris = mne.decimate_surface(points=my_surf['rr'],
                                            triangles=my_surf['tris'],
                                            n_triangles=n_tri)
        out_fif = fif.replace('dense', level)
        print('%i.2 Creating %s' % (ii, out_fif))
        surf_fname = '/tmp/tmp-surf.surf'
        # convert points to meters, make mne_analyze happy
        mne.write_surface(surf_fname, points * 1e3, tris)
        # XXX for some reason --check does not work here.
        try:
            cmd = ['mne_surf2bem', '--surf', surf_fname, '--id', '4',
                   '--force', '--fif', out_fif]
            run_subprocess(cmd, env=this_env, stdout=None, stderr=None)
        finally:
            os.remove(surf_fname)

is_main = (__name__ == '__main__')
if is_main:
    run()
