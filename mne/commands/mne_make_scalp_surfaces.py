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
import os.path as op
import sys
import mne

if __name__ == '__main__':
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite previously computed surface')
    parser.add_option('-s', '--subject', dest='subject',
                      help='The name of the subject', type='str')
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force transformation of surface into bem.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true',
                      help='Print the debug messages.')

    options, args = parser.parse_args()

    env = os.environ
    subject = vars(options).get('subject', env.get('SUBJECT'))
    if subject is None:
        parser.print_help()
        sys.exit(1)

    overwrite = options.overwrite
    verbose = options.verbose
    force = '--force' if options.force else '--check'

    from mne.commands.utils import get_status_output
    def my_run_cmd(cmd, err_msg):
        sig, out, error = get_status_output(cmd)
        if verbose:
            print(out, error)
        if sig != 0:
            print(err_msg)
            sys.exit(1)

    if not 'SUBJECTS_DIR' in env:
        print('The environment variable SUBJECTS_DIR should be set')
        sys.exit(1)

    if not op.isabs(env['SUBJECTS_DIR']):
        env['SUBJECTS_DIR'] = op.abspath(env['SUBJECTS_DIR'])
    subj_dir = env['SUBJECTS_DIR']

    if not 'MNE_ROOT' in env:
        print('MNE_ROOT environment variable is not set')
        sys.exit(1)

    if not 'FREESURFER_HOME' in env:
        print('The FreeSurfer environment needs to be set up for this script')
        sys.exit(1)

    subj_path = op.join(subj_dir, subject)
    if not op.exists(subj_path):
        print(('%s does not exits. Please check your subject directory '
               'path.' % subj_path))
        sys.exit(1)

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
        cmd = 'mkheadsurf -subjid %s -srcvol %s >/dev/null' % (subject, mri)
        my_run_cmd(cmd, 'mkheadsurf failed')
    else:
        print('%s/surf/%s already there' % (subj_path, my_seghead))
        if not overwrite:
            print('Use the --overwrite option to replace exisiting surfaces.')
            sys.exit()

    surf = check_seghead()
    if surf is None:
        print('mkheadsurf did not produce the standard output file.')
        sys.exit(1)

    fif = '{0}/{1}/bem/{1}-head-dense.fif'.format(subj_dir, subject)
    print('2. Creating %s ...' % fif)
    cmd = 'mne_surf2bem --surf %s --id 4 %s --fif %s' % (surf, force, fif)
    my_run_cmd(cmd, 'Failed to create %s, see above' % fif)
    levels = 'medium', 'sparse'
    for ii, (n_tri, level) in enumerate(zip([30000, 2500], levels), 3):
        my_surf = mne.read_bem_surfaces(fif)[0]
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
        cmd = 'mne_surf2bem --surf %s --id 4 --force --fif %s'
        cmd %= (surf_fname, out_fif)
        my_run_cmd(cmd, 'Failed to create %s, see above' % out_fif)
        os.remove(surf_fname)

    sys.exit(0)
