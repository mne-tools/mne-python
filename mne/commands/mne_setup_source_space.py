#!/usr/bin/env python
# Authors: Victor Ferat  <victor.ferat@live.fr>
'''Set up bilateral hemisphere surface-based source space with subsampling
 using :func:`mne.setup_source_space`

Examples
--------
.. code-block:: console

    $ mne setup_source_space -s sample -f sources-src.fif

'''

import sys

import mne


def run():
    '''Run command.'''
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option('-s', '--subject',
                      dest='subject',
                      help='Subject name (required)',
                      default=None)
    parser.add_option('-f', dest='fname',
                      help="Output FIF file (if not set, suffix '-src.fif'"
                           ' will be used)',
                      metavar='FILE', default=None)
    parser.add_option('-d', '--subjects-dir',
                      dest='subjects_dir',
                      help='Subjects directory',
                      default=None)
    parser.add_option('--spacing',
                      dest='spacing',
                      help="The spacing to use. Can be 'ico#' "
                           "for a recursively subdivided icosahedron, 'oct#'"
                           " for a recursively subdivided octahedron, 'all'"
                           " for all points, or an integer to use appoximate"
                           " distance-based spacing (in mm)."
                           " Changed in version 0.18: Support for integers"
                           " for distance-based spacing.",
                      default='oct6')
    parser.add_option('--surface',
                      dest='surface',
                      help='The surface to use.',
                      default='white',
                      type='string')
    parser.add_option('-a', '--add-dist',
                      dest='add_dist',
                      help='Add distance and patch information '
                           'to the source space.',
                      default=True)
    parser.add_option('-n', '--n-jobs',
                      dest='n_jobs',
                      help='The number of jobs to run in parallel '
                            '(default 1). Requires the joblib package. '
                            'Will use at most 2 jobs'
                            '(one for each hemisphere).',
                      default=1,
                      type='int')
    parser.add_option('--verbose',
                      dest='verbose',
                      help='Turn on verbose mode.',
                      default=None)
    parser.add_option('-o', '--overwrite',
                      dest='overwrite',
                      help='If True, the destination file (if it exists)'
                            ' will be overwritten. If False (default),'
                            ' an error will be raised if the file exists.',
                      default=False)

    options, args = parser.parse_args()

    if options.subject is None:
        parser.print_help()
        sys.exit(1)

    subject = options.subject
    fname = options.fname
    subjects_dir = options.subjects_dir
    spacing = options.spacing
    surface = options.surface
    add_dist = options.add_dist
    n_jobs = options.n_jobs
    verbose = options.verbose
    overwrite = options.overwrite

    if not (fname.endswith('_src.fif') or fname.endswith('-src.fif')):
        fname = fname + "-src.fif"

    src = mne.setup_source_space(subject=subject, spacing=spacing,
                                 surface=surface, subjects_dir=subjects_dir,
                                 add_dist=add_dist, n_jobs=n_jobs,
                                 verbose=verbose)
    src.save(fname=fname, overwrite=overwrite)

mne.utils.run_command_if_main()
