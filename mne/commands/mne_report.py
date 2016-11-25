#!/usr/bin/env python
r"""Create mne report for a folder.

Example usage

mne report -p MNE-sample-data/ -i \
MNE-sample-data/MEG/sample/sample_audvis-ave.fif -d MNE-sample-data/subjects/ \
-s sample

"""

import sys
import time

from mne.report import Report
from mne.utils import verbose, logger


@verbose
def log_elapsed(t, verbose=None):
    """Log elapsed time."""
    logger.info('Report complete in %s seconds' % round(t, 1))


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-p", "--path", dest="path",
                      help="Path to folder who MNE-Report must be created")
    parser.add_option("-i", "--info", dest="info_fname",
                      help="File from which info dictionary is to be read",
                      metavar="FILE")
    parser.add_option("-c", "--cov", dest="cov_fname",
                      help="File from which noise covariance is to be read",
                      metavar="FILE")
    parser.add_option("--bmin", dest="bmin",
                      help="Time at which baseline correction starts for "
                      "evokeds", default=None)
    parser.add_option("--bmax", dest="bmax",
                      help="Time at which baseline correction stops for "
                      "evokeds", default=None)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="The subjects directory")
    parser.add_option("-s", "--subject", dest="subject",
                      help="The subject name")
    parser.add_option("-v", "--verbose", dest="verbose",
                      action='store_true', help="run in verbose mode")
    parser.add_option("--no-browser", dest="no_browser", action='store_false',
                      help="Do not open MNE-Report in browser")
    parser.add_option("--overwrite", dest="overwrite", action='store_false',
                      help="Overwrite html report if it already exists")
    parser.add_option("-j", "--jobs", dest="n_jobs", help="Number of jobs to"
                      " run in parallel")
    parser.add_option("-m", "--mri-decim", type="int", dest="mri_decim",
                      default=2, help="Integer factor used to decimate "
                      "BEM plots")

    options, args = parser.parse_args()
    path = options.path
    if path is None:
        parser.print_help()
        sys.exit(1)
    info_fname = options.info_fname
    cov_fname = options.cov_fname
    subjects_dir = options.subjects_dir
    subject = options.subject
    mri_decim = int(options.mri_decim)
    verbose = True if options.verbose is not None else False
    open_browser = False if options.no_browser is not None else True
    overwrite = True if options.overwrite is not None else False
    n_jobs = int(options.n_jobs) if options.n_jobs is not None else 1

    bmin = float(options.bmin) if options.bmin is not None else None
    bmax = float(options.bmax) if options.bmax is not None else None
    # XXX: this means (None, None) cannot be specified through command line
    if bmin is None and bmax is None:
        baseline = None
    else:
        baseline = (bmin, bmax)

    t0 = time.time()
    report = Report(info_fname, subjects_dir=subjects_dir,
                    subject=subject, baseline=baseline,
                    cov_fname=cov_fname, verbose=verbose)
    report.parse_folder(path, verbose=verbose, n_jobs=n_jobs,
                        mri_decim=mri_decim)
    log_elapsed(time.time() - t0, verbose=verbose)
    report.save(open_browser=open_browser, overwrite=overwrite)

is_main = (__name__ == '__main__')
if is_main:
    run()
