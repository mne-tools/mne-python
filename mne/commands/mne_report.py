# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

r"""Create mne report for a folder.

Examples
--------
Before getting started with ``mne report``, make sure the files you want to
render follow the filename conventions defined by MNE:

.. highlight:: console

.. cssclass:: table-bordered
.. rst-class:: midvalign

============ ==============================================================
Data object  Filename convention (ends with)
============ ==============================================================
raw          -raw.fif(.gz), -raw_sss.fif(.gz), -raw_tsss.fif(.gz),
             _meg.fif(.gz), _eeg.fif(.gz), _ieeg.fif(.gz)
events       -eve.fif(.gz)
epochs       -epo.fif(.gz)
evoked       -ave.fif(.gz)
covariance   -cov.fif(.gz)
trans        -trans.fif(.gz)
forward      -fwd.fif(.gz)
inverse      -inv.fif(.gz)
============ ==============================================================

To generate a barebones report from all the \*.fif files in the sample
dataset, invoke the following command in a system (e.g., Bash) shell::

    $ mne report --path MNE-sample-data/ --verbose

On successful creation of the report, it will open the HTML in a new tab in
the browser. To disable this, use the ``--no-browser`` option.

TO generate a report for a single subject, give the ``SUBJECT`` name and
the ``SUBJECTS_DIR`` and this will generate the MRI slices (with BEM
contours overlaid on top if available)::

    $ mne report --path MNE-sample-data/ --subject sample --subjects-dir \
        MNE-sample-data/subjects --verbose

To properly render ``trans`` and ``covariance`` files, add the measurement
information::

    $ mne report --path MNE-sample-data/ \
        --info MNE-sample-data/MEG/sample/sample_audvis-ave.fif \
        --subject sample --subjects-dir MNE-sample-data/subjects --verbose

To render whitened ``evoked`` files with baseline correction, add the noise
covariance file::

    $ mne report --path MNE-sample-data/ \
        --info MNE-sample-data/MEG/sample/sample_audvis-ave.fif \
        --cov MNE-sample-data/MEG/sample/sample_audvis-cov.fif --bmax 0 \
        --subject sample --subjects-dir MNE-sample-data/subjects --verbose

To generate the report in parallel::

    $ mne report --path MNE-sample-data/ \
        --info MNE-sample-data/MEG/sample/sample_audvis-ave.fif \
        --subject sample --subjects-dir MNE-sample-data/subjects \
        --verbose --jobs 6

For help on all the available options, do::

    $ mne report --help
"""

import sys
import time

import mne
from mne.report import Report
from mne.utils import logger, verbose


@verbose
def log_elapsed(t, verbose=None):
    """Log elapsed time."""
    logger.info(f"Report complete in {round(t, 1)} seconds")


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "-p",
        "--path",
        dest="path",
        help="Path to folder who MNE-Report must be created",
    )
    parser.add_option(
        "-i",
        "--info",
        dest="info_fname",
        help="File from which info dictionary is to be read",
        metavar="FILE",
    )
    parser.add_option(
        "-c",
        "--cov",
        dest="cov_fname",
        help="File from which noise covariance is to be read",
        metavar="FILE",
    )
    parser.add_option(
        "--bmin",
        dest="bmin",
        help="Time at which baseline correction starts for evokeds",
        default=None,
    )
    parser.add_option(
        "--bmax",
        dest="bmax",
        help="Time at which baseline correction stops for evokeds",
        default=None,
    )
    parser.add_option(
        "-d", "--subjects-dir", dest="subjects_dir", help="The subjects directory"
    )
    parser.add_option("-s", "--subject", dest="subject", help="The subject name")
    parser.add_option(
        "--no-browser",
        dest="no_browser",
        action="store_false",
        help="Do not open MNE-Report in browser",
    )
    parser.add_option(
        "--overwrite",
        dest="overwrite",
        action="store_false",
        help="Overwrite html report if it already exists",
    )
    parser.add_option(
        "-j", "--jobs", dest="n_jobs", help="Number of jobs to run in parallel"
    )
    parser.add_option(
        "-m",
        "--mri-decim",
        type="int",
        dest="mri_decim",
        default=2,
        help="Integer factor used to decimate BEM plots",
    )
    parser.add_option(
        "--image-format",
        type="str",
        dest="image_format",
        default="png",
        help="Image format to use (can be 'png' or 'svg')",
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()
    path = options.path
    if path is None:
        parser.print_help()
        sys.exit(1)
    info_fname = options.info_fname
    cov_fname = options.cov_fname
    subjects_dir = options.subjects_dir
    subject = options.subject
    image_format = options.image_format
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
    report = Report(
        info_fname,
        subjects_dir=subjects_dir,
        subject=subject,
        baseline=baseline,
        cov_fname=cov_fname,
        verbose=verbose,
        image_format=image_format,
    )
    report.parse_folder(path, verbose=verbose, n_jobs=n_jobs, mri_decim=mri_decim)
    log_elapsed(time.time() - t0, verbose=verbose)
    report.save(open_browser=open_browser, overwrite=overwrite)


mne.utils.run_command_if_main()
