#!/usr/bin/env python
"""Create mne report for a folder

Example usage

mne report -p MNE-sample-data/ -i \
MNE-sample-data/MEG/sample/sample_audvis-ave.fif -d MNE-sample-data/subjects/ \
-s sample

"""

from mne.report import Report


if __name__ == '__main__':

    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-p", "--path", dest="path",
                      help="Path to folder who MNE-Report must be created")
    parser.add_option("-i", "--info", dest="info_fname",
                      help="File from which info dictionary is to be read",
                      metavar="FILE")
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="The subjects directory")
    parser.add_option("-s", "--subject", dest="subject",
                      help="The subject name")
    parser.add_option("-x", "--interactive", dest="interactive",
                      action='store_true',
                      help="make matplotlib plots interactive")
    parser.add_option("-v", "--verbose", dest="verbose",
                      action='store_true', help="run in verbose mode")
    parser.add_option("--no-browser", dest="no_browser", action='store_false',
                      help="Do not open MNE-Report in browser")
    parser.add_option("--overwrite", dest="overwrite", action='store_false',
                      help="Overwrite html report if it already exists")

    options, args = parser.parse_args()
    path = options.path
    info_fname = options.info_fname
    subjects_dir = options.subjects_dir
    subject = options.subject
    interactive = True if options.interactive is not None else False
    verbose = True if options.verbose is not None else False
    open_browser = False if options.no_browser is not None else True
    overwrite = True if options.overwrite is not None else False

    report = Report(info_fname, subjects_dir=subjects_dir, subject=subject,
                    verbose=verbose)
    report.parse_folder(path, interactive=interactive, verbose=verbose)
    report.save('report.html', open_browser=open_browser, overwrite=overwrite)
