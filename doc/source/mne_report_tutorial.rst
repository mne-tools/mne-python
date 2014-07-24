.. _mne_report_tutorial:

=======================================
Getting started with MNE report command
=======================================

This quick start will show you how to run the `mne report` command on the
sample data set provided with MNE.

First ensure that the files you want to render follow the filename conventions
defined by MNE:

==================   ====================================================
Data object          Filename convention (ends with)
==================   ====================================================
raw                  -raw.fif(.gz), -raw_sss.fif(.gz), -raw_tsss.fif(.gz)
events               -eve.fif(.gz)
epochs               -epo.fif(.gz)
evoked               -ave.fif(.gz)
covariance           -cov.fif(.gz)
trans                -trans.fif(.gz)
forward              -fwd.fif(.gz)
inverse              -inv.fif(.gz)
==================   ====================================================

The command line interface
--------------------------

To generate a barebones report from all the \*.fif files in the sample dataset,
invoke the following command::

    mne report --path MNE-sample-data/ --verbose

On successful creation of the report, it will open the html in a new tab in the browser.
To disable this, use the `--no-browser` option.

If the report is generated for a single subject, give the SUBJECT name and the
SUBJECTS_DIR and this will generate the MRI slices (with BEM contours overlaid on top
if available)::

    mne report --path MNE-sample-data/ --subject sample --subjects-dir MNE-sample-data/subjects --verbose

To properly render `trans` and `covariance` files, add the measurement information::

    mne report --path MNE-sample-data/ --info MNE-sample-data/MEG/sample/sample_audvis-ave.fif \ 
        --subject sample --subjects_dir MNE-sample-data/subjects --verbose

To generate the report in parallel::

    mne report --path MNE-sample-data/ --info MNE-sample-data/MEG/sample/sample_audvis-ave.fif \ 
        --subject sample --subjects_dir MNE-sample-data/subjects --verbose --jobs 6

The Python interface
--------------------

The same functionality can also be achieved using the Python interface. Import
the required functions:

    >>> from mne.report import Report
    >>> from mne.datasets import sample

Generate the report:

    >>> path = sample.data_path()
    >>> report = Report(subjects_dir=path + '/subjects', subject='sample')
    >>> report.parse_folder(data_path=path)
    >>> report.save('report.html')

There is greater flexibility compared to the command line interface. 
Custom plots can be added to the report. Let us first generate a custom plot:

    >>> from mne import read_evokeds
    >>> fname = path + '/MEG/sample/sample_audvis-ave.fif'
    >>> evoked = read_evokeds(fname, condition='Left Auditory', baseline=(None, 0))
    >>> fig = evoked.plot()

To add the custom plot to the report, do:

    >>> report.add_section(fig, captions='evoked response', section='subject 1')
    >>> report.save('report.html', overwrite=True)

The MNE report command internally manages the sections so that plots belonging to the same section
are rendered consecutively. Within a section, the plots are ordered in the same order that they were 
added using the `add_section` command. Each section is identified by a toggle button in the navigation 
bar of the report which can be used to show or hide the contents of the section.

That's it!
