#!/usr/bin/env python
"""Clean a raw file from EOG and ECG artifacts with PCA (ie SSP).

Examples
--------
.. code-block:: console

    $ mne clean_eog_ecg -i in_raw.fif -o clean_raw.fif -e -c

"""
# Authors : Dr Engr. Sheraz Khan,  P.Eng, Ph.D.
#           Engr. Nandita Shetty,  MS.
#           Alexandre Gramfort, Ph.D.


import sys

import mne


def clean_ecg_eog(in_fif_fname, out_fif_fname=None, eog=True, ecg=True,
                  ecg_proj_fname=None, eog_proj_fname=None,
                  ecg_event_fname=None, eog_event_fname=None, in_path='.',
                  quiet=False):
    """Clean ECG from raw fif file.

    Parameters
    ----------
    in_fif_fname : str
        Raw fif File
    eog_event_fname : str
        name of EOG event file required.
    eog : bool
        Reject or not EOG artifacts.
    ecg : bool
        Reject or not ECG artifacts.
    ecg_event_fname : str
        name of ECG event file required.
    in_path : str
        Path where all the files are.
    """
    if not eog and not ecg:
        raise Exception("EOG and ECG cannot be both disabled")

    # Reading fif File
    raw_in = mne.io.read_raw_fif(in_fif_fname)

    if in_fif_fname.endswith('_raw.fif') or in_fif_fname.endswith('-raw.fif'):
        prefix = in_fif_fname[:-8]
    else:
        prefix = in_fif_fname[:-4]

    if out_fif_fname is None:
        out_fif_fname = prefix + '_clean_ecg_eog_raw.fif'
    if ecg_proj_fname is None:
        ecg_proj_fname = prefix + '_ecg-proj.fif'
    if eog_proj_fname is None:
        eog_proj_fname = prefix + '_eog-proj.fif'
    if ecg_event_fname is None:
        ecg_event_fname = prefix + '_ecg-eve.fif'
    if eog_event_fname is None:
        eog_event_fname = prefix + '_eog-eve.fif'

    print('Implementing ECG and EOG artifact rejection on data')

    kwargs = dict() if quiet else dict(stdout=None, stderr=None)
    if ecg:
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(
            raw_in, reject_by_annotation=True)
        print("Writing ECG events in %s" % ecg_event_fname)
        mne.write_events(ecg_event_fname, ecg_events)
        print('Computing ECG projector')
        command = ('mne_process_raw', '--cd', in_path, '--raw', in_fif_fname,
                   '--events', ecg_event_fname, '--makeproj',
                   '--projtmin', '-0.08', '--projtmax', '0.08',
                   '--saveprojtag', '_ecg-proj', '--projnmag', '2',
                   '--projngrad', '1', '--projevent', '999', '--highpass', '5',
                   '--lowpass', '35', '--projmagrej', '4000',
                   '--projgradrej', '3000')
        mne.utils.run_subprocess(command, **kwargs)
    if eog:
        eog_events = mne.preprocessing.find_eog_events(raw_in)
        print("Writing EOG events in %s" % eog_event_fname)
        mne.write_events(eog_event_fname, eog_events)
        print('Computing EOG projector')
        command = ('mne_process_raw', '--cd', in_path, '--raw', in_fif_fname,
                   '--events', eog_event_fname, '--makeproj',
                   '--projtmin', '-0.15', '--projtmax', '0.15',
                   '--saveprojtag', '_eog-proj', '--projnmag', '2',
                   '--projngrad', '2', '--projevent', '998', '--lowpass', '35',
                   '--projmagrej', '4000', '--projgradrej', '3000')
        mne.utils.run_subprocess(command, **kwargs)

    if out_fif_fname is not None:
        # Applying the ECG EOG projector
        print('Applying ECG EOG projector')
        command = ('mne_process_raw', '--cd', in_path, '--raw', in_fif_fname,
                   '--proj', in_fif_fname, '--projoff', '--save',
                   out_fif_fname, '--filteroff',
                   '--proj', ecg_proj_fname, '--proj', eog_proj_fname)
        mne.utils.run_subprocess(command, **kwargs)
        print('Done removing artifacts.')
        print("Cleaned raw data saved in: %s" % out_fif_fname)
        print('IMPORTANT : Please eye-ball the data !!')
    else:
        print('Projection not applied to raw data.')


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-i", "--in", dest="raw_in",
                      help="Input raw FIF file", metavar="FILE")
    parser.add_option("-o", "--out", dest="raw_out",
                      help="Output raw FIF file", metavar="FILE",
                      default=None)
    parser.add_option("-e", "--no-eog", dest="eog", action="store_false",
                      help="Remove EOG", default=True)
    parser.add_option("-c", "--no-ecg", dest="ecg", action="store_false",
                      help="Remove ECG", default=True)
    parser.add_option("-q", "--quiet", dest="quiet", action="store_true",
                      help="Suppress mne_process_raw output", default=False)

    options, args = parser.parse_args()

    if options.raw_in is None:
        parser.print_help()
        sys.exit(1)

    raw_in = options.raw_in
    raw_out = options.raw_out
    eog = options.eog
    ecg = options.ecg
    quiet = options.quiet

    clean_ecg_eog(raw_in, raw_out, eog=eog, ecg=ecg, quiet=quiet)


mne.utils.run_command_if_main()
