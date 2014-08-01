#!/usr/bin/env python
"""Clean a raw file from EOG and ECG artifacts with PCA (ie SSP)
"""
from __future__ import print_function

# Authors : Dr Engr. Sheraz Khan,  P.Eng, Ph.D.
#           Engr. Nandita Shetty,  MS.
#           Alexandre Gramfort, Ph.D.


import os
import sys

import mne


def clean_ecg_eog(in_fif_fname, out_fif_fname=None, eog=True, ecg=True,
                  ecg_proj_fname=None, eog_proj_fname=None,
                  ecg_event_fname=None, eog_event_fname=None, in_path='.'):
    """Clean ECG from raw fif file

    Parameters
    ----------
    in_fif_fname : string
        Raw fif File
    eog_event_fname : string
        name of EOG event file required.
    eog : bool
        Reject or not EOG artifacts.
    ecg : bool
        Reject or not ECG artifacts.
    ecg_event_fname : string
        name of ECG event file required.
    in_path :
        Path where all the files are.
    """
    if not eog and not ecg:
        raise Exception("EOG and ECG cannot be both disabled")

    # Reading fif File
    raw_in = mne.io.Raw(in_fif_fname)

    if in_fif_fname.endswith('_raw.fif') or in_fif_fname.endswith('-raw.fif'):
        prefix = in_fif_fname[:-8]
    else:
        prefix = in_fif_fname[:-4]

    if out_fif_fname is None:
        out_fif_fname = prefix + '_clean_ecg_eog_raw.fif'
    if ecg_proj_fname is None:
        ecg_proj_fname = prefix + '_ecg_proj.fif'
    if eog_proj_fname is None:
        eog_proj_fname = prefix + '_eog_proj.fif'
    if ecg_event_fname is None:
        ecg_event_fname = prefix + '_ecg-eve.fif'
    if eog_event_fname is None:
        eog_event_fname = prefix + '_eog-eve.fif'

    print('Implementing ECG and EOG artifact rejection on data')

    if ecg:
        ecg_events, _, _  = mne.preprocessing.find_ecg_events(raw_in)
        print("Writing ECG events in %s" % ecg_event_fname)
        mne.write_events(ecg_event_fname, ecg_events)

        print('Computing ECG projector')

        command = ('mne_process_raw --cd %s --raw %s --events %s --makeproj '
                   '--projtmin -0.08 --projtmax 0.08 --saveprojtag _ecg_proj '
                   '--projnmag 2 --projngrad 1 --projevent 999 --highpass 5 '
                   '--lowpass 35 --projmagrej 4000  --projgradrej 3000'
                   % (in_path, in_fif_fname, ecg_event_fname))
        st = os.system(command)

        if st != 0:
            print("Error while running : %s" % command)

    if eog:
        eog_events = mne.preprocessing.find_eog_events(raw_in)
        print("Writing EOG events in %s" % eog_event_fname)
        mne.write_events(eog_event_fname, eog_events)

        print('Computing EOG projector')

        command = ('mne_process_raw --cd %s --raw %s --events %s --makeproj '
                   '--projtmin -0.15 --projtmax 0.15 --saveprojtag _eog_proj '
                   '--projnmag 2 --projngrad 2 --projevent 998 --lowpass 35 '
                   '--projmagrej 4000  --projgradrej 3000' % (in_path,
                   in_fif_fname, eog_event_fname))

        print('Running : %s' % command)

        st = os.system(command)
        if st != 0:
            raise ValueError('Problem while running : %s' % command)

    if out_fif_fname is not None:
        # Applying the ECG EOG projector
        print('Applying ECG EOG projector')

        command = ('mne_process_raw --cd %s --raw %s '
                   '--proj %s --projoff --save %s --filteroff'
                   % (in_path, in_fif_fname, in_fif_fname, out_fif_fname))
        command += ' --proj %s --proj %s' % (ecg_proj_fname, eog_proj_fname)

        print('Command executed: %s' % command)

        st = os.system(command)

        if st != 0:
            raise ValueError('Pb while running : %s' % command)

        print('Done removing artifacts.')
        print("Cleaned raw data saved in: %s" % out_fif_fname)
        print('IMPORTANT : Please eye-ball the data !!')
    else:
        print('Projection not applied to raw data.')


if __name__ == '__main__':

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

    options, args = parser.parse_args()

    if options.raw_in is None:
        parser.print_help()
        sys.exit(1)

    raw_in = options.raw_in
    raw_out = options.raw_out
    eog = options.eog
    ecg = options.ecg

    clean_ecg_eog(raw_in, raw_out, eog=eog, ecg=ecg)
