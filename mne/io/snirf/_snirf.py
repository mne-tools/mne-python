# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import re as re
import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, warn, requires_h5py


@requires_h5py
@fill_doc
def read_raw_snirf(fname, preload=False, verbose=None):
    """Reader for a continuous wave SNIRF data.

    .. note:: This reader supports the .snirf file type only,
              not the .jnirs version.
              Currently only support for continuous wave SNIRF data is
              provided.

    Parameters
    ----------
    fname : str
        Path to the SNIRF data file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawSNIRF
        A Raw object containing SNIRF data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawSNIRF(fname, preload, verbose)


def _open(fname):
    return open(fname, 'r', encoding='latin-1')


@requires_h5py
@fill_doc
class RawSNIRF(BaseRaw):
    """Raw object from a SNIRF fNIRS file.

    Parameters
    ----------
    fname : str
        Path to the SNIRF data file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        from ...externals.pymatreader.utils import _import_h5py
        h5py = _import_h5py()
        logger.info('Loading %s' % fname)

        with h5py.File(fname, 'r') as dat:

            if 'data2' in dat['nirs'].keys():
                warn("File contains multiple recordings. "
                     "MNE does not support this feature.")

            if np.array(dat.get('nirs/data1/measurementList1/dataType')) != 1:
                warn("File does not contain continuous wave data. "
                     "MNE only supports reading "
                     "continuous wave amplitude SNIRF files.")

            data = np.array(dat.get('/nirs/data1/dataTimeSeries')).T
            last_samps = data.shape[1] - 1

            samplingrate_raw = np.array(dat.get('nirs/data1/time'))
            sampling_rate = 0
            if samplingrate_raw.shape == (2, 1):
                # specified as onset/samplerate
                warn("Onset/sample rate SNIRF not yet supported.")
            else:
                # specified as time points
                fs_diff = np.around(np.diff(samplingrate_raw), decimals=4)
                if len(np.unique(fs_diff)) == 1:
                    # Uniformly sampled data
                    sampling_rate = 1. / np.unique(fs_diff)
                else:
                    # print(np.unique(fs_diff))
                    warn("Non uniform sampled data not supported.")
            if sampling_rate == 0:
                warn("Unable to extract sample rate from SNIRF file.")

            sources = np.array(dat.get('nirs/probe/sourceLabels'))
            detectors = np.array(dat.get('nirs/probe/detectorLabels'))
            sources = [s.decode('UTF-8') for s in sources]
            detectors = [d.decode('UTF-8') for d in detectors]

            # Extract source and detector locations
            detPos3D = np.array(dat.get('nirs/probe/detectorPos3D'))
            srcPos3D = np.array(dat.get('nirs/probe/sourcePos3D'))

            assert len(sources) == srcPos3D.shape[0]
            assert len(detectors) == detPos3D.shape[0]

            # Extract wavelengths
            fnirs_wavelengths = np.array(dat.get('nirs/probe/wavelengths'))
            fnirs_wavelengths = [int(w) for w in fnirs_wavelengths]

            # Extract channels
            def atoi(text):
                return int(text) if text.isdigit() else text

            def natural_keys(text):
                return [atoi(c) for c in re.split(r'(\d+)', text)]

            channels = np.array([name for name in dat['nirs']['data1'].keys()])
            channels_idx = np.array(['measurementList' in n for n in channels])
            channels = channels[channels_idx]
            channels = sorted(channels, key=natural_keys)

            chnames = []
            for chan in channels:
                src_idx = int(np.array(dat.get('nirs/data1/' +
                                               chan + '/sourceIndex'))[0])
                det_idx = int(np.array(dat.get('nirs/data1/' +
                                               chan + '/detectorIndex'))[0])
                wve_idx = int(np.array(dat.get('nirs/data1/' +
                                               chan + '/wavelengthIndex'))[0])
                ch_name = sources[src_idx - 1] + '_' +\
                    detectors[det_idx - 1] + ' ' +\
                    str(fnirs_wavelengths[wve_idx - 1])
                chnames.append(ch_name)

            # Create mne structure
            info = create_info(chnames,
                               sampling_rate,
                               ch_types='fnirs_cw_amplitude')

            subject_info = {}
            names = np.array(dat.get('nirs/metaDataTags/SubjectID'))
            subject_info['first_name'] = names[0].decode('UTF-8')
            info.update(subject_info=subject_info)

            LengthUnit = np.array(dat.get('/nirs/metaDataTags/LengthUnit'))
            LengthUnit = LengthUnit[0].decode('UTF-8')
            scal = 1
            if "cm" in LengthUnit:
                scal = 100
            elif "mm" in LengthUnit:
                scal = 1000

            for idx, chan in enumerate(channels):
                src_idx = int(np.array(dat.get('nirs/data1/' +
                                               chan + '/sourceIndex'))[0])
                det_idx = int(np.array(dat.get('nirs/data1/' +
                                               chan + '/detectorIndex'))[0])
                wve_idx = int(np.array(dat.get('nirs/data1/' +
                                               chan + '/wavelengthIndex'))[0])
                info['chs'][idx]['loc'][0:3] = srcPos3D[src_idx - 1, :] / scal
                info['chs'][idx]['loc'][3:6] = srcPos3D[src_idx - 1, :] / scal
                info['chs'][idx]['loc'][6:9] = detPos3D[det_idx - 1, :] / scal
                info['chs'][idx]['loc'][9] = fnirs_wavelengths[wve_idx - 1]

            super(RawSNIRF, self).__init__(info, preload, filenames=[fname],
                                           last_samps=[last_samps],
                                           verbose=verbose)

            # Extract annotations
            annot = Annotations([], [], [])
            for key in dat['nirs'].keys():
                if 'stim' in key:
                    data = np.array(dat.get('/nirs/' + key + '/data'))
                    if not data.shape == ():
                        annot.append(data[:, 0], 1.0, key[4:])
            self.set_annotations(annot)

            # Reorder channels to match expected ordering in MNE
            num_chans = len(self.ch_names)
            chans = []
            for idx in range(int(num_chans / 2)):
                chans.append(idx)
                chans.append(idx + int(num_chans / 2))
            self.pick(picks=chans)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        from ...externals.pymatreader.utils import _import_h5py
        h5py = _import_h5py()

        with h5py.File(self._filenames[0], 'r') as dat:
            data[:] = np.asarray(
                dat.get('/nirs/data1/dataTimeSeries')[start:stop].T[idx])

        return data
