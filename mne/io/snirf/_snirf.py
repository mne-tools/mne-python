# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import re
import numpy as np
import datetime

from ..base import BaseRaw
from ..meas_info import create_info
from ..utils import _mult_cal_one
from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, warn
from ...utils.check import _require_version
from ..constants import FIFF
from .._digitization import _make_dig_points
from ...transforms import _frame_to_str


@fill_doc
def read_raw_snirf(fname, preload=False, verbose=None):
    """Reader for a continuous wave SNIRF data.

    .. note:: This reader supports the .snirf file type only,
              not the .jnirs version.

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


@fill_doc
class RawSNIRF(BaseRaw):
    """Raw object from a continuous wave SNIRF file.

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
        _require_version('h5py', 'read raw SNIRF data')
        from ...externals.pymatreader.utils import _import_h5py
        h5py = _import_h5py()

        logger.info('Loading %s' % fname)

        with h5py.File(fname, 'r') as dat:

            if 'data2' in dat['nirs']:
                warn("File contains multiple recordings. "
                     "MNE does not support this feature. "
                     "Only the first dataset will be processed.")

            if np.array(dat.get('nirs/data1/measurementList1/dataType')) != 1:
                raise RuntimeError('File does not contain continuous wave '
                                   'data. MNE only supports reading continuous'
                                   ' wave amplitude SNIRF files. Expected type'
                                   ' code 1 but received type code %d' %
                                   (np.array(dat.get(
                                       'nirs/data1/measurementList1/dataType'
                                   ))))

            last_samps = dat.get('/nirs/data1/dataTimeSeries').shape[0] - 1

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
            # Read non standard (but allowed) custom metadata tags
            if 'lastName' in dat.get('nirs/metaDataTags/'):
                ln = dat.get('/nirs/metaDataTags/lastName')[0].decode('UTF-8')
                subject_info['last_name'] = ln
            if 'middleName' in dat.get('nirs/metaDataTags/'):
                m = dat.get('/nirs/metaDataTags/middleName')[0].decode('UTF-8')
                subject_info['middle_name'] = m
            if 'sex' in dat.get('nirs/metaDataTags/'):
                s = dat.get('/nirs/metaDataTags/sex')[0].decode('UTF-8')
                if s in {'M', 'Male', '1', 'm'}:
                    subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
                elif s in {'F', 'Female', '2', 'f'}:
                    subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
                elif s in {'0', 'u'}:
                    subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
            # End non standard name reading
            # Update info
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
                info['chs'][idx]['loc'][3:6] = srcPos3D[src_idx - 1, :] / scal
                info['chs'][idx]['loc'][6:9] = detPos3D[det_idx - 1, :] / scal
                # Store channel as mid point
                midpoint = (info['chs'][idx]['loc'][3:6] +
                            info['chs'][idx]['loc'][6:9]) / 2
                info['chs'][idx]['loc'][0:3] = midpoint
                info['chs'][idx]['loc'][9] = fnirs_wavelengths[wve_idx - 1]

            if 'MNE_coordFrame' in dat.get('nirs/metaDataTags/'):
                coord_frame = int(dat.get('/nirs/metaDataTags/MNE_coordFrame')
                                  [0])
            else:
                coord_frame = FIFF.FIFFV_COORD_UNKNOWN

            if 'landmarkPos3D' in dat.get('nirs/probe/'):
                diglocs = np.array(dat.get('/nirs/probe/landmarkPos3D'))
                digname = np.array(dat.get('/nirs/probe/landmarkLabels'))
                nasion, lpa, rpa, hpi = None, None, None, None
                extra_ps = dict()
                for idx, dign in enumerate(digname):
                    if dign == b'LPA':
                        lpa = diglocs[idx, :]
                    elif dign == b'NASION':
                        nasion = diglocs[idx, :]
                    elif dign == b'RPA':
                        rpa = diglocs[idx, :]
                    else:
                        extra_ps[f'EEG{len(extra_ps) + 1:03d}'] = diglocs[idx]
                info['dig'] = _make_dig_points(nasion=nasion, lpa=lpa, rpa=rpa,
                                               hpi=hpi, dig_ch_pos=extra_ps,
                                               coord_frame=_frame_to_str[
                                                   coord_frame])

            str_date = np.array((dat.get(
                '/nirs/metaDataTags/MeasurementDate')))[0].decode('UTF-8')
            str_time = np.array((dat.get(
                '/nirs/metaDataTags/MeasurementTime')))[0].decode('UTF-8')
            str_datetime = str_date + str_time

            # Several formats have been observed so we try each in turn
            for dt_code in ['%Y-%m-%d%H:%M:%SZ',
                            '%Y-%m-%d%H:%M:%S']:
                try:
                    meas_date = datetime.datetime.strptime(
                        str_datetime, dt_code)
                except ValueError:
                    pass
                else:
                    break
            else:
                warn("Extraction of measurement date from SNIRF file failed. "
                     "The date is being set to January 1st, 2000, "
                     f"instead of {str_datetime}")
                meas_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
            meas_date = meas_date.replace(tzinfo=datetime.timezone.utc)
            info['meas_date'] = meas_date

            if 'DateOfBirth' in dat.get('nirs/metaDataTags/'):
                str_birth = np.array((dat.get('/nirs/metaDataTags/'
                                              'DateOfBirth')))[0].decode()
                birth_matched = re.fullmatch(r'(\d+)-(\d+)-(\d+)', str_birth)
                if birth_matched is not None:
                    info["subject_info"]['birthday'] = (
                        int(birth_matched.groups()[0]),
                        int(birth_matched.groups()[1]),
                        int(birth_matched.groups()[2]))

            super(RawSNIRF, self).__init__(info, preload, filenames=[fname],
                                           last_samps=[last_samps],
                                           verbose=verbose)

            # Extract annotations
            annot = Annotations([], [], [])
            for key in dat['nirs']:
                if 'stim' in key:
                    data = np.array(dat.get('/nirs/' + key + '/data'))
                    if data.size > 0:
                        annot.append(data[:, 0], 1.0, key[4:])
            self.set_annotations(annot)

            # Reorder channels to match expected ordering in MNE
            num_chans = len(self.ch_names)
            chans = []
            for idx in range(num_chans // 2):
                chans.append(idx)
                chans.append(idx + num_chans // 2)
            self.pick(picks=chans)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        import h5py

        with h5py.File(self._filenames[0], 'r') as dat:
            one = dat['/nirs/data1/dataTimeSeries'][start:stop].T

        _mult_cal_one(data, one, idx, cals, mult)
