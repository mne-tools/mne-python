# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from configparser import ConfigParser, RawConfigParser
import glob as glob
import re as re
import os.path as op
import datetime as dt

import numpy as np

from ..base import BaseRaw
from ..utils import _mult_cal_one
from ..constants import FIFF
from ..meas_info import create_info, _format_dig_points
from ...annotations import Annotations
from ...transforms import apply_trans, _get_trans
from ...utils import logger, verbose, fill_doc, warn


@fill_doc
def read_raw_nirx(fname, preload=False, verbose=None):
    """Reader for a NIRX fNIRS recording.

    This function has only been tested with NIRScout devices.

    Parameters
    ----------
    fname : str
        Path to the NIRX data folder or header file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawNIRX
        A Raw object containing NIRX data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawNIRX(fname, preload, verbose)


def _open(fname):
    return open(fname, 'r', encoding='latin-1')


@fill_doc
class RawNIRX(BaseRaw):
    """Raw object from a NIRX fNIRS file.

    Parameters
    ----------
    fname : str
        Path to the NIRX data folder or header file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        from ...externals.pymatreader import read_mat
        from ...coreg import get_mni_fiducials  # avoid circular import prob
        logger.info('Loading %s' % fname)

        if fname.endswith('.hdr'):
            fname = op.dirname(op.abspath(fname))

        if not op.isdir(fname):
            raise FileNotFoundError('The path you specified does not exist.')

        # Check if required files exist and store names for later use
        files = dict()
        keys = ('hdr', 'inf', 'set', 'tpl', 'wl1', 'wl2',
                'config.txt', 'probeInfo.mat')
        for key in keys:
            files[key] = glob.glob('%s/*%s' % (fname, key))
            if len(files[key]) != 1:
                raise RuntimeError('Expect one %s file, got %d' %
                                   (key, len(files[key]),))
            files[key] = files[key][0]
        if len(glob.glob('%s/*%s' % (fname, 'dat'))) != 1:
            warn("A single dat file was expected in the specified path, but "
                 "got %d. This may indicate that the file structure has been "
                 "modified since the measurement was saved." %
                 (len(glob.glob('%s/*%s' % (fname, 'dat')))))

        # Read number of rows/samples of wavelength data
        last_sample = -1
        with _open(files['wl1']) as fid:
            for line in fid:
                last_sample += 1

        # Read header file
        # The header file isn't compliant with the configparser. So all the
        # text between comments must be removed before passing to parser
        with _open(files['hdr']) as f:
            hdr_str = f.read()
        hdr_str = re.sub('#.*?#', '', hdr_str, flags=re.DOTALL)
        hdr = RawConfigParser()
        hdr.read_string(hdr_str)

        # Check that the file format version is supported
        if hdr['GeneralInfo']['NIRStar'] not in ['"15.0"', '"15.2"', '"15.3"']:
            raise RuntimeError('MNE does not support this NIRStar version'
                               ' (%s)' % (hdr['GeneralInfo']['NIRStar'],))
        if "NIRScout" not in hdr['GeneralInfo']['Device']:
            warn("Only import of data from NIRScout devices have been "
                 "thoroughly tested. You are using a %s device. " %
                 hdr['GeneralInfo']['Device'])

        # Parse required header fields

        # Extract measurement date and time
        datetime_str = hdr['GeneralInfo']['Date'] + hdr['GeneralInfo']['Time']
        meas_date = None
        # Several formats have been observed so we try each in turn
        for dt_code in ['"%a, %b %d, %Y""%H:%M:%S.%f"',
                        '"%a, %d %b %Y""%H:%M:%S.%f"']:
            try:
                meas_date = dt.datetime.strptime(datetime_str, dt_code)
                meas_date = meas_date.replace(tzinfo=dt.timezone.utc)
                break
            except ValueError:
                pass
        if meas_date is None:
            warn("Extraction of measurement date from NIRX file failed. "
                 "This can be caused by files saved in certain locales. "
                 "Please report this as a github issue. "
                 "The date is being set to January 1st, 2000, "
                 "instead of {}".format(datetime_str))
            meas_date = dt.datetime(2000, 1, 1, 0, 0, 0,
                                    tzinfo=dt.timezone.utc)

        # Extract frequencies of light used by machine
        fnirs_wavelengths = [int(s) for s in
                             re.findall(r'(\d+)',
                                        hdr['ImagingParameters'][
                                            'Wavelengths'])]

        # Extract source-detectors
        sources = np.asarray([int(s) for s in re.findall(r'(\d+)-\d+:\d+',
                                                         hdr['DataStructure'][
                                                             'S-D-Key'])], int)
        detectors = np.asarray([int(s) for s in re.findall(r'\d+-(\d+):\d+',
                                                           hdr['DataStructure']
                                                           ['S-D-Key'])],
                               int)

        # Extract sampling rate
        samplingrate = float(hdr['ImagingParameters']['SamplingRate'])

        # Read participant information file
        inf = ConfigParser(allow_no_value=True)
        inf.read(files['inf'])
        inf = inf._sections['Subject Demographics']

        # Store subject information from inf file in mne format
        # Note: NIRX also records "Study Type", "Experiment History",
        #       "Additional Notes", "Contact Information" and this information
        #       is currently discarded
        # NIRStar does not record an id, or handedness by default
        subject_info = {}
        names = inf['name'].split()
        if len(names) > 0:
            subject_info['first_name'] = \
                inf['name'].split()[0].replace("\"", "")
        if len(names) > 1:
            subject_info['last_name'] = \
                inf['name'].split()[-1].replace("\"", "")
        if len(names) > 2:
            subject_info['middle_name'] = \
                inf['name'].split()[-2].replace("\"", "")
        subject_info['sex'] = inf['gender'].replace("\"", "")
        # Recode values
        if subject_info['sex'] in {'M', 'Male', '1'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
        elif subject_info['sex'] in {'F', 'Female', '2'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
        else:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
        subject_info['birthday'] = (meas_date.year - int(inf['age']),
                                    meas_date.month,
                                    meas_date.day)

        # Read information about probe/montage/optodes
        # A word on terminology used here:
        #   Sources produce light
        #   Detectors measure light
        #   Sources and detectors are both called optodes
        #   Each source - detector pair produces a channel
        #   Channels are defined as the midpoint between source and detector
        mat_data = read_mat(files['probeInfo.mat'], uint16_codec=None)
        requested_channels = mat_data['probeInfo']['probes']['index_c']
        src_locs = mat_data['probeInfo']['probes']['coords_s3'] / 100.
        det_locs = mat_data['probeInfo']['probes']['coords_d3'] / 100.
        ch_locs = mat_data['probeInfo']['probes']['coords_c3'] / 100.

        # These are all in MNI coordinates, so let's transform them to
        # the Neuromag head coordinate frame
        mri_head_t, _ = _get_trans('fsaverage', 'mri', 'head')
        src_locs = apply_trans(mri_head_t, src_locs)
        det_locs = apply_trans(mri_head_t, det_locs)
        ch_locs = apply_trans(mri_head_t, ch_locs)

        # Set up digitization
        dig = get_mni_fiducials('fsaverage', verbose=False)
        for fid in dig:
            fid['r'] = apply_trans(mri_head_t, fid['r'])
            fid['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        for ii, ch_loc in enumerate(ch_locs, 1):
            dig.append(dict(
                kind=FIFF.FIFFV_POINT_EEG,  # misnomer but probably okay
                r=ch_loc,
                ident=ii,
                coord_frame=FIFF.FIFFV_COORD_HEAD,
            ))
        dig = _format_dig_points(dig)
        del mri_head_t

        # Determine requested channel indices
        # The wl1 and wl2 files include all possible source - detector pairs.
        # But most of these are not relevant. We want to extract only the
        # subset requested in the probe file
        req_ind = np.array([], int)
        for req_idx in range(requested_channels.shape[0]):
            sd_idx = np.where((sources == requested_channels[req_idx][0]) &
                              (detectors == requested_channels[req_idx][1]))
            req_ind = np.concatenate((req_ind, sd_idx[0]))
        req_ind = req_ind.astype(int)

        # Generate meaningful channel names
        def prepend(li, str):
            str += '{0}'
            li = [str.format(i) for i in li]
            return li

        snames = prepend(sources[req_ind], 'S')
        dnames = prepend(detectors[req_ind], '_D')
        sdnames = [m + str(n) for m, n in zip(snames, dnames)]
        sd1 = [s + ' ' + str(fnirs_wavelengths[0]) for s in sdnames]
        sd2 = [s + ' ' + str(fnirs_wavelengths[1]) for s in sdnames]
        chnames = [val for pair in zip(sd1, sd2) for val in pair]

        # Create mne structure
        info = create_info(chnames,
                           samplingrate,
                           ch_types='fnirs_cw_amplitude')
        info.update(subject_info=subject_info, dig=dig)
        info['meas_date'] = meas_date

        # Store channel, source, and detector locations
        # The channel location is stored in the first 3 entries of loc.
        # The source location is stored in the second 3 entries of loc.
        # The detector location is stored in the third 3 entries of loc.
        # NIRx NIRSite uses MNI coordinates.
        # Also encode the light frequency in the structure.
        for ch_idx2 in range(requested_channels.shape[0]):
            # Find source and store location
            src = int(requested_channels[ch_idx2, 0]) - 1
            info['chs'][ch_idx2 * 2]['loc'][3:6] = src_locs[src, :]
            info['chs'][ch_idx2 * 2 + 1]['loc'][3:6] = src_locs[src, :]
            # Find detector and store location
            det = int(requested_channels[ch_idx2, 1]) - 1
            info['chs'][ch_idx2 * 2]['loc'][6:9] = det_locs[det, :]
            info['chs'][ch_idx2 * 2 + 1]['loc'][6:9] = det_locs[det, :]
            # Store channel location as midpoint between source and detector.
            midpoint = (src_locs[src, :] + det_locs[det, :]) / 2
            info['chs'][ch_idx2 * 2]['loc'][:3] = midpoint
            info['chs'][ch_idx2 * 2 + 1]['loc'][:3] = midpoint
            info['chs'][ch_idx2 * 2]['loc'][9] = fnirs_wavelengths[0]
            info['chs'][ch_idx2 * 2 + 1]['loc'][9] = fnirs_wavelengths[1]

        # Extract the start/stop numbers for samples in the CSV. In theory the
        # sample bounds should just be 10 * the number of channels, but some
        # files have mixed \n and \n\r endings (!) so we can't rely on it, and
        # instead make a single pass over the entire file at the beginning so
        # that we know how to seek and read later.
        bounds = dict()
        for key in ('wl1', 'wl2'):
            offset = 0
            bounds[key] = [offset]
            with open(files[key], 'rb') as fid:
                for line in fid:
                    offset += len(line)
                    bounds[key].append(offset)
                assert offset == fid.tell()

        # Extras required for reading data
        raw_extras = {
            'sd_index': req_ind,
            'files': files,
            'bounds': bounds,
        }

        super(RawNIRX, self).__init__(
            info, preload, filenames=[fname], last_samps=[last_sample],
            raw_extras=[raw_extras], verbose=verbose)

        # Read triggers from event file
        if op.isfile(files['hdr'][:-3] + 'evt'):
            with _open(files['hdr'][:-3] + 'evt') as fid:
                t = [re.findall(r'(\d+)', line) for line in fid]
            onset = np.zeros(len(t), float)
            duration = np.zeros(len(t), float)
            description = [''] * len(t)
            for t_idx in range(len(t)):
                binary_value = ''.join(t[t_idx][1:])[::-1]
                trigger_frame = float(t[t_idx][0])
                onset[t_idx] = (trigger_frame) * (1.0 / samplingrate)
                duration[t_idx] = 1.0  # No duration info stored in files
                description[t_idx] = int(binary_value, 2) * 1.
            annot = Annotations(onset, duration, description)
            self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        The NIRX machine records raw data as two different wavelengths.
        The returned data interleaves the wavelengths.
        """
        sdindex = self._raw_extras[fi]['sd_index']

        wls = [
            _read_csv_rows_cols(
                self._raw_extras[fi]['files'][key],
                start, stop, sdindex,
                self._raw_extras[fi]['bounds'][key]).T
            for key in ('wl1', 'wl2')
        ]

        # TODO: Make this more efficient by only indexing above what we need.
        # For now let's just construct the full data matrix and index.
        # Interleave wavelength 1 and 2 to match channel names:
        this_data = np.zeros((len(wls[0]) * 2, stop - start))
        this_data[0::2, :] = wls[0]
        this_data[1::2, :] = wls[1]
        _mult_cal_one(data, this_data, idx, cals, mult)
        return data


def _read_csv_rows_cols(fname, start, stop, cols, bounds):
    with open(fname, 'rb') as fid:
        fid.seek(bounds[start])
        data = fid.read(bounds[stop] - bounds[start]).decode('latin-1')
        x = np.fromstring(data, float, sep=' ')
    x.shape = (stop - start, -1)
    x = x[:, cols]
    return x
