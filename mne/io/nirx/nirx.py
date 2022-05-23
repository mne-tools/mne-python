# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD-3-Clause

from configparser import ConfigParser, RawConfigParser
import glob as glob
import re as re
import os.path as op
import datetime as dt
import json

import numpy as np

from ._localized_abbr import _localized_abbr
from ..base import BaseRaw
from ..utils import _mult_cal_one
from ..constants import FIFF
from ..meas_info import create_info, _format_dig_points
from ...annotations import Annotations
from ..._freesurfer import get_mni_fiducials
from ...transforms import apply_trans, _get_trans
from ...utils import (logger, verbose, fill_doc, warn, _check_fname,
                      _validate_type, _check_option, _mask_to_onsets_offsets)


@fill_doc
def read_raw_nirx(fname, saturated='annotate', preload=False, verbose=None):
    """Reader for a NIRX fNIRS recording.

    Parameters
    ----------
    fname : str
        Path to the NIRX data folder or header file.
    %(saturated)s
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawNIRX
        A Raw object containing NIRX data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    %(nirx_notes)s
    """
    return RawNIRX(fname, saturated, preload, verbose)


def _open(fname):
    return open(fname, 'r', encoding='latin-1')


@fill_doc
class RawNIRX(BaseRaw):
    """Raw object from a NIRX fNIRS file.

    Parameters
    ----------
    fname : str
        Path to the NIRX data folder or header file.
    %(saturated)s
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    %(nirx_notes)s
    """

    @verbose
    def __init__(self, fname, saturated, preload=False, verbose=None):
        from scipy.io import loadmat
        logger.info('Loading %s' % fname)
        _validate_type(fname, 'path-like', 'fname')
        _validate_type(saturated, str, 'saturated')
        _check_option('saturated', saturated, ('annotate', 'nan', 'ignore'))
        fname = str(fname)
        if fname.endswith('.hdr'):
            fname = op.dirname(op.abspath(fname))

        fname = _check_fname(fname, 'read', True, 'fname', need_dir=True)

        json_config = glob.glob('%s/*%s' % (fname, "config.json"))
        if len(json_config):
            is_aurora = True
        else:
            is_aurora = False

        if is_aurora:
            # NIRSport2 devices using Aurora software
            keys = ('hdr', 'config.json', 'description.json',
                    'wl1', 'wl2', 'probeInfo.mat', 'tri')
        else:
            # NIRScout devices and NIRSport1 devices
            keys = ('hdr', 'inf', 'set', 'tpl', 'wl1', 'wl2',
                    'config.txt', 'probeInfo.mat')
            n_dat = len(glob.glob('%s/*%s' % (fname, 'dat')))
            if n_dat != 1:
                warn("A single dat file was expected in the specified path, "
                     f"but got {n_dat}. This may indicate that the file "
                     "structure has been modified since the measurement "
                     "was saved.")

        # Check if required files exist and store names for later use
        files = dict()
        nan_mask = dict()
        for key in keys:
            files[key] = glob.glob('%s/*%s' % (fname, key))
            fidx = 0
            if len(files[key]) != 1:
                if key not in ('wl1', 'wl2'):
                    raise RuntimeError(
                        f'Need one {key} file, got {len(files[key])}')
                noidx = np.where(['nosatflags_' in op.basename(x)
                                  for x in files[key]])[0]
                if len(noidx) != 1 or len(files[key]) != 2:
                    raise RuntimeError(
                        f'Need one nosatflags and one standard {key} file, '
                        f'got {len(files[key])}')
                # Here two files have been found, one that is called
                # no sat flags. The nosatflag file has no NaNs in it.
                noidx = noidx[0]
                if saturated == 'ignore':
                    # Ignore NaN and return values
                    fidx = noidx
                elif saturated == 'nan':
                    # Return NaN
                    fidx = 0 if noidx == 1 else 1
                else:
                    assert saturated == 'annotate'  # guaranteed above
                    fidx = noidx
                    nan_mask[key] = files[key][0 if noidx == 1 else 1]
            files[key] = files[key][fidx]

        # Read number of rows/samples of wavelength data
        with _open(files['wl1']) as fid:
            last_sample = fid.read().count('\n') - 1

        # Read header file
        # The header file isn't compliant with the configparser. So all the
        # text between comments must be removed before passing to parser
        with _open(files['hdr']) as f:
            hdr_str_all = f.read()
        hdr_str = re.sub('#.*?#', '', hdr_str_all, flags=re.DOTALL)
        if is_aurora:
            hdr_str = re.sub('(\\[DataStructure].*)', '',
                             hdr_str, flags=re.DOTALL)
        hdr = RawConfigParser()
        hdr.read_string(hdr_str)

        # Check that the file format version is supported
        if is_aurora:
            # We may need to ease this requirement back
            if hdr['GeneralInfo']['Version'] not in ['2021.4.0-34-ge9fdbbc8',
                                                     '2021.9.0-5-g3eb32851',
                                                     '2021.9.0-6-g14ef4a71']:
                warn("MNE has not been tested with Aurora version "
                     f"{hdr['GeneralInfo']['Version']}")
        else:
            if hdr['GeneralInfo']['NIRStar'] not in ['"15.0"', '"15.2"',
                                                     '"15.3"']:
                raise RuntimeError('MNE does not support this NIRStar version'
                                   ' (%s)' % (hdr['GeneralInfo']['NIRStar'],))
            if "NIRScout" not in hdr['GeneralInfo']['Device'] \
                    and "NIRSport" not in hdr['GeneralInfo']['Device']:
                warn("Only import of data from NIRScout devices have been "
                     "thoroughly tested. You are using a %s device. " %
                     hdr['GeneralInfo']['Device'])

        # Parse required header fields

        # Extract measurement date and time
        if is_aurora:
            datetime_str = hdr['GeneralInfo']['Date']
        else:
            datetime_str = hdr['GeneralInfo']['Date'] + \
                hdr['GeneralInfo']['Time']

        meas_date = None
        # Several formats have been observed so we try each in turn
        for loc, translations in _localized_abbr.items():
            do_break = False
            # So far we are lucky in that all the formats below, if they
            # include %a (weekday abbr), always come first. Thus we can use
            # a .split(), replace, and rejoin.
            loc_datetime_str = datetime_str.split(' ')
            for key, val in translations['weekday'].items():
                loc_datetime_str[0] = loc_datetime_str[0].replace(key, val)
            for ii in range(1, len(loc_datetime_str)):
                for key, val in translations['month'].items():
                    loc_datetime_str[ii] = \
                        loc_datetime_str[ii].replace(key, val)
            loc_datetime_str = ' '.join(loc_datetime_str)
            logger.debug(f'Trying {loc} datetime: {loc_datetime_str}')
            for dt_code in ['"%a, %b %d, %Y""%H:%M:%S.%f"',
                            '"%a %d %b %Y""%H:%M:%S.%f"',
                            '"%a, %d %b %Y""%H:%M:%S.%f"',
                            '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    meas_date = dt.datetime.strptime(loc_datetime_str, dt_code)
                except ValueError:
                    pass
                else:
                    meas_date = meas_date.replace(tzinfo=dt.timezone.utc)
                    do_break = True
                    logger.debug(
                        f'Measurement date language {loc} detected: {dt_code}')
                    break
            if do_break:
                break
        if meas_date is None:
            warn("Extraction of measurement date from NIRX file failed. "
                 "This can be caused by files saved in certain locales "
                 f"(currently only {list(_localized_abbr)} supported). "
                 "Please report this as a github issue. "
                 "The date is being set to January 1st, 2000, "
                 f"instead of {repr(datetime_str)}.")
            meas_date = dt.datetime(2000, 1, 1, 0, 0, 0,
                                    tzinfo=dt.timezone.utc)

        # Extract frequencies of light used by machine
        if is_aurora:
            fnirs_wavelengths = [760, 850]
        else:
            fnirs_wavelengths = [int(s) for s in
                                 re.findall(r'(\d+)',
                                            hdr['ImagingParameters'][
                                                'Wavelengths'])]

        # Extract source-detectors
        if is_aurora:
            sources = re.findall(r'(\d+)-\d+', hdr_str_all.split("\n")[-2])
            detectors = re.findall(r'\d+-(\d+)', hdr_str_all.split("\n")[-2])
            sources = [int(s) + 1 for s in sources]
            detectors = [int(d) + 1 for d in detectors]

        else:
            sources = np.asarray([int(s) for s in
                                  re.findall(r'(\d+)-\d+:\d+',
                                             hdr['DataStructure']
                                             ['S-D-Key'])], int)
            detectors = np.asarray([int(s) for s in
                                    re.findall(r'\d+-(\d+):\d+',
                                               hdr['DataStructure']
                                               ['S-D-Key'])], int)

        # Extract sampling rate
        if is_aurora:
            samplingrate = float(hdr['GeneralInfo']['Sampling rate'])
        else:
            samplingrate = float(hdr['ImagingParameters']['SamplingRate'])

        # Read participant information file
        if is_aurora:
            with open(files['description.json']) as f:
                inf = json.load(f)
        else:
            inf = ConfigParser(allow_no_value=True)
            inf.read(files['inf'])
            inf = inf._sections['Subject Demographics']

        # Store subject information from inf file in mne format
        # Note: NIRX also records "Study Type", "Experiment History",
        #       "Additional Notes", "Contact Information" and this information
        #       is currently discarded
        # NIRStar does not record an id, or handedness by default
        # The name field is used to populate the his_id variable.
        subject_info = {}
        if is_aurora:
            names = inf["subject"].split()
        else:
            names = inf['name'].replace('"', "").split()
        subject_info['his_id'] = "_".join(names)
        if len(names) > 0:
            subject_info['first_name'] = \
                names[0].replace("\"", "")
        if len(names) > 1:
            subject_info['last_name'] = \
                names[-1].replace("\"", "")
        if len(names) > 2:
            subject_info['middle_name'] = \
                names[-2].replace("\"", "")
        subject_info['sex'] = inf['gender'].replace("\"", "")
        # Recode values
        if subject_info['sex'] in {'M', 'Male', '1'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
        elif subject_info['sex'] in {'F', 'Female', '2'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
        else:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
        if inf['age'] != '':
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
        mat_data = loadmat(files['probeInfo.mat'])
        probes = mat_data['probeInfo']['probes'][0, 0]
        requested_channels = probes['index_c'][0, 0]
        src_locs = probes['coords_s3'][0, 0] / 100.
        det_locs = probes['coords_d3'][0, 0] / 100.
        ch_locs = probes['coords_c3'][0, 0] / 100.

        # These are all in MNI coordinates, so let's transform them to
        # the Neuromag head coordinate frame
        src_locs, det_locs, ch_locs, mri_head_t = _convert_fnirs_to_head(
            'fsaverage', 'mri', 'head', src_locs, det_locs, ch_locs)

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

        snames = [f"S{sources[idx]}" for idx in req_ind]
        dnames = [f"_D{detectors[idx]}" for idx in req_ind]
        sdnames = [m + str(n) for m, n in zip(snames, dnames)]
        sd1 = [s + ' ' + str(fnirs_wavelengths[0]) for s in sdnames]
        sd2 = [s + ' ' + str(fnirs_wavelengths[1]) for s in sdnames]
        chnames = [val for pair in zip(sd1, sd2) for val in pair]

        # Create mne structure
        info = create_info(chnames,
                           samplingrate,
                           ch_types='fnirs_cw_amplitude')
        with info._unlock():
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
            # Find detector and store location
            det = int(requested_channels[ch_idx2, 1]) - 1
            # Store channel location as midpoint between source and detector.
            midpoint = (src_locs[src, :] + det_locs[det, :]) / 2
            for ii in range(2):
                ch_idx3 = ch_idx2 * 2 + ii
                info['chs'][ch_idx3]['loc'][3:6] = src_locs[src, :]
                info['chs'][ch_idx3]['loc'][6:9] = det_locs[det, :]
                info['chs'][ch_idx3]['loc'][:3] = midpoint
                info['chs'][ch_idx3]['loc'][9] = fnirs_wavelengths[ii]
                info['chs'][ch_idx3]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

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
            'nan_mask': nan_mask,
        }
        # Get our saturated mask
        annot_mask = None
        for ki, key in enumerate(('wl1', 'wl2')):
            if nan_mask.get(key, None) is None:
                continue
            mask = np.isnan(_read_csv_rows_cols(
                nan_mask[key], 0, last_sample + 1, req_ind, {0: 0, 1: None}).T)
            if saturated == 'nan':
                nan_mask[key] = mask
            else:
                assert saturated == 'annotate'
                if annot_mask is None:
                    annot_mask = np.zeros(
                        (len(info['ch_names']) // 2, last_sample + 1), bool)
                annot_mask |= mask
                nan_mask[key] = None  # shouldn't need again

        super(RawNIRX, self).__init__(
            info, preload, filenames=[fname], last_samps=[last_sample],
            raw_extras=[raw_extras], verbose=verbose)

        # make onset/duration/description
        onset, duration, description, ch_names = list(), list(), list(), list()
        if annot_mask is not None:
            for ci, mask in enumerate(annot_mask):
                on, dur = _mask_to_onsets_offsets(mask)
                on = on / info['sfreq']
                dur = dur / info['sfreq']
                dur -= on
                onset.extend(on)
                duration.extend(dur)
                description.extend(['BAD_SATURATED'] * len(on))
                ch_names.extend([self.ch_names[2 * ci:2 * ci + 2]] * len(on))

        # Read triggers from event file
        if not is_aurora:
            files['tri'] = files['hdr'][:-3] + 'evt'
        if op.isfile(files['tri']):
            with _open(files['tri']) as fid:
                t = [re.findall(r'(\d+)', line) for line in fid]
            if is_aurora:
                tf_idx, desc_idx = _determine_tri_idxs(t[0])
            for t_ in t:
                if is_aurora:
                    trigger_frame = float(t_[tf_idx])
                    desc = float(t_[desc_idx])
                else:
                    binary_value = ''.join(t_[1:])[::-1]
                    desc = float(int(binary_value, 2))
                    trigger_frame = float(t_[0])
                onset.append(trigger_frame / samplingrate)
                duration.append(1.)  # No duration info stored in files
                description.append(desc)
                ch_names.append(list())
        annot = Annotations(onset, duration, description, ch_names=ch_names)
        self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        The NIRX machine records raw data as two different wavelengths.
        The returned data interleaves the wavelengths.
        """
        sd_index = self._raw_extras[fi]['sd_index']

        wls = list()
        for key in ('wl1', 'wl2'):
            d = _read_csv_rows_cols(
                self._raw_extras[fi]['files'][key],
                start, stop, sd_index,
                self._raw_extras[fi]['bounds'][key]).T
            nan_mask = self._raw_extras[fi]['nan_mask'].get(key, None)
            if nan_mask is not None:
                d[nan_mask[:, start:stop]] = np.nan
            wls.append(d)

        # TODO: Make this more efficient by only indexing above what we need.
        # For now let's just construct the full data matrix and index.
        # Interleave wavelength 1 and 2 to match channel names:
        this_data = np.zeros((len(wls[0]) * 2, stop - start))
        this_data[0::2, :] = wls[0]
        this_data[1::2, :] = wls[1]
        _mult_cal_one(data, this_data, idx, cals, mult)
        return data


def _read_csv_rows_cols(fname, start, stop, cols, bounds,
                        sep=' ', replace=None):
    with open(fname, 'rb') as fid:
        fid.seek(bounds[start])
        args = list()
        if bounds[1] is not None:
            args.append(bounds[stop] - bounds[start])
        data = fid.read(*args).decode('latin-1')
        if replace is not None:
            data = replace(data)
        x = np.fromstring(data, float, sep=sep)
    x.shape = (stop - start, -1)
    x = x[:, cols]
    return x


def _convert_fnirs_to_head(trans, fro, to, src_locs, det_locs, ch_locs):
    mri_head_t, _ = _get_trans(trans, fro, to)
    src_locs = apply_trans(mri_head_t, src_locs)
    det_locs = apply_trans(mri_head_t, det_locs)
    ch_locs = apply_trans(mri_head_t, ch_locs)
    return src_locs, det_locs, ch_locs, mri_head_t


def _determine_tri_idxs(trigger):
    """Determine tri file indexes for frame and description."""
    if len(trigger) == 12:
        # Aurora version 2021.9.6 or greater
        trigger_frame_idx = 7
        desc_idx = 10
    elif len(trigger) == 9:
        # Aurora version 2021.9.5 or earlier
        trigger_frame_idx = 7
        desc_idx = 8
    else:
        raise RuntimeError("Unable to read trigger file.")

    return trigger_frame_idx, desc_idx
