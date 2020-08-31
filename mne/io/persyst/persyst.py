# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

from configparser import ConfigParser, RawConfigParser
import glob as glob
import re as re
import os.path as op
import time

import numpy as np

from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import create_info, _format_dig_points
from ...annotations import Annotations
from ...transforms import apply_trans, _get_trans
from ...utils import logger, verbose, fill_doc
from ...utils import warn


@fill_doc
def read_raw_persyst(fname, preload=False, verbose=None):
    """Reader for a Persyst (.lay/.dat) recording.

    Parameters
    ----------
    fname : str
        Path to the Persyst header (.lay) file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawPeryst
        A Raw object containing Persyst data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawPersyst(fname, preload, verbose)


def _open(fname):
    return open(fname, 'r', encoding='latin-1')


@fill_doc
class RawPersyst(BaseRaw):
    """Raw object from a Persyst file.

    Parameters
    ----------
    fname : str
        Path to the Persyst header (.lay) file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        logger.info('Loading %s' % fname)

        if not fname.endswith('.lay'):
            fname = fname + '.lay'
        lay_fname = op.basename(fname)
        # Assume it is the same location and has the same name as lay_fname
        dat_fname = lay_fname.replace('.lay', '.dat')

        if not op.exists(lay_fname):
            raise FileNotFoundError('The path you specified does not exist.')
        if not op.exists(dat_fname):
            raise FileNotFoundError('The data (.dat) path you specified '
                                    'does not exist for the lay path, %s'
                                    % lay_fname)

        # takes ~8 min for a 1.5GB file
        t = time.time()

        # get .ini file and replace [] with ''
        data, sections, subsections = inifile.inifile(layFileName,
                                                      'readall')  # sections and subsections currently unused
        for row in data:
            for entry in row:
                if entry == []:
                    entry = ''

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

        # Parse required header fields
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
        # subject_info['birthday'] = inf['age']  # TODO: not formatted properly
        subject_info['sex'] = inf['gender'].replace("\"", "")
        # Recode values
        if subject_info['sex'] in {'M', 'Male', '1'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
        elif subject_info['sex'] in {'F', 'Female', '2'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
        else:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN

        # Create mne structure
        info = create_info(chnames,
                           samplingrate,
                           ch_types='ch_types')
        info.update(subject_info=subject_info, dig=dig)

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

        super(RawPersyst, self).__init__(
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
        data[:] = this_data[idx]

        return data


def _read_ini_contents(fname):
    keys, sections, subsections = ReadAllKeys(fname)

    keys = []
    subsections = []
    sections = []
    with open(fname, 'r') as fin:
        for line in fin:
            status, key, val = _process_ini_line(line)


    return keys, sections, subsections


def _process_ini_line(line):
    """Processes a line read from the ini file.

    Parameters
    ----------
    line : str
        The actual line in the INI file.

    Returns
    -------
    status : int
        Returns the following integers based on status.
        -1  => unknown string found
        0   => empty line found
        1   => section found
        2   => subsection found
        3   => key-value pair found
        4   => comment line found (starting with ;)
    key : str
        The string before the ``'='`` character.
    value : str
        The string from the line after the ``'='`` character.
    """
    key = ''  # default; only return value possibly not set
    line = line.strip()  # remove leading and trailing spaces
    end_idx = len(line) - 1  # get the last index of the line

    if not line:  # empty sequence evaluates to false
        status = 0
        key = ''
        value = ''
        return status, key, value
    elif line[0] == ';':  # comment found
        status = 4
        value = line[1:end_idx + 1]
    elif (line[0] == '[') and (line[end_idx] == ']') and (end_idx + 1 >= 3):  # section found
        status = 1
        value = line[1:end_idx].lower()
    elif (line[0] == '{') and (line[end_idx] == '}') and (end_idx + 1 >= 3):  # subsection found
        status = 2
        value = line[1:end_idx].lower()
    else:  # key found
        if '=' not in line:
            raise RuntimeError('The line %s does not conform '
                               'to the standards. Please check the '
                               '.lay file.' % line)
        pos = line.index('=')
        status = 3
        key = line[0:pos].lower()
        key.strip()
        if not key:
            status = -1
            key = ''
            value = ''
        else:
            value = line[pos + 1:end_idx + 1].lower()
            value.strip()
    return status, value, key

def _read_csv_rows_cols(fname, start, stop, cols, bounds):
    with open(fname, 'rb') as fid:
        fid.seek(bounds[start])
        data = fid.read(bounds[stop] - bounds[start]).decode('latin-1')
        x = np.fromstring(data, float, sep=' ')
    x.shape = (stop - start, -1)
    x = x[:, cols]
    return x
