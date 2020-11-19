# Authors: Kyle Mathewson, Jonathan Kuziek <kuziek@ualberta.ca>
#
# License: BSD (3-clause)

import re as re

import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..utils import _mult_cal_one
from ...utils import logger, verbose, fill_doc
from ...annotations import Annotations


@fill_doc
def read_raw_boxy(fname, preload=False, verbose=None):
    """Reader for an optical imaging recording.

    This function has been tested using the ISS Imagent I and II systems
    and versions 0.40/0.84 of the BOXY recording software.

    Parameters
    ----------
    fname : str
        Path to the BOXY data file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawBOXY
        A Raw object containing BOXY data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawBOXY(fname, preload, verbose)


@fill_doc
class RawBOXY(BaseRaw):
    """Raw object from a BOXY optical imaging file.

    Parameters
    ----------
    fname : str
        Path to the BOXY data file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        logger.info('Loading %s' % fname)

        # Read header file and grab some info.
        start_line = np.inf
        col_names = mrk_col = filetype = mrk_data = end_line = None
        raw_extras = dict()
        raw_extras['offsets'] = list()  # keep track of our offsets
        sfreq = None
        with open(fname, 'r') as fid:
            line_num = 0
            i_line = fid.readline()
            while i_line:
                # most of our lines will be data lines, so check that first
                if line_num >= start_line:
                    assert col_names is not None
                    assert filetype is not None
                    if '#DATA ENDS' in i_line:
                        # Data ends just before this.
                        end_line = line_num
                        break
                    if mrk_col is not None:
                        if filetype == 'non-parsed':
                            # Non-parsed files have different lines lengths.
                            crnt_line = i_line.rsplit(' ')[0]
                            temp_data = re.findall(
                                r'[-+]?\d*\.?\d+', crnt_line)
                            if len(temp_data) == len(col_names):
                                mrk_data.append(float(
                                    re.findall(r'[-+]?\d*\.?\d+', crnt_line)
                                    [mrk_col]))
                        else:
                            crnt_line = i_line.rsplit(' ')[0]
                            mrk_data.append(float(re.findall(
                                r'[-+]?\d*\.?\d+', crnt_line)[mrk_col]))
                    raw_extras['offsets'].append(fid.tell())
                # now proceed with more standard header parsing
                elif 'BOXY.EXE:' in i_line:
                    boxy_ver = re.findall(r'\d*\.\d+',
                                          i_line.rsplit(' ')[-1])[0]
                    # Check that the BOXY version is supported
                    if boxy_ver not in ['0.40', '0.84']:
                        raise RuntimeError('MNE has not been tested with BOXY '
                                           'version (%s)' % boxy_ver)
                elif 'Detector Channels' in i_line:
                    raw_extras['detect_num'] = int(i_line.rsplit(' ')[0])
                elif 'External MUX Channels' in i_line:
                    raw_extras['source_num'] = int(i_line.rsplit(' ')[0])
                elif 'Update Rate (Hz)' in i_line or \
                        'Updata Rate (Hz)' in i_line:
                    # Version 0.40 of the BOXY recording software
                    # (and possibly other versions lower than 0.84) contains a
                    # typo in the raw data file where 'Update Rate' is spelled
                    # "Updata Rate. This will account for this typo.
                    sfreq = float(i_line.rsplit(' ')[0])
                elif '#DATA BEGINS' in i_line:
                    # Data should start a couple lines later.
                    start_line = line_num + 3
                elif line_num == start_line - 2:
                    # Grab names for each column of data.
                    raw_extras['col_names'] = col_names = re.findall(
                        r'\w+\-\w+|\w+\-\d+|\w+', i_line.rsplit(' ')[0])
                    if 'exmux' in col_names:
                        # Change filetype based on data organisation.
                        filetype = 'non-parsed'
                    else:
                        filetype = 'parsed'
                    if 'digaux' in col_names:
                        mrk_col = col_names.index('digaux')
                        mrk_data = list()
                    # raw_extras['offsets'].append(fid.tell())
                elif line_num == start_line - 1:
                    raw_extras['offsets'].append(fid.tell())
                line_num += 1
                i_line = fid.readline()
        assert sfreq is not None
        raw_extras.update(
            filetype=filetype, start_line=start_line, end_line=end_line)

        # Label each channel in our data, for each data type (DC, AC, Ph).
        # Data is organised by channels x timepoint, where the first
        # 'source_num' rows correspond to the first detector, the next
        # 'source_num' rows correspond to the second detector, and so on.
        ch_names = list()
        ch_types = list()
        cals = list()
        for det_num in range(raw_extras['detect_num']):
            for src_num in range(raw_extras['source_num']):
                for i_type, ch_type in [
                        ('DC', 'fnirs_cw_amplitude'),
                        ('AC', 'fnirs_fd_ac_amplitude'),
                        ('Ph', 'fnirs_fd_phase')]:
                    ch_names.append(
                        f'S{src_num + 1}_D{det_num + 1} {i_type}')
                    ch_types.append(ch_type)
                    cals.append(np.pi / 180. if i_type == 'Ph' else 1.)

        # Create info structure.
        info = create_info(ch_names, sfreq, ch_types)
        for ch, cal in zip(info['chs'], cals):
            ch['cal'] = cal

        # Determine how long our data is.
        delta = end_line - start_line
        assert len(raw_extras['offsets']) == delta + 1
        if filetype == 'non-parsed':
            delta //= (raw_extras['source_num'])
        super(RawBOXY, self).__init__(
            info, preload, filenames=[fname], first_samps=[0],
            last_samps=[delta - 1], raw_extras=[raw_extras], verbose=verbose)

        # Now let's grab our markers, if they are present.
        if mrk_data is not None:
            mrk_data = np.array(mrk_data, float)
            # We only want the first instance of each trigger.
            prev_mrk = 0
            mrk_idx = list()
            duration = list()
            tmp_dur = 0
            for i_num, i_mrk in enumerate(mrk_data):
                if i_mrk != 0 and i_mrk != prev_mrk:
                    mrk_idx.append(i_num)
                if i_mrk != 0 and i_mrk == prev_mrk:
                    tmp_dur += 1
                if i_mrk == 0 and i_mrk != prev_mrk:
                    duration.append((tmp_dur + 1) / sfreq)
                    tmp_dur = 0
                prev_mrk = i_mrk
            onset = np.array(mrk_idx) / sfreq
            description = mrk_data[mrk_idx]
            annot = Annotations(onset, duration, description)
            self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        Boxy file organises data in two ways, parsed or un-parsed.
        Regardless of type, output has (n_montages x n_sources x n_detectors
        + n_marker_channels) rows, and (n_timepoints x n_blocks) columns.
        """
        source_num = self._raw_extras[fi]['source_num']
        detect_num = self._raw_extras[fi]['detect_num']
        start_line = self._raw_extras[fi]['start_line']
        end_line = self._raw_extras[fi]['end_line']
        filetype = self._raw_extras[fi]['filetype']
        col_names = self._raw_extras[fi]['col_names']
        offsets = self._raw_extras[fi]['offsets']
        boxy_file = self._filenames[fi]

        # Non-parsed multiplexes sources, so we need source_num times as many
        # lines in that case
        if filetype == 'parsed':
            start_read = start_line + start
            stop_read = start_read + (stop - start)
        else:
            assert filetype == 'non-parsed'
            start_read = start_line + start * source_num
            stop_read = start_read + (stop - start) * source_num
        assert start_read >= start_line
        assert stop_read <= end_line

        # Possible detector names.
        detectors = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:detect_num]

        # Loop through our data.
        one = np.zeros((len(col_names), stop_read - start_read))
        with open(boxy_file, 'r') as fid:
            # Just a more efficient version of this:
            # ii = 0
            # for line_num, i_line in enumerate(fid):
            #     if line_num >= start_read:
            #         if line_num >= stop_read:
            #             break
            #         # Grab actual data.
            #         i_data = i_line.strip().split()
            #         one[:len(i_data), ii] = i_data
            #         ii += 1
            fid.seek(offsets[start_read - start_line], 0)
            for oo in one.T:
                i_data = fid.readline().strip().split()
                oo[:len(i_data)] = i_data

        # in theory we could index in the loop above, but it's painfully slow,
        # so let's just take a hopefully minor memory hit
        if filetype == 'non-parsed':
            ch_idxs = [col_names.index(f'{det}-{i_type}')
                       for det in detectors
                       for i_type in ['DC', 'AC', 'Ph']]
            one = one[ch_idxs].reshape(  # each "time point" multiplexes srcs
                len(detectors), 3, -1, source_num
            ).transpose(  # reorganize into (det, source, DC/AC/Ph, t) order
                0, 3, 1, 2
            ).reshape(  # reshape the way we store it (det x source x DAP, t)
                len(detectors) * source_num * 3, -1)
        else:
            assert filetype == 'parsed'
            ch_idxs = [col_names.index(f'{det}-{i_type}{si + 1}')
                       for det in detectors
                       for si in range(source_num)
                       for i_type in ['DC', 'AC', 'Ph']]
            one = one[ch_idxs]

        # Place our data into the data object in place.
        _mult_cal_one(data, one, idx, cals, mult)
