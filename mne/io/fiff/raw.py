# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import copy
import warnings
import os
import os.path as op

import numpy as np

from ..constants import FIFF
from ..open import fiff_open, _fiff_get_fid
from ..meas_info import read_meas_info
from ..tree import dir_tree_find
from ..tag import read_tag
from ..proj import make_eeg_average_ref_proj, _needs_eeg_average_ref_proj
from ..compensator import get_current_comp, set_current_comp, make_compensator
from ..base import _BaseRaw, _RawShell, _check_raw_compatibility

from ...utils import check_fname, logger, verbose
from ...externals.six import string_types


class RawFIF(_BaseRaw):
    """Raw data

    Parameters
    ----------
    fnames : list, or string
        A list of the raw files to treat as a Raw instance, or a single
        raw file. For files that have automatically been split, only the
        name of the first file has to be specified. Filenames should end
        with raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz,
        raw_tsss.fif or raw_tsss.fif.gz.
    allow_maxshield : bool, (default False)
        allow_maxshield if True, allow loading of data that has been
        processed with Maxshield. Maxshield-processed data should generally
        not be loaded directly, but should be processed using SSS first.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    proj : bool
        Apply the signal space projection (SSP) operators present in
        the file to the data. Note: Once the projectors have been
        applied, they can no longer be removed. It is usually not
        recommended to apply the projectors at this point as they are
        applied automatically later on (e.g. when computing inverse
        solutions).
    compensation : None | int
        If None the compensation in the data is not modified.
        If set to n, e.g. 3, apply gradient compensation of grade n as
        for CTF systems.
    add_eeg_ref : bool
        If True, add average EEG reference projector (if it's not already
        present).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes
    ----------
    info : dict
        Measurement info.
    ch_names : list of string
        List of channels' names.
    n_times : int
        Total number of time points in the raw file.
    preload : bool
        Indicates whether raw data are in memory.
    verbose : bool, str, int, or None
        See above.
    """
    @verbose
    def __init__(self, fnames, allow_maxshield=False, preload=False,
                 proj=False, compensation=None, add_eeg_ref=True,
                 verbose=None):

        if not isinstance(fnames, list):
            fnames = [fnames]
        fnames = [op.realpath(f) for f in fnames]
        split_fnames = []

        raws = []
        for ii, fname in enumerate(fnames):
            do_check_fname = fname not in split_fnames
            raw, next_fname = self._read_raw_file(fname, allow_maxshield,
                                                  preload, compensation,
                                                  do_check_fname)
            raws.append(raw)
            if next_fname is not None:
                if not op.exists(next_fname):
                    logger.warning('Split raw file detected but next file %s '
                                   'does not exist.' % next_fname)
                    continue
                if next_fname in fnames:
                    # the user manually specified the split files
                    logger.info('Note: %s is part of a split raw file. It is '
                                'not necessary to manually specify the parts '
                                'in this case; simply construct Raw using '
                                'the name of the first file.' % next_fname)
                    continue

                # process this file next
                fnames.insert(ii + 1, next_fname)
                split_fnames.append(next_fname)

        _check_raw_compatibility(raws)

        super(RawFIF, self).__init__(
            copy.deepcopy(raws[0].info), False,
            [r.first_samp for r in raws], [r.last_samp for r in raws],
            [r.filename for r in raws], [r._rawdir for r in raws],
            copy.deepcopy(raws[0].comp), raws[0]._orig_comp_grade,
            raws[0].orig_format,
            verbose=verbose)

        # combine information from each raw file to construct self
        if add_eeg_ref and _needs_eeg_average_ref_proj(self.info):
            eeg_ref = make_eeg_average_ref_proj(self.info, activate=False)
            self.add_proj(eeg_ref)

        if preload:
            self._preload_data(preload)
        else:
            self.preload = False

        # setup the SSP projector
        if proj:
            self.apply_proj()

    @verbose
    def _read_raw_file(self, fname, allow_maxshield, preload, compensation,
                       do_check_fname=True, verbose=None):
        """Read in header information from a raw file"""
        logger.info('Opening raw data file %s...' % fname)

        if do_check_fname:
            check_fname(fname, 'raw', ('raw.fif', 'raw_sss.fif',
                                       'raw_tsss.fif', 'raw.fif.gz',
                                       'raw_sss.fif.gz', 'raw_tsss.fif.gz'))

        #   Read in the whole file if preload is on and .fif.gz (saves time)
        ext = os.path.splitext(fname)[1].lower()
        whole_file = preload if '.gz' in ext else False
        ff, tree, _ = fiff_open(fname, preload=whole_file)
        with ff as fid:
            #   Read the measurement info
            info, meas = read_meas_info(fid, tree)

            #   Locate the data of interest
            raw_node = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
            if len(raw_node) == 0:
                raw_node = dir_tree_find(meas, FIFF.FIFFB_CONTINUOUS_DATA)
                if (len(raw_node) == 0) and allow_maxshield:
                    raw_node = dir_tree_find(meas, FIFF.FIFFB_SMSH_RAW_DATA)
                    msg = ('This file contains raw Internal Active Shielding '
                           'data. It may be distorted. Elekta recommends it be'
                           ' run through MaxFilter to produce reliable '
                           'results. Consider closing the file and running '
                           'MaxFilter on the data.')
                    info['maxshield'] = True
                    warnings.warn(msg)

            if len(raw_node) == 0:
                raise ValueError('No raw data in %s' % fname)

            if len(raw_node) == 1:
                raw_node = raw_node[0]

            #   Set up the output structure
            info['filename'] = fname

            #   Process the directory
            directory = raw_node['directory']
            nent = raw_node['nent']
            nchan = int(info['nchan'])
            first = 0
            first_samp = 0
            first_skip = 0

            #   Get first sample tag if it is there
            if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, directory[first].pos)
                first_samp = int(tag.data)
                first += 1

            #   Omit initial skip
            if directory[first].kind == FIFF.FIFF_DATA_SKIP:
                # This first skip can be applied only after we know the bufsize
                tag = read_tag(fid, directory[first].pos)
                first_skip = int(tag.data)
                first += 1

            raw = _RawShell()
            raw.filename = fname
            raw.first_samp = first_samp

            #   Go through the remaining tags in the directory
            rawdir = list()
            nskip = 0
            orig_format = None
            for k in range(first, nent):
                ent = directory[k]
                if ent.kind == FIFF.FIFF_DATA_SKIP:
                    tag = read_tag(fid, ent.pos)
                    nskip = int(tag.data)
                elif ent.kind == FIFF.FIFF_DATA_BUFFER:
                    #   Figure out the number of samples in this buffer
                    if ent.type == FIFF.FIFFT_DAU_PACK16:
                        nsamp = ent.size // (2 * nchan)
                    elif ent.type == FIFF.FIFFT_SHORT:
                        nsamp = ent.size // (2 * nchan)
                    elif ent.type == FIFF.FIFFT_FLOAT:
                        nsamp = ent.size // (4 * nchan)
                    elif ent.type == FIFF.FIFFT_DOUBLE:
                        nsamp = ent.size // (8 * nchan)
                    elif ent.type == FIFF.FIFFT_INT:
                        nsamp = ent.size // (4 * nchan)
                    elif ent.type == FIFF.FIFFT_COMPLEX_FLOAT:
                        nsamp = ent.size // (8 * nchan)
                    elif ent.type == FIFF.FIFFT_COMPLEX_DOUBLE:
                        nsamp = ent.size // (16 * nchan)
                    else:
                        raise ValueError('Cannot handle data buffers of type '
                                         '%d' % ent.type)
                    if orig_format is None:
                        if ent.type == FIFF.FIFFT_DAU_PACK16:
                            orig_format = 'short'
                        elif ent.type == FIFF.FIFFT_SHORT:
                            orig_format = 'short'
                        elif ent.type == FIFF.FIFFT_FLOAT:
                            orig_format = 'single'
                        elif ent.type == FIFF.FIFFT_DOUBLE:
                            orig_format = 'double'
                        elif ent.type == FIFF.FIFFT_INT:
                            orig_format = 'int'
                        elif ent.type == FIFF.FIFFT_COMPLEX_FLOAT:
                            orig_format = 'single'
                        elif ent.type == FIFF.FIFFT_COMPLEX_DOUBLE:
                            orig_format = 'double'

                    #  Do we have an initial skip pending?
                    if first_skip > 0:
                        first_samp += nsamp * first_skip
                        raw.first_samp = first_samp
                        first_skip = 0

                    #  Do we have a skip pending?
                    if nskip > 0:
                        rawdir.append(dict(ent=None, first=first_samp,
                                           last=first_samp + nskip * nsamp - 1,
                                           nsamp=nskip * nsamp))
                        first_samp += nskip * nsamp
                        nskip = 0

                    #  Add a data buffer
                    rawdir.append(dict(ent=ent, first=first_samp,
                                       last=first_samp + nsamp - 1,
                                       nsamp=nsamp))
                    first_samp += nsamp

            # Try to get the next filename tag for split files
            nodes_list = dir_tree_find(tree, FIFF.FIFFB_REF)
            next_fname = None
            for nodes in nodes_list:
                next_fname = None
                for ent in nodes['directory']:
                    if ent.kind == FIFF.FIFF_REF_ROLE:
                        tag = read_tag(fid, ent.pos)
                        role = int(tag.data)
                        if role != FIFF.FIFFV_ROLE_NEXT_FILE:
                            next_fname = None
                            break
                    if ent.kind == FIFF.FIFF_REF_FILE_NAME:
                        tag = read_tag(fid, ent.pos)
                        next_fname = op.join(op.dirname(fname), tag.data)
                    if ent.kind == FIFF.FIFF_REF_FILE_NUM:
                        # Some files don't have the name, just the number. So
                        # we construct the name from the current name.
                        if next_fname is not None:
                            continue
                        next_num = read_tag(fid, ent.pos).data
                        path, base = op.split(fname)
                        idx = base.find('.')
                        idx2 = base.rfind('-')
                        if idx2 < 0 and next_num == 1:
                            # this is the first file, which may not be numbered
                            next_fname = op.join(
                                path, '%s-%d.%s' % (base[:idx], next_num,
                                                    base[idx + 1:]))
                            continue
                        num_str = base[idx2 + 1:idx]
                        if not num_str.isdigit():
                            continue
                        next_fname = op.join(path, '%s-%d.%s' % (base[:idx2],
                                             next_num, base[idx + 1:]))
                if next_fname is not None:
                    break

        raw.last_samp = first_samp - 1
        raw.orig_format = orig_format

        #   Add the calibration factors
        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        raw._cals = cals
        raw._rawdir = rawdir
        raw.comp = None
        raw._orig_comp_grade = None

        #   Set up the CTF compensator
        current_comp = get_current_comp(info)
        if current_comp is not None:
            logger.info('Current compensation grade : %d' % current_comp)

        if compensation is not None:
            raw.comp = make_compensator(info, current_comp, compensation)
            if raw.comp is not None:
                logger.info('Appropriate compensator added to change to '
                            'grade %d.' % (compensation))
                raw._orig_comp_grade = current_comp
                set_current_comp(info, compensation)

        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    raw.first_samp, raw.last_samp,
                    float(raw.first_samp) / info['sfreq'],
                    float(raw.last_samp) / info['sfreq']))

        # store the original buffer size
        info['buffer_size_sec'] = (np.median([r['nsamp'] for r in rawdir]) /
                                   info['sfreq'])

        raw.info = info
        raw.verbose = verbose

        logger.info('Ready.')

        return raw, next_fname

    @verbose
    def _read_segment(self, start=0, stop=None, sel=None, data_buffer=None,
                      projector=None, verbose=None):
        """Read a chunk of raw data"""
        #  Initial checks
        start = int(start)
        stop = self.n_times if stop is None else min([int(stop), self.n_times])

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(self.info['sfreq']),
                     (stop - 1) / float(self.info['sfreq'])))

        #  Initialize the data and calibration vector
        nchan = self.info['nchan']

        n_sel_channels = nchan if sel is None else len(sel)
        # convert sel to a slice if possible for efficiency
        if sel is not None and len(sel) > 1 and np.all(np.diff(sel) == 1):
            sel = slice(sel[0], sel[-1] + 1)
        idx = slice(None, None, None) if sel is None else sel
        data_shape = (n_sel_channels, stop - start)
        if isinstance(data_buffer, np.ndarray):
            if data_buffer.shape != data_shape:
                raise ValueError('data_buffer has incorrect shape')
            data = data_buffer
        else:
            data = None  # we will allocate it later, once we know the type

        mult = list()
        for ri in range(len(self._first_samps)):
            mult.append(np.diag(self._cals.ravel()))
            if self.comp is not None:
                mult[ri] = np.dot(self.comp, mult[ri])
            if projector is not None:
                mult[ri] = np.dot(projector, mult[ri])
            mult[ri] = mult[ri][idx]

        # deal with having multiple files accessed by the raw object
        cumul_lens = np.concatenate(([0], np.array(self._raw_lengths,
                                                   dtype='int')))
        cumul_lens = np.cumsum(cumul_lens)
        files_used = np.logical_and(np.less(start, cumul_lens[1:]),
                                    np.greater_equal(stop - 1,
                                                     cumul_lens[:-1]))

        first_file_used = False
        s_off = 0
        dest = 0
        if isinstance(idx, slice):
            cals = self._cals.ravel()[idx][:, np.newaxis]
        else:
            cals = self._cals.ravel()[:, np.newaxis]

        for fi in np.nonzero(files_used)[0]:
            start_loc = self._first_samps[fi]
            # first iteration (only) could start in the middle somewhere
            if not first_file_used:
                first_file_used = True
                start_loc += start - cumul_lens[fi]
            stop_loc = np.min([stop - 1 - cumul_lens[fi] +
                               self._first_samps[fi], self._last_samps[fi]])
            if start_loc < self._first_samps[fi]:
                raise ValueError('Bad array indexing, could be a bug')
            if stop_loc > self._last_samps[fi]:
                raise ValueError('Bad array indexing, could be a bug')
            if stop_loc < start_loc:
                raise ValueError('Bad array indexing, could be a bug')
            len_loc = stop_loc - start_loc + 1
            fid = _fiff_get_fid(self._filenames[fi])

            for this in self._rawdirs[fi]:

                #  Do we need this buffer
                if this['last'] >= start_loc:
                    #  The picking logic is a bit complicated
                    if stop_loc > this['last'] and start_loc < this['first']:
                        #    We need the whole buffer
                        first_pick = 0
                        last_pick = this['nsamp']
                        logger.debug('W')

                    elif start_loc >= this['first']:
                        first_pick = start_loc - this['first']
                        if stop_loc <= this['last']:
                            #   Something from the middle
                            last_pick = this['nsamp'] + stop_loc - this['last']
                            logger.debug('M')
                        else:
                            #   From the middle to the end
                            last_pick = this['nsamp']
                            logger.debug('E')
                    else:
                        #    From the beginning to the middle
                        first_pick = 0
                        last_pick = stop_loc - this['first'] + 1
                        logger.debug('B')

                    #   Now we are ready to pick
                    picksamp = last_pick - first_pick
                    if picksamp > 0:
                        # only read data if it exists
                        if this['ent'] is not None:
                            one = read_tag(fid, this['ent'].pos,
                                           shape=(this['nsamp'], nchan),
                                           rlims=(first_pick, last_pick)).data
                            if np.isrealobj(one):
                                dtype = np.float
                            else:
                                dtype = np.complex128
                            one.shape = (picksamp, nchan)
                            one = one.T.astype(dtype)
                            # use proj + cal factors in mult
                            if mult is not None:
                                one[idx] = np.dot(mult[fi], one)
                            else:  # apply just the calibration factors
                                # this logic is designed to limit memory copies
                                if isinstance(idx, slice):
                                    # This is a view operation, so it's fast
                                    one[idx] *= cals
                                else:
                                    # Extra operations are actually faster here
                                    # than creating a new array
                                    # (fancy indexing)
                                    one *= cals

                            # if not already done, allocate array with
                            # right type
                            data = _allocate_data(data, data_buffer,
                                                  data_shape, dtype)
                            if isinstance(idx, slice):
                                # faster to slice in data than doing
                                # one = one[idx] sooner
                                data[:, dest:(dest + picksamp)] = one[idx]
                            else:
                                # faster than doing one = one[idx]
                                data_view = data[:, dest:(dest + picksamp)]
                                for ii, ix in enumerate(idx):
                                    data_view[ii] = one[ix]
                        dest += picksamp

                #   Done?
                if this['last'] >= stop_loc:
                    # if not already done, allocate array with float dtype
                    data = _allocate_data(data, data_buffer, data_shape,
                                          np.float)
                    break

            fid.close()  # clean it up
            s_off += len_loc
            # double-check our math
            if not s_off == dest:
                raise ValueError('Incorrect file reading')

        logger.info('[done]')
        times = np.arange(start, stop) / self.info['sfreq']

        return data, times


def _allocate_data(data, data_buffer, data_shape, dtype):
    if data is None:
        # if not already done, allocate array with right type
        if isinstance(data_buffer, string_types):
            # use a memmap
            data = np.memmap(data_buffer, mode='w+',
                             dtype=dtype, shape=data_shape)
        else:
            data = np.zeros(data_shape, dtype=dtype)
    return data


def read_raw_fif(fnames, allow_maxshield=False, preload=False,
                 proj=False, compensation=None, add_eeg_ref=True,
                 verbose=None):
    """Reader function for Raw FIF data

    Parameters
    ----------
    fnames : list, or string
        A list of the raw files to treat as a Raw instance, or a single
        raw file. For files that have automatically been split, only the
        name of the first file has to be specified. Filenames should end
        with raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz,
        raw_tsss.fif or raw_tsss.fif.gz.
    allow_maxshield : bool, (default False)
        allow_maxshield if True, allow loading of data that has been
        processed with Maxshield. Maxshield-processed data should generally
        not be loaded directly, but should be processed using SSS first.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    proj : bool
        Apply the signal space projection (SSP) operators present in
        the file to the data. Note: Once the projectors have been
        applied, they can no longer be removed. It is usually not
        recommended to apply the projectors at this point as they are
        applied automatically later on (e.g. when computing inverse
        solutions).
    compensation : None | int
        If None the compensation in the data is not modified.
        If set to n, e.g. 3, apply gradient compensation of grade n as
        for CTF systems.
    add_eeg_ref : bool
        If True, add average EEG reference projector (if it's not already
        present).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawFIF
        A Raw object containing FIF data.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    return RawFIF(fnames=fnames, allow_maxshield=allow_maxshield,
                  preload=preload, proj=proj, compensation=compensation,
                  add_eeg_ref=add_eeg_ref, verbose=verbose)
