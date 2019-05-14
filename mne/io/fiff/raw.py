# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import copy
import os
import os.path as op

import numpy as np

from ..constants import FIFF
from ..open import fiff_open, _fiff_get_fid, _get_next_fname
from ..meas_info import read_meas_info
from ..tree import dir_tree_find
from ..tag import read_tag, read_tag_info
from ..base import (BaseRaw, _RawShell, _check_raw_compatibility,
                    _check_maxshield)
from ..utils import _mult_cal_one

from ...annotations import Annotations, _read_annotations_fif

from ...event import AcqParserFIF
from ...utils import check_fname, logger, verbose, warn, fill_doc


@fill_doc
class Raw(BaseRaw):
    """Raw data in FIF format.

    Parameters
    ----------
    fname : str | file-like
        The raw filename to load. For files that have automatically been split,
        the split part will be automatically loaded. Filenames should end
        with raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz,
        raw_tsss.fif or raw_tsss.fif.gz. If a file-like object is provided,
        preloading must be used.

        .. versionchanged:: 0.18
           Support for file-like objects.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    %(verbose)s

    Attributes
    ----------
    info : dict
        :class:`Measurement info <mne.Info>`.
    ch_names : list of string
        List of channels' names.
    n_times : int
        Total number of time points in the raw file.
    times :  ndarray
        Time vector in seconds. Starts from 0, independently of `first_samp`
        value. Time interval between consecutive time samples is equal to the
        inverse of the sampling frequency.
    preload : bool
        Indicates whether raw data are in memory.
    %(verbose)s
    """

    @verbose
    def __init__(self, fname, allow_maxshield=False, preload=False,
                 verbose=None):  # noqa: D102
        raws = []
        do_check_fname = isinstance(fname, str)
        next_fname = fname
        while next_fname is not None:
            raw, next_fname, buffer_size_sec = \
                self._read_raw_file(next_fname, allow_maxshield,
                                    preload, do_check_fname)
            do_check_fname = False
            raws.append(raw)
            if next_fname is not None:
                if not op.exists(next_fname):
                    warn('Split raw file detected but next file %s does not '
                         'exist.' % next_fname)
                    break
        if not isinstance(fname, str):
            # avoid serialization error when copying file-like
            fname = None  # noqa

        _check_raw_compatibility(raws)
        super(Raw, self).__init__(
            copy.deepcopy(raws[0].info), False,
            [r.first_samp for r in raws], [r.last_samp for r in raws],
            [r.filename for r in raws], [r._raw_extras for r in raws],
            raws[0].orig_format, None, buffer_size_sec=buffer_size_sec,
            verbose=verbose)

        # combine annotations
        self.set_annotations(raws[0].annotations, emit_warning=False)

        # Add annotations for in-data skips
        for extra in self._raw_extras:
            start = np.array([e['first'] for e in extra if e['ent'] is None])
            stop = np.array([e['last'] for e in extra if e['ent'] is None])
            duration = (stop - start + 1.) / self.info['sfreq']

            annot = Annotations(onset=(start / self.info['sfreq']),
                                duration=duration,
                                description='BAD_ACQ_SKIP',
                                orig_time=self.info['meas_date'])

            self._annotations += annot

        if preload:
            self._preload_data(preload)
        else:
            self.preload = False
        # If using a file-like object, fix the filenames to be representative
        # strings now instead of the file-like objects
        self._filenames = [_get_fname_rep(fname) for fname in self._filenames]

    @verbose
    def _read_raw_file(self, fname, allow_maxshield, preload,
                       do_check_fname=True, verbose=None):
        """Read in header information from a raw file."""
        logger.info('Opening raw data file %s...' % fname)

        if do_check_fname:
            check_fname(fname, 'raw', ('raw.fif', 'raw_sss.fif',
                                       'raw_tsss.fif', 'raw.fif.gz',
                                       'raw_sss.fif.gz', 'raw_tsss.fif.gz'))

        #   Read in the whole file if preload is on and .fif.gz (saves time)
        if isinstance(fname, str):
            # filename
            fname = op.realpath(fname)
            ext = os.path.splitext(fname)[1].lower()
            whole_file = preload if '.gz' in ext else False
            del ext
        else:
            # file-like
            if not preload:
                raise ValueError('preload must be used with file-like objects')
            whole_file = True
        fname_rep = _get_fname_rep(fname)
        ff, tree, _ = fiff_open(fname, preload=whole_file)
        with ff as fid:
            #   Read the measurement info

            info, meas = read_meas_info(fid, tree, clean_bads=True)
            annotations = _read_annotations_fif(fid, tree)

            #   Locate the data of interest
            raw_node = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
            if len(raw_node) == 0:
                raw_node = dir_tree_find(meas, FIFF.FIFFB_CONTINUOUS_DATA)
                if (len(raw_node) == 0):
                    raw_node = dir_tree_find(meas, FIFF.FIFFB_SMSH_RAW_DATA)
                    if (len(raw_node) == 0):
                        raise ValueError('No raw data in %s' % fname_rep)
                    _check_maxshield(allow_maxshield)
                    info['maxshield'] = True

            if len(raw_node) == 1:
                raw_node = raw_node[0]

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
                _check_entry(first, nent)

            #   Omit initial skip
            if directory[first].kind == FIFF.FIFF_DATA_SKIP:
                # This first skip can be applied only after we know the bufsize
                tag = read_tag(fid, directory[first].pos)
                first_skip = int(tag.data)
                first += 1
                _check_entry(first, nent)

            raw = _RawShell()
            raw.filename = fname
            raw.first_samp = first_samp
            raw.set_annotations(annotations)

            #   Go through the remaining tags in the directory
            raw_extras = list()
            nskip = 0
            orig_format = None

            for k in range(first, nent):
                ent = directory[k]
                # There can be skips in the data (e.g., if the user unclicked)
                # an re-clicked the button
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
                        raw_extras.append(dict(
                            ent=None, first=first_samp, nsamp=nskip * nsamp,
                            last=first_samp + nskip * nsamp - 1))
                        first_samp += nskip * nsamp
                        nskip = 0

                    #  Add a data buffer
                    raw_extras.append(dict(ent=ent, first=first_samp,
                                           last=first_samp + nsamp - 1,
                                           nsamp=nsamp))
                    first_samp += nsamp

            next_fname = _get_next_fname(fid, fname_rep, tree)

        raw.last_samp = first_samp - 1
        raw.orig_format = orig_format

        #   Add the calibration factors
        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        raw._cals = cals
        raw._raw_extras = raw_extras
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    raw.first_samp, raw.last_samp,
                    float(raw.first_samp) / info['sfreq'],
                    float(raw.last_samp) / info['sfreq']))

        # store the original buffer size
        buffer_size_sec = np.median(
            [r['nsamp'] for r in raw_extras]) / info['sfreq']

        raw.info = info
        raw.verbose = verbose

        logger.info('Ready.')

        return raw, next_fname, buffer_size_sec

    @property
    def _dtype(self):
        """Get the dtype to use to store data from disk."""
        if self._dtype_ is not None:
            return self._dtype_
        dtype = None
        for raw_extra, filename in zip(self._raw_extras, self._filenames):
            for this in raw_extra:
                if this['ent'] is not None:
                    with _fiff_get_fid(filename) as fid:
                        fid.seek(this['ent'].pos, 0)
                        tag = read_tag_info(fid)
                        if tag is not None:
                            if tag.type in (FIFF.FIFFT_COMPLEX_FLOAT,
                                            FIFF.FIFFT_COMPLEX_DOUBLE):
                                dtype = np.complex128
                            else:
                                dtype = np.float64
                    if dtype is not None:
                        break
            if dtype is not None:
                break
        if dtype is None:
            raise RuntimeError('bug in reading')
        self._dtype_ = dtype
        return dtype

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        stop -= 1
        offset = 0
        with _fiff_get_fid(self._filenames[fi]) as fid:
            for this in self._raw_extras[fi]:
                #  Do we need this buffer
                if this['last'] >= start:
                    #  The picking logic is a bit complicated
                    if stop > this['last'] and start < this['first']:
                        #    We need the whole buffer
                        first_pick = 0
                        last_pick = this['nsamp']
                        logger.debug('W')

                    elif start >= this['first']:
                        first_pick = start - this['first']
                        if stop <= this['last']:
                            #   Something from the middle
                            last_pick = this['nsamp'] + stop - this['last']
                            logger.debug('M')
                        else:
                            #   From the middle to the end
                            last_pick = this['nsamp']
                            logger.debug('E')
                    else:
                        #    From the beginning to the middle
                        first_pick = 0
                        last_pick = stop - this['first'] + 1
                        logger.debug('B')

                    #   Now we are ready to pick
                    picksamp = last_pick - first_pick
                    if picksamp > 0:
                        # only read data if it exists
                        if this['ent'] is not None:
                            one = read_tag(fid, this['ent'].pos,
                                           shape=(this['nsamp'],
                                                  self.info['nchan']),
                                           rlims=(first_pick, last_pick)).data
                            one.shape = (picksamp, self.info['nchan'])
                            _mult_cal_one(data[:, offset:(offset + picksamp)],
                                          one.T, idx, cals, mult)
                        offset += picksamp

                #   Done?
                if this['last'] >= stop:
                    break

    def fix_mag_coil_types(self):
        """Fix Elekta magnetometer coil types.

        Returns
        -------
        raw : instance of Raw
            The raw object. Operates in place.

        Notes
        -----
        This function changes magnetometer coil types 3022 (T1: SQ20483N) and
        3023 (T2: SQ20483-A) to 3024 (T3: SQ20950N) in the channel definition
        records in the info structure.

        Neuromag Vectorview systems can contain magnetometers with two
        different coil sizes (3022 and 3023 vs. 3024). The systems
        incorporating coils of type 3024 were introduced last and are used at
        the majority of MEG sites. At some sites with 3024 magnetometers,
        the data files have still defined the magnetometers to be of type
        3022 to ensure compatibility with older versions of Neuromag software.
        In the MNE software as well as in the present version of Neuromag
        software coil type 3024 is fully supported. Therefore, it is now safe
        to upgrade the data files to use the true coil type.

        .. note:: The effect of the difference between the coil sizes on the
                  current estimates computed by the MNE software is very small.
                  Therefore the use of mne_fix_mag_coil_types is not mandatory.
        """
        from ...channels import fix_mag_coil_types
        fix_mag_coil_types(self.info)
        return self

    @property
    def acqparser(self):
        """The AcqParserFIF for the measurement info.

        See Also
        --------
        mne.AcqParserFIF
        """
        if getattr(self, '_acqparser', None) is None:
            self._acqparser = AcqParserFIF(self.info)
        return self._acqparser


def _get_fname_rep(fname):
    if isinstance(fname, str):
        return fname
    else:
        return 'File-like %r' % (fname,)


def _check_entry(first, nent):
    """Sanity check entries."""
    if first >= nent:
        raise IOError('Could not read data, perhaps this is a corrupt file')


@fill_doc
def read_raw_fif(fname, allow_maxshield=False, preload=False, verbose=None):
    """Reader function for Raw FIF data.

    Parameters
    ----------
    fname : str | file-like
        The raw filename to load. For files that have automatically been split,
        the split part will be automatically loaded. Filenames should end
        with raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz,
        raw_tsss.fif or raw_tsss.fif.gz. If a file-like object is provided,
        preloading must be used.

        .. versionchanged:: 0.18
           Support for file-like objects.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be "yes" to load without eliciting a warning.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        A Raw object containing FIF data.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    return Raw(fname=fname, allow_maxshield=allow_maxshield,
               preload=preload, verbose=verbose)
