# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np

from .constants import FIFF
from .open import fiff_open
from .tag import read_tag
from .tree import dir_tree_find
from .meas_info import read_meas_info, write_meas_info
from .proj import make_projector_info
from ..baseline import rescale

from .write import start_file, start_block, end_file, end_block, \
                   write_int, write_string, write_float_matrix, \
                   write_id


class Evoked(object):
    """Evoked data

    Parameters
    ----------
    fname : string
        Name of evoked/average FIF file to load.
        If None no data is loaded.

    setno : int
        Dataset ID number. Optional if there is only one data set
        in file.

    proj : bool, optional
        Apply SSP projection vectors

    Attributes
    ----------
    info: dict
        Measurement info

    ch_names: list of string
        List of channels' names

    nave : int
        Number of averaged epochs

    aspect_kind :
        aspect_kind

    first : int
        First time sample

    last : int
        Last time sample

    comment : string
        Comment on dataset. Can be the condition.

    times : array
        Array of time instants in seconds

    data : 2D array of shape [n_channels x n_times]
        Evoked response.
    """

    def __init__(self, fname, setno=None, baseline=None, proj=True):
        if fname is None:
            return

        print 'Reading %s ...' % fname
        fid, tree, _ = fiff_open(fname)

        #   Read the measurement info
        info, meas = read_meas_info(fid, tree)
        info['filename'] = fname

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        if len(processed) == 0:
            fid.close()
            raise ValueError('Could not find processed data')

        evoked_node = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
        if len(evoked_node) == 0:
            fid.close()
            raise ValueError('Could not find evoked data')

        if setno is None:
            if len(evoked_node) > 1:
                fid.close()
                raise ValueError('%d datasets present. '
                                 'setno parameter mush be set'
                                 % len(evoked_node))
            else:
                setno = 0

        if setno >= len(evoked_node) or setno < 0:
            fid.close()
            raise ValueError('Data set selector out of range')

        my_evoked = evoked_node[setno]

        # Identify the aspects
        aspects = dir_tree_find(my_evoked, FIFF.FIFFB_ASPECT)
        if len(aspects) > 1:
            print 'Multiple aspects found. Taking first one.'
        my_aspect = aspects[0]

        # Now find the data in the evoked block
        nchan = 0
        sfreq = -1
        chs = []
        comment = None
        for k in range(my_evoked['nent']):
            kind = my_evoked['directory'][k].kind
            pos = my_evoked['directory'][k].pos
            if kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comment = tag.data
            elif kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, pos)
                first = int(tag.data)
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                tag = read_tag(fid, pos)
                last = int(tag.data)
            elif kind == FIFF.FIFF_NCHAN:
                tag = read_tag(fid, pos)
                nchan = int(tag.data)
            elif kind == FIFF.FIFF_SFREQ:
                tag = read_tag(fid, pos)
                sfreq = float(tag.data)
            elif kind == FIFF.FIFF_CH_INFO:
                tag = read_tag(fid, pos)
                chs.append(tag.data)

        if comment is None:
            comment = 'No comment'

        #   Local channel information?
        if nchan > 0:
            if chs is None:
                fid.close()
                raise ValueError('Local channel information was not found '
                                 'when it was expected.')

            if len(chs) != nchan:
                fid.close()
                raise ValueError('Number of channels and number of channel '
                                 'definitions are different')

            info['chs'] = chs
            info['nchan'] = nchan
            print ('    Found channel information in evoked data. nchan = %d'
                                                                    % nchan)
            if sfreq > 0:
                info['sfreq'] = sfreq

        nsamp = last - first + 1
        print '    Found the data of interest:'
        print '        t = %10.2f ... %10.2f ms (%s)' % (
            1000 * first / info['sfreq'], 1000 * last / info['sfreq'], comment)
        if info['comps'] is not None:
            print ('        %d CTF compensation matrices available'
                                                          % len(info['comps']))

        # Read the data in the aspect block
        nave = 1
        epoch = []
        for k in range(my_aspect['nent']):
            kind = my_aspect['directory'][k].kind
            pos = my_aspect['directory'][k].pos
            if kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comment = tag.data
            elif kind == FIFF.FIFF_ASPECT_KIND:
                tag = read_tag(fid, pos)
                aspect_kind = int(tag.data)
            elif kind == FIFF.FIFF_NAVE:
                tag = read_tag(fid, pos)
                nave = int(tag.data)
            elif kind == FIFF.FIFF_EPOCH:
                tag = read_tag(fid, pos)
                epoch.append(tag)

        print '        nave = %d - aspect type = %d' % (nave, aspect_kind)

        nepoch = len(epoch)
        if nepoch != 1 and nepoch != info['nchan']:
            fid.close()
            raise ValueError('Number of epoch tags is unreasonable '
                         '(nepoch = %d nchan = %d)' % (nepoch, info['nchan']))

        if nepoch == 1:
            # Only one epoch
            all_data = epoch[0].data
            # May need a transpose if the number of channels is one
            if all_data.shape[1] == 1 and info['nchan'] == 1:
                all_data = all_data.T
        else:
            # Put the old style epochs together
            all_data = np.concatenate([e.data[None, :] for e in epoch], axis=0)

        if all_data.shape[1] != nsamp:
            fid.close()
            raise ValueError('Incorrect number of samples (%d instead of %d)'
                              % (all_data.shape[1], nsamp))

        # Calibrate
        cals = np.array([info['chs'][k]['cal'] for k in range(info['nchan'])])
        all_data = cals[:, None] * all_data

        times = np.arange(first, last + 1, dtype=np.float) / info['sfreq']

        # Set up projection
        if info['projs'] is None or not proj:
            print 'No projector specified for these data'
            self.proj = None
        else:
            #   Activate the projection items
            for proj in info['projs']:
                proj['active'] = True

            print '%d projection items activated' % len(info['projs'])

            #   Create the projector
            proj, nproj = make_projector_info(info)
            if nproj == 0:
                print 'The projection vectors do not apply to these channels'
                self.proj = None
            else:
                print ('Created an SSP operator (subspace dimension = %d)'
                                                                    % nproj)
                self.proj = proj

        if self.proj is not None:
            print "SSP projectors applied..."
            all_data = np.dot(self.proj, all_data)

        # Run baseline correction
        all_data = rescale(all_data, times, baseline, 'mean', verbose=True,
                        copy=False)

        # Put it all together
        self.info = info
        self.nave = nave
        self.aspect_kind = aspect_kind
        self.first = first
        self.last = last
        self.comment = comment
        self.times = times
        self.data = all_data

        fid.close()

    def save(self, fname):
        """Save dataset to file.

        Parameters
        ----------
        fname : string
            Name of the file where to save the data.
        """

        # Create the file and save the essentials
        fid = start_file(fname)

        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        if self.info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, self.info['meas_id'])

        # Write measurement info
        write_meas_info(fid, self.info)

        # One or more evoked data sets
        start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        start_block(fid, FIFF.FIFFB_EVOKED)

        # Comment is optional
        if len(self.comment) > 0:
            write_string(fid, FIFF.FIFF_COMMENT, self.comment)

        # First and last sample
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, self.first)
        write_int(fid, FIFF.FIFF_LAST_SAMPLE, self.last)

        # The epoch itself
        start_block(fid, FIFF.FIFFB_ASPECT)

        write_int(fid, FIFF.FIFF_ASPECT_KIND, self.aspect_kind)
        write_int(fid, FIFF.FIFF_NAVE, self.nave)

        decal = np.zeros((self.info['nchan'], self.info['nchan']))
        for k in range(self.info['nchan']):
            decal[k, k] = 1.0 / self.info['chs'][k]['cal']

        write_float_matrix(fid, FIFF.FIFF_EPOCH, np.dot(decal, self.data))
        end_block(fid, FIFF.FIFFB_ASPECT)
        end_block(fid, FIFF.FIFFB_EVOKED)
        end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        end_block(fid, FIFF.FIFFB_MEAS)
        end_file(fid)

    def __repr__(self):
        s = "comment : %s" % self.comment
        s += ", time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", n_epochs : %d" % self.nave
        s += ", n_channels x n_times : %s x %s" % self.data.shape
        return "Evoked (%s)" % s

    @property
    def ch_names(self):
        return self.info['ch_names']

    def crop(self, tmin=None, tmax=None):
        """Crop data to a given time interval
        """
        times = self.times
        mask = np.ones(len(times), dtype=np.bool)
        if tmin is not None:
            mask = mask & (times >= tmin)
        if tmax is not None:
            mask = mask & (times <= tmax)
        self.times = times[mask]
        self.first = int(self.times[0] * self.info['sfreq'])
        self.last = len(self.times) + self.first - 1
        self.data = self.data[:, mask]

    def __add__(self, evoked):
        """Add evoked taking into account number of epochs"""
        out = merge_evoked([self, evoked])
        out.comment = self.comment + " + " + evoked.comment
        return out

    def __sub__(self, evoked):
        """Add evoked taking into account number of epochs"""
        this_evoked = deepcopy(evoked)
        this_evoked.data *= -1.
        out = merge_evoked([self, this_evoked])
        out.comment = self.comment + " - " + this_evoked.comment
        return out


def merge_evoked(all_evoked):
    """Merge/concat evoked data

    Data should have the same channels and the time instants.

    Parameters
    ----------
    all_evoked : list of Evoked
        The evoked datasets

    Returns
    -------
    evoked : Evoked
        The merged evoked data
    """
    evoked = deepcopy(all_evoked[0])

    ch_names = evoked.ch_names
    for e in all_evoked[1:]:
        assert e.ch_names == ch_names, ValueError("%s and %s do not contain "
                        "the same channels" % (evoked, e))
        assert np.max(np.abs(e.times - evoked.times)) < 1e-7, \
                ValueError("%s and %s do not "
                           "contain the same time instants" % (evoked, e))

    all_nave = sum(e.nave for e in all_evoked)
    evoked.data = sum(e.nave * e.data for e in all_evoked) / all_nave
    evoked.nave = all_nave
    return evoked


def read_evoked(fname, setno=0, baseline=None):
    """Read an evoked dataset

    Parameters
    ----------
    fname: string
        The file name.

    setno: int
        The index of the evoked dataset to read. FIF
        file can contain multiple datasets.

    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.

    Returns
    -------
    data: dict
        The evoked dataset
    """
    return Evoked(fname, setno, baseline=baseline)


def write_evoked(name, evoked):
    """Write an evoked dataset to a file

    Parameters
    ----------
    name: string
        The file name.

    evoked: object of type Evoked
        The evoked dataset to save
    """
    evoked.save(name)
