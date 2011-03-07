# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import fiff


class Epochs(object):
    """List of Epochs

    Parameters
    ----------
    raw : Raw object
        Returned by the setup_read_raw function

    events : array, of shape [n_events, 3]
        Returned by the read_events function

    event_id : int
        The id of the event to consider

    tmin : float
        Start time before event

    tmax : float
        End time after event

    keep_comp : boolean
        Apply CTF gradient compensation

    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.

    preload : boolean
        Load all epochs from disk when creating the object
        or wait before accessing each epoch (more memory
        efficient but can be slower).

    Methods
    -------
    get_epoch(i) : self
        Return the ith epoch as a 2D array [n_channels x n_times].

    get_data() : self
        Return all epochs as a 3D array [n_epochs x n_channels x n_times].

    average() : self
        Return averaged epochs as a 2D array [n_channels x n_times].

    """

    def __init__(self, raw, events, event_id, tmin, tmax,
                picks=None, keep_comp=False,
                dest_comp=0, baseline=(None, 0),
                preload=True):
        self.raw = raw
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.picks = picks
        self.keep_comp = keep_comp
        self.dest_comp = dest_comp
        self.baseline = baseline
        self.preload = preload

        if picks is None:
            picks = range(len(raw['info']['ch_names']))
            self.ch_names = raw['info']['ch_names']
        else:
            self.ch_names = [raw['info']['ch_names'][k] for k in picks]

        #   Set up projection
        if raw['info']['projs'] is None:
            print 'No projector specified for these data'
            raw['proj'] = []
        else:
            #   Activate the projection items
            for proj in raw['info']['projs']:
                proj['active'] = True

            print '%d projection items activated' % len(raw['info']['projs'])

            #   Create the projector
            proj, nproj = fiff.proj.make_projector_info(raw['info'])
            if nproj == 0:
                print 'The projection vectors do not apply to these channels'
                raw['proj'] = None
            else:
                print ('Created an SSP operator (subspace dimension = %d)'
                                                                    % nproj)
                raw['proj'] = proj

        #   Set up the CTF compensator
        current_comp = fiff.get_current_comp(raw['info'])
        if current_comp > 0:
            print 'Current compensation grade : %d' % current_comp

        if keep_comp:
            dest_comp = current_comp

        if current_comp != dest_comp:
            raw.comp = fiff.raw.make_compensator(raw['info'], current_comp,
                                                 dest_comp)
            print 'Appropriate compensator added to change to grade %d.' % (
                                                                    dest_comp)

        #    Select the desired events
        selected = np.logical_and(events[:, 1] == 0, events[:, 2] == event_id)
        self.events = events[selected]
        n_events = len(self.events)

        if n_events > 0:
            print '%d matching events found' % n_events
        else:
            raise ValueError, 'No desired events found.'

        # Handle times
        sfreq = raw['info']['sfreq']
        self.times = np.arange(int(tmin*sfreq), int(tmax*sfreq),
                          dtype=np.float) / sfreq

        if self.preload:
            self._data = self._get_data()

    def __len__(self):
        return len(self.events)

    def get_epoch(self, idx):
        """Load one epoch from disk"""
        sfreq = self.raw['info']['sfreq']
        event_samp = self.events[idx, 0]

        # Read a data segment
        start = int(event_samp + self.tmin*sfreq)
        stop = start + len(self.times)
        epoch, _ = self.raw[self.picks, start:stop]

        # Run baseline correction
        times = self.times
        baseline = self.baseline
        if baseline is not None:
            print "Applying baseline correction ..."
            bmin = baseline[0]
            bmax = baseline[1]
            if bmin is None:
                imin = 0
            else:
                imin = int(np.where(times >= bmin)[0][0])
            if bmax is None:
                imax = len(times)
            else:
                imax = int(np.where(times <= bmax)[0][-1]) + 1
            epoch -= np.mean(epoch[:, imin:imax], axis=1)[:, None]
        else:
            print "No baseline correction applied..."

        return epoch

    def _get_data(self):
        """Load all data from disk
        """
        n_channels = len(self.ch_names)
        n_times = len(self.times)
        n_events = len(self.events)
        data = np.empty((n_events, n_channels, n_times))
        for k, e in enumerate(self):
            data[k] = e
        return data

    def get_data(self):
        """Get all epochs as a 3D array

        Returns
        -------
        data : array of shape [n_epochs, n_channels, n_times]
            The epochs data
        """
        if self.preload:
            return self._data
        else:
            return self._get_data()

    def __iter__(self):
        """To iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self):
        """To iteration over epochs easy.
        """
        if self._current >= len(self.events):
            raise StopIteration

        epoch = self.get_epoch(self._current)

        self._current += 1
        return epoch

    def average(self):
        """Compute average of epochs

        Returns
        -------
        data : array of shape [n_channels, n_times]
            The averaged epochs
        """
        n_channels = len(self.ch_names)
        n_times = len(self.times)
        n_events = len(self.events)
        data = np.zeros((n_channels, n_times))
        for e in self:
            data += e
        return data / n_events

