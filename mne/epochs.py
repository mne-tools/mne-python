# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import fiff


def read_epochs(raw, events, event_id, tmin, tmax, picks=None,
                 keep_comp=False, dest_comp=0, baseline=None):
    """Read epochs from a raw dataset

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

    Returns
    -------
    data : list of epochs
        An epoch is a dict with key:
            epoch    the epoch, channel by channel
            event    event #
            tmin     starting time in the raw data file (initial skip omitted)
            tmax     ending stime in the raw data file (initial skip omitted)

    times : array
        The time points of the samples, in seconds

    ch_names : list of strings
        Names of the channels included

    Notes
    -----
    NOTE 1: The purpose of this function is to demonstrate the raw data reading
    routines. You may need to modify this for your purposes

    NOTE 2: You need to run mne_process_raw once as

    mne_process_raw --raw <fname> --projoff

    to create the fif-format event file (or open the file in mne_browse_raw).
    """

    ch_names = [raw['info']['ch_names'][k] for k in picks]
    sfreq = raw['info']['sfreq']

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
            print 'Created an SSP operator (subspace dimension = %d)' % nproj
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

    # #  Read the events
    # if event_fname is None:
    #     if fname.endswith('.fif'):
    #         event_name = '%s-eve.fif' % fname[:-4]
    #     else:
    #         raise ValueError, 'Raw file name does not end properly'
    #
    #     events = fiff.read_events(event_name)
    #     print 'Events read from %s' % event_name
    # else:
    #     #   Binary file
    #     if event_name.endswith('-eve.fif'):
    #         events = fiff.read_events(event_name)
    #         print 'Binary event file %s read' % event_name
    #     else:
    #         #   Text file
    #         events = np.loadtxt(event_name)
    #         if events.shape[0] < 1:
    #             raise ValueError, 'No data in the event file'
    #
    #         #   Convert time to samples if sample number is negative
    #         events[events[:,0] < 0,0] = events[:,1] * sfreq
    #         #    Select the columns of interest (convert to integers)
    #         events = np.array(events[:,[0, 2, 3]], dtype=np.int32)
    #         #    New format?
    #         if events[0,1] == 0 and events[0,2] == 0:
    #             print 'The text event file %s is in the new format' % event_name
    #             if events[0,0] != raw['first_samp']:
    #                 raise ValueError, ('This new format event file is not compatible'
    #                                    ' with the raw data')
    #         else:
    #             print 'The text event file %s is in the old format' % event_name
    #             #   Offset with first sample
    #             events[:,0] += raw['first_samp']

    #    Select the desired events
    selected = np.logical_and(events[:, 1] == 0, events[:, 2] == event_id)
    n_events = np.sum(selected)
    if n_events > 0:
        print '%d matching events found' % n_events
    else:
        raise ValueError, 'No desired events found.'

    data = list()

    for p, event_samp in enumerate(events[selected, 0]):
        #       Read a data segment
        start = event_samp + tmin*sfreq
        stop = event_samp + tmax*sfreq
        epoch, _ = raw[picks, start:stop]

        if p == 0:
            times = np.arange(start - event_samp, stop - event_samp,
                              dtype=np.float) / sfreq

        # Run baseline correction
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
            epoch -= np.mean(epoch[:, imin:imax], axis=1)[:,None]
        else:
            print "No baseline correction applied..."

        d = dict()
        d['epoch'] = epoch
        d['event'] = event_id
        d['tmin'] = (float(start) - float(raw['first_samp'])) / sfreq
        d['tmax'] = (float(stop) - float(raw['first_samp'])) / sfreq
        data.append(d)


    print 'Read %d epochs, %d samples each.' % (len(data),
                                                data[0]['epoch'].shape[1])

    #   Remember to close the file descriptor
    # raw['fid'].close()
    # print 'File closed.'

    return data, times, ch_names
