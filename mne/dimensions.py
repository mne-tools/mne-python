'''
Created on Oct 25, 2012

@author: christian
'''
import numpy as np

from label import Label, BiHemiLabel



class DimensionMismatchError(Exception):
    pass



class Dimension():
    pass



class SourceSpace:
    name = 'source'
    """
    Indexing
    --------

    besides numpy indexing, the following indexes are possible:

     - mne Label objects
     - 'lh' or 'rh' to select an entire hemisphere

    """
    def __init__(self, vertno, subject='fsaverage'):
        """
        vertno : list of array
            The indices of the dipoles in the different source spaces.
            Each array has shape [n_dipoles] for in each source space]
        subject : str
            The mri-subject (used to load brain).

        """
        self.vertno = vertno
        self.lh_vertno = vertno[0]
        self.rh_vertno = vertno[1]
        self.lh_n = len(self.lh_vertno)
        self.rh_n = len(self.rh_vertno)
        self.subject = subject

    def __repr__(self):
        return "<dim source_space: %i (lh), %i (rh)>" % (self.lh_n, self.rh_n)

    def __len__(self):
        return self.lh_n + self.rh_n

    def __getitem__(self, index):
        vert = np.hstack(self.vertno)
        hemi = np.zeros(len(vert))
        hemi[self.lh_n:] = 1

        vert = vert[index]
        hemi = hemi[index]

        new_vert = (vert[hemi == 0], vert[hemi == 1])
        dim = SourceSpace(new_vert, subject=self.subject)
        return dim

    def dimindex(self, obj):
        if isinstance(obj, (Label, BiHemiLabel)):
            return self.label_index(obj)
        elif isinstance(obj, str):
            if obj == 'lh':
                if self.lh_n:
                    return slice(None, self.lh_n)
                else:
                    raise IndexError("lh is empty")
            if obj == 'rh':
                if self.rh_n:
                    return slice(self.lh_n, None)
                else:
                    raise IndexError("rh is empty")
            else:
                raise IndexError('%r' % obj)
        else:
            return obj

    def _hemilabel_index(self, label):
        if label.hemi == 'lh':
            stc_vertices = self.vertno[0]
            base = 0
        else:
            stc_vertices = self.vertno[1]
            base = len(self.vertno[0])

        idx = np.nonzero(map(label.vertices.__contains__, stc_vertices))[0]
        return idx + base

    def label_index(self, label):
        """Returns a SourceEstimate object restricted to a label

        SourceEstimate contains the time course of
        activation of all sources inside the label.

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        """
        if label.hemi == 'both':
            lh_idx = self._hemilabel_index(label.lh)
            rh_idx = self._hemilabel_index(label.rh)
            idx = np.hstack((lh_idx, rh_idx))
        else:
            idx = self._hemilabel_index(label)

        if len(idx) == 0:
            raise ValueError('No vertices match the label in the stc file')

        return idx



def find_time_point(timevar, time):
    """
    Returns (index, time) for the closest point to ``time`` in ``timevar``

    timevar : array-like 1d
        Monotonically increasing values.
    time : scalar
        Time point for which to find a match.

    """
    if time in timevar:
        i = np.where(timevar == time)[0][0]
    else:
        gr = (timevar > time)
        if np.all(gr):
            if timevar[1] - timevar[0] > timevar[0] - time:
                return 0, timevar[0]
            else:
                name = repr(timevar.name) if hasattr(timevar, 'name') else ''
                raise ValueError("time=%s lies outside array %r" % (time, name))
        elif np.any(gr):
            i_next = np.where(gr)[0][0]
        elif timevar[-1] - timevar[-2] > time - timevar[-1]:
            return len(timevar) - 1, timevar[-1]
        else:
            name = repr(timevar.name) if hasattr(timevar, 'name') else ''
            raise ValueError("time=%s lies outside array %r" % (time, name))
        t_next = timevar[i_next]

        sm = timevar < time
        i_prev = np.where(sm)[0][-1]
        t_prev = timevar[i_prev]

        if (t_next - time) < (time - t_prev):
            i = i_next
            time = t_next
        else:
            i = i_prev
            time = t_prev
    return i, time
