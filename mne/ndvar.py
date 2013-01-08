'''
Created on Dec 21, 2012

@author: christian
'''
import numpy as np

from .dimensions import DimensionMismatchError, SourceSpace, UTS
from . import filter
from .source_estimate import SourceEstimate



class NdVar(object):
    "Container for n-dimensional data."
    def __init__(self, x, dims=('case',), info=None, name=None):
        """
        Parameters
        ----------

        x : array
            The data.
        dims : tuple
            The dimensions characterizing the axes of the data. If present,
            ``'case'`` should be provided as a :py:class:`str`, and should
            always occupy the first position.
        info : dict
            A dictionary with data properties.


        Notes
        -----

        ``x`` and ``dims`` are stored without copying. A shallow
        copy of ``properties`` is stored. Make sure the relevant objects
        are not modified externally later.


        Examples
        --------

        Importing 600 epochs of data for 80 time points:

            >>> time = var('time', range(-.2, .6, .01))
            >>> dims = ('case', time)
            >>> data.shape
            (600, 80)
            >>> Y = NdVar(data, dims=dims)

        """
        # check data shape
        dims = tuple(dims)
        ndim = len(dims)
        if ndim != x.ndim:
            err = ("Unequal number of dimensions (data: %i, dims: %i)" %
                   (x.ndim, ndim))
            raise DimensionMismatchError(err)

        # check dimensions
        d0 = dims[0]
        if isinstance(d0, basestring):
            if d0 == 'case':
                has_case = True
            else:
                err = ("String dimension needs to be 'case' (got %r)" % d0)
                raise ValueError(err)
        else:
            has_case = False

        for dim, n in zip(dims, x.shape)[has_case:]:
            if isinstance(dim, basestring):
                err = ("Invalid dimension: %r in %r. First dimension can be "
                       "'case', other dimensions need to be array-like" %
                       (dim, dims))
                raise TypeError(err)
            n_dim = len(dim)
            if n_dim != n:
                err = ("Dimension %r length mismatch: %i in data, "
                       "%i in dimension %r" % (dim.name, n, n_dim, dim.name))
                raise DimensionMismatchError(err)

        state = {'dims': dims,
                 'x': x,
                 'name': name}

        # store attributes
        if info is None:
            state['info'] = {}
        else:
            state['info'] = info.copy()

        self.__setstate__(state)

    def __setstate__(self, state):
        self.dims = dims = state['dims']
        self.has_case = (dims[0] == 'case')
        self._truedims = truedims = dims[self.has_case:]

        # dimnames
        self.dimnames = tuple(dim.name for dim in truedims)
        if self.has_case:
            self.dimnames = ('case',) + self.dimnames

        self.x = x = state['x']
        self.name = state['name']
        self.info = state['info']
        # derived
        self.ndim = len(dims)
        self._len = len(x)
        self._dim_2_ax = dict(zip(self.dimnames, xrange(self.ndim)))
        # attr
        for dim in truedims:
            if hasattr(self, dim.name):
                err = ("invalid dimension name: %r (already present as NdVar"
                       " attr)" % dim.name)
                raise ValueError(err)
            else:
                setattr(self, dim.name, dim)

    def __getstate__(self):
        state = {'dims': self.dims,
                 'x': self.x,
                 'name': self.name,
                 'info': self.info}
        return state

    # numeric ---
    def _align(self, other):
        "align data from 2 NdVars"
        if isinstance(other, NdVar):
            i_self = list(self.dimnames)
            dims = list(self.dims)
            i_other = []

            for dim in i_self:
                if dim in other.dimnames:
                    i_other.append(dim)
                else:
                    i_other.append(None)

            for dim in other.dimnames:
                if dim in i_self:
                    pass
                else:
                    i_self.append(None)
                    i_other.append(dim)
                    dims.append(other.get_dim(dim))

            x_self = self.get_data(i_self)
            x_other = other.get_data(i_other)
            return dims, x_self, x_other
        else:
            raise TypeError

    def _ialign(self, other):
        "align for self-modifying operations (+=, ...)"
        assert all(dim in self.dimnames for dim in other.dimnames)
        i_other = []
        for dim in self.dimnames:
            if dim in other.dimnames:
                i_other.append(dim)
            else:
                i_other.append(None)
        return other.get_data(i_other)

    def __add__(self, other):
        if isinstance(other, NdVar):
            dims, x_self, x_other = self._align(other)
            x = x_self + x_other
            name = '+'.join((self.name, other.name))
        elif np.isscalar(other):
            x = self.x + other
            dims = self.dims
            name = '+'.join((self.name, str(other)))
        else:
            raise ValueError("can't add %r" % other)
        return NdVar(x, dims=dims, name=name, info=self.info)

    def __iadd__(self, other):
        self.x += self._ialign(other)
        return self

    def __sub__(self, other):  # TODO: use dims
        if isinstance(other, NdVar):
            dims, x_self, x_other = self._align(other)
            x = x_self - x_other
            name = '-'.join((self.name, other.name))
        elif np.isscalar(other):
            x = self.x - other
            dims = self.dims
            name = '-'.join((self.name, str(other)))
        else:
            raise ValueError("can't subtract %r" % other)
        return NdVar(x, dims=dims, name=name, info=self.info)

    def __isub__(self, other):
        self.x -= self._ialign(other)
        return self

    def __rsub__(self, other):
        x = other - self.x
        return NdVar(x, self.dims, self.info, name=self.name)

    # container ---
    def __getitem__(self, index):
        if np.iterable(index) or isinstance(index, slice):
            x = self.x[index]
            if x.shape[1:] != self.x.shape[1:]:
                raise NotImplementedError("Use subdata method when dims are affected")
            return NdVar(x, dims=self.dims, name=self.name, info=self.info)
        else:
            index = int(index)
            x = self.x[index]
            dims = self.dims[1:]
            if self.name:
                name = '%s_%i' % (self.name, index)
            else:
                name = None
            return NdVar(x, dims=dims, name=name, info=self.info)

    def __len__(self):
        return self._len

    def __repr__(self):
        rep = '<NdVar%(name)s: %(dims)s>'
        if self.has_case:
            dims = [(self._len, 'case')]
        else:
            dims = []
        dims.extend([(len(dim), dim._dimrepr_()) for dim in self._truedims])

        dims = ' X '.join('%i (%s)' % fmt for fmt in dims)
        args = dict(dims=dims)
        args['name'] = repr(self.name) if self.name else ''
        return rep % args

    def assert_dims(self, dims):
        if self.dimnames != dims:
            err = "Dimensions of %r do not match %r" % (self, dims)
            raise DimensionMismatchError(err)

    def compress(self, X, func=np.mean, name='{name}'):
        """
        Return an NdVar with one case for each cell in ``X``.

        Parameters
        ----------
        X : categorial
            Categorial whose cells are used to compress the NdVar.
        func : function with axis argument
            Function that is used to create a summary of the cases falling
            into each cell of X. The function needs to accept the data as
            first argument and ``axis`` as keyword-argument. Default is
            ``numpy.mean``.
        name : str
            Name for the resulting NdVar. ``'{name}'`` is formatted to the
            current NdVar's ``.name``.

        """
        if not self.has_case:
            raise DimensionMismatchError("%r has no case dimension" % self)
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            idx = (X == cell)
            if np.sum(idx):
                x_cell = self.x[idx]
                x.append(func(x_cell, axis=0))

        # update info for summary
        info = self.info.copy()
        for key in self.info:
            if key.startswith('summary_') and (key != 'summary_func'):
                info[key[8:]] = info.pop(key)

        x = np.array(x)
        name = name.format(name=self.name)
        out = NdVar(x, self.dims, info=info, name=name)
        return out

    def copy(self, name='{name}'):
        "returns a deep copy of itself"
        x = self.x.copy()
        name = name.format(name=self.name)
        info = self.info.copy()
        return self.__class__(x, dims=self.dims, name=name,
                              info=info)

    def get_axis(self, dim):
        return self._dim_2_ax[dim]

    def get_data(self, dims):
        """
        returns the data with a specific ordering of dimension as indicated in
        ``dims``.

        Parameters
        ----------
        dims : sequence of str and None
            List of dimension names. The array that is returned will have axes
            in this order. None can be used to increase the insert a dimension
            with size 1. Accessing a single dimension can be abbreviated by
            providing a str.

        """
        if isinstance(dims, str):
            dims = (dims,)

        if set(dims).difference([None]) != set(self.dimnames):
            err = "Requested dimensions %r from %r" % (dims, self)
            raise DimensionMismatchError(err)

        dimnames = list(self.dimnames)
        x = self.x

        index = []
        dim_seq = []
        for dim in dims:
            if dim is None:
                index.append(None)
            else:
                index.append(slice(None))
                dim_seq.append(dim)

        for i_tgt, dim in enumerate(dim_seq):
            i_src = dimnames.index(dim)
            if i_tgt != i_src:
                x = x.swapaxes(i_src, i_tgt)
                dimnames[i_src], dimnames[i_tgt] = dimnames[i_tgt], dimnames[i_src]

        return x[index]

    def get_dim(self, name):
        "Returns the dimension var named ``name``"
        if self.has_dim(name):
            i = self._dim_2_ax[name]
            return self.dims[i]
        elif name == 'case':
            return np.arange(len(self))
        else:
            msg = "%r has no dimension named %r" % (self, name)
            raise DimensionMismatchError(msg)

    def get_dims(self, names):
        "Returns a tuple with the requested Dimension objects"
        return tuple(self.get_dim(name) for name in names)

    def has_dim(self, name):
        return name in self._dim_2_ax

    def repeat(self, repeats, dim='case', name='{name}'):
        """
        Analogous to :py:func:`numpy.repeat`

        """
        ax = self.get_axis(dim)
        x = self.x.repeat(repeats, axis=ax)

        repdim = self.dims[ax]
        if not isinstance(repdim, str):
            repdim = repdim.repeat(repeats)

        dims = self.dims[:ax] + (repdim,) + self.dims[ax + 1:]
        info = self.info.copy()
        name = name.format(name=self.name)
        return NdVar(x, dims, info=info, name=name)

    def summary(self, *dims, **regions):
        r"""
        Returns a new NdVar with specified dimensions collapsed.

        .. warning::
            Data is collapsed over the different dimensions in turn using the
            provided function with an axis argument. For certain functions
            this is not equivalent to collapsing over several axes concurrently
            (e.g., np.var).

        dimension:
            A whole dimension is specified as string argument. This
            dimension is collapsed over the whole range.
        range:
            A range within a dimension is specified through a keyword-argument.
            Only the data in the specified range is included. Use like the
            :py:meth:`.subdata` method.


        **additional kwargs:**

        func : callable
            Function used to collapse the data. Needs to accept an "axis"
            kwarg (default: np.mean)
        name : str
            Name for the new NdVar. Default: "{func}({name})".


        Examples
        --------

        Assuming UTS is a normal time series. Get the average in a time
        window::

            >>> Y = UTS.summary(time=(.1, .2))

        Get the peak in a time window::

            >>> Y = UTS.summary(time=(.1, .2), func=np.max)

        Assuming MEG is an NdVar with dimensions time and sensor. Get the
        average across sensors 5, 6, and 8 in a time window::

            >>> ROI = [5, 6, 8]
            >>> Y = MEG.summary(sensor=ROI, time=(.1, .2))

        Get the peak in the same data:

            >>> ROI = [5, 6, 8]
            >>> Y = MEG.summary(sensor=ROI, time=(.1, .2), func=np.max)

        Get the RMS over all sensors

            >>> MEG_RMS = MEG.summary('sensor', func=statfuncs.RMS)

        """
        func = regions.pop('func', self.info.get('summary_func', np.mean))
        name = regions.pop('name', '{func}({name})')
        name = name.format(func=func.__name__, name=self.name)
        if len(dims) + len(regions) == 0:
            dims = ('case',)

        if regions:
            dims = list(dims)
            dims.extend(dim for dim in regions if not np.isscalar(regions[dim]))
            data = self.subdata(**regions)
            return data.summary(*dims, func=func, name=name)
        else:
            x = self.x
            axes = [self._dim_2_ax[dim] for dim in np.unique(dims)]
            dims = list(self.dims)
            for axis in sorted(axes, reverse=True):
                x = func(x, axis=axis)
                dims.pop(axis)

            # update info for summary
            info = self.info.copy()
            for key in self.info:
                if key.startswith('summary_') and (key != 'summary_func'):
                    info[key[8:]] = info.pop(key)

            if len(dims) == 0:
                return x
            elif dims == ['case']:
                return x
            else:
                return NdVar(x, dims=dims, name=name, info=info)

    def subdata(self, **kwargs):
        """
        returns an NdVar object with a subset of the current NdVar's data.
        The slice is specified using kwargs, with dimensions as keywords and
        indexes as values, e.g.::

            >>> Y.subdata(time = 1)

        returns a slice for time point 1 (second). For dimensions whose values
        change monotonically, a tuple can be used to specify a window::

            >>> Y.subdata(time = (.2, .6))

        returns a slice containing all values for times .2 seconds to .6
        seconds.

        """
        info = self.info.copy()
        dims = list(self.dims)
        index = [slice(None)] * len(dims)

        for name, arg in kwargs.iteritems():
            try:
                dimax = self._dim_2_ax[name]
                dim = self.dims[dimax]
            except KeyError:
                err = ("Segment does not contain %r dimension." % name)
                raise DimensionMismatchError(err)

            if hasattr(dim, 'dimindex'):
                i = dim.dimindex(arg)
            else:
                i = arg

            index[dimax] = i
            if np.isscalar(i):
                dims[dimax] = None
                info[name] = arg
            else:
                dims[dimax] = dim[i]

        # create subdata object
        x = self.x[index]
        dims = tuple(dim for dim in dims if dim is not None)
        return NdVar(x, dims=dims, name=self.name, info=info)



def from_stc(stc, subject='fsaverage', name=None, check=True):
    """
    create an NdVar object from one or more mne SourceEstimate object(s)

    stc : SourceEstimate | list of SourceEstimates
        The source estimate object(s).
    subject : str
        MRI subject (used for loading MRI in PySurfer plotting)
    name : str | None
        Ndvar name.
    check : bool
        If multiple stcs are provided, check if all stcs have the same times
        and vertices.

    """
    if isinstance(stc, SourceEstimate):
        case = False
        x = stc.data
    else:
        case = True
        stcs = stc
        stc = stcs[0]
        if check:
            vert_lh, vert_rh = stc.vertno
            times = stc.times
            for stc_ in stcs[1:]:
                assert np.all(times == stc_.times)
                lh, rh = stc_.vertno
                assert np.all(vert_lh == lh)
                assert np.all(vert_rh == rh)
        x = np.array([s.data for s in stcs])

    time = UTS(stc.tmin, stc.tstep, len(stc.times) - 1)
    ss = SourceSpace(stc.vertno, subject=subject)
    if case:
        dims = ('case', ss, time)
    else:
        dims = (ss, time)

    return NdVar(x, dims, name=name)



def resample(ndvar, sfreq, npad=100, window='boxcar'):
    axis = ndvar.get_axis('time')
    old_sfreq = 1.0 / ndvar.time.tstep
    x = filter.resample(ndvar.x, sfreq, old_sfreq, npad, axis, window)
    tstep = 1. / sfreq
    time = UTS(ndvar.time.tmin, tstep, x.shape[axis] - 1)
    dims = ndvar.dims[:axis] + (time,) + ndvar.dims[axis + 1:]
    return NdVar(x, dims=dims, info=ndvar.info, name=ndvar.name)
