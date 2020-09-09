# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numpy as np

from .utils import _ensure_int, verbose, logger


###############################################################################
# Class for interpolation between adjacent points

class _Interp2(object):
    r"""Interpolate between two points.

    Parameters
    ----------
    control_points : array, shape (n_changes,)
        The control points (indices) to use.
    values : callable | array, shape (n_changes, ...)
        Callable that takes the control point and returns a list of
        arrays that must be interpolated.
    interp : str
        Can be 'zero', 'linear', 'hann', or 'cos2' (same as hann).

    Notes
    -----
    This will process data using overlapping windows of potentially
    different sizes to achieve a constant output value using different
    2-point interpolation schemes. For example, for linear interpolation,
    and window sizes of 6 and 17, this would look like::

        1 _     _
          |\   / '-.           .-'
          | \ /     '-.     .-'
          |  x         |-.-|
          | / \     .-'     '-.
          |/   \_.-'           '-.
        0 +----|----|----|----|---
          0    5   10   15   20   25

    """

    def __init__(self, control_points, values, interp='hann'):
        # set up interpolation
        self.control_points = np.array(control_points, int).ravel()
        if not np.array_equal(np.unique(self.control_points),
                              self.control_points):
            raise ValueError('Control points must be sorted and unique')
        if len(self.control_points) == 0:
            raise ValueError('Must be at least one control point')
        if not (self.control_points >= 0).all():
            raise ValueError('All control points must be positive (got %s)'
                             % (self.control_points[:3],))
        if isinstance(values, np.ndarray):
            values = [values]
        if isinstance(values, (list, tuple)):
            for v in values:
                if not (v is None or isinstance(v, np.ndarray)):
                    raise TypeError('All entries in "values" must be ndarray '
                                    'or None, got %s' % (type(v),))
                if v is not None and v.shape[0] != len(self.control_points):
                    raise ValueError('Values, if provided, must be the same '
                                     'length as the number of control points '
                                     '(%s), got %s'
                                     % (len(self.control_points), v.shape[0]))
            use_values = values

            def val(pt):
                idx = np.where(control_points == pt)[0][0]
                return [v[idx] if v is not None else None for v in use_values]
            values = val
        self.values = values
        self.n_last = None
        self._position = 0  # start at zero
        self._left_idx = 0
        self._left = self._right = self._use_interp = None
        known_types = ('cos2', 'linear', 'zero', 'hann')
        if interp not in known_types:
            raise ValueError('interp must be one of %s, got "%s"'
                             % (known_types, interp))
        self._interp = interp

    def feed_generator(self, n_pts):
        """Feed data and get interpolators as a generator."""
        self.n_last = 0
        n_pts = _ensure_int(n_pts, 'n_pts')
        original_position = self._position
        stop = self._position + n_pts
        logger.debug('Feed %s (%s-%s)' % (n_pts, self._position, stop))
        used = np.zeros(n_pts, bool)
        if self._left is None:  # first one
            logger.debug('  Eval @ %s (%s)' % (0, self.control_points[0]))
            self._left = self.values(self.control_points[0])
            if len(self.control_points) == 1:
                self._right = self._left
        n_used = 0

        # Left zero-order hold condition
        if self._position < self.control_points[self._left_idx]:
            n_use = min(self.control_points[self._left_idx] - self._position,
                        n_pts)
            logger.debug('  Left ZOH %s' % n_use)
            this_sl = slice(None, n_use)
            assert used[this_sl].size == n_use
            assert not used[this_sl].any()
            used[this_sl] = True
            yield [this_sl, self._left, None, None]
            self._position += n_use
            n_used += n_use
            self.n_last += 1

        # Standard interpolation condition
        stop_right_idx = np.where(self.control_points >= stop)[0]
        if len(stop_right_idx) == 0:
            stop_right_idx = [len(self.control_points) - 1]
        stop_right_idx = stop_right_idx[0]
        left_idxs = np.arange(self._left_idx, stop_right_idx)
        self.n_last += max(len(left_idxs) - 1, 0)
        for bi, left_idx in enumerate(left_idxs):
            if left_idx != self._left_idx or self._right is None:
                if self._right is not None:
                    assert left_idx == self._left_idx + 1
                    self._left = self._right
                    self._left_idx += 1
                    self._use_interp = None  # need to recreate it
                eval_pt = self.control_points[self._left_idx + 1]
                logger.debug('  Eval @ %s (%s)'
                             % (self._left_idx + 1, eval_pt))
                self._right = self.values(eval_pt)
            assert self._right is not None
            left_point = self.control_points[self._left_idx]
            right_point = self.control_points[self._left_idx + 1]
            if self._use_interp is None:
                interp_span = right_point - left_point
                if self._interp == 'zero':
                    self._use_interp = None
                elif self._interp == 'linear':
                    self._use_interp = np.linspace(1., 0., interp_span,
                                                   endpoint=False)
                else:  # self._interp in ('cos2', 'hann'):
                    self._use_interp = np.cos(
                        np.linspace(0, np.pi / 2., interp_span,
                                    endpoint=False))
                    self._use_interp *= self._use_interp
            n_use = min(stop, right_point) - self._position
            if n_use > 0:
                logger.debug('  Interp %s %s (%s-%s)' % (self._interp, n_use,
                             left_point, right_point))
                interp_start = self._position - left_point
                assert interp_start >= 0
                if self._use_interp is None:
                    this_interp = None
                else:
                    this_interp = \
                        self._use_interp[interp_start:interp_start + n_use]
                    assert this_interp.size == n_use
                this_sl = slice(n_used, n_used + n_use)
                assert used[this_sl].size == n_use
                assert not used[this_sl].any()
                used[this_sl] = True
                yield [this_sl, self._left, self._right, this_interp]
                self._position += n_use
                n_used += n_use

        # Right zero-order hold condition
        if self.control_points[self._left_idx] <= self._position:
            n_use = stop - self._position
            if n_use > 0:
                logger.debug('  Right ZOH %s' % n_use)
                this_sl = slice(n_pts - n_use, None)
                assert not used[this_sl].any()
                used[this_sl] = True
                assert self._right is not None
                yield [this_sl, self._right, None, None]
                self._position += n_use
                n_used += n_use
                self.n_last += 1
        assert self._position == stop
        assert n_used == n_pts
        assert used.all()
        assert self._position == original_position + n_pts

    def feed(self, n_pts):
        """Feed data and get interpolated values."""
        # Convenience function for assembly
        out_arrays = None
        for o in self.feed_generator(n_pts):
            if out_arrays is None:
                out_arrays = [np.empty(v.shape + (n_pts,))
                              if v is not None else None for v in o[1]]
            for ai, arr in enumerate(out_arrays):
                if arr is not None:
                    if o[3] is None:
                        arr[..., o[0]] = o[1][ai][..., np.newaxis]
                    else:
                        arr[..., o[0]] = (
                            o[1][ai][..., np.newaxis] * o[3] +
                            o[2][ai][..., np.newaxis] * (1. - o[3]))
        assert out_arrays is not None
        return out_arrays


###############################################################################
# Constant overlap-add processing class


def _check_store(store):
    if isinstance(store, np.ndarray):
        store = [store]
    if isinstance(store, (list, tuple)) and all(isinstance(s, np.ndarray)
                                                for s in store):
        store = _Storer(*store)
    if not callable(store):
        raise TypeError('store must be callable, got type %s'
                        % (type(store),))
    return store


class _COLA(object):
    r"""Constant overlap-add processing helper.

    Parameters
    ----------
    process : callable
        A function that takes a chunk of input data with shape
        ``(n_channels, n_samples)`` and processes it.
    store : callable | ndarray
        A function that takes a completed chunk of output data.
        Can also be an ``ndarray``, in which case it is treated as the
        output data in which to store the results.
    n_total : int
        The total number of samples.
    n_samples : int
        The number of samples per window.
    n_overlap : int
        The overlap between windows.
    window : str
        The window to use. Default is "hann".
    tol : float
        The tolerance for COLA checking.

    Notes
    -----
    This will process data using overlapping windows to achieve a constant
    output value. For example, for ``n_total=27``, ``n_samples=10``,
    ``n_overlap=5`` and ``window='triang'``::

        1 _____               _______
          |    \   /\   /\   /
          |     \ /  \ /  \ /
          |      x    x    x
          |     / \  / \  / \
          |    /   \/   \/   \
        0 +----|----|----|----|----|-
          0    5   10   15   20   25

    This produces four windows: the first three are the requested length
    (10 samples) and the last one is longer (12 samples). The first and last
    window are asymmetric.
    """

    @verbose
    def __init__(self, process, store, n_total, n_samples, n_overlap,
                 sfreq, window='hann', tol=1e-10, verbose=None):
        from scipy.signal import get_window
        n_samples = _ensure_int(n_samples, 'n_samples')
        n_overlap = _ensure_int(n_overlap, 'n_overlap')
        n_total = _ensure_int(n_total, 'n_total')
        if n_samples <= 0:
            raise ValueError('n_samples must be > 0, got %s' % (n_samples,))
        if n_overlap < 0:
            raise ValueError('n_overlap must be >= 0, got %s' % (n_overlap,))
        if n_total < 0:
            raise ValueError('n_total must be >= 0, got %s' % (n_total,))
        self._n_samples = int(n_samples)
        self._n_overlap = int(n_overlap)
        del n_samples, n_overlap
        if n_total < self._n_samples:
            raise ValueError('Number of samples per window (%d) must be at '
                             'most the total number of samples (%s)'
                             % (self._n_samples, n_total))
        if not callable(process):
            raise TypeError('process must be callable, got type %s'
                            % (type(process),))
        self._process = process
        self._step = self._n_samples - self._n_overlap
        self._store = _check_store(store)
        self._idx = 0
        self._in_buffers = self._out_buffers = None

        # Create our window boundaries
        window_name = window if isinstance(window, str) else 'custom'
        self._window = get_window(window, self._n_samples,
                                  fftbins=(self._n_samples - 1) % 2)
        self._window /= _check_cola(self._window, self._n_samples, self._step,
                                    window_name, tol=tol)
        self.starts = np.arange(0, n_total - self._n_samples + 1, self._step)
        self.stops = self.starts + self._n_samples
        delta = n_total - self.stops[-1]
        self.stops[-1] = n_total
        sfreq = float(sfreq)
        pl = 's' if len(self.starts) != 1 else ''
        logger.info('    Processing %4d data chunk%s of (at least) %0.1f sec '
                    'with %0.1f sec overlap and %s windowing'
                    % (len(self.starts), pl, self._n_samples / sfreq,
                       self._n_overlap / sfreq, window_name))
        del window, window_name
        if delta > 0:
            logger.info('    The final %0.3f sec will be lumped into the '
                        'final window' % (delta / sfreq,))
        self.verbose = verbose

    @property
    def _in_offset(self):
        """Compute from current processing window start and buffer len."""
        return self.starts[self._idx] + self._in_buffers[0].shape[-1]

    @verbose
    def feed(self, *datas, **kwargs):
        """Pass in a chunk of data."""
        # Append to our input buffer
        if self._in_buffers is None:
            self._in_buffers = [None] * len(datas)
        if len(datas) != len(self._in_buffers):
            raise ValueError('Got %d array(s), needed %d'
                             % (len(datas), len(self._in_buffers)))
        for di, data in enumerate(datas):
            if not isinstance(data, np.ndarray) or data.ndim < 1:
                raise TypeError('data entry %d must be an 2D ndarray, got %s'
                                % (di, type(data),))
            if self._in_buffers[di] is None:
                # In practice, users can give large chunks, so we use
                # dynamic allocation of the in buffer. We could save some
                # memory allocation by only ever processing max_len at once,
                # but this would increase code complexity.
                self._in_buffers[di] = np.empty(
                    data.shape[:-1] + (0,), data.dtype)
            if data.shape[:-1] != self._in_buffers[di].shape[:-1] or \
                    self._in_buffers[di].dtype != data.dtype:
                raise TypeError('data must dtype %s and shape[:-1]==%s, '
                                'got dtype %s shape[:-1]=%s'
                                % (self._in_buffers[di].dtype,
                                   self._in_buffers[di].shape[:-1],
                                   data.dtype, data.shape[:-1]))
            logger.debug('    + Appending %d->%d'
                         % (self._in_offset, self._in_offset + data.shape[-1]))
            self._in_buffers[di] = np.concatenate(
                [self._in_buffers[di], data], -1)
            if self._in_offset > self.stops[-1]:
                raise ValueError('data (shape %s) exceeded expected total '
                                 'buffer size (%s > %s)'
                                 % (data.shape, self._in_offset,
                                    self.stops[-1]))
        # Check to see if we can process the next chunk and dump outputs
        while self._idx < len(self.starts) and \
                self._in_offset >= self.stops[self._idx]:
            start, stop = self.starts[self._idx], self.stops[self._idx]
            this_len = stop - start
            this_window = self._window.copy()
            if self._idx == len(self.starts) - 1:
                this_window = np.pad(
                    self._window, (0, this_len - len(this_window)), 'constant')
                for offset in range(self._step, len(this_window), self._step):
                    n_use = len(this_window) - offset
                    this_window[offset:] += self._window[:n_use]
            if self._idx == 0:
                for offset in range(self._n_samples - self._step, 0,
                                    -self._step):
                    this_window[:offset] += self._window[-offset:]
            logger.debug('    * Processing %d->%d' % (start, stop))
            this_proc = [in_[..., :this_len].copy()
                         for in_ in self._in_buffers]
            if not all(proc.shape[-1] == this_len == this_window.size
                       for proc in this_proc):
                raise RuntimeError('internal indexing error')
            outs = self._process(*this_proc, **kwargs)
            if self._out_buffers is None:
                max_len = np.max(self.stops - self.starts)
                self._out_buffers = [np.zeros(o.shape[:-1] + (max_len,),
                                              o.dtype) for o in outs]
            for oi, out in enumerate(outs):
                out *= this_window
                self._out_buffers[oi][..., :stop - start] += out
            self._idx += 1
            if self._idx < len(self.starts):
                next_start = self.starts[self._idx]
            else:
                next_start = self.stops[-1]
            delta = next_start - self.starts[self._idx - 1]
            for di in range(len(self._in_buffers)):
                self._in_buffers[di] = self._in_buffers[di][..., delta:]
            logger.debug('    - Shifting input/output buffers by %d samples'
                         % (delta,))
            self._store(*[o[..., :delta] for o in self._out_buffers])
            for ob in self._out_buffers:
                ob[..., :-delta] = ob[..., delta:]
                ob[..., -delta:] = 0.


def _check_cola(win, nperseg, step, window_name, tol=1e-10):
    """Check whether the Constant OverLap Add (COLA) constraint is met."""
    # adapted from SciPy
    binsums = np.sum([win[ii * step:(ii + 1) * step]
                      for ii in range(nperseg // step)], axis=0)
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]
    const = np.median(binsums)
    deviation = np.max(np.abs(binsums - const))
    if deviation > tol:
        raise ValueError('segment length %d with step %d for %s window '
                         'type does not provide a constant output '
                         '(%g%% deviation)'
                         % (nperseg, step, window_name,
                            100 * deviation / const))
    return const


class _Storer(object):
    """Store data in chunks."""

    def __init__(self, *outs, picks=None):
        for oi, out in enumerate(outs):
            if not isinstance(out, np.ndarray) or out.ndim < 1:
                raise TypeError('outs[oi] must be >= 1D ndarray, got %s'
                                % (out,))
        self.outs = outs
        self.idx = 0
        self.picks = picks

    def __call__(self, *outs):
        if (len(outs) != len(self.outs) or
                not all(out.shape[-1] == outs[0].shape[-1] for out in outs)):
            raise ValueError('Bad outs')
        idx = (Ellipsis,)
        if self.picks is not None:
            idx += (self.picks,)
        stop = self.idx + outs[0].shape[-1]
        idx += (slice(self.idx, stop),)
        for o1, o2 in zip(self.outs, outs):
            o1[idx] = o2
        self.idx = stop
