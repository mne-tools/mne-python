# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import traceback

import numpy as np

from ...utils import fill_doc_static, logger, warn
from ..utils import _generate_default_filename


class _MovieMixin:
    """Mixin providing movie export for Brain."""

    def _save_movie(
        self,
        filename,
        time_dilation=4.0,
        tmin=None,
        tmax=None,
        framerate=24,
        interpolation=None,
        codec=None,
        bitrate=None,
        callback=None,
        time_viewer=False,
        **kwargs,
    ):
        import imageio

        with self._renderer._disabled_interaction():
            images = self._make_movie_frames(
                time_dilation,
                tmin,
                tmax,
                framerate,
                interpolation,
                callback,
                time_viewer,
            )
        # find imageio FFMPEG parameters
        if "fps" not in kwargs:
            kwargs["fps"] = framerate
        if codec is not None:
            kwargs["codec"] = codec
        if bitrate is not None:
            kwargs["bitrate"] = bitrate
        # when using GIF we need to convert FPS to duration in milliseconds for Pillow
        if str(filename).endswith(".gif"):
            kwargs["duration"] = 1000 * len(images) / kwargs.pop("fps")
        imageio.mimwrite(filename, images, **kwargs)

    def _save_movie_tv(
        self,
        filename,
        time_dilation=4.0,
        tmin=None,
        tmax=None,
        framerate=24,
        interpolation=None,
        codec=None,
        bitrate=None,
        callback=None,
        time_viewer=False,
        **kwargs,
    ):
        def frame_callback(frame, n_frames):
            if frame == n_frames:
                # On the ImageIO step
                self.status_msg.set_value(f"Saving with ImageIO: {filename}")
                self.status_msg.show()
                self.status_progress.hide()
                self._renderer._status_bar_update()
            else:
                self.status_msg.set_value(
                    f"Rendering images (frame {frame + 1} / {n_frames}) ..."
                )
                self.status_msg.show()
                self.status_progress.show()
                self.status_progress.set_range([0, n_frames - 1])
                self.status_progress.set_value(frame)
                self.status_progress.update()
            self.status_msg.update()
            self._renderer._status_bar_update()

        # set cursor to busy
        default_cursor = self._renderer._window_get_cursor()
        self._renderer._window_set_cursor(
            self._renderer._window_new_cursor("WaitCursor")
        )

        try:
            self._save_movie(
                filename,
                time_dilation,
                tmin,
                tmax,
                framerate,
                interpolation,
                codec,
                bitrate,
                frame_callback,
                time_viewer,
                **kwargs,
            )
        except (Exception, KeyboardInterrupt):
            warn("Movie saving aborted:\n" + traceback.format_exc())
        finally:
            self._renderer._window_set_cursor(default_cursor)

    @fill_doc_static("interpolation_brain_time", "time_viewer_brain_screenshot")
    def save_movie(
        self,
        filename=None,
        time_dilation=4.0,
        tmin=None,
        tmax=None,
        framerate=24,
        interpolation=None,
        codec=None,
        bitrate=None,
        callback=None,
        time_viewer=False,
        **kwargs,
    ):
        """Save a movie (for data with a time axis).

        The movie is created through the :mod:`imageio` module. The format is
        determined by the extension, and additional options can be specified
        through keyword arguments that depend on the format, see
        :doc:`imageio's format page <imageio:formats/index>`.

        .. Warning::
            This method assumes that time is specified in seconds when adding
            data. If time is specified in milliseconds this will result in
            movies 1000 times longer than expected.

        Parameters
        ----------
        filename : str
            Path at which to save the movie. The extension determines the
            format (e.g., ``'*.mov'``, ``'*.gif'``, ...; see the :mod:`imageio`
            documentation for available formats).
        time_dilation : float
            Factor by which to stretch time (default 4). For example, an epoch
            from -100 to 600 ms lasts 700 ms. With ``time_dilation=4`` this
            would result in a 2.8 s long movie.
        tmin : float
            First time point to include (default: all data).
        tmax : float
            Last time point to include (default: all data).
        framerate : float
            Framerate of the movie (frames per second, default 24).
        interpolation : str | None
            Interpolation method (:class:`scipy.interpolate.interp1d` parameter).
            Must be one of ``'linear'``, ``'nearest'``, ``'zero'``, ``'slinear'``,
            ``'quadratic'`` or ``'cubic'``.
            If None, it uses the current ``brain.interpolation``,
            which defaults to ``'nearest'``. Defaults to None.
        codec : str | None
            The codec to use.
        bitrate : float | None
            The bitrate to use.
        callback : callable | None
            A function to call on each iteration. Useful for status message
            updates. It will be passed keyword arguments ``frame`` and
            ``n_frames``.
        time_viewer : bool
            If True, include time viewer traces. Only used if
            ``time_viewer=True`` and ``separate_canvas=False``.
        **kwargs : dict
            Specify additional options for :mod:`imageio`.
        """
        if filename is None:
            filename = _generate_default_filename(".mp4")
        func = self._save_movie_tv if self.time_viewer else self._save_movie
        func(
            filename,
            time_dilation,
            tmin,
            tmax,
            framerate,
            interpolation,
            codec,
            bitrate,
            callback,
            time_viewer,
            **kwargs,
        )

    def _make_movie_frames(
        self, time_dilation, tmin, tmax, framerate, interpolation, callback, time_viewer
    ):
        from math import floor

        # find tmin
        if tmin is None:
            tmin = self._times[0]
        elif tmin < self._times[0]:
            raise ValueError(
                f"tmin={repr(tmin)} is smaller than the first time point "
                f"({repr(self._times[0])})"
            )

        # find indexes at which to create frames
        if tmax is None:
            tmax = self._times[-1]
        elif tmax > self._times[-1]:
            raise ValueError(
                f"tmax={repr(tmax)} is greater than the latest time point "
                f"({repr(self._times[-1])})"
            )
        n_frames = floor((tmax - tmin) * time_dilation * framerate)
        times = np.arange(n_frames, dtype=float)
        times /= framerate * time_dilation
        times += tmin
        time_idx = np.interp(times, self._times, np.arange(self._n_times))

        n_times = len(time_idx)
        if n_times == 0:
            raise ValueError("No time points selected")

        logger.debug(f"Save movie for time points/samples\n{times}\n{time_idx}")
        # Sometimes the first screenshot is rendered with a different
        # resolution on OS X
        self.screenshot(time_viewer=time_viewer)
        old_mode = self.time_interpolation
        if interpolation is not None:
            self.set_time_interpolation(interpolation)
        try:
            images = [
                self.screenshot(time_viewer=time_viewer)
                for _ in self._iter_time(time_idx, callback)
            ]
        finally:
            self.set_time_interpolation(old_mode)
        if callback is not None:
            callback(frame=len(time_idx), n_frames=len(time_idx))
        return images

    def _iter_time(self, time_idx, callback):
        """Iterate through time points, then reset to current time.

        Parameters
        ----------
        time_idx : array_like
            Time point indexes through which to iterate.
        callback : callable | None
            Callback to call before yielding each frame.

        Yields
        ------
        idx : int | float
            Current index.

        Notes
        -----
        Used by movie and image sequence saving functions.
        """
        current_time_idx = self._data["time_idx"]
        for ii, idx in enumerate(time_idx):
            self.set_time_point(idx)
            if callback is not None:
                callback(frame=ii, n_frames=len(time_idx))
            yield idx

        # Restore original time index
        self.set_time_point(current_time_idx)
