# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import contextlib
import copy
import os.path as op
from types import GeneratorType

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist, pdist

from ._fiff.constants import FIFF
from ._fiff.meas_info import Info
from ._fiff.pick import _picks_to_idx, pick_types
from ._freesurfer import _get_atlas_values, _get_mri_info_data, read_freesurfer_lut
from .baseline import rescale
from .cov import Covariance
from .evoked import _get_peak
from .filter import FilterMixin, _check_fun, resample
from .fixes import _eye_array, _safe_svd
from .parallel import parallel_func
from .source_space._source_space import (
    SourceSpaces,
    _check_volume_labels,
    _ensure_src,
    _ensure_src_subject,
    _get_morph_src_reordering,
    _get_src_nn,
    get_decimated_surfaces,
)
from .surface import _get_ico_surface, _project_onto_surface, mesh_edges, read_surface
from .transforms import _get_trans, apply_trans
from .utils import (
    TimeMixin,
    _build_data_frame,
    _check_fname,
    _check_option,
    _check_pandas_index_arguments,
    _check_pandas_installed,
    _check_preload,
    _check_src_normal,
    _check_stc_units,
    _check_subject,
    _check_time_format,
    _convert_times,
    _ensure_int,
    _import_h5io_funcs,
    _import_nibabel,
    _path_like,
    _pl,
    _time_mask,
    _validate_type,
    copy_function_doc_to_method_doc,
    fill_doc,
    get_subjects_dir,
    logger,
    object_size,
    sizeof_fmt,
    verbose,
    warn,
)
from .viz import (
    plot_source_estimates,
    plot_vector_source_estimates,
    plot_volume_source_estimates,
)


def _read_stc(filename):
    """Aux Function."""
    with open(filename, "rb") as fid:
        buf = fid.read()

    stc = dict()
    offset = 0
    num_bytes = 4

    # read tmin in ms
    stc["tmin"] = (
        float(np.frombuffer(buf, dtype=">f4", count=1, offset=offset).item()) / 1000.0
    )
    offset += num_bytes

    # read sampling rate in ms
    stc["tstep"] = (
        float(np.frombuffer(buf, dtype=">f4", count=1, offset=offset).item()) / 1000.0
    )
    offset += num_bytes

    # read number of vertices/sources
    vertices_n = int(np.frombuffer(buf, dtype=">u4", count=1, offset=offset).item())
    offset += num_bytes

    # read the source vector
    stc["vertices"] = np.frombuffer(buf, dtype=">u4", count=vertices_n, offset=offset)
    offset += num_bytes * vertices_n

    # read the number of timepts
    data_n = int(np.frombuffer(buf, dtype=">u4", count=1, offset=offset).item())
    offset += num_bytes

    if (
        vertices_n
        and (  # vertices_n can be 0 (empty stc)
            (len(buf) // 4 - 4 - vertices_n) % (data_n * vertices_n)
        )
        != 0
    ):
        raise ValueError("incorrect stc file size")

    # read the data matrix
    stc["data"] = np.frombuffer(
        buf, dtype=">f4", count=vertices_n * data_n, offset=offset
    )
    stc["data"] = stc["data"].reshape([data_n, vertices_n]).T

    return stc


def _write_stc(filename, tmin, tstep, vertices, data):
    """Write an STC file.

    Parameters
    ----------
    filename : path-like
        The name of the STC file.
    tmin : float
        The first time point of the data in seconds.
    tstep : float
        Time between frames in seconds.
    vertices : array of integers
        Vertex indices (0 based).
    data : 2D array
        The data matrix (nvert * ntime).
    """
    with open(filename, "wb") as fid:
        # write start time in ms
        fid.write(np.array(1000 * tmin, dtype=">f4").tobytes())
        # write sampling rate in ms
        fid.write(np.array(1000 * tstep, dtype=">f4").tobytes())
        # write number of vertices
        fid.write(np.array(vertices.shape[0], dtype=">u4").tobytes())
        # write the vertex indices
        fid.write(np.array(vertices, dtype=">u4").tobytes())
        # write the number of timepts
        fid.write(np.array(data.shape[1], dtype=">u4").tobytes())
        # write the data
        fid.write(np.array(data.T, dtype=">f4").tobytes())


def _read_3(fid):
    """Read 3 byte integer from file."""
    data = np.fromfile(fid, dtype=np.uint8, count=3).astype(np.int32)

    out = np.left_shift(data[0], 16) + np.left_shift(data[1], 8) + data[2]

    return out


def _read_w(filename):
    """Read a w file.

    w files contain activations or source reconstructions for a single time
    point.

    Parameters
    ----------
    filename : path-like
        The name of the w file.

    Returns
    -------
    data: dict
        The w structure. It has the following keys:
           vertices       vertex indices (0 based)
           data           The data matrix (nvert long)
    """
    with open(filename, "rb", buffering=0) as fid:  # buffering=0 for np bug
        # skip first 2 bytes
        fid.read(2)

        # read number of vertices/sources (3 byte integer)
        vertices_n = int(_read_3(fid))

        vertices = np.zeros((vertices_n), dtype=np.int32)
        data = np.zeros((vertices_n), dtype=np.float32)

        # read the vertices and data
        for i in range(vertices_n):
            vertices[i] = _read_3(fid)
            data[i] = np.fromfile(fid, dtype=">f4", count=1).item()

        w = dict()
        w["vertices"] = vertices
        w["data"] = data

    return w


def _write_3(fid, val):
    """Write 3 byte integer to file."""
    f_bytes = np.zeros((3), dtype=np.uint8)
    f_bytes[0] = (val >> 16) & 255
    f_bytes[1] = (val >> 8) & 255
    f_bytes[2] = val & 255
    fid.write(f_bytes.tobytes())


def _write_w(filename, vertices, data):
    """Write a w file.

    w files contain activations or source reconstructions for a single time
    point.

    Parameters
    ----------
    filename: path-like
        The name of the w file.
    vertices: array of int
        Vertex indices (0 based).
    data: 1D array
        The data array (nvert).
    """
    assert len(vertices) == len(data)

    with open(filename, "wb") as fid:
        # write 2 zero bytes
        fid.write(np.zeros((2), dtype=np.uint8).tobytes())

        # write number of vertices/sources (3 byte integer)
        vertices_n = len(vertices)
        _write_3(fid, vertices_n)

        # write the vertices and data
        for i in range(vertices_n):
            _write_3(fid, vertices[i])
            # XXX: without float() endianness is wrong, not sure why
            fid.write(np.array(float(data[i]), dtype=">f4").tobytes())


def read_source_estimate(fname, subject=None):
    """Read a source estimate object.

    Parameters
    ----------
    fname : path-like
        Path to (a) source-estimate file(s).
    subject : str | None
        Name of the subject the source estimate(s) is (are) from.
        It is good practice to set this attribute to avoid combining
        incompatible labels and SourceEstimates (e.g., ones from other
        subjects). Note that due to file specification limitations, the
        subject name isn't saved to or loaded from files written to disk.

    Returns
    -------
    stc : SourceEstimate | VectorSourceEstimate | VolSourceEstimate | MixedSourceEstimate
        The source estimate object loaded from file.

    Notes
    -----
     - for volume source estimates, ``fname`` should provide the path to a
       single file named ``'*-vl.stc``` or ``'*-vol.stc'``
     - for surface source estimates, ``fname`` should either provide the
       path to the file corresponding to a single hemisphere (``'*-lh.stc'``,
       ``'*-rh.stc'``) or only specify the asterisk part in these patterns. In
       any case, the function expects files for both hemisphere with names
       following this pattern.
     - for vector surface source estimates, only HDF5 files are supported.
     - for mixed source estimates, only HDF5 files are supported.
     - for single time point ``.w`` files, ``fname`` should follow the same
       pattern as for surface estimates, except that files are named
       ``'*-lh.w'`` and ``'*-rh.w'``.
    """  # noqa: E501
    fname_arg = fname

    # expand `~` without checking whether the file actually exists – we'll
    # take care of that later, as it's complicated by the different suffixes
    # STC files can have
    fname = str(_check_fname(fname=fname, overwrite="read", must_exist=False))

    # make sure corresponding file(s) can be found
    ftype = None
    if op.exists(fname):
        if fname.endswith(("-vl.stc", "-vol.stc", "-vl.w", "-vol.w")):
            ftype = "volume"
        elif fname.endswith(".stc"):
            ftype = "surface"
            if fname.endswith(("-lh.stc", "-rh.stc")):
                fname = fname[:-7]
            else:
                err = (
                    f"Invalid .stc filename: {fname!r}; needs to end with "
                    "hemisphere tag ('...-lh.stc' or '...-rh.stc')"
                )
                raise OSError(err)
        elif fname.endswith(".w"):
            ftype = "w"
            if fname.endswith(("-lh.w", "-rh.w")):
                fname = fname[:-5]
            else:
                err = (
                    f"Invalid .w filename: {fname!r}; needs to end with "
                    "hemisphere tag ('...-lh.w' or '...-rh.w')"
                )
                raise OSError(err)
        elif fname.endswith(".h5"):
            ftype = "h5"
            fname = fname[:-3]
        else:
            raise RuntimeError(f"Unknown extension for file {fname_arg}")

    if ftype != "volume":
        stc_exist = [op.exists(f) for f in [fname + "-rh.stc", fname + "-lh.stc"]]
        w_exist = [op.exists(f) for f in [fname + "-rh.w", fname + "-lh.w"]]
        if all(stc_exist) and ftype != "w":
            ftype = "surface"
        elif all(w_exist):
            ftype = "w"
        elif op.exists(fname + ".h5"):
            ftype = "h5"
        elif op.exists(fname + "-stc.h5"):
            ftype = "h5"
            fname += "-stc"
        elif any(stc_exist) or any(w_exist):
            raise OSError(f"Hemisphere missing for {fname_arg!r}")
        else:
            raise OSError(f"SourceEstimate File(s) not found for: {fname_arg!r}")

    # read the files
    if ftype == "volume":  # volume source space
        if fname.endswith(".stc"):
            kwargs = _read_stc(fname)
        elif fname.endswith(".w"):
            kwargs = _read_w(fname)
            kwargs["data"] = kwargs["data"][:, np.newaxis]
            kwargs["tmin"] = 0.0
            kwargs["tstep"] = 0.0
        else:
            raise OSError("Volume source estimate must end with .stc or .w")
        kwargs["vertices"] = [kwargs["vertices"]]
    elif ftype == "surface":  # stc file with surface source spaces
        lh = _read_stc(fname + "-lh.stc")
        rh = _read_stc(fname + "-rh.stc")
        assert lh["tmin"] == rh["tmin"]
        assert lh["tstep"] == rh["tstep"]
        kwargs = lh.copy()
        kwargs["data"] = np.r_[lh["data"], rh["data"]]
        kwargs["vertices"] = [lh["vertices"], rh["vertices"]]
    elif ftype == "w":  # w file with surface source spaces
        lh = _read_w(fname + "-lh.w")
        rh = _read_w(fname + "-rh.w")
        kwargs = lh.copy()
        kwargs["data"] = np.atleast_2d(np.r_[lh["data"], rh["data"]]).T
        kwargs["vertices"] = [lh["vertices"], rh["vertices"]]
        # w files only have a single time point
        kwargs["tmin"] = 0.0
        kwargs["tstep"] = 1.0
        ftype = "surface"
    elif ftype == "h5":
        read_hdf5, _ = _import_h5io_funcs()
        kwargs = read_hdf5(fname + ".h5", title="mnepython")
        ftype = kwargs.pop("src_type", "surface")
        if isinstance(kwargs["vertices"], np.ndarray):
            kwargs["vertices"] = [kwargs["vertices"]]

    if ftype != "volume":
        # Make sure the vertices are ordered
        vertices = kwargs["vertices"]
        if any(np.any(np.diff(v.astype(int)) <= 0) for v in vertices):
            sidx = [np.argsort(verts) for verts in vertices]
            vertices = [verts[idx] for verts, idx in zip(vertices, sidx)]
            data = kwargs["data"][np.r_[sidx[0], len(sidx[0]) + sidx[1]]]
            kwargs["vertices"] = vertices
            kwargs["data"] = data

    if "subject" not in kwargs:
        kwargs["subject"] = subject
    if subject is not None and subject != kwargs["subject"]:
        raise RuntimeError(
            f'provided subject name "{subject}" does not match '
            f'subject name from the file "{kwargs["subject"]}'
        )

    if ftype in ("volume", "discrete"):
        klass = VolVectorSourceEstimate
    elif ftype == "mixed":
        klass = MixedVectorSourceEstimate
    else:
        assert ftype == "surface"
        klass = VectorSourceEstimate
    if kwargs["data"].ndim < 3:
        klass = klass._scalar_class
    return klass(**kwargs)


def _get_src_type(src, vertices, warn_text=None):
    src_type = None
    if src is None:
        if warn_text is None:
            warn("src should not be None for a robust guess of stc type.")
        else:
            warn(warn_text)
        if isinstance(vertices, list) and len(vertices) == 2:
            src_type = "surface"
        elif (
            isinstance(vertices, np.ndarray)
            or isinstance(vertices, list)
            and len(vertices) == 1
        ):
            src_type = "volume"
        elif isinstance(vertices, list) and len(vertices) > 2:
            src_type = "mixed"
    else:
        src_type = src.kind
    assert src_type in ("surface", "volume", "mixed", "discrete")
    return src_type


def _make_stc(
    data,
    vertices,
    src_type=None,
    tmin=None,
    tstep=None,
    subject=None,
    vector=False,
    source_nn=None,
    warn_text=None,
):
    """Generate a surface, vector-surface, volume or mixed source estimate."""

    def guess_src_type():
        return _get_src_type(src=None, vertices=vertices, warn_text=warn_text)

    src_type = guess_src_type() if src_type is None else src_type

    if vector and src_type == "surface" and source_nn is None:
        raise RuntimeError("No source vectors supplied.")

    # infer Klass from src_type
    if src_type == "surface":
        Klass = VectorSourceEstimate if vector else SourceEstimate
    elif src_type in ("volume", "discrete"):
        Klass = VolVectorSourceEstimate if vector else VolSourceEstimate
    elif src_type == "mixed":
        Klass = MixedVectorSourceEstimate if vector else MixedSourceEstimate
    else:
        raise ValueError(
            "vertices has to be either a list with one or more arrays or an array"
        )

    # Rotate back for vector source estimates
    if vector:
        n_vertices = sum(len(v) for v in vertices)
        assert data.shape[0] in (n_vertices, n_vertices * 3)
        if len(data) == n_vertices:
            assert src_type == "surface"  # should only be possible for this
            assert source_nn.shape == (n_vertices, 3)
            data = data[:, np.newaxis] * source_nn[:, :, np.newaxis]
        else:
            data = data.reshape((-1, 3, data.shape[-1]))
            assert source_nn.shape in ((n_vertices, 3, 3), (n_vertices * 3, 3))
            # This will be an identity transform for volumes, but let's keep
            # the code simple and general and just do the matrix mult
            data = np.matmul(
                np.transpose(source_nn.reshape(n_vertices, 3, 3), axes=[0, 2, 1]), data
            )

    return Klass(data=data, vertices=vertices, tmin=tmin, tstep=tstep, subject=subject)


def _verify_source_estimate_compat(a, b):
    """Make sure two SourceEstimates are compatible for arith. operations."""
    compat = False
    if type(a) is not type(b):
        raise ValueError(f"Cannot combine {type(a)} and {type(b)}.")
    if len(a.vertices) == len(b.vertices):
        if all(np.array_equal(av, vv) for av, vv in zip(a.vertices, b.vertices)):
            compat = True
    if not compat:
        raise ValueError(
            "Cannot combine source estimates that do not have "
            "the same vertices. Consider using stc.expand()."
        )
    if a.subject != b.subject:
        raise ValueError(
            "source estimates do not have the same subject "
            f"names, {repr(a.subject)} and {repr(b.subject)}"
        )


class _BaseSourceEstimate(TimeMixin, FilterMixin):
    _data_ndim = 2

    @verbose
    def __init__(self, data, vertices, tmin, tstep, subject=None, verbose=None):
        assert hasattr(self, "_data_ndim"), self.__class__.__name__
        assert hasattr(self, "_src_type"), self.__class__.__name__
        assert hasattr(self, "_src_count"), self.__class__.__name__
        kernel, sens_data = None, None
        if isinstance(data, tuple):
            if len(data) != 2:
                raise ValueError("If data is a tuple it has to be length 2")
            kernel, sens_data = data
            data = None
            if kernel.shape[1] != sens_data.shape[0]:
                raise ValueError(
                    f"kernel ({kernel.shape}) and sens_data ({sens_data.shape}) "
                    "have invalid dimensions"
                )
            if sens_data.ndim != 2:
                raise ValueError(
                    "The sensor data must have 2 dimensions, got {sens_data.ndim}"
                )

        _validate_type(vertices, list, "vertices")
        if self._src_count is not None:
            if len(vertices) != self._src_count:
                raise ValueError(
                    f"vertices must be a list with {self._src_count} entries, "
                    f"got {len(vertices)}."
                )
        vertices = [np.array(v, np.int64) for v in vertices]  # makes copy
        if any(np.any(np.diff(v) <= 0) for v in vertices):
            raise ValueError("Vertices must be ordered in increasing order.")

        n_src = sum([len(v) for v in vertices])

        # safeguard the user against doing something silly
        if data is not None:
            if data.ndim not in (self._data_ndim, self._data_ndim - 1):
                raise ValueError(
                    f"Data (shape {data.shape}) must have {self._data_ndim} "
                    f"dimensions for {self.__class__.__name__}"
                )
            if data.shape[0] != n_src:
                raise ValueError(
                    f"Number of vertices ({n_src}) and stc.data.shape[0] "
                    f"({data.shape[0]}) must match"
                )
            if self._data_ndim == 3:
                if data.shape[1] != 3:
                    raise ValueError(
                        "Data for VectorSourceEstimate must have "
                        f"shape[1] == 3, got shape {data.shape}"
                    )
            if data.ndim == self._data_ndim - 1:  # allow upbroadcasting
                data = data[..., np.newaxis]

        self._data = data
        self._tmin = tmin
        self._tstep = tstep
        self.vertices = vertices
        self._kernel = kernel
        self._sens_data = sens_data
        self._kernel_removed = False
        self._times = None
        self._update_times()
        self.subject = _check_subject(None, subject, raise_error=False)

    def __repr__(self):  # noqa: D105
        s = f"{sum(len(v) for v in self.vertices)} vertices"
        if self.subject is not None:
            s += f", subject : {self.subject}"
        s += ", tmin : %s (ms)" % (1e3 * self.tmin)
        s += ", tmax : %s (ms)" % (1e3 * self.times[-1])
        s += ", tstep : %s (ms)" % (1e3 * self.tstep)
        s += f", data shape : {self.shape}"
        sz = sum(object_size(x) for x in (self.vertices + [self.data]))
        s += f", ~{sizeof_fmt(sz)}"
        return f"<{type(self).__name__} | {s}>"

    @fill_doc
    def get_peak(
        self, tmin=None, tmax=None, mode="abs", vert_as_index=False, time_as_index=False
    ):
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        %(get_peak_parameters)s

        Returns
        -------
        pos : int
            The vertex exhibiting the maximum response, either ID or index.
        latency : float
            The latency in seconds.
        """
        stc = self.magnitude() if self._data_ndim == 3 else self
        if self._n_vertices == 0:
            raise RuntimeError("Cannot find peaks with no vertices")
        vert_idx, time_idx, _ = _get_peak(stc.data, self.times, tmin, tmax, mode)
        if not vert_as_index:
            vert_idx = np.concatenate(self.vertices)[vert_idx]
        if not time_as_index:
            time_idx = self.times[time_idx]
        return vert_idx, time_idx

    @verbose
    def extract_label_time_course(
        self, labels, src, mode="auto", allow_empty=False, verbose=None
    ):
        """Extract label time courses for lists of labels.

        This function will extract one time course for each label. The way the
        time courses are extracted depends on the mode parameter.

        Parameters
        ----------
        %(labels_eltc)s
        %(src_eltc)s
        %(mode_eltc)s
        %(allow_empty_eltc)s
        %(verbose)s

        Returns
        -------
        %(label_tc_el_returns)s

        See Also
        --------
        extract_label_time_course : Extract time courses for multiple STCs.

        Notes
        -----
        %(eltc_mode_notes)s
        """
        return extract_label_time_course(
            self,
            labels,
            src,
            mode=mode,
            return_generator=False,
            allow_empty=allow_empty,
            verbose=verbose,
        )

    @verbose
    def apply_function(
        self, fun, picks=None, dtype=None, n_jobs=None, verbose=None, **kwargs
    ):
        """Apply a function to a subset of vertices.

        %(applyfun_summary_stc)s

        Parameters
        ----------
        %(fun_applyfun_stc)s
        %(picks_all)s
        %(dtype_applyfun)s
        %(n_jobs)s Ignored if ``vertice_wise=False`` as the workload
            is split across vertices.
        %(verbose)s
        %(kwargs_fun)s

        Returns
        -------
        self : instance of SourceEstimate
            The SourceEstimate object with transformed data.
        """
        _check_preload(self, "source_estimate.apply_function")
        picks = _picks_to_idx(len(self._data), picks, exclude=(), with_ref_meg=False)

        if not callable(fun):
            raise ValueError("fun needs to be a function")

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        # check the dimension of the source estimate data
        _check_option("source_estimate.ndim", self._data.ndim, [2, 3])

        parallel, p_fun, n_jobs = parallel_func(_check_fun, n_jobs)
        if n_jobs == 1:
            # modify data inplace to save memory
            for idx in picks:
                self._data[idx, :] = _check_fun(fun, data_in[idx, :], **kwargs)
        else:
            # use parallel function
            data_picks_new = parallel(
                p_fun(fun, data_in[p, :], **kwargs) for p in picks
            )
            for pp, p in enumerate(picks):
                self._data[p, :] = data_picks_new[pp]

        return self

    @verbose
    def apply_baseline(self, baseline=(None, 0), *, verbose=None):
        """Baseline correct source estimate data.

        Parameters
        ----------
        %(baseline_stc)s
            Defaults to ``(None, 0)``, i.e. beginning of the the data until
            time point zero.
        %(verbose)s

        Returns
        -------
        stc : instance of SourceEstimate
            The baseline-corrected source estimate object.

        Notes
        -----
        Baseline correction can be done multiple times.
        """
        self.data = rescale(self.data, self.times, baseline, copy=False)
        return self

    @verbose
    def save(self, fname, ftype="h5", *, overwrite=False, verbose=None):
        """Save the full source estimate to an HDF5 file.

        Parameters
        ----------
        fname : path-like
            The file name to write the source estimate to, should end in
            ``'-stc.h5'``.
        ftype : str
            File format to use. Currently, the only allowed values is ``"h5"``.
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose)s
        """
        fname = _check_fname(fname=fname, overwrite=True)  # check below
        if ftype != "h5":
            raise ValueError(
                f"{self.__class__.__name__} objects can only be written as HDF5 files."
            )
        _, write_hdf5 = _import_h5io_funcs()
        if fname.suffix != ".h5":
            fname = fname.with_name(f"{fname.name}-stc.h5")
        fname = _check_fname(fname=fname, overwrite=overwrite)
        write_hdf5(
            fname,
            dict(
                vertices=self.vertices,
                data=self.data,
                tmin=self.tmin,
                tstep=self.tstep,
                subject=self.subject,
                src_type=self._src_type,
            ),
            title="mnepython",
            overwrite=True,
        )

    @copy_function_doc_to_method_doc(plot_source_estimates)
    def plot(
        self,
        subject=None,
        surface="inflated",
        hemi="lh",
        colormap="auto",
        time_label="auto",
        smoothing_steps=10,
        transparent=True,
        alpha=1.0,
        time_viewer="auto",
        *,
        subjects_dir=None,
        figure=None,
        views="auto",
        colorbar=True,
        clim="auto",
        cortex="classic",
        size=800,
        background="black",
        foreground=None,
        initial_time=None,
        time_unit="s",
        backend="auto",
        spacing="oct6",
        title=None,
        show_traces="auto",
        src=None,
        volume_options=1.0,
        view_layout="vertical",
        add_data_kwargs=None,
        brain_kwargs=None,
        verbose=None,
    ):
        brain = plot_source_estimates(
            self,
            subject,
            surface=surface,
            hemi=hemi,
            colormap=colormap,
            time_label=time_label,
            smoothing_steps=smoothing_steps,
            transparent=transparent,
            alpha=alpha,
            time_viewer=time_viewer,
            subjects_dir=subjects_dir,
            figure=figure,
            views=views,
            colorbar=colorbar,
            clim=clim,
            cortex=cortex,
            size=size,
            background=background,
            foreground=foreground,
            initial_time=initial_time,
            time_unit=time_unit,
            backend=backend,
            spacing=spacing,
            title=title,
            show_traces=show_traces,
            src=src,
            volume_options=volume_options,
            view_layout=view_layout,
            add_data_kwargs=add_data_kwargs,
            brain_kwargs=brain_kwargs,
            verbose=verbose,
        )
        return brain

    @property
    def sfreq(self):
        """Sample rate of the data."""
        return 1.0 / self.tstep

    @property
    def _n_vertices(self):
        return sum(len(v) for v in self.vertices)

    def _remove_kernel_sens_data_(self):
        """Remove kernel and sensor space data and compute self._data."""
        if self._kernel is not None or self._sens_data is not None:
            self._kernel_removed = True
            self._data = np.dot(self._kernel, self._sens_data)
            self._kernel = None
            self._sens_data = None

    @fill_doc
    def crop(self, tmin=None, tmax=None, include_tmax=True):
        """Restrict SourceEstimate to a time interval.

        Parameters
        ----------
        tmin : float | None
            The first time point in seconds. If None the first present is used.
        tmax : float | None
            The last time point in seconds. If None the last present is used.
        %(include_tmax)s

        Returns
        -------
        stc : instance of SourceEstimate
            The cropped source estimate.
        """
        mask = _time_mask(
            self.times, tmin, tmax, sfreq=self.sfreq, include_tmax=include_tmax
        )
        self.tmin = self.times[np.where(mask)[0][0]]
        if self._kernel is not None and self._sens_data is not None:
            self._sens_data = self._sens_data[..., mask]
        else:
            self.data = self.data[..., mask]

        return self  # return self for chaining methods

    @verbose
    def resample(
        self,
        sfreq,
        *,
        npad=100,
        method="fft",
        window="auto",
        pad="auto",
        n_jobs=None,
        verbose=None,
    ):
        """Resample data.

        If appropriate, an anti-aliasing filter is applied before resampling.
        See :ref:`resampling-and-decimating` for more information.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        npad : int | str
            Amount to pad the start and end of the data.
            Can also be "auto" to use a padding that will result in
            a power-of-two size (can be much faster).
        %(method_resample)s

            .. versionadded:: 1.7
        %(window_resample)s

            .. versionadded:: 1.7
        %(pad_resample_auto)s

            .. versionadded:: 1.7
        %(n_jobs)s
        %(verbose)s

        Returns
        -------
        stc : instance of SourceEstimate
            The resampled source estimate.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!

        Note that the sample rate of the original data is inferred from tstep.
        """
        from .filter import _check_resamp_noop

        o_sfreq = 1.0 / self.tstep
        if _check_resamp_noop(sfreq, o_sfreq):
            return self

        # resampling in sensor instead of source space gives a somewhat
        # different result, so we don't allow it
        self._remove_kernel_sens_data_()

        data = self.data
        if data.dtype == np.float32:
            data = data.astype(np.float64)
        self.data = resample(
            data, sfreq, o_sfreq, npad=npad, window=window, n_jobs=n_jobs, method=method
        )

        # adjust indirectly affected variables
        self.tstep = 1.0 / sfreq
        return self

    @property
    def data(self):
        """Numpy array of source estimate data."""
        if self._data is None:
            # compute the solution the first time the data is accessed and
            # remove the kernel and sensor data
            self._remove_kernel_sens_data_()
        return self._data

    @data.setter
    def data(self, value):
        value = np.asarray(value)
        if self._data is not None and value.ndim != self._data.ndim:
            raise ValueError(f"Data array should have {self._data.ndim} dimensions.")
        n_verts = sum(len(v) for v in self.vertices)
        if value.shape[0] != n_verts:
            raise ValueError(
                "The first dimension of the data array must match the number of "
                f"vertices ({value.shape[0]} != {n_verts})."
            )
        self._data = value
        self._update_times()

    @property
    def shape(self):
        """Shape of the data."""
        if self._data is not None:
            return self._data.shape
        return (self._kernel.shape[0], self._sens_data.shape[1])

    @property
    def tmin(self):
        """The first timestamp."""
        return self._tmin

    @tmin.setter
    def tmin(self, value):
        self._tmin = float(value)
        self._update_times()

    @property
    def tstep(self):
        """The change in time between two consecutive samples (1 / sfreq)."""
        return self._tstep

    @tstep.setter
    def tstep(self, value):
        if value <= 0:
            raise ValueError(".tstep must be greater than 0.")
        self._tstep = float(value)
        self._update_times()

    @property
    def times(self):
        """A timestamp for each sample."""
        return self._times

    @times.setter
    def times(self, value):
        raise ValueError(
            "You cannot write to the .times attribute directly. "
            "This property automatically updates whenever "
            ".tmin, .tstep or .data changes."
        )

    def _update_times(self):
        """Update the times attribute after changing tmin, tmax, or tstep."""
        self._times = self.tmin + (self.tstep * np.arange(self.shape[-1]))
        self._times.flags.writeable = False

    def __add__(self, a):
        """Add source estimates."""
        stc = self.copy()
        stc += a
        return stc

    def __iadd__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data += a.data
        else:
            self.data += a
        return self

    def mean(self):
        """Make a summary stc file with mean over time points.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The modified stc.
        """
        out = self.sum()
        out /= len(self.times)
        return out

    def sum(self):
        """Make a summary stc file with sum over time points.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The modified stc.
        """
        data = self.data
        tmax = self.tmin + self.tstep * data.shape[-1]
        tmin = (self.tmin + tmax) / 2.0
        tstep = tmax - self.tmin
        sum_stc = self.__class__(
            self.data.sum(axis=-1, keepdims=True),
            vertices=self.vertices,
            tmin=tmin,
            tstep=tstep,
            subject=self.subject,
        )
        return sum_stc

    def __sub__(self, a):
        """Subtract source estimates."""
        stc = self.copy()
        stc -= a
        return stc

    def __isub__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data -= a.data
        else:
            self.data -= a
        return self

    def __truediv__(self, a):  # noqa: D105
        return self.__div__(a)

    def __div__(self, a):  # noqa: D105
        """Divide source estimates."""
        stc = self.copy()
        stc /= a
        return stc

    def __itruediv__(self, a):  # noqa: D105
        return self.__idiv__(a)

    def __idiv__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data /= a.data
        else:
            self.data /= a
        return self

    def __mul__(self, a):
        """Multiply source estimates."""
        stc = self.copy()
        stc *= a
        return stc

    def __imul__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data *= a.data
        else:
            self.data *= a
        return self

    def __pow__(self, a):  # noqa: D105
        stc = self.copy()
        stc **= a
        return stc

    def __ipow__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        self.data **= a
        return self

    def __radd__(self, a):  # noqa: D105
        return self + a

    def __rsub__(self, a):  # noqa: D105
        return self - a

    def __rmul__(self, a):  # noqa: D105
        return self * a

    def __rdiv__(self, a):  # noqa: D105
        return self / a

    def __neg__(self):  # noqa: D105
        """Negate the source estimate."""
        stc = self.copy()
        stc._remove_kernel_sens_data_()
        stc.data *= -1
        return stc

    def __pos__(self):  # noqa: D105
        return self

    def __abs__(self):
        """Compute the absolute value of the data.

        Returns
        -------
        stc : instance of _BaseSourceEstimate
            A version of the source estimate, where the data attribute is set
            to abs(self.data).
        """
        stc = self.copy()
        stc._remove_kernel_sens_data_()
        stc._data = abs(stc._data)
        return stc

    def sqrt(self):
        """Take the square root.

        Returns
        -------
        stc : instance of SourceEstimate
            A copy of the SourceEstimate with sqrt(data).
        """
        return self ** (0.5)

    def copy(self):
        """Return copy of source estimate instance.

        Returns
        -------
        stc : instance of SourceEstimate
            A copy of the source estimate.
        """
        return copy.deepcopy(self)

    def bin(self, width, tstart=None, tstop=None, func=np.mean):
        """Return a source estimate object with data summarized over time bins.

        Time bins of ``width`` seconds. This method is intended for
        visualization only. No filter is applied to the data before binning,
        making the method inappropriate as a tool for downsampling data.

        Parameters
        ----------
        width : scalar
            Width of the individual bins in seconds.
        tstart : scalar | None
            Time point where the first bin starts. The default is the first
            time point of the stc.
        tstop : scalar | None
            Last possible time point contained in a bin (if the last bin would
            be shorter than width it is dropped). The default is the last time
            point of the stc.
        func : callable
            Function that is applied to summarize the data. Needs to accept a
            numpy.array as first input and an ``axis`` keyword argument.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The binned source estimate.
        """
        if tstart is None:
            tstart = self.tmin
        if tstop is None:
            tstop = self.times[-1]

        times = np.arange(tstart, tstop + self.tstep, width)
        nt = len(times) - 1
        data = np.empty(self.shape[:-1] + (nt,), dtype=self.data.dtype)
        for i in range(nt):
            idx = (self.times >= times[i]) & (self.times < times[i + 1])
            data[..., i] = func(self.data[..., idx], axis=-1)

        tmin = times[0] + width / 2.0
        stc = self.copy()
        stc._data = data
        stc.tmin = tmin
        stc.tstep = width
        return stc

    def transform_data(self, func, idx=None, tmin_idx=None, tmax_idx=None):
        """Get data after a linear (time) transform has been applied.

        The transform is applied to each source time course independently.

        Parameters
        ----------
        func : callable
            The transform to be applied, including parameters (see, e.g.,
            :func:`functools.partial`). The first parameter of the function is
            the input data. The first return value is the transformed data,
            remaining outputs are ignored. The first dimension of the
            transformed data has to be the same as the first dimension of the
            input data.
        idx : array | None
            Indicices of source time courses for which to compute transform.
            If None, all time courses are used.
        tmin_idx : int | None
            Index of first time point to include. If None, the index of the
            first time point is used.
        tmax_idx : int | None
            Index of the first time point not to include. If None, time points
            up to (and including) the last time point are included.

        Returns
        -------
        data_t : ndarray
            The transformed data.

        Notes
        -----
        Applying transforms can be significantly faster if the
        SourceEstimate object was created using "(kernel, sens_data)", for
        the "data" parameter as the transform is applied in sensor space.
        Inverse methods, e.g., "apply_inverse_epochs", or "apply_lcmv_epochs"
        do this automatically (if possible).
        """
        if idx is None:
            # use all time courses by default
            idx = slice(None, None)

        if self._kernel is None and self._sens_data is None:
            if self._kernel_removed:
                warn(
                    "Performance can be improved by not accessing the data "
                    "attribute before calling this method."
                )

            # transform source space data directly
            data_t = func(self.data[idx, ..., tmin_idx:tmax_idx])

            if isinstance(data_t, tuple):
                # use only first return value
                data_t = data_t[0]
        else:
            # apply transform in sensor space
            sens_data_t = func(self._sens_data[:, tmin_idx:tmax_idx])

            if isinstance(sens_data_t, tuple):
                # use only first return value
                sens_data_t = sens_data_t[0]

            # apply inverse
            data_shape = sens_data_t.shape
            if len(data_shape) > 2:
                # flatten the last dimensions
                sens_data_t = sens_data_t.reshape(
                    data_shape[0], np.prod(data_shape[1:])
                )

            data_t = np.dot(self._kernel[idx, :], sens_data_t)

            # restore original shape if necessary
            if len(data_shape) > 2:
                data_t = data_t.reshape(data_t.shape[0], *data_shape[1:])

        return data_t

    def transform(self, func, idx=None, tmin=None, tmax=None, copy=False):
        """Apply linear transform.

        The transform is applied to each source time course independently.

        Parameters
        ----------
        func : callable
            The transform to be applied, including parameters (see, e.g.,
            :func:`functools.partial`). The first parameter of the function is
            the input data. The first two dimensions of the transformed data
            should be (i) vertices and (ii) time.  See Notes for details.
        idx : array | None
            Indices of source time courses for which to compute transform.
            If None, all time courses are used.
        tmin : float | int | None
            First time point to include (ms). If None, self.tmin is used.
        tmax : float | int | None
            Last time point to include (ms). If None, self.tmax is used.
        copy : bool
            If True, return a new instance of SourceEstimate instead of
            modifying the input inplace.

        Returns
        -------
        stcs : SourceEstimate | VectorSourceEstimate | list
            The transformed stc or, in the case of transforms which yield
            N-dimensional output (where N > 2), a list of stcs. For a list,
            copy must be True.

        Notes
        -----
        Transforms which yield 3D
        output (e.g. time-frequency transforms) are valid, so long as the
        first two dimensions are vertices and time.  In this case, the
        copy parameter must be True and a list of
        SourceEstimates, rather than a single instance of SourceEstimate,
        will be returned, one for each index of the 3rd dimension of the
        transformed data.  In the case of transforms yielding 2D output
        (e.g. filtering), the user has the option of modifying the input
        inplace (copy = False) or returning a new instance of
        SourceEstimate (copy = True) with the transformed data.

        Applying transforms can be significantly faster if the
        SourceEstimate object was created using "(kernel, sens_data)", for
        the "data" parameter as the transform is applied in sensor space.
        Inverse methods, e.g., "apply_inverse_epochs", or "apply_lcmv_epochs"
        do this automatically (if possible).
        """
        # min and max data indices to include
        times = 1000.0 * self.times
        t_idx = np.where(_time_mask(times, tmin, tmax, sfreq=self.sfreq))[0]
        if tmin is None:
            tmin_idx = None
        else:
            tmin_idx = t_idx[0]

        if tmax is None:
            tmax_idx = None
        else:
            # +1, because upper boundary needs to include the last sample
            tmax_idx = t_idx[-1] + 1

        data_t = self.transform_data(
            func, idx=idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx
        )

        # account for change in n_vertices
        if idx is not None:
            idx_lh = idx[idx < len(self.lh_vertno)]
            idx_rh = idx[idx >= len(self.lh_vertno)] - len(self.lh_vertno)
            verts_lh = self.lh_vertno[idx_lh]
            verts_rh = self.rh_vertno[idx_rh]
        else:
            verts_lh = self.lh_vertno
            verts_rh = self.rh_vertno
        verts = [verts_lh, verts_rh]

        tmin_idx = 0 if tmin_idx is None else tmin_idx
        tmin = self.times[tmin_idx]

        if data_t.ndim > 2:
            # return list of stcs if transformed data has dimensionality > 2
            if copy:
                stcs = [
                    SourceEstimate(
                        data_t[:, :, a], verts, tmin, self.tstep, self.subject
                    )
                    for a in range(data_t.shape[-1])
                ]
            else:
                raise ValueError(
                    "copy must be True if transformed data has more than 2 dimensions"
                )
        else:
            # return new or overwritten stc
            stcs = self if not copy else self.copy()
            stcs.vertices = verts
            stcs.data = data_t
            stcs.tmin = tmin

        return stcs

    @verbose
    def to_data_frame(
        self,
        index=None,
        scalings=None,
        long_format=False,
        time_format=None,
        *,
        verbose=None,
    ):
        """Export data in tabular structure as a pandas DataFrame.

        Vertices are converted to columns in the DataFrame. By default,
        an additional column "time" is added, unless ``index='time'``
        (in which case time values form the DataFrame's index).

        Parameters
        ----------
        %(index_df_evk)s
            Defaults to ``None``.
        %(scalings_df)s
        %(long_format_df_stc)s
        %(time_format_df)s

            .. versionadded:: 0.20
        %(verbose)s

        Returns
        -------
        %(df_return)s
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ["time", "subject"]
        valid_time_formats = ["ms", "timedelta"]
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        data = self.data.T
        times = self.times
        # prepare extra columns / multiindex
        mindex = list()
        default_index = ["time"]
        if self.subject is not None:
            default_index = ["subject", "time"]
            mindex.append(("subject", np.repeat(self.subject, data.shape[0])))
        times = _convert_times(times, time_format)
        mindex.append(("time", times))
        # triage surface vs volume source estimates
        col_names = list()
        kinds = ["VOL"] * len(self.vertices)
        if isinstance(self, _BaseSurfaceSourceEstimate | _BaseMixedSourceEstimate):
            kinds[:2] = ["LH", "RH"]
        for kind, vertno in zip(kinds, self.vertices):
            col_names.extend([f"{kind}_{vert}" for vert in vertno])
        # build DataFrame
        df = _build_data_frame(
            self,
            data,
            None,
            long_format,
            mindex,
            index,
            default_index=default_index,
            col_names=col_names,
            col_kind="source",
        )
        return df


def _center_of_mass(
    vertices, values, hemi, surf, subject, subjects_dir, restrict_vertices
):
    """Find the center of mass on a surface."""
    if (values == 0).all() or (values < 0).any():
        raise ValueError(
            "All values must be non-negative and at least one "
            "must be non-zero, cannot compute COM"
        )
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    surf = read_surface(subjects_dir / subject / "surf" / f"{hemi}.{surf}")
    if restrict_vertices is True:
        restrict_vertices = vertices
    elif restrict_vertices is False:
        restrict_vertices = np.arange(surf[0].shape[0])
    elif isinstance(restrict_vertices, SourceSpaces):
        idx = 1 if restrict_vertices.kind == "surface" and hemi == "rh" else 0
        restrict_vertices = restrict_vertices[idx]["vertno"]
    else:
        restrict_vertices = np.array(restrict_vertices, int)
    pos = surf[0][vertices, :].T
    c_o_m = np.sum(pos * values, axis=1) / np.sum(values)
    vertex = np.argmin(
        np.sqrt(np.mean((surf[0][restrict_vertices, :] - c_o_m) ** 2, axis=1))
    )
    vertex = restrict_vertices[vertex]
    return vertex


@fill_doc
class _BaseSurfaceSourceEstimate(_BaseSourceEstimate):
    """Abstract base class for surface source estimates.

    Parameters
    ----------
    data : array
        The data in source space.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    data : array
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).
    """

    _src_type = "surface"
    _src_count = 2

    @property
    def lh_data(self):
        """Left hemisphere data."""
        return self.data[: len(self.lh_vertno)]

    @property
    def rh_data(self):
        """Right hemisphere data."""
        return self.data[len(self.lh_vertno) :]

    @property
    def lh_vertno(self):
        """Left hemisphere vertno."""
        return self.vertices[0]

    @property
    def rh_vertno(self):
        """Right hemisphere vertno."""
        return self.vertices[1]

    def _hemilabel_stc(self, label):
        if label.hemi == "lh":
            stc_vertices = self.vertices[0]
        else:
            stc_vertices = self.vertices[1]

        # find index of the Label's vertices
        idx = np.nonzero(np.isin(stc_vertices, label.vertices))[0]

        # find output vertices
        vertices = stc_vertices[idx]

        # find data
        if label.hemi == "rh":
            values = self.data[idx + len(self.vertices[0])]
        else:
            values = self.data[idx]

        return vertices, values

    def in_label(self, label):
        """Get a source estimate object restricted to a label.

        SourceEstimate contains the time course of
        activation of all sources inside the label.

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The source estimate restricted to the given label.
        """
        # make sure label and stc are compatible
        from .label import BiHemiLabel, Label

        _validate_type(label, (Label, BiHemiLabel), "label")
        if (
            label.subject is not None
            and self.subject is not None
            and label.subject != self.subject
        ):
            raise RuntimeError(
                "label and stc must have same subject names, "
                f'currently "{label.subject}" and "{self.subject}"'
            )

        if label.hemi == "both":
            lh_vert, lh_val = self._hemilabel_stc(label.lh)
            rh_vert, rh_val = self._hemilabel_stc(label.rh)
            vertices = [lh_vert, rh_vert]
            values = np.vstack((lh_val, rh_val))
        elif label.hemi == "lh":
            lh_vert, values = self._hemilabel_stc(label)
            vertices = [lh_vert, np.array([], int)]
        else:
            assert label.hemi == "rh"
            rh_vert, values = self._hemilabel_stc(label)
            vertices = [np.array([], int), rh_vert]

        if sum([len(v) for v in vertices]) == 0:
            raise ValueError("No vertices match the label in the stc file")

        label_stc = self.__class__(
            values,
            vertices=vertices,
            tmin=self.tmin,
            tstep=self.tstep,
            subject=self.subject,
        )
        return label_stc

    def save_as_surface(self, fname, src, *, scale=1, scale_rr=1e3):
        """Save a surface source estimate (stc) as a GIFTI file.

        Parameters
        ----------
        fname : path-like
            Filename basename to save files as.
            Will write anatomical GIFTI plus time series GIFTI for both lh/rh,
            for example ``"basename"`` will write ``"basename.lh.gii"``,
            ``"basename.lh.time.gii"``, ``"basename.rh.gii"``, and
            ``"basename.rh.time.gii"``.
        src : instance of SourceSpaces
            The source space of the forward solution.
        scale : float
            Scale factor to apply to the data (functional) values.
        scale_rr : float
            Scale factor for the source vertex positions. The default (1e3) will
            scale from meters to millimeters, which is more standard for GIFTI files.

        Notes
        -----
        .. versionadded:: 1.7
        """
        nib = _import_nibabel()
        _check_option("src.kind", src.kind, ("surface", "mixed"))
        ss = get_decimated_surfaces(src)
        assert len(ss) == 2  # should be guaranteed by _check_option above

        # Create lists to put DataArrays into
        hemis = ("lh", "rh")
        for s, hemi in zip(ss, hemis):
            darrays = list()
            darrays.append(
                nib.gifti.gifti.GiftiDataArray(
                    data=(s["rr"] * scale_rr).astype(np.float32),
                    intent="NIFTI_INTENT_POINTSET",
                    datatype="NIFTI_TYPE_FLOAT32",
                )
            )

            # Make the topology DataArray
            darrays.append(
                nib.gifti.gifti.GiftiDataArray(
                    data=s["tris"].astype(np.int32),
                    intent="NIFTI_INTENT_TRIANGLE",
                    datatype="NIFTI_TYPE_INT32",
                )
            )

            # Make the output GIFTI for anatomicals
            topo_gi_hemi = nib.gifti.gifti.GiftiImage(darrays=darrays)

            # actually save the file
            nib.save(topo_gi_hemi, f"{fname}-{hemi}.gii")

            # Make the Time Series data arrays
            ts = []
            data = getattr(self, f"{hemi}_data") * scale
            ts = [
                nib.gifti.gifti.GiftiDataArray(
                    data=data[:, idx].astype(np.float32),
                    intent="NIFTI_INTENT_POINTSET",
                    datatype="NIFTI_TYPE_FLOAT32",
                )
                for idx in range(data.shape[1])
            ]

            # save the time series
            ts_gi = nib.gifti.gifti.GiftiImage(darrays=ts)
            nib.save(ts_gi, f"{fname}-{hemi}.time.gii")

    def expand(self, vertices):
        """Expand SourceEstimate to include more vertices.

        This will add rows to stc.data (zero-filled) and modify stc.vertices
        to include all vertices in stc.vertices and the input vertices.

        Parameters
        ----------
        vertices : list of array
            New vertices to add. Can also contain old values.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The modified stc (note: method operates inplace).
        """
        if not isinstance(vertices, list):
            raise TypeError("vertices must be a list")
        if not len(self.vertices) == len(vertices):
            raise ValueError("vertices must have the same length as stc.vertices")

        # can no longer use kernel and sensor data
        self._remove_kernel_sens_data_()

        inserters = list()
        offsets = [0]
        for vi, (v_old, v_new) in enumerate(zip(self.vertices, vertices)):
            v_new = np.setdiff1d(v_new, v_old)
            inds = np.searchsorted(v_old, v_new)
            # newer numpy might overwrite inds after np.insert, copy here
            inserters += [inds.copy()]
            offsets += [len(v_old)]
            self.vertices[vi] = np.insert(v_old, inds, v_new)
        inds = [ii + offset for ii, offset in zip(inserters, offsets[:-1])]
        inds = np.concatenate(inds)
        new_data = np.zeros((len(inds),) + self.data.shape[1:])
        self.data = np.insert(self.data, inds, new_data, axis=0)
        return self

    @verbose
    def to_original_src(
        self, src_orig, subject_orig=None, subjects_dir=None, verbose=None
    ):
        """Get a source estimate from morphed source to the original subject.

        Parameters
        ----------
        src_orig : instance of SourceSpaces
            The original source spaces that were morphed to the current
            subject.
        subject_orig : str | None
            The original subject. For most source spaces this shouldn't need
            to be provided, since it is stored in the source space itself.
        %(subjects_dir)s
        %(verbose)s

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The transformed source estimate.

        See Also
        --------
        morph_source_spaces

        Notes
        -----
        .. versionadded:: 0.10.0
        """
        if self.subject is None:
            raise ValueError("stc.subject must be set")
        src_orig = _ensure_src(src_orig, kind="surface")
        subject_orig = _ensure_src_subject(src_orig, subject_orig)
        data_idx, vertices = _get_morph_src_reordering(
            self.vertices, src_orig, subject_orig, self.subject, subjects_dir
        )
        return self.__class__(
            self._data[data_idx], vertices, self.tmin, self.tstep, subject_orig
        )

    @fill_doc
    def get_peak(
        self,
        hemi=None,
        tmin=None,
        tmax=None,
        mode="abs",
        vert_as_index=False,
        time_as_index=False,
    ):
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        hemi : {'lh', 'rh', None}
            The hemi to be considered. If None, the entire source space is
            considered.
        %(get_peak_parameters)s

        Returns
        -------
        pos : int
            The vertex exhibiting the maximum response, either ID or index.
        latency : float | int
            The time point of the maximum response, either latency in seconds
            or index.
        """
        _check_option("hemi", hemi, ("lh", "rh", None))
        vertex_offset = 0
        if hemi is not None:
            if hemi == "lh":
                data = self.lh_data
                vertices = [self.lh_vertno, []]
            else:
                vertex_offset = len(self.vertices[0])
                data = self.rh_data
                vertices = [[], self.rh_vertno]
            meth = self.__class__(data, vertices, self.tmin, self.tstep).get_peak
        else:
            meth = super().get_peak
        out = meth(
            tmin=tmin,
            tmax=tmax,
            mode=mode,
            vert_as_index=vert_as_index,
            time_as_index=time_as_index,
        )
        if vertex_offset and vert_as_index:
            out = (out[0] + vertex_offset, out[1])
        return out


@fill_doc
class SourceEstimate(_BaseSurfaceSourceEstimate):
    """Container for surface source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. When it is a single array, the
        left hemisphere is stored in data[:len(vertices[0])] and the right
        hemisphere is stored in data[-len(vertices[1]):].
        When data is a tuple, it contains two arrays:

        - "kernel" shape (n_vertices, n_sensors) and
        - "sens_data" shape (n_sensors, n_times).

        In this case, the source space data corresponds to
        ``np.dot(kernel, sens_data)``.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : list of array, shape (2,)
        The indices of the dipoles in the left and right source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    VectorSourceEstimate : A container for vector surface source estimates.
    VolSourceEstimate : A container for volume source estimates.
    VolVectorSourceEstimate : A container for volume vector source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.
    """

    @verbose
    def save(self, fname, ftype="stc", *, overwrite=False, verbose=None):
        """Save the source estimates to a file.

        Parameters
        ----------
        fname : path-like
            The stem of the file name. The file names used for surface source
            spaces are obtained by adding ``"-lh.stc"`` and ``"-rh.stc"`` (or
            ``"-lh.w"`` and ``"-rh.w"``) to the stem provided, for the left and
            the right hemisphere, respectively.
        ftype : str
            File format to use. Allowed values are ``"stc"`` (default),
            ``"w"``, and ``"h5"``. The ``"w"`` format only supports a single
            time point.
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose)s
        """
        fname = str(_check_fname(fname=fname, overwrite=True))  # checked below
        _check_option("ftype", ftype, ["stc", "w", "h5"])

        lh_data = self.data[: len(self.lh_vertno)]
        rh_data = self.data[-len(self.rh_vertno) :]

        if ftype == "stc":
            if np.iscomplexobj(self.data):
                raise ValueError(
                    "Cannot save complex-valued STC data in "
                    "FIFF format; please set ftype='h5' to save "
                    "in HDF5 format instead, or cast the data to "
                    "real numbers before saving."
                )
            logger.info("Writing STC to disk...")
            fname_l = str(_check_fname(fname + "-lh.stc", overwrite=overwrite))
            fname_r = str(_check_fname(fname + "-rh.stc", overwrite=overwrite))
            _write_stc(
                fname_l,
                tmin=self.tmin,
                tstep=self.tstep,
                vertices=self.lh_vertno,
                data=lh_data,
            )
            _write_stc(
                fname_r,
                tmin=self.tmin,
                tstep=self.tstep,
                vertices=self.rh_vertno,
                data=rh_data,
            )
        elif ftype == "w":
            if self.shape[1] != 1:
                raise ValueError("w files can only contain a single time point.")
            logger.info("Writing STC to disk (w format)...")
            fname_l = str(_check_fname(fname + "-lh.w", overwrite=overwrite))
            fname_r = str(_check_fname(fname + "-rh.w", overwrite=overwrite))
            _write_w(fname_l, vertices=self.lh_vertno, data=lh_data[:, 0])
            _write_w(fname_r, vertices=self.rh_vertno, data=rh_data[:, 0])
        elif ftype == "h5":
            super().save(fname, overwrite=overwrite)
        logger.info("[done]")

    @verbose
    def estimate_snr(self, info, fwd, cov, verbose=None):
        r"""Compute time-varying SNR in the source space.

        This function should only be used with source estimates with units
        nanoAmperes (i.e., MNE-like solutions, *not* dSPM or sLORETA).
        See also :footcite:`GoldenholzEtAl2009`.

        .. warning:: This function currently only works properly for fixed
                     orientation.

        Parameters
        ----------
        %(info_not_none)s
        fwd : instance of Forward
            The forward solution used to create the source estimate.
        cov : instance of Covariance
            The noise covariance used to estimate the resting cortical
            activations. Should be an evoked covariance, not empty room.
        %(verbose)s

        Returns
        -------
        snr_stc : instance of SourceEstimate
            The source estimate with the SNR computed.

        Notes
        -----
        We define the SNR in decibels for each source location at each
        time point as:

        .. math::

            {\rm SNR} = 10\log_10[\frac{a^2}{N}\sum_k\frac{b_k^2}{s_k^2}]

        where :math:`\\b_k` is the signal on sensor :math:`k` provided by the
        forward model for a source with unit amplitude, :math:`a` is the
        source amplitude, :math:`N` is the number of sensors, and
        :math:`s_k^2` is the noise variance on sensor :math:`k`.

        References
        ----------
        .. footbibliography::
        """
        from .forward import Forward, convert_forward_solution
        from .minimum_norm.inverse import _prepare_forward

        _validate_type(fwd, Forward, "fwd")
        _validate_type(info, Info, "info")
        _validate_type(cov, Covariance, "cov")
        _check_stc_units(self)
        if (self.data >= 0).all():
            warn(
                "This STC appears to be from free orientation, currently SNR"
                " function is valid only for fixed orientation"
            )

        fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=False)

        # G is gain matrix [ch x src], cov is noise covariance [ch x ch]
        G, _, _, _, _, _, _, cov, _ = _prepare_forward(
            fwd,
            info,
            cov,
            fixed=True,
            loose=0,
            rank=None,
            pca=False,
            use_cps=True,
            exp=None,
            limit_depth_chs=False,
            combine_xyz="fro",
            allow_fixed_depth=False,
            limit=None,
        )
        G = G["sol"]["data"]
        n_channels = cov["dim"]  # number of sensors/channels
        b_k2 = (G * G).T
        s_k2 = np.diag(cov["data"])
        scaling = (1 / n_channels) * np.sum(b_k2 / s_k2, axis=1, keepdims=True)
        snr_stc = self.copy()
        snr_stc._data[:] = 10 * np.log10((self.data * self.data) * scaling)
        return snr_stc

    @fill_doc
    def center_of_mass(
        self,
        subject=None,
        hemi=None,
        restrict_vertices=False,
        subjects_dir=None,
        surf="sphere",
    ):
        """Compute the center of mass of activity.

        This function computes the spatial center of mass on the surface
        as well as the temporal center of mass as in :footcite:`LarsonLee2013`.

        .. note:: All activity must occur in a single hemisphere, otherwise
                  an error is raised. The "mass" of each point in space for
                  computing the spatial center of mass is computed by summing
                  across time, and vice-versa for each point in time in
                  computing the temporal center of mass. This is useful for
                  quantifying spatio-temporal cluster locations, especially
                  when combined with :func:`mne.vertex_to_mni`.

        Parameters
        ----------
        subject : str | None
            The subject the stc is defined for.
        hemi : int, or None
            Calculate the center of mass for the left (0) or right (1)
            hemisphere. If None, one of the hemispheres must be all zeroes,
            and the center of mass will be calculated for the other
            hemisphere (useful for getting COM for clusters).
        restrict_vertices : bool | array of int | instance of SourceSpaces
            If True, returned vertex will be one from stc. Otherwise, it could
            be any vertex from surf. If an array of int, the returned vertex
            will come from that array. If instance of SourceSpaces (as of
            0.13), the returned vertex will be from the given source space.
            For most accuruate estimates, do not restrict vertices.
        %(subjects_dir)s
        surf : str
            The surface to use for Euclidean distance center of mass
            finding. The default here is "sphere", which finds the center
            of mass on the spherical surface to help avoid potential issues
            with cortical folding.

        Returns
        -------
        vertex : int
            Vertex of the spatial center of mass for the inferred hemisphere,
            with each vertex weighted by the sum of the stc across time. For a
            boolean stc, then, this would be weighted purely by the duration
            each vertex was active.
        hemi : int
            Hemisphere the vertex was taken from.
        t : float
            Time of the temporal center of mass (weighted by the sum across
            source vertices).

        See Also
        --------
        mne.Label.center_of_mass
        mne.vertex_to_mni

        References
        ----------
        .. footbibliography::
        """
        if not isinstance(surf, str):
            raise TypeError(f"surf must be a string, got {type(surf)}")
        subject = _check_subject(self.subject, subject)
        if np.any(self.data < 0):
            raise ValueError("Cannot compute COM with negative values")
        values = np.sum(self.data, axis=1)  # sum across time
        vert_inds = [
            np.arange(len(self.vertices[0])),
            np.arange(len(self.vertices[1])) + len(self.vertices[0]),
        ]
        if hemi is None:
            hemi = np.where(np.array([np.sum(values[vi]) for vi in vert_inds]))[0]
            if not len(hemi) == 1:
                raise ValueError("Could not infer hemisphere")
            hemi = hemi[0]
        _check_option("hemi", hemi, [0, 1])
        vertices = self.vertices[hemi]
        values = values[vert_inds[hemi]]  # left or right
        del vert_inds
        vertex = _center_of_mass(
            vertices,
            values,
            hemi=["lh", "rh"][hemi],
            surf=surf,
            subject=subject,
            subjects_dir=subjects_dir,
            restrict_vertices=restrict_vertices,
        )
        # do time center of mass by using the values across space
        masses = np.sum(self.data, axis=0).astype(float)
        t_ind = np.sum(masses * np.arange(self.shape[1])) / np.sum(masses)
        t = self.tmin + self.tstep * t_ind
        return vertex, hemi, t


class _BaseVectorSourceEstimate(_BaseSourceEstimate):
    _data_ndim = 3

    @verbose
    def __init__(
        self, data, vertices=None, tmin=None, tstep=None, subject=None, verbose=None
    ):
        assert hasattr(self, "_scalar_class")
        super().__init__(data, vertices, tmin, tstep, subject, verbose)

    def magnitude(self):
        """Compute magnitude of activity without directionality.

        Returns
        -------
        stc : instance of SourceEstimate
            The source estimate without directionality information.
        """
        data_mag = np.linalg.norm(self.data, axis=1)
        return self._scalar_class(
            data_mag, self.vertices, self.tmin, self.tstep, self.subject
        )

    def _get_src_normals(self, src, use_cps):
        normals = np.vstack(
            [_get_src_nn(s, use_cps, v) for s, v in zip(src, self.vertices)]
        )
        return normals

    @fill_doc
    def project(self, directions, src=None, use_cps=True):
        """Project the data for each vertex in a given direction.

        Parameters
        ----------
        directions : ndarray, shape (n_vertices, 3) | str
            Can be:

            - ``'normal'``
                Project onto the source space normals.
            - ``'pca'``
                SVD will be used to project onto the direction of maximal
                power for each source.
            - :class:`~numpy.ndarray`, shape (n_vertices, 3)
                Projection directions for each source.
        src : instance of SourceSpaces | None
            The source spaces corresponding to the source estimate.
            Not used when ``directions`` is an array, optional when
            ``directions='pca'``.
        %(use_cps)s
            Should be the same value that was used when the forward model
            was computed (typically True).

        Returns
        -------
        stc : instance of SourceEstimate
            The projected source estimate.
        directions : ndarray, shape (n_vertices, 3)
            The directions that were computed (or just used).

        Notes
        -----
        When using SVD, there is a sign ambiguity for the direction of maximal
        power. When ``src is None``, the direction is chosen that makes the
        resulting time waveform sum positive (i.e., have positive amplitudes).
        When ``src`` is provided, the directions are flipped in the direction
        of the source normals, i.e., outward from cortex for surface source
        spaces and in the +Z / superior direction for volume source spaces.

        .. versionadded:: 0.21
        """
        _validate_type(directions, (str, np.ndarray), "directions")
        _validate_type(src, (None, SourceSpaces), "src")
        if isinstance(directions, str):
            _check_option("directions", directions, ("normal", "pca"), extra="when str")

            if directions == "normal":
                if src is None:
                    raise ValueError('If directions="normal", src cannot be None')
                _check_src_normal("normal", src)
                directions = self._get_src_normals(src, use_cps)
            else:
                assert directions == "pca"
                x = self.data
                if not np.isrealobj(self.data):
                    _check_option(
                        "stc.data.dtype", self.data.dtype, (np.complex64, np.complex128)
                    )
                    dtype = np.float32 if x.dtype == np.complex64 else np.float64
                    x = x.view(dtype)
                    assert x.shape[-1] == 2 * self.data.shape[-1]
                u, _, v = np.linalg.svd(x, full_matrices=False)
                directions = u[:, :, 0]
                # The sign is arbitrary, so let's flip it in the direction that
                # makes the resulting time series the most positive:
                if src is None:
                    signs = np.sum(v[:, 0].real, axis=1, keepdims=True)
                else:
                    normals = self._get_src_normals(src, use_cps)
                    signs = np.sum(directions * normals, axis=1, keepdims=True)
                assert signs.shape == (self.data.shape[0], 1)
                signs = np.sign(signs)
                signs[signs == 0] = 1.0
                directions *= signs
        _check_option("directions.shape", directions.shape, [(self.data.shape[0], 3)])
        data_norm = np.matmul(directions[:, np.newaxis], self.data)[:, 0]
        stc = self._scalar_class(
            data_norm, self.vertices, self.tmin, self.tstep, self.subject
        )
        return stc, directions

    @copy_function_doc_to_method_doc(plot_vector_source_estimates)
    def plot(
        self,
        subject=None,
        hemi="lh",
        colormap="hot",
        time_label="auto",
        smoothing_steps=10,
        transparent=True,
        brain_alpha=0.4,
        overlay_alpha=None,
        vector_alpha=1.0,
        scale_factor=None,
        time_viewer="auto",
        *,
        subjects_dir=None,
        figure=None,
        views="lateral",
        colorbar=True,
        clim="auto",
        cortex="classic",
        size=800,
        background="black",
        foreground=None,
        initial_time=None,
        time_unit="s",
        title=None,
        show_traces="auto",
        src=None,
        volume_options=1.0,
        view_layout="vertical",
        add_data_kwargs=None,
        brain_kwargs=None,
        verbose=None,
    ):
        return plot_vector_source_estimates(
            self,
            subject=subject,
            hemi=hemi,
            colormap=colormap,
            time_label=time_label,
            smoothing_steps=smoothing_steps,
            transparent=transparent,
            brain_alpha=brain_alpha,
            overlay_alpha=overlay_alpha,
            vector_alpha=vector_alpha,
            scale_factor=scale_factor,
            time_viewer=time_viewer,
            subjects_dir=subjects_dir,
            figure=figure,
            views=views,
            colorbar=colorbar,
            clim=clim,
            cortex=cortex,
            size=size,
            background=background,
            foreground=foreground,
            initial_time=initial_time,
            time_unit=time_unit,
            title=title,
            show_traces=show_traces,
            src=src,
            volume_options=volume_options,
            view_layout=view_layout,
            add_data_kwargs=add_data_kwargs,
            brain_kwargs=brain_kwargs,
            verbose=verbose,
        )


class _BaseVolSourceEstimate(_BaseSourceEstimate):
    _src_type = "volume"
    _src_count = None

    @copy_function_doc_to_method_doc(plot_source_estimates)
    def plot_3d(
        self,
        subject=None,
        surface="white",
        hemi="both",
        colormap="auto",
        time_label="auto",
        smoothing_steps=10,
        transparent=True,
        alpha=0.1,
        time_viewer="auto",
        subjects_dir=None,
        figure=None,
        views="axial",
        colorbar=True,
        clim="auto",
        cortex="classic",
        size=800,
        background="black",
        foreground=None,
        initial_time=None,
        time_unit="s",
        backend="auto",
        spacing="oct6",
        title=None,
        show_traces="auto",
        src=None,
        volume_options=1.0,
        view_layout="vertical",
        add_data_kwargs=None,
        brain_kwargs=None,
        verbose=None,
    ):
        return super().plot(
            subject=subject,
            surface=surface,
            hemi=hemi,
            colormap=colormap,
            time_label=time_label,
            smoothing_steps=smoothing_steps,
            transparent=transparent,
            alpha=alpha,
            time_viewer=time_viewer,
            subjects_dir=subjects_dir,
            figure=figure,
            views=views,
            colorbar=colorbar,
            clim=clim,
            cortex=cortex,
            size=size,
            background=background,
            foreground=foreground,
            initial_time=initial_time,
            time_unit=time_unit,
            backend=backend,
            spacing=spacing,
            title=title,
            show_traces=show_traces,
            src=src,
            volume_options=volume_options,
            view_layout=view_layout,
            add_data_kwargs=add_data_kwargs,
            brain_kwargs=brain_kwargs,
            verbose=verbose,
        )

    @copy_function_doc_to_method_doc(plot_volume_source_estimates)
    def plot(
        self,
        src,
        subject=None,
        subjects_dir=None,
        mode="stat_map",
        bg_img="T1.mgz",
        colorbar=True,
        colormap="auto",
        clim="auto",
        transparent="auto",
        show=True,
        initial_time=None,
        initial_pos=None,
        verbose=None,
    ):
        data = self.magnitude() if self._data_ndim == 3 else self
        return plot_volume_source_estimates(
            data,
            src=src,
            subject=subject,
            subjects_dir=subjects_dir,
            mode=mode,
            bg_img=bg_img,
            colorbar=colorbar,
            colormap=colormap,
            clim=clim,
            transparent=transparent,
            show=show,
            initial_time=initial_time,
            initial_pos=initial_pos,
            verbose=verbose,
        )

    # Override here to provide the volume-specific options
    @verbose
    def extract_label_time_course(
        self,
        labels,
        src,
        mode="auto",
        allow_empty=False,
        *,
        mri_resolution=True,
        verbose=None,
    ):
        """Extract label time courses for lists of labels.

        This function will extract one time course for each label. The way the
        time courses are extracted depends on the mode parameter.

        Parameters
        ----------
        %(labels_eltc)s
        %(src_eltc)s
        %(mode_eltc)s
        %(allow_empty_eltc)s
        %(mri_resolution_eltc)s
        %(verbose)s

        Returns
        -------
        %(label_tc_el_returns)s

        See Also
        --------
        extract_label_time_course : Extract time courses for multiple STCs.

        Notes
        -----
        %(eltc_mode_notes)s
        """
        return extract_label_time_course(
            self,
            labels,
            src,
            mode=mode,
            return_generator=False,
            allow_empty=allow_empty,
            mri_resolution=mri_resolution,
            verbose=verbose,
        )

    @verbose
    def in_label(self, label, mri, src, *, verbose=None):
        """Get a source estimate object restricted to a label.

        SourceEstimate contains the time course of
        activation of all sources inside the label.

        Parameters
        ----------
        label : str | int
            The label to use. Can be the name of a label if using a standard
            FreeSurfer atlas, or an integer value to extract from the ``mri``.
        mri : str
            Path to the atlas to use.
        src : instance of SourceSpaces
            The volumetric source space. It must be a single, whole-brain
            volume.
        %(verbose)s

        Returns
        -------
        stc : VolSourceEstimate | VolVectorSourceEstimate
            The source estimate restricted to the given label.

        Notes
        -----
        .. versionadded:: 0.21.0
        """
        if len(self.vertices) != 1:
            raise RuntimeError(
                "This method can only be used with whole-brain volume source spaces"
            )
        _validate_type(label, (str, "int-like"), "label")
        if isinstance(label, str):
            volume_label = [label]
        else:
            volume_label = {f"Volume ID {label}": _ensure_int(label)}
        label = _volume_labels(src, (mri, volume_label), mri_resolution=False)
        assert len(label) == 1
        label = label[0]
        vertices = label.vertices
        keep = np.isin(self.vertices[0], label.vertices)
        values, vertices = self.data[keep], [self.vertices[0][keep]]
        label_stc = self.__class__(
            values,
            vertices=vertices,
            tmin=self.tmin,
            tstep=self.tstep,
            subject=self.subject,
        )
        return label_stc

    @verbose
    def save_as_volume(
        self,
        fname,
        src,
        dest="mri",
        mri_resolution=False,
        format="nifti1",  # noqa: A002
        *,
        overwrite=False,
        verbose=None,
    ):
        """Save a volume source estimate in a NIfTI file.

        Parameters
        ----------
        fname : path-like
            The name of the generated nifti file.
        src : list
            The list of source spaces (should all be of type volume).
        dest : ``'mri'`` | ``'surf'``
            If ``'mri'`` the volume is defined in the coordinate system of
            the original T1 image. If ``'surf'`` the coordinate system
            of the FreeSurfer surface is used (Surface RAS).
        mri_resolution : bool
            It True the image is saved in MRI resolution.

            .. warning: If you have many time points the file produced can be
                        huge. The default is ``mri_resolution=False``.
        format : str
            Either ``'nifti1'`` (default) or ``'nifti2'``.

            .. versionadded:: 0.17
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose)s

            .. versionadded:: 1.0

        Returns
        -------
        img : instance Nifti1Image
            The image object.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        nib = _import_nibabel()
        fname = _check_fname(fname=fname, overwrite=overwrite)
        img = self.as_volume(
            src, dest=dest, mri_resolution=mri_resolution, format=format
        )
        nib.save(img, fname)

    def as_volume(
        self,
        src,
        dest="mri",
        mri_resolution=False,
        format="nifti1",  # noqa: A002
    ):
        """Export volume source estimate as a nifti object.

        Parameters
        ----------
        src : instance of SourceSpaces
            The source spaces (should all be of type volume, or part of a
            mixed source space).
        dest : ``'mri'`` | ``'surf'``
            If ``'mri'`` the volume is defined in the coordinate system of
            the original T1 image. If 'surf' the coordinate system
            of the FreeSurfer surface is used (Surface RAS).
        mri_resolution : bool
            It True the image is saved in MRI resolution.

            .. warning: If you have many time points the file produced can be
                        huge. The default is ``mri_resolution=False``.
        format : str
            Either 'nifti1' (default) or 'nifti2'.

        Returns
        -------
        img : instance of Nifti1Image
            The image object.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from .morph import _interpolate_data

        data = self.magnitude() if self._data_ndim == 3 else self
        return _interpolate_data(
            data, src, mri_resolution=mri_resolution, mri_space=True, output=format
        )


@fill_doc
class VolSourceEstimate(_BaseVolSourceEstimate):
    """Container for volume source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to ``np.dot(kernel, sens_data)``.
    %(vertices_volume)s
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    %(vertices_volume)s
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VectorSourceEstimate : A container for vector surface source estimates.
    VolVectorSourceEstimate : A container for volume vector source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    @verbose
    def save(self, fname, ftype="stc", *, overwrite=False, verbose=None):
        """Save the source estimates to a file.

        Parameters
        ----------
        fname : path-like
            The stem of the file name. The stem is extended with ``"-vl.stc"``
            or ``"-vl.w"``.
        ftype : str
            File format to use. Allowed values are ``"stc"`` (default),
            ``"w"``, and ``"h5"``. The ``"w"`` format only supports a single
            time point.
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose)s
        """
        # check overwrite individually below
        fname = str(_check_fname(fname=fname, overwrite=True))  # checked below
        _check_option("ftype", ftype, ["stc", "w", "h5"])
        if ftype != "h5" and len(self.vertices) != 1:
            raise ValueError(
                "Can only write to .stc or .w if a single volume "
                "source space was used, use .h5 instead"
            )
        if ftype != "h5" and self.data.dtype == "complex":
            raise ValueError(
                "Can only write non-complex data to .stc or .w, use .h5 instead"
            )
        if ftype == "stc":
            logger.info("Writing STC to disk...")
            if not fname.endswith(("-vl.stc", "-vol.stc")):
                fname += "-vl.stc"
            fname = str(_check_fname(fname, overwrite=overwrite))
            _write_stc(
                fname,
                tmin=self.tmin,
                tstep=self.tstep,
                vertices=self.vertices[0],
                data=self.data,
            )
        elif ftype == "w":
            logger.info("Writing STC to disk (w format)...")
            if not fname.endswith(("-vl.w", "-vol.w")):
                fname += "-vl.w"
            fname = str(_check_fname(fname, overwrite=overwrite))
            _write_w(fname, vertices=self.vertices[0], data=self.data[:, 0])
        elif ftype == "h5":
            super().save(fname, "h5", overwrite=overwrite)
        logger.info("[done]")


@fill_doc
class VolVectorSourceEstimate(_BaseVolSourceEstimate, _BaseVectorSourceEstimate):
    """Container for volume source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, 3, n_times)
        The data in source space. Each dipole contains three vectors that
        denote the dipole strength in X, Y and Z directions over time.
    %(vertices_volume)s
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    %(vertices_volume)s
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VectorSourceEstimate : A container for vector surface source estimates.
    VolSourceEstimate : A container for volume source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    _scalar_class = VolSourceEstimate

    # defaults differ: hemi='both', views='axial'
    @copy_function_doc_to_method_doc(plot_vector_source_estimates)
    def plot_3d(
        self,
        subject=None,
        hemi="both",
        colormap="hot",
        time_label="auto",
        smoothing_steps=10,
        transparent=True,
        brain_alpha=0.4,
        overlay_alpha=None,
        vector_alpha=1.0,
        scale_factor=None,
        time_viewer="auto",
        *,
        subjects_dir=None,
        figure=None,
        views="axial",
        colorbar=True,
        clim="auto",
        cortex="classic",
        size=800,
        background="black",
        foreground=None,
        initial_time=None,
        time_unit="s",
        title=None,
        show_traces="auto",
        src=None,
        volume_options=1.0,
        view_layout="vertical",
        add_data_kwargs=None,
        brain_kwargs=None,
        verbose=None,
    ):
        return _BaseVectorSourceEstimate.plot(
            self,
            subject=subject,
            hemi=hemi,
            colormap=colormap,
            time_label=time_label,
            smoothing_steps=smoothing_steps,
            transparent=transparent,
            brain_alpha=brain_alpha,
            overlay_alpha=overlay_alpha,
            vector_alpha=vector_alpha,
            scale_factor=scale_factor,
            time_viewer=time_viewer,
            subjects_dir=subjects_dir,
            figure=figure,
            views=views,
            colorbar=colorbar,
            clim=clim,
            cortex=cortex,
            size=size,
            background=background,
            foreground=foreground,
            initial_time=initial_time,
            time_unit=time_unit,
            title=title,
            show_traces=show_traces,
            src=src,
            volume_options=volume_options,
            view_layout=view_layout,
            add_data_kwargs=add_data_kwargs,
            brain_kwargs=brain_kwargs,
            verbose=verbose,
        )


@fill_doc
class VectorSourceEstimate(_BaseVectorSourceEstimate, _BaseSurfaceSourceEstimate):
    """Container for vector surface source estimates.

    For each vertex, the magnitude of the current is defined in the X, Y and Z
    directions.

    Parameters
    ----------
    data : array of shape (n_dipoles, 3, n_times)
        The data in source space. Each dipole contains three vectors that
        denote the dipole strength in X, Y and Z directions over time.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VolSourceEstimate : A container for volume source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.15
    """

    _scalar_class = SourceEstimate


###############################################################################
# Mixed source estimate (two cortical surfs plus other stuff)


class _BaseMixedSourceEstimate(_BaseSourceEstimate):
    _src_type = "mixed"
    _src_count = None

    @verbose
    def __init__(
        self, data, vertices=None, tmin=None, tstep=None, subject=None, verbose=None
    ):
        if not isinstance(vertices, list) or len(vertices) < 2:
            raise ValueError(
                "Vertices must be a list of numpy arrays with "
                "one array per source space."
            )
        super().__init__(
            data,
            vertices=vertices,
            tmin=tmin,
            tstep=tstep,
            subject=subject,
            verbose=verbose,
        )

    @property
    def _n_surf_vert(self):
        return sum(len(v) for v in self.vertices[:2])

    def surface(self):
        """Return the cortical surface source estimate.

        Returns
        -------
        stc : instance of SourceEstimate or VectorSourceEstimate
            The surface source estimate.
        """
        if self._data_ndim == 3:
            klass = VectorSourceEstimate
        else:
            klass = SourceEstimate
        return klass(
            self.data[: self._n_surf_vert],
            self.vertices[:2],
            self.tmin,
            self.tstep,
            self.subject,
        )

    def volume(self):
        """Return the volume surface source estimate.

        Returns
        -------
        stc : instance of VolSourceEstimate or VolVectorSourceEstimate
            The volume source estimate.
        """
        if self._data_ndim == 3:
            klass = VolVectorSourceEstimate
        else:
            klass = VolSourceEstimate
        return klass(
            self.data[self._n_surf_vert :],
            self.vertices[2:],
            self.tmin,
            self.tstep,
            self.subject,
        )


@fill_doc
class MixedSourceEstimate(_BaseMixedSourceEstimate):
    """Container for mixed surface and volume source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to ``np.dot(kernel, sens_data)``.
    vertices : list of array
        Vertex numbers corresponding to the data. The list contains arrays
        with one array per source space.
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : list of array
        Vertex numbers corresponding to the data. The list contains arrays
        with one array per source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VectorSourceEstimate : A container for vector surface source estimates.
    VolSourceEstimate : A container for volume source estimates.
    VolVectorSourceEstimate : A container for Volume vector source estimates.

    Notes
    -----
    .. versionadded:: 0.9.0
    """


@fill_doc
class MixedVectorSourceEstimate(_BaseVectorSourceEstimate, _BaseMixedSourceEstimate):
    """Container for volume source estimates.

    Parameters
    ----------
    data : array, shape (n_dipoles, 3, n_times)
        The data in source space. Each dipole contains three vectors that
        denote the dipole strength in X, Y and Z directions over time.
    vertices : list of array, shape (n_src,)
        Vertex numbers corresponding to the data.
    %(tmin)s
    %(tstep)s
    %(subject_optional)s
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array, shape (n_times,)
        The time vector.
    vertices : array of shape (n_dipoles,)
        The indices of the dipoles in the source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.21.0
    """

    _scalar_class = MixedSourceEstimate


###############################################################################
# Morphing


def _get_vol_mask(src):
    """Get the volume source space mask."""
    assert len(src) == 1  # not a mixed source space
    shape = src[0]["shape"][::-1]
    mask = np.zeros(shape, bool)
    mask.flat[src[0]["vertno"]] = True
    return mask


def _spatio_temporal_src_adjacency_vol(src, n_times):
    from sklearn.feature_extraction import grid_to_graph

    mask = _get_vol_mask(src)
    edges = grid_to_graph(*mask.shape, mask=mask)
    adjacency = _get_adjacency_from_edges(edges, n_times)
    return adjacency


def _spatio_temporal_src_adjacency_surf(src, n_times):
    if src[0]["use_tris"] is None:
        # XXX It would be nice to support non oct source spaces too...
        raise RuntimeError(
            "The source space does not appear to be an ico "
            "surface. adjacency cannot be extracted from"
            " non-ico source spaces."
        )
    used_verts = [np.unique(s["use_tris"]) for s in src]
    offs = np.cumsum([0] + [len(u_v) for u_v in used_verts])[:-1]
    tris = np.concatenate(
        [
            np.searchsorted(u_v, s["use_tris"]) + off
            for u_v, s, off in zip(used_verts, src, offs)
        ]
    )
    adjacency = spatio_temporal_tris_adjacency(tris, n_times)

    # deal with source space only using a subset of vertices
    masks = [np.isin(u, s["vertno"]) for s, u in zip(src, used_verts)]
    if sum(u.size for u in used_verts) != adjacency.shape[0] / n_times:
        raise ValueError("Used vertices do not match adjacency shape")
    if [np.sum(m) for m in masks] != [len(s["vertno"]) for s in src]:
        raise ValueError("Vertex mask does not match number of vertices")
    masks = np.concatenate(masks)
    missing = 100 * float(len(masks) - np.sum(masks)) / len(masks)
    if missing:
        warn(
            f"{missing:0.1f}% of original source space vertices have been"
            " omitted, tri-based adjacency will have holes.\n"
            "Consider using distance-based adjacency or "
            "morphing data to all source space vertices."
        )
        masks = np.tile(masks, n_times)
        masks = np.where(masks)[0]
        adjacency = adjacency.tocsr()
        adjacency = adjacency[masks]
        adjacency = adjacency[:, masks]
        # return to original format
        adjacency = adjacency.tocoo()
    return adjacency


@verbose
def spatio_temporal_src_adjacency(src, n_times, dist=None, verbose=None):
    """Compute adjacency for a source space activation over time.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space. It can be a surface source space or a
        volume source space.
    n_times : int
        Number of time instants.
    dist : float, or None
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors. If None, immediate neighbors
        are extracted from an ico surface.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    # XXX we should compute adjacency for each source space and then
    # use scipy.sparse.block_diag to concatenate them
    if src[0]["type"] == "vol":
        if dist is not None:
            raise ValueError(
                f"dist must be None for a volume source space. Got {dist}."
            )

        adjacency = _spatio_temporal_src_adjacency_vol(src, n_times)
    elif dist is not None:
        # use distances computed and saved in the source space file
        adjacency = spatio_temporal_dist_adjacency(src, n_times, dist)
    else:
        adjacency = _spatio_temporal_src_adjacency_surf(src, n_times)
    return adjacency


@verbose
def grade_to_tris(grade, verbose=None):
    """Get tris defined for a certain grade.

    Parameters
    ----------
    grade : int
        Grade of an icosahedral mesh.
    %(verbose)s

    Returns
    -------
    tris : list
        2-element list containing Nx3 arrays of tris, suitable for use in
        spatio_temporal_tris_adjacency.
    """
    a = _get_ico_tris(grade, None, False)
    tris = np.concatenate((a, a + (np.max(a) + 1)))
    return tris


@verbose
def spatio_temporal_tris_adjacency(tris, n_times, remap_vertices=False, verbose=None):
    """Compute adjacency from triangles and time instants.

    Parameters
    ----------
    tris : array
        N x 3 array defining triangles.
    n_times : int
        Number of time points.
    remap_vertices : bool
        Reassign vertex indices based on unique values. Useful
        to process a subset of triangles. Defaults to False.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    if remap_vertices:
        logger.info("Reassigning vertex indices.")
        tris = np.searchsorted(np.unique(tris), tris)

    edges = mesh_edges(tris)
    edges = (edges + _eye_array(edges.shape[0])).tocoo()
    return _get_adjacency_from_edges(edges, n_times)


@verbose
def spatio_temporal_dist_adjacency(src, n_times, dist, verbose=None):
    """Compute adjacency from distances in a source space and time instants.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space must have distances between vertices computed, such
        that src['dist'] exists and is useful. This can be obtained
        with a call to :func:`mne.setup_source_space` with the
        ``add_dist=True`` option.
    n_times : int
        Number of time points.
    dist : float
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    if src[0]["dist"] is None:
        raise RuntimeError(
            "src must have distances included, consider using "
            "setup_source_space with add_dist=True"
        )
    blocks = [s["dist"][s["vertno"], :][:, s["vertno"]] for s in src]
    # Ensure we keep explicit zeros; deal with changes in SciPy
    for block in blocks:
        if isinstance(block, np.ndarray):
            block[block == 0] = -np.inf
        else:
            block.data[block.data == 0] == -1
    edges = sparse.block_diag(blocks)
    edges.data[:] = np.less_equal(edges.data, dist)
    # clean it up and put it in coo format
    edges = edges.tocsr()
    edges.eliminate_zeros()
    edges = edges.tocoo()
    return _get_adjacency_from_edges(edges, n_times)


@verbose
def spatial_src_adjacency(src, dist=None, verbose=None):
    """Compute adjacency for a source space activation.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space. It can be a surface source space or a
        volume source space.
    dist : float, or None
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors. If None, immediate neighbors
        are extracted from an ico surface.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatial graph structure.
    """
    return spatio_temporal_src_adjacency(src, 1, dist)


@verbose
def spatial_tris_adjacency(tris, remap_vertices=False, verbose=None):
    """Compute adjacency from triangles.

    Parameters
    ----------
    tris : array
        N x 3 array defining triangles.
    remap_vertices : bool
        Reassign vertex indices based on unique values. Useful
        to process a subset of triangles. Defaults to False.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatial graph structure.
    """
    return spatio_temporal_tris_adjacency(tris, 1, remap_vertices)


@verbose
def spatial_dist_adjacency(src, dist, verbose=None):
    """Compute adjacency from distances in a source space.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space must have distances between vertices computed, such
        that src['dist'] exists and is useful. This can be obtained
        with a call to :func:`mne.setup_source_space` with the
        ``add_dist=True`` option.
    dist : float
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatial graph structure.
    """
    return spatio_temporal_dist_adjacency(src, 1, dist)


@verbose
def spatial_inter_hemi_adjacency(src, dist, verbose=None):
    """Get vertices on each hemisphere that are close to the other hemisphere.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space. Must be surface type.
    dist : float
        Maximal Euclidean distance (in m) between vertices in one hemisphere
        compared to the other to consider neighbors.
    %(verbose)s

    Returns
    -------
    adjacency : ~scipy.sparse.coo_array
        The adjacency matrix describing the spatial graph structure.
        Typically this should be combined (addititively) with another
        existing intra-hemispheric adjacency matrix, e.g. computed
        using geodesic distances.
    """
    src = _ensure_src(src, kind="surface")
    adj = cdist(src[0]["rr"][src[0]["vertno"]], src[1]["rr"][src[1]["vertno"]])
    adj = sparse.csr_array(adj <= dist, dtype=int)
    empties = [sparse.csr_array((nv, nv), dtype=int) for nv in adj.shape]
    adj = sparse.vstack(
        [sparse.hstack([empties[0], adj]), sparse.hstack([adj.T, empties[1]])]
    )
    return adj


@verbose
def _get_adjacency_from_edges(edges, n_times, verbose=None):
    """Given edges sparse matrix, create adjacency matrix."""
    n_vertices = edges.shape[0]
    logger.info("-- number of adjacent vertices : %d", n_vertices)
    nnz = edges.col.size
    aux = n_vertices * np.tile(np.arange(n_times)[:, None], (1, nnz))
    col = (edges.col[None, :] + aux).ravel()
    row = (edges.row[None, :] + aux).ravel()
    if n_times > 1:  # add temporal edges
        o = (
            n_vertices * np.arange(n_times - 1)[:, None]
            + np.arange(n_vertices)[None, :]
        ).ravel()
        d = (
            n_vertices * np.arange(1, n_times)[:, None] + np.arange(n_vertices)[None, :]
        ).ravel()
        row = np.concatenate((row, o, d))
        col = np.concatenate((col, d, o))
    data = np.ones(
        edges.data.size * n_times + 2 * n_vertices * (n_times - 1), dtype=np.int64
    )
    adjacency = sparse.coo_array((data, (row, col)), shape=(n_times * n_vertices,) * 2)
    return adjacency


@verbose
def _get_ico_tris(grade, verbose=None, return_surf=False):
    """Get triangles for ico surface."""
    ico = _get_ico_surface(grade)
    if not return_surf:
        return ico["tris"]
    else:
        return ico


def _pca_flip(flip, data):
    U, s, V = _safe_svd(data, full_matrices=False)
    # determine sign-flip
    sign = np.sign(np.dot(U[:, 0], flip))
    # use average power in label for scaling
    scale = np.linalg.norm(s) / np.sqrt(len(data))
    return sign * scale * V[0]


_label_funcs = {
    "mean": lambda flip, data: np.mean(data, axis=0),
    "mean_flip": lambda flip, data: np.mean(flip * data, axis=0),
    "max": lambda flip, data: np.max(np.abs(data), axis=0),
    "pca_flip": _pca_flip,
    None: lambda flip, data: data,  # Return Identity: Preserves all vertices.
}


@contextlib.contextmanager
def _temporary_vertices(src, vertices):
    orig_vertices = [s["vertno"] for s in src]
    for s, v in zip(src, vertices):
        s["vertno"] = v
    try:
        yield
    finally:
        for s, v in zip(src, orig_vertices):
            s["vertno"] = v


def _check_stc_src(stc, src):
    if stc is not None and src is not None:
        _check_subject(
            src._subject,
            stc.subject,
            raise_error=False,
            first_kind="source space subject",
            second_kind="stc.subject",
        )
        for s, v, hemi in zip(src, stc.vertices, ("left", "right")):
            n_missing = (~np.isin(v, s["vertno"])).sum()
            if n_missing:
                raise ValueError(
                    f"{n_missing}/{len(v)} {hemi} hemisphere stc vertices "
                    "missing from the source space, likely mismatch"
                )


def _prepare_label_extraction(stc, labels, src, mode, allow_empty, use_sparse):
    """Prepare indices and flips for extract_label_time_course."""
    # If src is a mixed src space, the first 2 src spaces are surf type and
    # the other ones are vol type. For mixed source space n_labels will be
    # given by the number of ROIs of the cortical parcellation plus the number
    # of vol src space.
    # If stc=None (i.e. no activation time courses provided) and mode='mean',
    # only computes vertex indices and label_flip will be list of None.
    from .label import BiHemiLabel, Label, label_sign_flip

    # if source estimate provided in stc, get vertices from source space and
    # check that they are the same as in the stcs
    _check_stc_src(stc, src)
    vertno = [s["vertno"] for s in src] if stc is None else stc.vertices
    nvert = [len(vn) for vn in vertno]

    # initialization
    label_flip = list()
    label_vertidx = list()

    bad_labels = list()
    for li, label in enumerate(labels):
        subject = label["subject"] if use_sparse else label.subject
        # stc and src can each be None
        _check_subject(
            subject,
            getattr(stc, "subject", None),
            raise_error=False,
            first_kind="label.subject",
            second_kind="stc.subject",
        )
        _check_subject(
            subject,
            getattr(src, "_subject", None),
            raise_error=False,
            first_kind="label.subject",
            second_kind="source space subject",
        )
        if use_sparse:
            assert isinstance(label, dict)
            vertidx = label["csr"]
            # This can happen if some labels aren't present in the space
            if vertidx.shape[0] == 0:
                bad_labels.append(label["name"])
                vertidx = None
            # Efficiency shortcut: use linearity early to avoid redundant
            # calculations
            elif mode == "mean":
                vertidx = sparse.csr_array(vertidx.mean(axis=0)[np.newaxis])
            label_vertidx.append(vertidx)
            label_flip.append(None)
            continue
        # standard case
        _validate_type(label, (Label, BiHemiLabel), f"labels[{li}]")

        if label.hemi == "both":
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if slabel.hemi == "lh":
                this_vertices = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertices)
            elif slabel.hemi == "rh":
                this_vertices = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertices)
            else:
                raise ValueError(f"label {label.name} has invalid hemi")
            this_vertidx.append(vertidx)

        # convert it to an array
        this_vertidx = np.concatenate(this_vertidx)
        this_flip = None
        if len(this_vertidx) == 0:
            bad_labels.append(label.name)
            this_vertidx = None  # to later check if label is empty
        elif mode not in ("mean", "max"):  # mode-dependent initialization
            # label_sign_flip uses two properties:
            #
            # - src[ii]['nn']
            # - src[ii]['vertno']
            #
            # So if we override vertno with the stc vertices, it will pick
            # the correct normals.
            with _temporary_vertices(src, stc.vertices):
                this_flip = label_sign_flip(label, src[:2])[:, None]

        label_vertidx.append(this_vertidx)
        label_flip.append(this_flip)

    if len(bad_labels):
        msg = (
            f"source space does not contain any vertices for {len(bad_labels)} "
            f"label{_pl(bad_labels)}:\n{bad_labels}"
        )
        if not allow_empty:
            raise ValueError(msg)
        else:
            msg += "\nAssigning all-zero time series."
            if allow_empty == "ignore":
                logger.info(msg)
            else:
                warn(msg)

    return label_vertidx, label_flip


def _vol_src_rr(src):
    return apply_trans(
        src[0]["src_mri_t"],
        np.array(
            [
                d.ravel(order="F")
                for d in np.meshgrid(
                    *(np.arange(s) for s in src[0]["shape"]), indexing="ij"
                )
            ],
            float,
        ).T,
    )


def _volume_labels(src, labels, mri_resolution):
    # This will create Label objects that should do the right thing for our
    # given volumetric source space when used with extract_label_time_course
    from .label import Label

    assert src.kind == "volume"
    subject = src._subject
    extra = " when using a volume source space"
    _import_nibabel("use volume atlas labels")
    _validate_type(labels, ("path-like", list, tuple), "labels" + extra)
    if _path_like(labels):
        mri = labels
        infer_labels = True
    else:
        if len(labels) != 2:
            raise ValueError(
                "labels, if list or tuple, must have length 2, got {len(labels)}"
            )
        mri, labels = labels
        infer_labels = False
        _validate_type(mri, "path-like", "labels[0]" + extra)
    logger.info(f"Reading atlas {mri}")
    vol_info = _get_mri_info_data(str(mri), data=True)
    atlas_data = vol_info["data"]
    atlas_values = np.unique(atlas_data)
    if atlas_values.dtype.kind == "f":  # MGZ will be 'i'
        atlas_values = atlas_values[np.isfinite(atlas_values)]
        if not (atlas_values == np.round(atlas_values)).all():
            raise RuntimeError("Non-integer values present in atlas, cannot labelize")
        atlas_values = np.round(atlas_values).astype(np.int64)
    if infer_labels:
        labels = {
            k: v for k, v in read_freesurfer_lut()[0].items() if v in atlas_values
        }
    labels = _check_volume_labels(labels, mri, name="labels[1]")
    assert isinstance(labels, dict)
    del atlas_values

    vox_mri_t = vol_info["vox_mri_t"]
    want = src[0].get("vox_mri_t", None)
    if want is None:
        raise RuntimeError(
            "Cannot use volumetric atlas if no mri was supplied during "
            "source space creation"
        )
    vox_mri_t, want = vox_mri_t["trans"], want["trans"]
    if not np.allclose(vox_mri_t, want, atol=1e-6):
        raise RuntimeError(
            "atlas vox_mri_t does not match that used to create the source space"
        )
    src_shape = tuple(src[0]["mri_" + k] for k in ("width", "height", "depth"))
    atlas_shape = atlas_data.shape
    if atlas_shape != src_shape:
        raise RuntimeError(
            f"atlas shape {atlas_shape} does not match source space MRI "
            f"shape {src_shape}"
        )
    atlas_data = atlas_data.ravel(order="F")
    if mri_resolution:
        # Upsample then just index
        out_labels = list()
        nnz = 0
        interp = src[0]["interpolator"]
        # should be guaranteed by size checks above and our src interp code
        assert interp.shape[0] == np.prod(src_shape)
        assert interp.shape == (atlas_data.size, len(src[0]["rr"]))
        interp = interp[:, src[0]["vertno"]]
        for k, v in labels.items():
            mask = atlas_data == v
            csr = interp[mask]
            out_labels.append(dict(csr=csr, name=k, subject=subject))
            nnz += csr.shape[0] > 0
    else:
        # Use nearest values
        vertno = src[0]["vertno"]
        rr = _vol_src_rr(src)
        del src
        src_values = _get_atlas_values(vol_info, rr[vertno])
        vertices = [vertno[src_values == val] for val in labels.values()]
        out_labels = [
            Label(v, hemi="lh", name=val, subject=subject)
            for v, val in zip(vertices, labels.keys())
        ]
        nnz = sum(len(v) != 0 for v in vertices)
    logger.info(
        "%d/%d atlas regions had at least one vertex in the source space",
        nnz,
        len(out_labels),
    )
    return out_labels


def _get_default_label_modes():
    return sorted(_label_funcs.keys(), key=lambda x: (x is None, x)) + ["auto"]


def _get_allowed_label_modes(stc):
    if isinstance(stc, _BaseVolSourceEstimate | _BaseVectorSourceEstimate):
        return ("mean", "max", "auto")
    else:
        return _get_default_label_modes()


def _gen_extract_label_time_course(
    stcs,
    labels,
    src,
    *,
    mode="mean",
    allow_empty=False,
    mri_resolution=True,
    verbose=None,
):
    # loop through source estimates and extract time series
    if src is None and mode in ["mean", "max"]:
        kind = "surface"
    else:
        _validate_type(src, SourceSpaces)
        kind = src.kind
    _check_option("mode", mode, _get_default_label_modes())

    if kind in ("surface", "mixed"):
        if not isinstance(labels, list):
            labels = [labels]
        use_sparse = False
    else:
        labels = _volume_labels(src, labels, mri_resolution)
        use_sparse = bool(mri_resolution)
    n_mode = len(labels)  # how many processed with the given mode
    n_mean = len(src[2:]) if kind == "mixed" else 0
    n_labels = n_mode + n_mean
    vertno = func = None
    for si, stc in enumerate(stcs):
        _validate_type(stc, _BaseSourceEstimate, f"stcs[{si}]", "source estimate")
        _check_option(
            "mode",
            mode,
            _get_allowed_label_modes(stc),
            "when using a vector and/or volume source estimate",
        )
        if isinstance(stc, _BaseVolSourceEstimate | _BaseVectorSourceEstimate):
            mode = "mean" if mode == "auto" else mode
        else:
            mode = "mean_flip" if mode == "auto" else mode
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)  # avoid keeping a ref
            nvert = np.array([len(v) for v in vertno])
            label_vertidx, src_flip = _prepare_label_extraction(
                stc, labels, src, mode, allow_empty, use_sparse
            )
            func = _label_funcs[mode]
        # make sure the stc is compatible with the source space
        if len(vertno) != len(stc.vertices):
            raise ValueError("stc not compatible with source space")
        for vn, svn in zip(vertno, stc.vertices):
            if len(vn) != len(svn):
                raise ValueError(
                    "stc not compatible with source space. "
                    f"stc has {len(svn)} time series but there are {len(vn)} "
                    "vertices in source space. Ensure you used "
                    "src from the forward or inverse operator, "
                    "as forward computation can exclude vertices."
                )
            if not np.array_equal(svn, vn):
                raise ValueError("stc not compatible with source space")

        logger.info("Extracting time courses for %d labels (mode: %s)", n_labels, mode)

        # do the extraction
        if mode is None:
            # prepopulate an empty list for easy array-like index-based assignment
            label_tc = [None] * max(len(label_vertidx), len(src_flip))
        else:
            # For other modes, initialize the label_tc array
            label_tc = np.zeros((n_labels,) + stc.data.shape[1:], dtype=stc.data.dtype)
        for i, (vertidx, flip) in enumerate(zip(label_vertidx, src_flip)):
            if vertidx is not None:
                if isinstance(vertidx, sparse.csr_array):
                    assert mri_resolution
                    assert vertidx.shape[1] == stc.data.shape[0]
                    this_data = np.reshape(stc.data, (stc.data.shape[0], -1))
                    this_data = vertidx @ this_data
                    this_data.shape = (this_data.shape[0],) + stc.data.shape[1:]
                else:
                    this_data = stc.data[vertidx]
                label_tc[i] = func(flip, this_data)

        if mode is not None:
            offset = nvert[:-n_mean].sum()  # effectively :2 or :0
            for i, nv in enumerate(nvert[2:]):
                if nv != 0:
                    v2 = offset + nv
                    label_tc[n_mode + i] = np.mean(stc.data[offset:v2], axis=0)
                    offset = v2
        yield label_tc


@verbose
def extract_label_time_course(
    stcs,
    labels,
    src,
    mode="auto",
    allow_empty=False,
    return_generator=False,
    *,
    mri_resolution=True,
    verbose=None,
):
    """Extract label time course for lists of labels and source estimates.

    This function will extract one time course for each label and source
    estimate. The way the time courses are extracted depends on the mode
    parameter (see Notes).

    Parameters
    ----------
    stcs : SourceEstimate | list (or generator) of SourceEstimate
        The source estimates from which to extract the time course.
    %(labels_eltc)s
    %(src_eltc)s
    %(mode_eltc)s
    %(allow_empty_eltc)s
    return_generator : bool
        If True, a generator instead of a list is returned.
    %(mri_resolution_eltc)s
    %(verbose)s

    Returns
    -------
    %(label_tc_el_returns)s

    Notes
    -----
    %(eltc_mode_notes)s

    If encountering a ``ValueError`` due to mismatch between number of
    source points in the subject source space and computed ``stc`` object set
    ``src`` argument to ``fwd['src']`` or ``inv['src']`` to ensure the source
    space is the one actually used by the inverse to compute the source
    time courses.
    """
    # convert inputs to lists
    if not isinstance(stcs, list | tuple | GeneratorType):
        stcs = [stcs]
        return_several = False
        return_generator = False
    else:
        return_several = True

    label_tc = _gen_extract_label_time_course(
        stcs,
        labels,
        src,
        mode=mode,
        allow_empty=allow_empty,
        mri_resolution=mri_resolution,
    )

    if not return_generator:
        # do the extraction and return a list
        label_tc = list(label_tc)

    if not return_several:
        # input was a single SoureEstimate, return single array
        label_tc = label_tc[0]

    return label_tc


@verbose
def stc_near_sensors(
    evoked,
    trans,
    subject,
    distance=0.01,
    mode="sum",
    project=True,
    subjects_dir=None,
    src=None,
    picks=None,
    surface="auto",
    verbose=None,
):
    """Create a STC from ECoG, sEEG and DBS sensor data.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data. Must contain ECoG, sEEG or DBS channels.
    %(trans)s

        .. versionchanged:: 0.19
            Support for 'fsaverage' argument.
    subject : str
        The subject name.
    distance : float
        Distance (m) defining the activation "ball" of the sensor.
    mode : str
        Can be ``"sum"`` to do a linear sum of weights, ``"weighted"`` to make
        this a weighted sum, ``"nearest"`` to use only the weight of the
        nearest sensor, or ``"single"`` to do a distance-weight of the nearest
        sensor. Default is ``"sum"``. See Notes.

        .. versionchanged:: 0.24
           Added "weighted" option.
    project : bool
        If True, project the sensors to the nearest ``'pial`` surface
        vertex before computing distances. Only used when doing a
        surface projection.
    %(subjects_dir)s
    src : instance of SourceSpaces
        The source space.

        .. warning:: If a surface source space is used, make sure that
                     ``surface='pial'`` was used during construction,
                     or that you set ``surface='pial'`` here.
    %(picks_base)s good sEEG, ECoG, and DBS channels.

        .. versionadded:: 0.24
    surface : str | None
        The surface to use. If ``src=None``, defaults to the pial surface.
        Otherwise, the source space surface will be used.

        .. versionadded:: 0.24.1
    %(verbose)s

    Returns
    -------
    stc : instance of SourceEstimate
        The surface source estimate. If src is None, a surface source
        estimate will be produced, and the number of vertices will equal
        the number of pial-surface vertices that were close enough to
        the sensors to take on a non-zero volue. If src is not None,
        a surface, volume, or mixed source estimate will be produced
        (depending on the kind of source space passed) and the
        vertices will match those of src (i.e., there may be me
        many all-zero values in stc.data).

    Notes
    -----
    For surface projections, this function projects the ECoG sensors to
    the pial surface (if ``project``), then the activation at each pial
    surface vertex is given by the mode:

    - ``'sum'``
        Activation is the sum across each sensor weighted by the fractional
        ``distance`` from each sensor. A sensor with zero distance gets weight
        1 and a sensor at ``distance`` meters away (or larger) gets weight 0.
        If ``distance`` is less than half the distance between any two
        sensors, this will be the same as ``'single'``.
    - ``'single'``
        Same as ``'sum'`` except that only the nearest sensor is used,
        rather than summing across sensors within the ``distance`` radius.
        As ``'nearest'`` for vertices with distance zero to the projected
        sensor.
    - ``'nearest'``
        The value is given by the value of the nearest sensor, up to a
        ``distance`` (beyond which it is zero).
    - ``'weighted'``
        The value is given by the same as ``sum`` but the total weight for
        each vertex is 1. (i.e., it's a weighted sum based on proximity).

    If creating a Volume STC, ``src`` must be passed in, and this
    function will project sEEG and DBS sensors to nearby surrounding vertices.
    Then the activation at each volume vertex is given by the mode
    in the same way as ECoG surface projections.

    .. versionadded:: 0.22
    """
    from .evoked import Evoked

    _validate_type(evoked, Evoked, "evoked")
    _validate_type(mode, str, "mode")
    _validate_type(src, (None, SourceSpaces), "src")
    _check_option("mode", mode, ("sum", "single", "nearest", "weighted"))
    if surface == "auto":
        surface = "pial" if src is None or src.kind == "surface" else None

    # create a copy of Evoked using ecog, seeg and dbs
    if picks is None:
        picks = pick_types(evoked.info, ecog=True, seeg=True, dbs=True)
    evoked = evoked.copy().pick(picks)
    frames = set(ch["coord_frame"] for ch in evoked.info["chs"])
    if not frames == {FIFF.FIFFV_COORD_HEAD}:
        raise RuntimeError(
            f"Channels must be in the head coordinate frame, got {sorted(frames)}"
        )

    # get channel positions that will be used to pinpoint where
    # in the Source space we will use the evoked data
    pos = evoked._get_channel_positions()

    # remove nan channels
    nan_inds = np.where(np.isnan(pos).any(axis=1))[0]
    nan_chs = [evoked.ch_names[idx] for idx in nan_inds]
    if len(nan_chs):
        evoked.drop_channels(nan_chs)
    pos = [pos[idx] for idx in range(len(pos)) if idx not in nan_inds]

    # coord_frame transformation from native mne "head" to MRI coord_frame
    trans, _ = _get_trans(trans, "head", "mri", allow_none=True)

    # convert head positions -> coord_frame MRI
    pos = apply_trans(trans, pos)

    subject = _check_subject(None, subject, raise_error=False)
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if surface is not None:
        surf_rr = [
            read_surface(subjects_dir / subject / "surf" / f"{hemi}.{surface}")[0]
            / 1000.0
            for hemi in ("lh", "rh")
        ]
    if src is None:  # fake a full surface one
        _validate_type(surface, str, "surface", "when src is None")
        src = SourceSpaces(
            [
                dict(
                    rr=rr,
                    vertno=np.arange(len(rr)),
                    type="surf",
                    coord_frame=FIFF.FIFFV_COORD_MRI,
                )
                for rr in surf_rr
            ]
        )
        rrs = np.concatenate([s_rr[s["vertno"]] for s_rr, s in zip(surf_rr, src)])
        keep_all = False
    else:
        if surface is None:
            rrs = np.concatenate([s["rr"][s["vertno"]] for s in src])
            if src[0]["coord_frame"] == FIFF.FIFFV_COORD_HEAD:
                rrs = apply_trans(trans, rrs)
        else:
            rrs = np.concatenate([s_rr[s["vertno"]] for s_rr, s in zip(surf_rr, src)])
        keep_all = True
    # ensure it's a usable one
    klass = dict(
        surface=SourceEstimate,
        volume=VolSourceEstimate,
        mixed=MixedSourceEstimate,
    )
    _check_option("src.kind", src.kind, sorted(klass.keys()))
    klass = klass[src.kind]
    # projection will only occur with surfaces
    logger.info(
        f"Projecting data from {len(pos)} sensor{_pl(pos)} onto {len(rrs)} "
        f"{src.kind} vertices: {mode} mode"
    )
    if project and src.kind == "surface":
        logger.info("    Projecting sensors onto surface")
        pos = _project_onto_surface(
            pos, dict(rr=rrs), project_rrs=True, method="nearest"
        )[2]

    min_dist = pdist(pos).min() * 1000
    logger.info(
        f"    Minimum {'projected ' if project else ''}intra-sensor distance: "
        f"{min_dist:0.1f} mm"
    )

    # compute pairwise distance between source space points and sensors
    dists = cdist(rrs, pos)
    assert dists.shape == (len(rrs), len(pos))

    # only consider vertices within our "epsilon-ball"
    # characterized by distance kwarg
    vertices = np.where((dists <= distance).any(-1))[0]
    logger.info(f"    {len(vertices)} / {len(rrs)} non-zero vertices")
    w = np.maximum(1.0 - dists[vertices] / distance, 0)
    # now we triage based on mode
    if mode in ("single", "nearest"):
        range_ = np.arange(w.shape[0])
        idx = np.argmax(w, axis=1)
        vals = w[range_, idx] if mode == "single" else 1.0
        w.fill(0)
        w[range_, idx] = vals
    elif mode == "weighted":
        norms = w.sum(-1, keepdims=True)
        norms[norms == 0] = 1.0
        w /= norms
    missing = np.where(~np.any(w, axis=0))[0]
    if len(missing):
        warn(
            f"Channel{_pl(missing)} missing in STC: "
            f"{', '.join(evoked.ch_names[mi] for mi in missing)}"
        )

    nz_data = w @ evoked.data
    if keep_all:
        data = np.zeros(
            (sum(len(s["vertno"]) for s in src), len(evoked.times)), dtype=nz_data.dtype
        )
        data[vertices] = nz_data
        vertices = [s["vertno"].copy() for s in src]
    else:
        assert src.kind == "surface"
        data = nz_data
        offset = len(src[0]["vertno"])
        vertices = [vertices[vertices < offset], vertices[vertices >= offset] - offset]

    return klass(
        data,
        vertices,
        evoked.times[0],
        1.0 / evoked.info["sfreq"],
        subject=subject,
        verbose=verbose,
    )
