# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..._fiff.meas_info import Info
from ..._fiff.pick import pick_types
from ..._freesurfer import (
    _get_aseg,
    _get_head_surface,
    _get_skull_surface,
    read_freesurfer_lut,
)
from ...defaults import DEFAULTS
from ...surface import _marching_cubes
from ...transforms import (
    Transform,
    _frame_to_str,
    _get_trans,
    _get_transforms_to_coord_frame,
    apply_trans,
)
from ...utils import (
    _check_option,
    _to_rgb,
    _validate_type,
    fill_doc_static,
    verbose_static,
    warn,
)
from .._3d import _handle_sensor_types, _plot_forward, _plot_helmet, _plot_sensors_3d
from .surface import _Surface
from .view import views_dicts


class _SceneMixin:
    """Mixin providing the scene objects (sensors, head, foci, ...) of Brain."""

    @fill_doc_static("fwd", "trans_not_none", "alpha")
    def add_forward(self, fwd, trans, alpha=1, scale=None):
        """Add a quiver to render positions of dipoles.

        Parameters
        ----------
        fwd : instance of Forward
            The forward solution. If present, the orientations of the dipoles
            present in the forward solution are displayed.
        trans : str | dict | instance of Transform
            If str, the path to the head<->MRI transform ``*-trans.fif`` file produced
            during coregistration. Can also be ``'fsaverage'`` to use the built-in
            fsaverage transformation.
        alpha : float in [0, 1]
            Alpha level to control opacity.
            Default 1.
        scale : None | float
            The size of the arrow representing the dipoles in
            :class:`mne.viz.Brain` units. Default 1.5mm.

        Notes
        -----
        .. versionadded:: 1.0
        """
        head_mri_t = _get_trans(trans, "head", "mri", allow_none=False)[0]
        del trans
        if scale is None:
            scale = 1.5 if self._units == "mm" else 1.5e-3
        error_msg = (
            'Unexpected forward model coordinate frame {}, must be "head" or "mri"'
        )
        if fwd["coord_frame"] in _frame_to_str:
            fwd_frame = _frame_to_str[fwd["coord_frame"]]
            if fwd_frame == "mri":
                fwd_trans = Transform("mri", "mri")
            elif fwd_frame == "head":
                fwd_trans = head_mri_t
            else:
                raise RuntimeError(error_msg.format(fwd_frame))
        else:
            raise RuntimeError(error_msg.format(fwd["coord_frame"]))
        for actor in _plot_forward(
            self._renderer,
            fwd,
            fwd_trans,
            fwd_scale=1e3 if self._units == "mm" else 1,
            scale=scale,
            alpha=alpha,
        ):
            self._add_actor("forward", actor)

        self._renderer._update()

    def remove_forward(self):
        """Remove forward sources from the rendered scene."""
        self._remove("forward", render=True)

    @fill_doc_static("trans_not_none", "alpha")
    def add_dipole(
        self, dipole, trans, colors="red", alpha=1, scales=None, *, mode="arrow"
    ):
        """Add a quiver to render positions of dipoles.

        Parameters
        ----------
        dipole : instance of Dipole
            Dipole object containing position, orientation and amplitude of
            one or more dipoles or in the forward solution.
        trans : str | dict | instance of Transform
            If str, the path to the head<->MRI transform ``*-trans.fif`` file produced
            during coregistration. Can also be ``'fsaverage'`` to use the built-in
            fsaverage transformation.
        colors : list | matplotlib-style color | None
            A single color or list of anything matplotlib accepts:
            string, RGB, hex, etc. Default red.
        alpha : float in [0, 1]
            Alpha level to control opacity.
            Default 1.
        scales : list | float | None
            The size of the arrow representing the dipole in
            :class:`mne.viz.Brain` units. Default 5mm.
        mode : "2darrow" | "arrow" | "cone" | "cylinder" | "sphere" | "oct"
            The drawing mode for the dipole to render.
            Defaults to ``"arrow"``.

        Notes
        -----
        .. versionadded:: 1.0
        """
        head_mri_t = _get_trans(trans, "head", "mri", allow_none=False)[0]
        del trans
        n_dipoles = len(dipole)
        if not isinstance(colors, list | tuple):
            colors = [colors] * n_dipoles  # make into list
        if len(colors) != n_dipoles:
            raise ValueError(
                f"The number of colors ({len(colors)}) "
                f"and dipoles ({n_dipoles}) must match"
            )
        colors = [
            _to_rgb(color, name=f"colors[{ci}]") for ci, color in enumerate(colors)
        ]
        if scales is None:
            scales = 5 if self._units == "mm" else 5e-3
        if not isinstance(scales, list | tuple):
            scales = [scales] * n_dipoles  # make into list
        if len(scales) != n_dipoles:
            raise ValueError(
                f"The number of scales ({len(scales)}) "
                f"and dipoles ({n_dipoles}) must match"
            )
        pos = apply_trans(head_mri_t, dipole.pos)
        pos *= 1e3 if self._units == "mm" else 1
        for _ in self._iter_views("vol"):
            for this_pos, this_ori, color, scale in zip(
                pos, dipole.ori, colors, scales
            ):
                actor, _ = self._renderer.quiver3d(
                    *this_pos,
                    *this_ori,
                    color=color,
                    opacity=alpha,
                    mode=mode,
                    scale=scale,
                )
                self._add_actor("dipole", actor)

        self._renderer._update()

    def remove_dipole(self):
        """Remove dipole objects from the rendered scene."""
        self._remove("dipole", render=True)

    @fill_doc_static("color_matplotlib", "alpha")
    def add_head(self, dense=True, color="gray", alpha=0.5):
        """Add a mesh to render the outer head surface.

        Parameters
        ----------
        dense : bool
            Whether to plot the dense head (``seghead``) or the less dense head
            (``head``).
        color : color
            A list of anything matplotlib accepts: string, RGB, hex, etc.
        alpha : float in [0, 1]
            Alpha level to control opacity.

        Notes
        -----
        .. versionadded:: 0.24
        """
        # load head
        surf = _get_head_surface(
            "seghead" if dense else "head", self._subject, self._subjects_dir
        )
        verts, triangles = surf["rr"], surf["tris"]
        verts *= 1e3 if self._units == "mm" else 1
        color = _to_rgb(color)

        for _ in self._iter_views("vol"):
            actor, _ = self._renderer.mesh(
                *verts.T,
                triangles=triangles,
                color=color,
                opacity=alpha,
                render=False,
            )
            self._add_actor("head", actor)

        self._renderer._update()

    def remove_head(self):
        """Remove head objects from the rendered scene."""
        self._remove("head", render=True)

    @fill_doc_static("color_matplotlib", "alpha")
    def add_skull(self, outer=True, color="gray", alpha=0.5):
        """Add a mesh to render the skull surface.

        Parameters
        ----------
        outer : bool
            Adds the outer skull if ``True``, otherwise adds the inner skull.
        color : color
            A list of anything matplotlib accepts: string, RGB, hex, etc.
        alpha : float in [0, 1]
            Alpha level to control opacity.

        Notes
        -----
        .. versionadded:: 0.24
        """
        surf = _get_skull_surface(
            "outer" if outer else "inner", self._subject, self._subjects_dir
        )
        verts, triangles = surf["rr"], surf["tris"]
        verts *= 1e3 if self._units == "mm" else 1
        color = _to_rgb(color)

        for _ in self._iter_views("vol"):
            actor, _ = self._renderer.mesh(
                *verts.T,
                triangles=triangles,
                color=color,
                opacity=alpha,
                reset_camera=False,
                render=False,
            )
            self._add_actor("skull", actor)

        self._renderer._update()

    def remove_skull(self):
        """Remove skull objects from the rendered scene."""
        self._remove("skull", render=True)

    @fill_doc_static("aseg", "labels_aseg", "alpha", "smooth")
    def add_volume_labels(
        self,
        aseg="auto",
        labels=None,
        colors=None,
        alpha=0.5,
        smooth=0.9,
        fill_hole_size=None,
        legend=None,
    ):
        """Add labels to the rendering from an anatomical segmentation.

        Parameters
        ----------
        aseg : str
            The anatomical segmentation file. Default ``auto`` uses ``aparc+aseg``
            if available and ``wmparc`` if not. This may be any anatomical
            segmentation file in the mri subdirectory of the FreeSurfer subject
            directory.

            .. versionchanged:: 1.8
               Added support for the new default ``'auto'``.
        labels : list of str | None
            Labeled regions of interest to plot. See
            :func:`mne.get_montage_volume_labels` for one way to determine regions of
            interest. Regions can also be chosen from the :term:`FreeSurfer LUT`. If
            ``None``, all labels that are defined in the segmentation file are used.
        colors : list | matplotlib-style color | None
            A list of anything matplotlib accepts: string, RGB, hex, etc.
            (default :term:`FreeSurfer LUT` colors).
        alpha : float in [0, 1]
            Alpha level to control opacity.
        smooth : float in [0, 1)
            The smoothing factor to be applied. Default 0 is no smoothing.
        fill_hole_size : int | None
            The size of holes to remove in the mesh in voxels. Default is None,
            no holes are removed. Warning, this dilates the boundaries of the
            surface by ``fill_hole_size`` number of voxels so use the minimal
            size.
        legend : bool | None | dict
            Add a legend displaying the names of the ``labels``. Default (None)
            is ``True`` if the number of ``labels`` is 10 or fewer.
            Can also be a dict of ``kwargs`` to pass to
            ``pyvista.Plotter.add_legend``.

        Notes
        -----
        .. versionadded:: 0.24
        """
        aseg, aseg_data = _get_aseg(aseg, self._subject, self._subjects_dir)

        vox_mri_t = aseg.header.get_vox2ras_tkr()
        mult = 1e-3 if self._units == "m" else 1
        vox_mri_t[:3] *= mult
        del aseg

        # read freesurfer lookup table
        lut, fs_colors = read_freesurfer_lut()
        if labels is None:  # assign default ROI labels based on indices
            lut_r = {v: k for k, v in lut.items()}
            labels = [lut_r[idx] for idx in DEFAULTS["volume_label_indices"]]

        _validate_type(fill_hole_size, (int, None), "fill_hole_size")
        _validate_type(legend, (bool, None, dict), "legend")
        if legend is None:
            legend = len(labels) < 11

        if colors is None:
            colors = [fs_colors[label] / 255 for label in labels]
        elif not isinstance(colors, list | tuple):
            colors = [colors] * len(labels)  # make into list
        colors = [
            _to_rgb(color, name=f"colors[{ci}]") for ci, color in enumerate(colors)
        ]
        surfs = _marching_cubes(
            aseg_data,
            [lut[label] for label in labels],
            smooth=smooth,
            fill_hole_size=fill_hole_size,
        )
        for label, color, (verts, triangles) in zip(labels, colors, surfs):
            if len(verts) == 0:  # not in aseg vals
                warn(
                    f"Value {lut[label]} not found for label "
                    f"{repr(label)} in anatomical segmentation file "
                )
                continue
            verts = apply_trans(vox_mri_t, verts)
            for _ in self._iter_views("vol"):
                actor, _ = self._renderer.mesh(
                    *verts.T,
                    triangles=triangles,
                    color=color,
                    opacity=alpha,
                    reset_camera=False,
                    render=False,
                )
                self._add_actor("volume_labels", actor)

        if legend or isinstance(legend, dict):
            # use empty kwargs for legend = True
            legend = legend if isinstance(legend, dict) else dict()
            self._renderer.plotter.add_legend(list(zip(labels, colors)), **legend)

        self._renderer._update()

    def remove_volume_labels(self):
        """Remove the volume labels from the rendered scene."""
        self._remove("volume_labels", render=True)
        self._renderer.plotter.remove_legend()

    @fill_doc_static("color_matplotlib", "alpha")
    def add_foci(
        self,
        coords,
        coords_as_verts=False,
        map_surface=None,
        scale_factor=1,
        color="white",
        alpha=1,
        name=None,
        hemi=None,
        resolution=50,
    ):
        """Add spherical foci, possibly mapping to displayed surf.

        The foci spheres can be displayed at the coordinates given, or
        mapped through a surface geometry. In other words, coordinates
        from a volume-based analysis in MNI space can be displayed on an
        inflated average surface by finding the closest vertex on the
        white surface and mapping to that vertex on the inflated mesh.

        Parameters
        ----------
        coords : ndarray, shape (n_coords, 3)
            Coordinates in stereotaxic space (default) or array of
            vertex ids (with ``coord_as_verts=True``).
        coords_as_verts : bool
            Whether the coords parameter should be interpreted as vertex ids.
        map_surface : str | None
            Surface to project the coordinates to, or None to use raw coords.
            When set to a surface, each foci is positioned at the closest
            vertex in the mesh.
        scale_factor : float
            Controls the size of the foci spheres (relative to 1cm).
        color : color
            A list of anything matplotlib accepts: string, RGB, hex, etc.
        alpha : float in [0, 1]
            Alpha level to control opacity.
            Default is 1.
        name : str
            Internal name to use.
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        resolution : int
            The resolution of the spheres.
        """
        from scipy.spatial.distance import cdist

        hemi = self._check_hemi(hemi, extras=["vol"])

        # Figure out how to interpret the first parameter
        if coords_as_verts:
            coords = self.geo[hemi].coords[coords]
            map_surface = None

        # Possibly map the foci coords through a surface
        if map_surface is not None:
            foci_surf = _Surface(
                self._subject,
                hemi,
                map_surface,
                self._subjects_dir,
                offset=0,
                units=self._units,
                x_dir=self._rigid[0, :3],
            )
            foci_surf.load_geometry()
            foci_vtxs = np.argmin(cdist(foci_surf.coords, coords), axis=0)
            coords = self.geo[hemi].coords[foci_vtxs]

        # Convert the color code
        color = _to_rgb(color)

        if self._units == "m":
            scale_factor = scale_factor / 1000.0

        for _, _, v in self._iter_views(hemi):
            self._renderer.sphere(
                center=coords,
                color=color,
                scale=(10.0 * scale_factor),
                opacity=alpha,
                resolution=resolution,
            )
            self._set_camera(**views_dicts[hemi][v])
        self._renderer._update()

        # Store the foci separately from overlay data
        data_foci = coords
        if "foci" in self._foci_data.get(hemi, {}):
            data_foci = np.vstack((self._foci_data[hemi]["foci"], data_foci))
        self._foci_data.setdefault(hemi, {})["foci"] = data_foci

    @verbose_static(
        "info_not_none",
        "trans_not_none",
        "meg",
        "eeg",
        "fnirs",
        "ecog",
        "seeg",
        "dbs",
        "max_dist_ieeg",
        "sensor_colors",
        "sensor_scales",
    )
    def add_sensors(
        self,
        info,
        trans,
        meg=None,
        eeg="original",
        fnirs=True,
        ecog=True,
        seeg=True,
        dbs=True,
        max_dist=0.004,
        *,
        sensor_colors=None,
        sensor_scales=None,
        verbose=None,
    ):
        """Add mesh objects to represent sensor positions.

        Parameters
        ----------
        info : mne.Info
            The :class:`mne.Info` object with information about the
            sensors and methods of measurement.
        trans : str | dict | instance of Transform
            If str, the path to the head<->MRI transform ``*-trans.fif`` file produced
            during coregistration. Can also be ``'fsaverage'`` to use the built-in
            fsaverage transformation.
        meg : str | list | dict | bool | None
            Can be "helmet", "sensors" or "ref" to show the MEG helmet, sensors or
            reference sensors respectively, or a combination like ``('helmet',
            'sensors')`` (same as None, default). True translates to ``('helmet',
            'sensors', 'ref')``. Can also be a dict to specify alpha values, e.g.
            ``{"helmet": 0.1, "sensors": 0.8}``.

            .. versionchanged:: 1.6
               Added support for specifying alpha values as a dict.
        eeg : bool | str | list | dict
            String options are:

            - "original" (default; equivalent to ``True``)
                Shows EEG sensors using their digitized locations (after
                transformation to the chosen ``coord_frame``)
            - "projected"
                The EEG locations projected onto the scalp, as is done in
                forward modeling

            Can also be a list of these options, or a dict to specify the alpha values
            to use, e.g. ``dict(original=0.2, projected=0.8)``.

            .. versionchanged:: 1.6
               Added support for specifying alpha values as a dict.
        fnirs : str | list | dict | bool | None
            Can be "channels", "pairs", "detectors", and/or "sources" to show the
            fNIRS channel locations, optode locations, or line between
            source-detector pairs, or a combination like ``('pairs', 'channels')``.
            True translates to ``('pairs',)``. A dict can also be used to specify
            alpha values (but only "channels" and "pairs" will be used), e.g.
            ``dict(channels=0.2, pairs=0.7)``.

            .. versionchanged:: 1.6
               Added support for specifying alpha values as a dict.
        ecog : bool
            If True (default), show ECoG sensors.
        seeg : bool
            If True (default), show sEEG electrodes.
        dbs : bool
            If True (default), show DBS (deep brain stimulation) electrodes.
        max_dist : float
            The maximum distance to project a sensor to the pial surface in meters.
            Sensors that are greater than this distance from the pial surface will
            not be assigned locations. Projections can be done to the inflated or
            flat brain.
        sensor_colors : array-like of color | dict | None
            Colors to use for the sensor glyphs. Can be None (default) to use default
            colors. A dict should provide the colors (values) for each channel type
            (keys), e.g.::

                dict(eeg=eeg_colors)

            Where the value (``eeg_colors`` above) can be broadcast to an array of
            colors with length that matches the number of channels of that type, i.e.,
            is compatible with :func:`matplotlib.colors.to_rgba_array`. A few examples
            of this for the case above are the string ``"k"``, a list of ``n_eeg`` color
            strings, or an NumPy ndarray of shape ``(n_eeg, 3)`` or ``(n_eeg, 4)``.

            .. versionadded:: 1.6
        sensor_scales : int | float | array-like | dict | None
            Scale to use for the sensor glyphs. Can be None (default) to use default
            scale. A dict should provide the Scale (values) for each channel type
            (keys), e.g.::

                dict(eeg=eeg_scales)

            Where the value (``eeg_scales`` above) can be broadcast to an array of
            values with length that matches the number of channels of that type. A few
            examples of this for the case above are the value ``10e-3``, a list of
            ``n_eeg`` values, or an NumPy ndarray of shape ``(n_eeg,)``.

            .. versionadded:: 1.9
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Notes
        -----
        .. versionadded:: 0.24
        """
        from ...preprocessing.ieeg._projection import _project_sensors_onto_inflated

        _validate_type(info, Info, "info")
        meg, eeg, fnirs, warn_meg, sensor_alpha = _handle_sensor_types(meg, eeg, fnirs)
        picks = pick_types(
            info,
            meg=("sensors" in meg),
            ref_meg=("ref" in meg),
            eeg=(len(eeg) > 0),
            ecog=ecog,
            seeg=seeg,
            dbs=dbs,
            fnirs=(len(fnirs) > 0),
        )
        head_mri_t = _get_trans(trans, "head", "mri", allow_none=False)[0]
        if self._surf in ("inflated", "flat"):
            for modality, check in dict(seeg=seeg, ecog=ecog).items():
                if pick_types(info, **{modality: check}).size > 0:
                    info = _project_sensors_onto_inflated(
                        info.copy(),
                        head_mri_t,
                        subject=self._subject,
                        subjects_dir=self._subjects_dir,
                        picks=modality,
                        max_dist=max_dist,
                        flat=self._surf == "flat",
                    )
        del trans
        # get transforms to "mri" window
        to_cf_t = _get_transforms_to_coord_frame(info, head_mri_t, coord_frame="mri")
        if pick_types(info, eeg=True, exclude=()).size > 0 and "projected" in eeg:
            head_surf = _get_head_surface("seghead", self._subject, self._subjects_dir)
        else:
            head_surf = None
        # Do the main plotting
        for _ in self._iter_views("vol"):
            if picks.size > 0:
                sensors_actors = _plot_sensors_3d(
                    self._renderer,
                    info,
                    to_cf_t,
                    picks,
                    meg,
                    eeg,
                    fnirs,
                    warn_meg,
                    head_surf,
                    self._units,
                    sensor_alpha=sensor_alpha,
                    sensor_colors=sensor_colors,
                    sensor_scales=sensor_scales,
                )
                # sensors_actors can still be None
                for item, actors in (sensors_actors or {}).items():
                    for actor in actors:
                        self._add_actor(item, actor)

            if "helmet" in meg and pick_types(info, meg=True).size > 0:
                actor, _, _ = _plot_helmet(
                    self._renderer,
                    info,
                    to_cf_t,
                    head_mri_t,
                    "mri",
                    alpha=sensor_alpha["meg_helmet"],
                    scale=1 if self._units == "m" else 1e3,
                )
                self._add_actor("helmet", actor)

        self._renderer._update()

    def remove_sensors(self, kind=None):
        """Remove sensors from the rendered scene.

        Parameters
        ----------
        kind : str | list | None
            If None, removes all sensor-related data including the helmet.
            Can be "meg", "eeg", "fnirs", "ecog", "seeg", "dbs" or "helmet"
            to remove that item.
        """
        all_kinds = ("meg", "eeg", "fnirs", "ecog", "seeg", "dbs", "helmet")
        if kind is None:
            for item in all_kinds:
                self._remove(item, render=False)
        else:
            if isinstance(kind, str):
                kind = [kind]
            for this_kind in kind:
                _check_option("kind", this_kind, all_kinds)
            self._remove(this_kind, render=False)
        self._renderer._update()

    def add_text(
        self,
        x,
        y,
        text,
        name=None,
        color=None,
        opacity=1.0,
        row=0,
        col=0,
        font_size=None,
        justification=None,
        font_file=None,
    ):
        """Add a text to the visualization.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        text : str
            Text to add.
        name : str
            Name of the text (text label can be updated using update_text()).
        color : tuple
            Color of the text. Default is the foreground color set during
            initialization (default is black or white depending on the
            background color).
        opacity : float
            Opacity of the text (default 1.0).
        row : int | None
            Row index of which brain to use. Default is the top row.
        col : int | None
            Column index of which brain to use. Default is the left-most
            column.
        font_size : float | None
            The font size to use.
        justification : str | None
            The text justification.
        font_file : str | None
            Path to an absolute path of a font file to use for rendering
            the text. See https://freetype.org/freetype2/docs/index.html for a list of
            supported font file formats.
        """
        _validate_type(name, (str, None), "name")
        name = text if name is None else name
        if "text" in self._actors and name in self._actors["text"]:
            raise ValueError(f"Text with the name {name} already exists")
        if color is None:
            color = self._fg_color
        for ri, ci, _ in self._iter_views("vol"):
            if (row is None or row == ri) and (col is None or col == ci):
                actor = self._renderer.text2d(
                    x_window=x,
                    y_window=y,
                    text=text,
                    color=color,
                    size=font_size,
                    justification=justification,
                    font_file=font_file,
                )
                if "text" not in self._actors:
                    self._actors["text"] = dict()
                self._actors["text"][name] = actor

    def remove_text(self, name=None):
        """Remove text from the rendered scene.

        Parameters
        ----------
        name : str | None
            Remove specific text by name. If None, all text will be removed.
        """
        _validate_type(name, (str, None), "name")
        if name is None:
            for actor in self._actors["text"].values():
                self._renderer.plotter.remove_actor(actor, render=False)
            self._actors.pop("text")
        else:
            names = [None]
            if "text" in self._actors:
                names += list(self._actors["text"].keys())
            _check_option("name", name, names)
            self._renderer.plotter.remove_actor(
                self._actors["text"][name], render=False
            )
            self._actors["text"].pop(name)
        self._renderer._update()
