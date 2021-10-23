# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD-3-Clause

from copy import deepcopy
from itertools import count
from math import sqrt

import numpy as np

from .tree import dir_tree_find
from .tag import find_tag, _rename_list
from .constants import FIFF
from .pick import pick_types, pick_info
from .write import (write_int, write_float, write_string, write_name_list,
                    write_float_matrix, end_block, start_block)
from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT
from ..utils import logger, verbose, warn, fill_doc, _validate_type


class Projection(dict):
    """Projection vector.

    A basic class to proj a meaningful print for projection vectors.
    """

    def __repr__(self):  # noqa: D105
        s = "%s" % self['desc']
        s += ", active : %s" % self['active']
        s += f", n_channels : {len(self['data']['col_names'])}"
        return "<Projection | %s>" % s

    # speed up info copy by taking advantage of mutability
    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.items():
            if k == 'data':
                v = v.copy()
                v['data'] = v['data'].copy()
                result[k] = v
            else:
                result[k] = v  # kind, active, desc, explained_var immutable
        return result

    @fill_doc
    def plot_topomap(self, info, cmap=None, sensors=True,
                     colorbar=False, res=64, size=1, show=True,
                     outlines='head', contours=6, image_interp='bilinear',
                     axes=None, vlim=(None, None), sphere=None,
                     border=_BORDER_DEFAULT):
        """Plot topographic maps of SSP projections.

        Parameters
        ----------
        %(info_not_none)s Used to determine the layout.
        %(proj_topomap_kwargs)s
        %(topomap_sphere_auto)s
        %(topomap_border)s

        Returns
        -------
        fig : instance of Figure
            Figure distributing one image per channel across sensor topography.

        Notes
        -----
        .. versionadded:: 0.15.0
        """  # noqa: E501
        from ..viz.topomap import plot_projs_topomap
        return plot_projs_topomap(self, info, cmap, sensors, colorbar,
                                  res, size, show, outlines,
                                  contours, image_interp, axes, vlim,
                                  sphere=sphere, border=border)


class ProjMixin(object):
    """Mixin class for Raw, Evoked, Epochs.

    Notes
    -----
    This mixin adds a proj attribute as a property to data containers.
    It is True if at least one proj is present and all of them are active.
    The projs might not be applied yet if data are not preloaded. In
    this case it's the _projector attribute that does the job.
    If a private _data attribute is present then the projs applied
    to it are the ones marked as active.

    A proj parameter passed in constructor of raw or epochs calls
    apply_proj and hence after the .proj attribute is True.

    As soon as you've applied the projs it will stay active in the
    remaining pipeline.

    The suggested pipeline is proj=True in epochs (it's cheaper than for raw).

    When you use delayed SSP in Epochs, projs are applied when you call
    get_data() method. They are not applied to the evoked._data unless you call
    apply_proj(). The reason is that you want to reject with projs although
    it's not stored in proj mode.
    """

    @property
    def proj(self):
        """Whether or not projections are active."""
        return (len(self.info['projs']) > 0 and
                all(p['active'] for p in self.info['projs']))

    @verbose
    def add_proj(self, projs, remove_existing=False, verbose=None):
        """Add SSP projection vectors.

        Parameters
        ----------
        projs : list
            List with projection vectors.
        remove_existing : bool
            Remove the projection vectors currently in the file.
        %(verbose_meth)s

        Returns
        -------
        self : instance of Raw | Epochs | Evoked
            The data container.
        """
        if isinstance(projs, Projection):
            projs = [projs]

        if (not isinstance(projs, list) and
                not all(isinstance(p, Projection) for p in projs)):
            raise ValueError('Only projs can be added. You supplied '
                             'something else.')

        # mark proj as inactive, as they have not been applied
        projs = deactivate_proj(projs, copy=True, verbose=self.verbose)
        if remove_existing:
            # we cannot remove the proj if they are active
            if any(p['active'] for p in self.info['projs']):
                raise ValueError('Cannot remove projectors that have '
                                 'already been applied')
            with self.info._unlock():
                self.info['projs'] = projs
        else:
            self.info['projs'].extend(projs)
        # We don't want to add projectors that are activated again.
        with self.info._unlock():
            self.info['projs'] = _uniquify_projs(self.info['projs'],
                                                 check_active=False,
                                                 sort=False)
        return self

    @verbose
    def apply_proj(self, verbose=None):
        """Apply the signal space projection (SSP) operators to the data.

        Parameters
        ----------
        %(verbose_meth)s

        Returns
        -------
        self : instance of Raw | Epochs | Evoked
            The instance.

        Notes
        -----
        Once the projectors have been applied, they can no longer be
        removed. It is usually not recommended to apply the projectors at
        too early stages, as they are applied automatically later on
        (e.g. when computing inverse solutions).
        Hint: using the copy method individual projection vectors
        can be tested without affecting the original data.
        With evoked data, consider the following example::

            projs_a = mne.read_proj('proj_a.fif')
            projs_b = mne.read_proj('proj_b.fif')
            # add the first, copy, apply and see ...
            evoked.add_proj(a).copy().apply_proj().plot()
            # add the second, copy, apply and see ...
            evoked.add_proj(b).copy().apply_proj().plot()
            # drop the first and see again
            evoked.copy().del_proj(0).apply_proj().plot()
            evoked.apply_proj()  # finally keep both
        """
        from ..epochs import BaseEpochs
        from ..evoked import Evoked
        from .base import BaseRaw
        if self.info['projs'] is None or len(self.info['projs']) == 0:
            logger.info('No projector specified for this dataset. '
                        'Please consider the method self.add_proj.')
            return self

        # Exit delayed mode if you apply proj
        if isinstance(self, BaseEpochs) and self._do_delayed_proj:
            logger.info('Leaving delayed SSP mode.')
            self._do_delayed_proj = False

        if all(p['active'] for p in self.info['projs']):
            logger.info('Projections have already been applied. '
                        'Setting proj attribute to True.')
            return self

        _projector, info = setup_proj(deepcopy(self.info), add_eeg_ref=False,
                                      activate=True, verbose=self.verbose)
        # let's not raise a RuntimeError here, otherwise interactive plotting
        if _projector is None:  # won't be fun.
            logger.info('The projections don\'t apply to these data.'
                        ' Doing nothing.')
            return self
        self._projector, self.info = _projector, info
        if isinstance(self, (BaseRaw, Evoked)):
            if self.preload:
                self._data = np.dot(self._projector, self._data)
        else:  # BaseEpochs
            if self.preload:
                for ii, e in enumerate(self._data):
                    self._data[ii] = self._project_epoch(e)
            else:
                self.load_data()  # will automatically apply
        logger.info('SSP projectors applied...')
        return self

    def del_proj(self, idx='all'):
        """Remove SSP projection vector.

        .. note:: The projection vector can only be removed if it is inactive
                  (has not been applied to the data).

        Parameters
        ----------
        idx : int | list of int | str
            Index of the projector to remove. Can also be "all" (default)
            to remove all projectors.

        Returns
        -------
        self : instance of Raw | Epochs | Evoked
            The instance.
        """
        if isinstance(idx, str) and idx == 'all':
            idx = list(range(len(self.info['projs'])))
        idx = np.atleast_1d(np.array(idx, int)).ravel()

        for ii in idx:
            proj = self.info['projs'][ii]
            if (proj['active'] and
                    set(self.info['ch_names']) &
                    set(proj['data']['col_names'])):
                msg = (f'Cannot remove projector that has already been '
                       f'applied, unless you first remove all channels it '
                       f'applies to. The problematic projector is: {proj}')
                raise ValueError(msg)

        keep = np.ones(len(self.info['projs']))
        keep[idx] = False  # works with negative indexing and does checks
        with self.info._unlock():
            self.info['projs'] = [p for p, k in zip(self.info['projs'], keep)
                                  if k]
        return self

    @fill_doc
    def plot_projs_topomap(self, ch_type=None, cmap=None,
                           sensors=True, colorbar=False, res=64, size=1,
                           show=True, outlines='head', contours=6,
                           image_interp='bilinear', axes=None,
                           vlim=(None, None), sphere=None,
                           extrapolate=_EXTRAPOLATE_DEFAULT,
                           border=_BORDER_DEFAULT):
        """Plot SSP vector.

        Parameters
        ----------
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None | list
            The channel type to plot. For 'grad', the gradiometers are collec-
            ted in pairs and the RMS for each pair is plotted. If None
            (default), it will return all channel types present. If a list of
            ch_types is provided, it will return multiple figures.
        %(proj_topomap_kwargs)s
        %(topomap_sphere_auto)s
        %(topomap_extrapolate)s

            .. versionadded:: 0.20
        %(topomap_border)s

        Returns
        -------
        fig : instance of Figure
            Figure distributing one image per channel across sensor topography.
        """
        if self.info['projs'] is not None or len(self.info['projs']) != 0:
            from ..viz.topomap import plot_projs_topomap
            fig = plot_projs_topomap(self.info['projs'], self.info, cmap=cmap,
                                     sensors=sensors, colorbar=colorbar,
                                     res=res, size=size, show=show,
                                     outlines=outlines, contours=contours,
                                     image_interp=image_interp, axes=axes,
                                     vlim=vlim, sphere=sphere,
                                     extrapolate=extrapolate, border=border)
        else:
            raise ValueError("Info is missing projs. Nothing to plot.")
        return fig

    def _reconstruct_proj(self, mode='accurate', origin='auto'):
        from ..forward import _map_meg_or_eeg_channels
        if len(self.info['projs']) == 0:
            return self
        self.apply_proj()
        for kind in ('meg', 'eeg'):
            kwargs = dict(meg=False)
            kwargs[kind] = True
            picks = pick_types(self.info, **kwargs)
            if len(picks) == 0:
                continue
            info_from = pick_info(self.info, picks)
            info_to = info_from.copy()
            with info_to._unlock():
                info_to['projs'] = []
                if kind == 'eeg' and _has_eeg_average_ref_proj(
                        info_from['projs']):
                    info_to['projs'] = [
                        make_eeg_average_ref_proj(info_to, verbose=False)]
            mapping = _map_meg_or_eeg_channels(
                info_from, info_to, mode=mode, origin=origin)
            self.data[..., picks, :] = np.matmul(
                mapping, self.data[..., picks, :])
        return self


def _proj_equal(a, b, check_active=True):
    """Test if two projectors are equal."""
    equal = ((a['active'] == b['active'] or not check_active) and
             a['kind'] == b['kind'] and
             a['desc'] == b['desc'] and
             a['data']['col_names'] == b['data']['col_names'] and
             a['data']['row_names'] == b['data']['row_names'] and
             a['data']['ncol'] == b['data']['ncol'] and
             a['data']['nrow'] == b['data']['nrow'] and
             np.all(a['data']['data'] == b['data']['data']))
    return equal


@verbose
def _read_proj(fid, node, *, ch_names_mapping=None, verbose=None):
    ch_names_mapping = {} if ch_names_mapping is None else ch_names_mapping
    projs = list()

    #   Locate the projection data
    nodes = dir_tree_find(node, FIFF.FIFFB_PROJ)
    if len(nodes) == 0:
        return projs

    # This might exist but we won't use it:
    # global_nchan = None
    # tag = find_tag(fid, nodes[0], FIFF.FIFF_NCHAN)
    # if tag is not None:
    #     global_nchan = int(tag.data)

    items = dir_tree_find(nodes[0], FIFF.FIFFB_PROJ_ITEM)
    for item in items:
        #   Find all desired tags in one item

        # This probably also exists but used to be written incorrectly
        # sometimes
        # tag = find_tag(fid, item, FIFF.FIFF_NCHAN)
        # if tag is not None:
        #     nchan = int(tag.data)
        # else:
        #     nchan = global_nchan

        tag = find_tag(fid, item, FIFF.FIFF_DESCRIPTION)
        if tag is not None:
            desc = tag.data
        else:
            tag = find_tag(fid, item, FIFF.FIFF_NAME)
            if tag is not None:
                desc = tag.data
            else:
                raise ValueError('Projection item description missing')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_KIND)
        if tag is not None:
            kind = int(tag.data)
        else:
            raise ValueError('Projection item kind missing')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_NVEC)
        if tag is not None:
            nvec = int(tag.data)
        else:
            raise ValueError('Number of projection vectors not specified')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST)
        if tag is not None:
            names = tag.data.split(':')
        else:
            raise ValueError('Projection item channel list missing')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_VECTORS)
        if tag is not None:
            data = tag.data
        else:
            raise ValueError('Projection item data missing')

        tag = find_tag(fid, item, FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE)
        if tag is not None:
            active = bool(tag.data)
        else:
            active = False

        tag = find_tag(fid, item, FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR)
        if tag is not None:
            explained_var = tag.data
        else:
            explained_var = None

        # handle the case when data is transposed for some reason
        if data.shape[0] == len(names) and data.shape[1] == nvec:
            data = data.T

        if data.shape[1] != len(names):
            raise ValueError('Number of channel names does not match the '
                             'size of data matrix')

        # just always use this, we used to have bugs with writing the
        # number correctly...
        nchan = len(names)
        names[:] = _rename_list(names, ch_names_mapping)
        #   Use exactly the same fields in data as in a named matrix
        one = Projection(kind=kind, active=active, desc=desc,
                         data=dict(nrow=nvec, ncol=nchan, row_names=None,
                                   col_names=names, data=data),
                         explained_var=explained_var)

        projs.append(one)

    if len(projs) > 0:
        logger.info('    Read a total of %d projection items:' % len(projs))
        for proj in projs:
            misc = 'active' if proj['active'] else ' idle'
            logger.info(f'        {proj["desc"]} '
                        f'({proj["data"]["nrow"]} x '
                        f'{len(proj["data"]["col_names"])}) {misc}')

    return projs


###############################################################################
# Write

def _write_proj(fid, projs, *, ch_names_mapping=None):
    """Write a projection operator to a file.

    Parameters
    ----------
    fid : file
        The file descriptor of the open file.
    projs : dict
        The projection operator.
    """
    if len(projs) == 0:
        return

    ch_names_mapping = dict() if ch_names_mapping is None else ch_names_mapping
    # validation
    _validate_type(projs, (list, tuple), 'projs')
    for pi, proj in enumerate(projs):
        _validate_type(proj, Projection, f'projs[{pi}]')

    start_block(fid, FIFF.FIFFB_PROJ)

    for proj in projs:
        start_block(fid, FIFF.FIFFB_PROJ_ITEM)
        write_int(fid, FIFF.FIFF_NCHAN, len(proj['data']['col_names']))
        names = _rename_list(proj['data']['col_names'], ch_names_mapping)
        write_name_list(fid, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST, names)
        write_string(fid, FIFF.FIFF_NAME, proj['desc'])
        write_int(fid, FIFF.FIFF_PROJ_ITEM_KIND, proj['kind'])
        if proj['kind'] == FIFF.FIFFV_PROJ_ITEM_FIELD:
            write_float(fid, FIFF.FIFF_PROJ_ITEM_TIME, 0.0)

        write_int(fid, FIFF.FIFF_PROJ_ITEM_NVEC, proj['data']['nrow'])
        write_int(fid, FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE, proj['active'])
        write_float_matrix(fid, FIFF.FIFF_PROJ_ITEM_VECTORS,
                           proj['data']['data'])
        if proj['explained_var'] is not None:
            write_float(fid, FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR,
                        proj['explained_var'])
        end_block(fid, FIFF.FIFFB_PROJ_ITEM)

    end_block(fid, FIFF.FIFFB_PROJ)


###############################################################################
# Utils

def _check_projs(projs, copy=True):
    """Check that projs is a list of Projection."""
    if not isinstance(projs, (list, tuple)):
        raise TypeError('projs must be a list or tuple, got %s'
                        % (type(projs),))
    for pi, p in enumerate(projs):
        if not isinstance(p, Projection):
            raise TypeError('All entries in projs list must be Projection '
                            'instances, but projs[%d] is type %s'
                            % (pi, type(p)))
    return deepcopy(projs) if copy else projs


def make_projector(projs, ch_names, bads=(), include_active=True):
    """Create an SSP operator from SSP projection vectors.

    Parameters
    ----------
    projs : list
        List of projection vectors.
    ch_names : list of str
        List of channels to include in the projection matrix.
    bads : list of str
        Some bad channels to exclude. If bad channels were marked
        in the raw file when projs were calculated using mne-python,
        they should not need to be included here as they will
        have been automatically omitted from the projectors.
    include_active : bool
        Also include projectors that are already active.

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data.
    nproj : int
        How many items in the projector.
    U : array
        The orthogonal basis of the projection vectors.
    """
    return _make_projector(projs, ch_names, bads, include_active)


def _make_projector(projs, ch_names, bads=(), include_active=True,
                    inplace=False):
    """Subselect projs based on ch_names and bads.

    Use inplace=True mode to modify ``projs`` inplace so that no
    warning will be raised next time projectors are constructed with
    the given inputs. If inplace=True, no meaningful data are returned.
    """
    from scipy import linalg
    nchan = len(ch_names)
    if nchan == 0:
        raise ValueError('No channel names specified')

    default_return = (np.eye(nchan, nchan), 0, np.empty((nchan, 0)))

    #   Check trivial cases first
    if projs is None:
        return default_return

    nvec = 0
    nproj = 0
    for p in projs:
        if not p['active'] or include_active:
            nproj += 1
            nvec += p['data']['nrow']

    if nproj == 0:
        return default_return

    #   Pick the appropriate entries
    vecs = np.zeros((nchan, nvec))
    nvec = 0
    nonzero = 0
    bads = set(bads)
    for k, p in enumerate(projs):
        if not p['active'] or include_active:
            if (len(p['data']['col_names']) !=
                    len(np.unique(p['data']['col_names']))):
                raise ValueError('Channel name list in projection item %d'
                                 ' contains duplicate items' % k)

            # Get the two selection vectors to pick correct elements from
            # the projection vectors omitting bad channels
            sel = []
            vecsel = []
            p_set = set(p['data']['col_names'])  # faster membership access
            for c, name in enumerate(ch_names):
                if name not in bads and name in p_set:
                    sel.append(c)
                    vecsel.append(p['data']['col_names'].index(name))

            # If there is something to pick, pickit
            nrow = p['data']['nrow']
            this_vecs = vecs[:, nvec:nvec + nrow]
            if len(sel) > 0:
                this_vecs[sel] = p['data']['data'][:, vecsel].T

            # Rescale for better detection of small singular values
            for v in range(p['data']['nrow']):
                psize = sqrt(np.sum(this_vecs[:, v] * this_vecs[:, v]))
                if psize > 0:
                    orig_n = p['data']['data'].any(axis=0).sum()
                    # Average ref still works if channels are removed
                    if len(vecsel) < 0.9 * orig_n and not inplace and \
                            (p['kind'] != FIFF.FIFFV_PROJ_ITEM_EEG_AVREF or
                             len(vecsel) == 1):
                        warn('Projection vector "%s" has magnitude %0.2f '
                             '(should be unity), applying projector with '
                             '%s/%s of the original channels available may '
                             'be dangerous, consider recomputing and adding '
                             'projection vectors for channels that are '
                             'eventually used. If this is intentional, '
                             'consider using info.normalize_proj()'
                             % (p['desc'], psize, len(vecsel), orig_n))
                    this_vecs[:, v] /= psize
                    nonzero += 1
            # If doing "inplace" mode, "fix" the projectors to only operate
            # on this subset of channels.
            if inplace:
                p['data']['data'] = this_vecs[sel].T
                p['data']['col_names'] = [p['data']['col_names'][ii]
                                          for ii in vecsel]
                p['data']['ncol'] = len(p['data']['col_names'])
            nvec += p['data']['nrow']

    #   Check whether all of the vectors are exactly zero
    if nonzero == 0 or inplace:
        return default_return

    # Reorthogonalize the vectors
    U, S, _ = linalg.svd(vecs[:, :nvec], full_matrices=False)

    # Throw away the linearly dependent guys
    nproj = np.sum((S / S[0]) > 1e-2)
    U = U[:, :nproj]

    # Here is the celebrated result
    proj = np.eye(nchan, nchan) - np.dot(U, U.T)
    if nproj >= nchan:  # e.g., 3 channels and 3 projectors
        raise RuntimeError('Application of %d projectors for %d channels '
                           'will yield no components.' % (nproj, nchan))

    return proj, nproj, U


def _normalize_proj(info):
    """Normalize proj after subselection to avoid warnings.

    This is really only useful for tests, and might not be needed
    eventually if we change or improve our handling of projectors
    with picks.
    """
    # Here we do info.get b/c info can actually be a noise cov
    _make_projector(info['projs'], info.get('ch_names', info.get('names')),
                    info['bads'], include_active=True, inplace=True)


@fill_doc
def make_projector_info(info, include_active=True):
    """Make an SSP operator using the measurement info.

    Calls make_projector on good channels.

    Parameters
    ----------
    %(info_not_none)s
    include_active : bool
        Also include projectors that are already active.

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data.
    nproj : int
        How many items in the projector.
    """
    proj, nproj, _ = make_projector(info['projs'], info['ch_names'],
                                    info['bads'], include_active)
    return proj, nproj


@verbose
def activate_proj(projs, copy=True, verbose=None):
    """Set all projections to active.

    Useful before passing them to make_projector.

    Parameters
    ----------
    projs : list
        The projectors.
    copy : bool
        Modify projs in place or operate on a copy.
    %(verbose)s

    Returns
    -------
    projs : list
        The projectors.
    """
    if copy:
        projs = deepcopy(projs)

    #   Activate the projection items
    for proj in projs:
        proj['active'] = True

    logger.info('%d projection items activated' % len(projs))

    return projs


@verbose
def deactivate_proj(projs, copy=True, verbose=None):
    """Set all projections to inactive.

    Useful before saving raw data without projectors applied.

    Parameters
    ----------
    projs : list
        The projectors.
    copy : bool
        Modify projs in place or operate on a copy.
    %(verbose)s

    Returns
    -------
    projs : list
        The projectors.
    """
    if copy:
        projs = deepcopy(projs)

    #   Deactivate the projection items
    for proj in projs:
        proj['active'] = False

    logger.info('%d projection items deactivated' % len(projs))

    return projs


@verbose
def make_eeg_average_ref_proj(info, activate=True, verbose=None):
    """Create an EEG average reference SSP projection vector.

    Parameters
    ----------
    %(info_not_none)s
    activate : bool
        If True projections are activated.
    %(verbose)s

    Returns
    -------
    eeg_proj: instance of Projection
        The SSP/PCA projector.
    """
    if info.get('custom_ref_applied', False):
        raise RuntimeError('A custom reference has been applied to the '
                           'data earlier. Please use the '
                           'mne.io.set_eeg_reference function to move from '
                           'one EEG reference to another.')

    logger.info("Adding average EEG reference projection.")
    eeg_sel = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude='bads')
    ch_names = info['ch_names']
    eeg_names = [ch_names[k] for k in eeg_sel]
    n_eeg = len(eeg_sel)
    if n_eeg == 0:
        raise ValueError('Cannot create EEG average reference projector '
                         '(no EEG data found)')
    vec = np.ones((1, n_eeg))
    vec /= n_eeg
    explained_var = None
    eeg_proj_data = dict(col_names=eeg_names, row_names=None,
                         data=vec, nrow=1, ncol=n_eeg)
    eeg_proj = Projection(active=activate, data=eeg_proj_data,
                          desc='Average EEG reference',
                          kind=FIFF.FIFFV_PROJ_ITEM_EEG_AVREF,
                          explained_var=explained_var)
    return eeg_proj


def _has_eeg_average_ref_proj(projs, check_active=False):
    """Determine if a list of projectors has an average EEG ref.

    Optionally, set check_active=True to additionally check if the CAR
    has already been applied.
    """
    for proj in projs:
        if (proj['desc'] == 'Average EEG reference' or
                proj['kind'] == FIFF.FIFFV_PROJ_ITEM_EEG_AVREF):
            if not check_active or proj['active']:
                return True
    return False


def _needs_eeg_average_ref_proj(info):
    """Determine if the EEG needs an averge EEG reference.

    This returns True if no custom reference has been applied and no average
    reference projection is present in the list of projections.
    """
    eeg_sel = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude='bads')
    return (len(eeg_sel) > 0 and
            not info['custom_ref_applied'] and
            not _has_eeg_average_ref_proj(info['projs']))


@verbose
def setup_proj(info, add_eeg_ref=True, activate=True, verbose=None):
    """Set up projection for Raw and Epochs.

    Parameters
    ----------
    %(info_not_none)s Warning: will be modified in-place.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless one
        already exists).
    activate : bool
        If True projections are activated.
    %(verbose)s

    Returns
    -------
    projector : array of shape [n_channels, n_channels]
        The projection operator to apply to the data.
    info : mne.Info
        The modified measurement info.
    """
    # Add EEG ref reference proj if necessary
    if add_eeg_ref and _needs_eeg_average_ref_proj(info):
        eeg_proj = make_eeg_average_ref_proj(info, activate=activate)
        info['projs'].append(eeg_proj)

    # Create the projector
    projector, nproj = make_projector_info(info)
    if nproj == 0:
        if verbose:
            logger.info('The projection vectors do not apply to these '
                        'channels')
        projector = None
    else:
        logger.info('Created an SSP operator (subspace dimension = %d)'
                    % nproj)

    # The projection items have been activated
    if activate:
        with info._unlock():
            info['projs'] = activate_proj(info['projs'], copy=False)

    return projector, info


def _uniquify_projs(projs, check_active=True, sort=True):
    """Make unique projs."""
    final_projs = []
    for proj in projs:  # flatten
        if not any(_proj_equal(p, proj, check_active) for p in final_projs):
            final_projs.append(proj)

    my_count = count(len(final_projs))

    def sorter(x):
        """Sort in a nice way."""
        digits = [s for s in x['desc'] if s.isdigit()]
        if digits:
            sort_idx = int(digits[-1])
        else:
            sort_idx = next(my_count)
        return (sort_idx, x['desc'])

    return sorted(final_projs, key=sorter) if sort else final_projs
