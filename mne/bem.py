# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Lorenzo De Santis <lorenzo.de-santis@u-psud.fr>
#
# License: BSD-3-Clause

# The computations in this code were primarily derived from Matti Hämäläinen's
# C code.

from collections import OrderedDict
from copy import deepcopy
from functools import partial
import glob
import json
import os
import os.path as op
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .io.constants import FIFF, FWD
from .io._digitization import _dig_kind_dict, _dig_kind_rev, _dig_kind_ints
from .io.write import (start_and_end_file, start_block, write_float, write_int,
                       write_float_matrix, write_int_matrix, end_block,
                       write_string)
from .io.tag import find_tag
from .io.tree import dir_tree_find
from .io.open import fiff_open
from .surface import (read_surface, write_surface, complete_surface_info,
                      _compute_nearest, _get_ico_surface, read_tri,
                      _fast_cross_nd_sum, _get_solids, _complete_sphere_surf,
                      decimate_surface, transform_surface_to)
from .transforms import _ensure_trans, apply_trans, Transform
from .utils import (verbose, logger, run_subprocess, get_subjects_dir, warn,
                    _pl, _validate_type, _TempDir, _check_freesurfer_home,
                    _check_fname, has_nibabel, _check_option, path_like,
                    _on_missing, _import_h5io_funcs, _ensure_int,
                    _path_like, _verbose_safe_false)


# ############################################################################
# Compute BEM solution

# The following approach is based on:
#
# de Munck JC: "A linear discretization of the volume conductor boundary
# integral equation using analytically integrated elements",
# IEEE Trans Biomed Eng. 1992 39(9) : 986 - 990
#


class ConductorModel(dict):
    """BEM or sphere model."""

    def __repr__(self):  # noqa: D105
        if self['is_sphere']:
            center = ', '.join('%0.1f' % (x * 1000.) for x in self['r0'])
            rad = self.radius
            if rad is None:  # no radius / MEG only
                extra = 'Sphere (no layers): r0=[%s] mm' % center
            else:
                extra = ('Sphere (%s layer%s): r0=[%s] R=%1.f mm'
                         % (len(self['layers']) - 1, _pl(self['layers']),
                            center, rad * 1000.))
        else:
            extra = ('BEM (%s layer%s)' % (len(self['surfs']),
                                           _pl(self['surfs'])))
            extra += " solver=%s" % self['solver']
        return '<ConductorModel | %s>' % extra

    def copy(self):
        """Return copy of ConductorModel instance."""
        return deepcopy(self)

    @property
    def radius(self):
        """Sphere radius if an EEG sphere model."""
        if not self['is_sphere']:
            raise RuntimeError('radius undefined for BEM')
        return None if len(self['layers']) == 0 else self['layers'][-1]['rad']


def _calc_beta(rk, rk_norm, rk1, rk1_norm):
    """Compute coefficients for calculating the magic vector omega."""
    rkk1 = rk1[0] - rk[0]
    size = np.linalg.norm(rkk1)
    rkk1 /= size
    num = rk_norm + np.dot(rk, rkk1)
    den = rk1_norm + np.dot(rk1, rkk1)
    res = np.log(num / den) / size
    return res


def _lin_pot_coeff(fros, tri_rr, tri_nn, tri_area):
    """Compute the linear potential matrix element computations."""
    omega = np.zeros((len(fros), 3))

    # we replicate a little bit of the _get_solids code here for speed
    # (we need some of the intermediate values later)
    v1 = tri_rr[np.newaxis, 0, :] - fros
    v2 = tri_rr[np.newaxis, 1, :] - fros
    v3 = tri_rr[np.newaxis, 2, :] - fros
    triples = _fast_cross_nd_sum(v1, v2, v3)
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    l3 = np.linalg.norm(v3, axis=1)
    ss = l1 * l2 * l3
    ss += np.einsum('ij,ij,i->i', v1, v2, l3)
    ss += np.einsum('ij,ij,i->i', v1, v3, l2)
    ss += np.einsum('ij,ij,i->i', v2, v3, l1)
    solids = np.arctan2(triples, ss)

    # We *could* subselect the good points from v1, v2, v3, triples, solids,
    # l1, l2, and l3, but there are *very* few bad points. So instead we do
    # some unnecessary calculations, and then omit them from the final
    # solution. These three lines ensure we don't get invalid values in
    # _calc_beta.
    bad_mask = np.abs(solids) < np.pi / 1e6
    l1[bad_mask] = 1.
    l2[bad_mask] = 1.
    l3[bad_mask] = 1.

    # Calculate the magic vector vec_omega
    beta = [_calc_beta(v1, l1, v2, l2)[:, np.newaxis],
            _calc_beta(v2, l2, v3, l3)[:, np.newaxis],
            _calc_beta(v3, l3, v1, l1)[:, np.newaxis]]
    vec_omega = (beta[2] - beta[0]) * v1
    vec_omega += (beta[0] - beta[1]) * v2
    vec_omega += (beta[1] - beta[2]) * v3

    area2 = 2.0 * tri_area
    n2 = 1.0 / (area2 * area2)
    # leave omega = 0 otherwise
    # Put it all together...
    yys = [v1, v2, v3]
    idx = [0, 1, 2, 0, 2]
    for k in range(3):
        diff = yys[idx[k - 1]] - yys[idx[k + 1]]
        zdots = _fast_cross_nd_sum(yys[idx[k + 1]], yys[idx[k - 1]], tri_nn)
        omega[:, k] = -n2 * (area2 * zdots * 2. * solids -
                             triples * (diff * vec_omega).sum(axis=-1))
    # omit the bad points from the solution
    omega[bad_mask] = 0.
    return omega


def _correct_auto_elements(surf, mat):
    """Improve auto-element approximation."""
    pi2 = 2.0 * np.pi
    tris_flat = surf['tris'].ravel()
    misses = pi2 - mat.sum(axis=1)
    for j, miss in enumerate(misses):
        # How much is missing?
        n_memb = len(surf['neighbor_tri'][j])
        assert n_memb > 0  # should be guaranteed by our surface checks
        # The node itself receives one half
        mat[j, j] = miss / 2.0
        # The rest is divided evenly among the member nodes...
        miss /= (4.0 * n_memb)
        members = np.where(j == tris_flat)[0]
        mods = members % 3
        offsets = np.array([[1, 2], [-1, 1], [-1, -2]])
        tri_1 = members + offsets[mods, 0]
        tri_2 = members + offsets[mods, 1]
        for t1, t2 in zip(tri_1, tri_2):
            mat[j, tris_flat[t1]] += miss
            mat[j, tris_flat[t2]] += miss
    return


def _fwd_bem_lin_pot_coeff(surfs):
    """Calculate the coefficients for linear collocation approach."""
    # taken from fwd_bem_linear_collocation.c
    nps = [surf['np'] for surf in surfs]
    np_tot = sum(nps)
    coeff = np.zeros((np_tot, np_tot))
    offsets = np.cumsum(np.concatenate(([0], nps)))
    for si_1, surf1 in enumerate(surfs):
        rr_ord = np.arange(nps[si_1])
        for si_2, surf2 in enumerate(surfs):
            logger.info("        %s (%d) -> %s (%d) ..." %
                        (_bem_surf_name[surf1['id']], nps[si_1],
                         _bem_surf_name[surf2['id']], nps[si_2]))
            tri_rr = surf2['rr'][surf2['tris']]
            tri_nn = surf2['tri_nn']
            tri_area = surf2['tri_area']
            submat = coeff[offsets[si_1]:offsets[si_1 + 1],
                           offsets[si_2]:offsets[si_2 + 1]]  # view
            for k in range(surf2['ntri']):
                tri = surf2['tris'][k]
                if si_1 == si_2:
                    skip_idx = ((rr_ord == tri[0]) |
                                (rr_ord == tri[1]) |
                                (rr_ord == tri[2]))
                else:
                    skip_idx = list()
                # No contribution from a triangle that
                # this vertex belongs to
                # if sidx1 == sidx2 and (tri == j).any():
                #     continue
                # Otherwise do the hard job
                coeffs = _lin_pot_coeff(fros=surf1['rr'], tri_rr=tri_rr[k],
                                        tri_nn=tri_nn[k], tri_area=tri_area[k])
                coeffs[skip_idx] = 0.
                submat[:, tri] -= coeffs
            if si_1 == si_2:
                _correct_auto_elements(surf1, submat)
    return coeff


def _fwd_bem_multi_solution(solids, gamma, nps):
    """Do multi surface solution.

    * Invert I - solids/(2*M_PI)
    * Take deflation into account
    * The matrix is destroyed after inversion
    * This is the general multilayer case
    """
    pi2 = 1.0 / (2 * np.pi)
    n_tot = np.sum(nps)
    assert solids.shape == (n_tot, n_tot)
    nsurf = len(nps)
    defl = 1.0 / n_tot
    # Modify the matrix
    offsets = np.cumsum(np.concatenate(([0], nps)))
    for si_1 in range(nsurf):
        for si_2 in range(nsurf):
            mult = pi2 if gamma is None else pi2 * gamma[si_1, si_2]
            slice_j = slice(offsets[si_1], offsets[si_1 + 1])
            slice_k = slice(offsets[si_2], offsets[si_2 + 1])
            solids[slice_j, slice_k] = defl - solids[slice_j, slice_k] * mult
    solids += np.eye(n_tot)
    return np.linalg.inv(solids)


def _fwd_bem_homog_solution(solids, nps):
    """Make a homogeneous solution."""
    return _fwd_bem_multi_solution(solids, gamma=None, nps=nps)


def _fwd_bem_ip_modify_solution(solution, ip_solution, ip_mult, n_tri):
    """Modify the solution according to the IP approach."""
    n_last = n_tri[-1]
    mult = (1.0 + ip_mult) / ip_mult

    logger.info('        Combining...')
    offsets = np.cumsum(np.concatenate(([0], n_tri)))
    for si in range(len(n_tri)):
        # Pick the correct submatrix (right column) and multiply
        sub = solution[offsets[si]:offsets[si + 1], np.sum(n_tri[:-1]):]
        # Multiply
        sub -= 2 * np.dot(sub, ip_solution)

    # The lower right corner is a special case
    sub[-n_last:, -n_last:] += mult * ip_solution

    # Final scaling
    logger.info('        Scaling...')
    solution *= ip_mult
    return


def _check_complete_surface(surf, copy=False, incomplete='raise', extra=''):
    surf = complete_surface_info(
        surf, copy=copy, verbose=_verbose_safe_false())
    fewer = np.where([len(t) < 3 for t in surf['neighbor_tri']])[0]
    if len(fewer) > 0:
        fewer = list(fewer)
        fewer = (fewer[:80] + ['...']) if len(fewer) > 80 else fewer
        fewer = ', '.join(str(f) for f in fewer)
        msg = ('Surface {} has topological defects: {:.0f} / {:.0f} vertices '
               'have fewer than three neighboring triangles [{}]{}'
               .format(_bem_surf_name[surf['id']], len(fewer), len(surf['rr']),
                       fewer, extra))
        _on_missing(on_missing=incomplete, msg=msg, name='on_defects')
    return surf


def _fwd_bem_linear_collocation_solution(bem):
    """Compute the linear collocation potential solution."""
    # first, add surface geometries
    logger.info('Computing the linear collocation solution...')
    logger.info('    Matrix coefficients...')
    coeff = _fwd_bem_lin_pot_coeff(bem['surfs'])
    bem['nsol'] = len(coeff)
    logger.info("    Inverting the coefficient matrix...")
    nps = [surf['np'] for surf in bem['surfs']]
    bem['solution'] = _fwd_bem_multi_solution(coeff, bem['gamma'], nps)
    if len(bem['surfs']) == 3:
        ip_mult = bem['sigma'][1] / bem['sigma'][2]
        if ip_mult <= FWD.BEM_IP_APPROACH_LIMIT:
            logger.info('IP approach required...')
            logger.info('    Matrix coefficients (homog)...')
            coeff = _fwd_bem_lin_pot_coeff([bem['surfs'][-1]])
            logger.info('    Inverting the coefficient matrix (homog)...')
            ip_solution = _fwd_bem_homog_solution(coeff,
                                                  [bem['surfs'][-1]['np']])
            logger.info('    Modify the original solution to incorporate '
                        'IP approach...')
            _fwd_bem_ip_modify_solution(bem['solution'], ip_solution, ip_mult,
                                        nps)
    bem['bem_method'] = FIFF.FIFFV_BEM_APPROX_LINEAR
    bem['solver'] = 'mne'


def _import_openmeeg(what='compute a BEM solution using OpenMEEG'):
    try:
        import openmeeg as om
    except Exception as exc:
        raise ImportError(
            f'The OpenMEEG module must be installed to {what}, but '
            f'"import openmeeg" resulted in: {exc}') from None
    return om


def _make_openmeeg_geometry(bem, mri_head_t=None):
    # OpenMEEG
    om = _import_openmeeg()
    meshes = []
    for surf in bem['surfs'][::-1]:
        if mri_head_t is not None:
            surf = transform_surface_to(surf, "head", mri_head_t, copy=True)
        points, faces = surf['rr'], surf['tris']
        faces = faces[:, [1, 0, 2]]  # swap faces
        meshes.append((points, faces))

    conductivity = bem['sigma'][::-1]
    # We should be able to do this:
    #
    # geom = om.make_nested_geometry(meshes, conductivity)
    #
    # But OpenMEEG's NumPy support is iffy. So let's use file IO for now :(

    def _write_tris(fname, mesh):
        from .surface import complete_surface_info
        mesh = dict(rr=mesh[0], tris=mesh[1])
        complete_surface_info(mesh, copy=False, do_neighbor_tri=False)
        with open(fname, 'w') as fid:
            fid.write(f'- {len(mesh["rr"])}\n')
            for r, n in zip(mesh['rr'], mesh['nn']):
                fid.write(f'{r[0]:.8f} {r[1]:.8f} {r[2]:.8f} '
                          f'{n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n')
            n_tri = len(mesh['tris'])
            fid.write(f'- {n_tri} {n_tri} {n_tri}\n')
            for t in mesh['tris']:
                fid.write(f'{t[0]} {t[1]} {t[2]}\n')

    assert len(conductivity) in (1, 3)
    # on Windows, the dir can't be cleaned up, presumably because OpenMEEG
    # does not let go of the file pointer (?). This is not great but hopefully
    # writing files is temporary, and/or we can fix the file pointer bug
    # in OpenMEEG soon.
    tmp_dir = tempfile.TemporaryDirectory(prefix='openmeeg-io-')
    tmp_path = Path(tmp_dir.name)
    # In 3.10+ we could use this as a context manager as there is a
    # ignore_cleanup_errors arg, but before this there is not.
    # so let's just try/finally
    try:
        tmp_path = Path(tmp_path)
        # write geom_file and three .tri files
        geom_file = tmp_path / 'tmp.geom'
        names = ['inner_skull', 'outer_skull', 'outer_skin']
        lines = [
            '# Domain Description 1.1',
            '',
            f'Interfaces {len(conductivity)}'
            '',
            f'Interface Cortex: "{names[0]}.tri"',
        ]
        if len(conductivity) == 3:
            lines.extend([
                f'Interface Skull: "{names[1]}.tri"',
                f'Interface Head: "{names[2]}.tri"',
            ])
        lines.extend([
            '',
            f'Domains {len(conductivity) + 1}',
            '',
            'Domain Brain: -Cortex',
        ])
        if len(conductivity) == 1:
            lines.extend([
                'Domain Air: Cortex',
            ])
        else:
            lines.extend([
                'Domain Skull: Cortex -Skull',
                'Domain Scalp: Skull -Head',
                'Domain Air: Head',
            ])
        with open(geom_file, 'w') as fid:
            fid.write('\n'.join(lines))
        for mesh, name in zip(meshes, names):
            _write_tris(tmp_path / f'{name}.tri', mesh)
        # write cond_file
        cond_file = tmp_path / 'tmp.cond'
        lines = [
            '# Properties Description 1.0 (Conductivities)',
            '',
            f'Brain       {conductivity[0]}',
        ]
        if len(conductivity) == 3:
            lines.extend([
                f'Skull       {conductivity[1]}',
                f'Scalp       {conductivity[2]}',
            ])
        lines.append('Air         0.0')
        with open(cond_file, 'w') as fid:
            fid.write('\n'.join(lines))
        geom = om.Geometry(str(geom_file), str(cond_file))
    finally:
        try:
            tmp_dir.cleanup()
        except Exception:
            pass  # ignore any cleanup errors (esp. on Windows)

    return geom


def _fwd_bem_openmeeg_solution(bem):
    om = _import_openmeeg()
    logger.info('Creating BEM solution using OpenMEEG')
    logger.info('Computing the openmeeg head matrix solution...')
    logger.info('    Matrix coefficients...')

    geom = _make_openmeeg_geometry(bem)

    hm = om.HeadMat(geom)
    bem['nsol'] = hm.nlin()

    logger.info("    Inverting the coefficient matrix...")
    hm.invert()  # invert inplace
    bem['solution'] = hm.array_flat()
    bem['bem_method'] = FIFF.FIFFV_BEM_APPROX_LINEAR
    bem['solver'] = 'openmeeg'


@verbose
def make_bem_solution(surfs, *, solver='mne', verbose=None):
    """Create a BEM solution using the linear collocation approach.

    Parameters
    ----------
    surfs : list of dict
        The BEM surfaces to use (from :func:`mne.make_bem_model`).
    solver : str
        Can be 'mne' (default) to use MNE-Python, or 'openmeeg' to use
        the :doc:`OpenMEEG <openmeeg:index>` package.

        .. versionadded:: 1.2
    %(verbose)s

    Returns
    -------
    bem : instance of ConductorModel
        The BEM solution.

    See Also
    --------
    make_bem_model
    read_bem_surfaces
    write_bem_surfaces
    read_bem_solution
    write_bem_solution

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    _validate_type(solver, str, 'solver')
    _check_option('method', solver.lower(), ('mne', 'openmeeg'))
    bem = _ensure_bem_surfaces(surfs)
    _add_gamma_multipliers(bem)
    if len(bem['surfs']) == 3:
        logger.info('Three-layer model surfaces loaded.')
    elif len(bem['surfs']) == 1:
        logger.info('Homogeneous model surface loaded.')
    else:
        raise RuntimeError('Only 1- or 3-layer BEM computations supported')
    _check_bem_size(bem['surfs'])
    for surf in bem['surfs']:
        _check_complete_surface(surf)
    if solver.lower() == 'openmeeg':
        _fwd_bem_openmeeg_solution(bem)
    else:
        assert solver.lower() == 'mne'
        _fwd_bem_linear_collocation_solution(bem)
    logger.info("Solution ready.")
    logger.info('BEM geometry computations complete.')
    return bem


# ############################################################################
# Make BEM model

def _ico_downsample(surf, dest_grade):
    """Downsample the surface if isomorphic to a subdivided icosahedron."""
    n_tri = len(surf['tris'])
    bad_msg = ("Cannot decimate to requested ico grade %d. The provided "
               "BEM surface has %d triangles, which cannot be isomorphic with "
               "a subdivided icosahedron. Consider manually decimating the "
               "surface to a suitable density and then use ico=None in "
               "make_bem_model." % (dest_grade, n_tri))
    if n_tri % 20 != 0:
        raise RuntimeError(bad_msg)
    n_tri = n_tri // 20
    found = int(round(np.log(n_tri) / np.log(4)))
    if n_tri != 4 ** found:
        raise RuntimeError(bad_msg)
    del n_tri

    if dest_grade > found:
        raise RuntimeError('For this surface, decimation grade should be %d '
                           'or less, not %s.' % (found, dest_grade))

    source = _get_ico_surface(found)
    dest = _get_ico_surface(dest_grade, patch_stats=True)
    del dest['tri_cent']
    del dest['tri_nn']
    del dest['neighbor_tri']
    del dest['tri_area']
    if not np.array_equal(source['tris'], surf['tris']):
        raise RuntimeError('The source surface has a matching number of '
                           'triangles but ordering is wrong')
    logger.info('Going from %dth to %dth subdivision of an icosahedron '
                '(n_tri: %d -> %d)' % (found, dest_grade, len(surf['tris']),
                                       len(dest['tris'])))
    # Find the mapping
    dest['rr'] = surf['rr'][_get_ico_map(source, dest)]
    return dest


def _get_ico_map(fro, to):
    """Get a mapping between ico surfaces."""
    nearest, dists = _compute_nearest(fro['rr'], to['rr'], return_dists=True)
    n_bads = (dists > 5e-3).sum()
    if n_bads > 0:
        raise RuntimeError('No matching vertex for %d destination vertices'
                           % (n_bads))
    return nearest


def _order_surfaces(surfs):
    """Reorder the surfaces."""
    if len(surfs) != 3:
        return surfs
    # we have three surfaces
    surf_order = [FIFF.FIFFV_BEM_SURF_ID_HEAD,
                  FIFF.FIFFV_BEM_SURF_ID_SKULL,
                  FIFF.FIFFV_BEM_SURF_ID_BRAIN]
    ids = np.array([surf['id'] for surf in surfs])
    if set(ids) != set(surf_order):
        raise RuntimeError('bad surface ids: %s' % ids)
    order = [np.where(ids == id_)[0][0] for id_ in surf_order]
    surfs = [surfs[idx] for idx in order]
    return surfs


def _assert_complete_surface(surf, incomplete='raise'):
    """Check the sum of solid angles as seen from inside."""
    # from surface_checks.c
    # Center of mass....
    cm = surf['rr'].mean(axis=0)
    logger.info('%s CM is %6.2f %6.2f %6.2f mm' %
                (_bem_surf_name[surf['id']],
                 1000 * cm[0], 1000 * cm[1], 1000 * cm[2]))
    tot_angle = _get_solids(surf['rr'][surf['tris']], cm[np.newaxis, :])[0]
    prop = tot_angle / (2 * np.pi)
    if np.abs(prop - 1.0) > 1e-5:
        msg = (f'Surface {_bem_surf_name[surf["id"]]} is not complete (sum of '
               f'solid angles yielded {prop}, should be 1.)')
        _on_missing(
            incomplete, msg, name='incomplete', error_klass=RuntimeError)


def _assert_inside(fro, to):
    """Check one set of points is inside a surface."""
    # this is "is_inside" in surface_checks.c
    fro_name = _bem_surf_name[fro["id"]]
    to_name = _bem_surf_name[to["id"]]
    logger.info(
        f'Checking that surface {fro_name} is inside surface {to_name} ...')
    tot_angle = _get_solids(to['rr'][to['tris']], fro['rr'])
    if (np.abs(tot_angle / (2 * np.pi) - 1.0) > 1e-5).any():
        raise RuntimeError(
            f'Surface {fro_name} is not completely inside surface {to_name}')


def _check_surfaces(surfs, incomplete='raise'):
    """Check that the surfaces are complete and non-intersecting."""
    for surf in surfs:
        _assert_complete_surface(surf, incomplete=incomplete)
    # Then check the topology
    for surf_1, surf_2 in zip(surfs[:-1], surfs[1:]):
        _assert_inside(surf_2, surf_1)


def _check_surface_size(surf):
    """Check that the coordinate limits are reasonable."""
    sizes = surf['rr'].max(axis=0) - surf['rr'].min(axis=0)
    if (sizes < 0.05).any():
        raise RuntimeError(
            f'Dimensions of the surface {_bem_surf_name[surf["id"]]} seem too '
            f'small ({1000 * sizes.min():9.5f}). Maybe the unit of measure'
            ' is meters instead of mm')


def _check_thicknesses(surfs):
    """Compute how close we are."""
    for surf_1, surf_2 in zip(surfs[:-1], surfs[1:]):
        min_dist = _compute_nearest(surf_1['rr'], surf_2['rr'],
                                    return_dists=True)[1]
        min_dist = min_dist.min()
        fro = _bem_surf_name[surf_1['id']]
        to = _bem_surf_name[surf_2['id']]
        logger.info(f'Checking distance between {fro} and {to} surfaces...')
        logger.info(f'Minimum distance between the {fro} and {to} surfaces is '
                    f'approximately {1000 * min_dist:6.1f} mm')


def _surfaces_to_bem(surfs, ids, sigmas, ico=None, rescale=True,
                     incomplete='raise', extra=''):
    """Convert surfaces to a BEM."""
    # equivalent of mne_surf2bem
    # surfs can be strings (filenames) or surface dicts
    if len(surfs) not in (1, 3) or not (len(surfs) == len(ids) ==
                                        len(sigmas)):
        raise ValueError('surfs, ids, and sigmas must all have the same '
                         'number of elements (1 or 3)')
    for si, surf in enumerate(surfs):
        if isinstance(surf, str):
            surfs[si] = surf = read_surface(surf, return_dict=True)[-1]
    # Downsampling if the surface is isomorphic with a subdivided icosahedron
    if ico is not None:
        for si, surf in enumerate(surfs):
            surfs[si] = _ico_downsample(surf, ico)
    for surf, id_ in zip(surfs, ids):
        # Do topology checks (but don't save data) to fail early
        surf['id'] = id_
        _check_complete_surface(surf, copy=True, incomplete=incomplete,
                                extra=extra)
        surf['coord_frame'] = surf.get('coord_frame', FIFF.FIFFV_COORD_MRI)
        surf.update(np=len(surf['rr']), ntri=len(surf['tris']))
        if rescale:
            surf['rr'] /= 1000.  # convert to meters

    # Shifting surfaces is not implemented here...

    # Order the surfaces for the benefit of the topology checks
    for surf, sigma in zip(surfs, sigmas):
        surf['sigma'] = sigma
    surfs = _order_surfaces(surfs)

    # Check topology as best we can
    _check_surfaces(surfs, incomplete=incomplete)
    for surf in surfs:
        _check_surface_size(surf)
    _check_thicknesses(surfs)
    logger.info('Surfaces passed the basic topology checks.')
    return surfs


@verbose
def make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3),
                   subjects_dir=None, verbose=None):
    """Create a BEM model for a subject.

    .. note:: To get a single layer bem corresponding to the --homog flag in
              the command line tool set the ``conductivity`` parameter
              to a list/tuple with a single value (e.g. [0.3]).

    Parameters
    ----------
    subject : str
        The subject.
    ico : int | None
        The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280.
        If None, no subsampling is applied.
    conductivity : array of int, shape (3,) or (1,)
        The conductivities to use for each shell. Should be a single element
        for a one-layer model, or three elements for a three-layer model.
        Defaults to ``[0.3, 0.006, 0.3]``. The MNE-C default for a
        single-layer model would be ``[0.3]``.
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    surfaces : list of dict
        The BEM surfaces. Use `make_bem_solution` to turn these into a
        `~mne.bem.ConductorModel` suitable for forward calculation.

    See Also
    --------
    make_bem_solution
    make_sphere_model
    read_bem_surfaces
    write_bem_surfaces

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    conductivity = np.array(conductivity, float)
    if conductivity.ndim != 1 or conductivity.size not in (1, 3):
        raise ValueError('conductivity must be 1D array-like with 1 or 3 '
                         'elements')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subject_dir = op.join(subjects_dir, subject)
    bem_dir = op.join(subject_dir, 'bem')
    inner_skull = op.join(bem_dir, 'inner_skull.surf')
    outer_skull = op.join(bem_dir, 'outer_skull.surf')
    outer_skin = op.join(bem_dir, 'outer_skin.surf')
    surfaces = [inner_skull, outer_skull, outer_skin]
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
           FIFF.FIFFV_BEM_SURF_ID_SKULL,
           FIFF.FIFFV_BEM_SURF_ID_HEAD]
    logger.info('Creating the BEM geometry...')
    if len(conductivity) == 1:
        surfaces = surfaces[:1]
        ids = ids[:1]
    surfaces = _surfaces_to_bem(surfaces, ids, conductivity, ico)
    _check_bem_size(surfaces)
    logger.info('Complete.\n')
    return surfaces


# ############################################################################
# Compute EEG sphere model

def _fwd_eeg_get_multi_sphere_model_coeffs(m, n_terms):
    """Get the model depended weighting factor for n."""
    nlayer = len(m['layers'])
    if nlayer in (0, 1):
        return 1.

    # Initialize the arrays
    c1 = np.zeros(nlayer - 1)
    c2 = np.zeros(nlayer - 1)
    cr = np.zeros(nlayer - 1)
    cr_mult = np.zeros(nlayer - 1)
    for k in range(nlayer - 1):
        c1[k] = m['layers'][k]['sigma'] / m['layers'][k + 1]['sigma']
        c2[k] = c1[k] - 1.0
        cr_mult[k] = m['layers'][k]['rel_rad']
        cr[k] = cr_mult[k]
        cr_mult[k] *= cr_mult[k]

    coeffs = np.zeros(n_terms - 1)
    for n in range(1, n_terms):
        # Increment the radius coefficients
        for k in range(nlayer - 1):
            cr[k] *= cr_mult[k]

        # Multiply the matrices
        M = np.eye(2)
        n1 = n + 1.0
        for k in range(nlayer - 2, -1, -1):
            M = np.dot([[n + n1 * c1[k], n1 * c2[k] / cr[k]],
                        [n * c2[k] * cr[k], n1 + n * c1[k]]], M)
        num = n * (2.0 * n + 1.0) ** (nlayer - 1)
        coeffs[n - 1] = num / (n * M[1, 1] + n1 * M[1, 0])
    return coeffs


def _compose_linear_fitting_data(mu, u):
    """Get the linear fitting data."""
    from scipy import linalg
    k1 = np.arange(1, u['nterms'])
    mu1ns = mu[0] ** k1
    # data to be fitted
    y = u['w'][:-1] * (u['fn'][1:] - mu1ns * u['fn'][0])
    # model matrix
    M = u['w'][:-1, np.newaxis] * (mu[1:] ** k1[:, np.newaxis] -
                                   mu1ns[:, np.newaxis])
    uu, sing, vv = linalg.svd(M, full_matrices=False)
    ncomp = u['nfit'] - 1
    uu, sing, vv = uu[:, :ncomp], sing[:ncomp], vv[:ncomp]
    return y, uu, sing, vv


def _compute_linear_parameters(mu, u):
    """Compute the best-fitting linear parameters."""
    y, uu, sing, vv = _compose_linear_fitting_data(mu, u)

    # Compute the residuals
    vec = np.dot(y, uu)
    resi = y - np.dot(uu, vec)
    vec /= sing

    lambda_ = np.zeros(u['nfit'])
    lambda_[1:] = np.dot(vec, vv)
    lambda_[0] = u['fn'][0] - np.sum(lambda_[1:])
    rv = np.dot(resi, resi) / np.dot(y, y)
    return rv, lambda_


def _one_step(mu, u):
    """Evaluate the residual sum of squares fit for one set of mu values."""
    if np.abs(mu).max() >= 1.0:
        return 100.0

    # Compose the data for the linear fitting, compute SVD, then residuals
    y, uu, sing, vv = _compose_linear_fitting_data(mu, u)
    resi = y - np.dot(uu, np.dot(y, uu))
    return np.dot(resi, resi)


def _fwd_eeg_fit_berg_scherg(m, nterms, nfit):
    """Fit the Berg-Scherg equivalent spherical model dipole parameters."""
    from scipy.optimize import fmin_cobyla
    assert nfit >= 2
    u = dict(nfit=nfit, nterms=nterms)

    # (1) Calculate the coefficients of the true expansion
    u['fn'] = _fwd_eeg_get_multi_sphere_model_coeffs(m, nterms + 1)

    # (2) Calculate the weighting
    f = (min([layer['rad'] for layer in m['layers']]) /
         max([layer['rad'] for layer in m['layers']]))

    # correct weighting
    k = np.arange(1, nterms + 1)
    u['w'] = np.sqrt((2.0 * k + 1) * (3.0 * k + 1.0) /
                     k) * np.power(f, (k - 1.0))
    u['w'][-1] = 0

    # Do the nonlinear minimization, constraining mu to the interval [-1, +1]
    mu_0 = np.zeros(3)
    fun = partial(_one_step, u=u)
    catol = 1e-6
    max_ = 1. - 2 * catol

    def cons(x):
        return max_ - np.abs(x)

    mu = fmin_cobyla(fun, mu_0, [cons], rhobeg=0.5, rhoend=1e-5, catol=catol)

    # (6) Do the final step: calculation of the linear parameters
    rv, lambda_ = _compute_linear_parameters(mu, u)
    order = np.argsort(mu)[::-1]
    mu, lambda_ = mu[order], lambda_[order]  # sort: largest mu first

    m['mu'] = mu
    # This division takes into account the actual conductivities
    m['lambda'] = lambda_ / m['layers'][-1]['sigma']
    m['nfit'] = nfit
    return rv


@verbose
def make_sphere_model(r0=(0., 0., 0.04), head_radius=0.09, info=None,
                      relative_radii=(0.90, 0.92, 0.97, 1.0),
                      sigmas=(0.33, 1.0, 0.004, 0.33), verbose=None):
    """Create a spherical model for forward solution calculation.

    Parameters
    ----------
    r0 : array-like | str
        Head center to use (in head coordinates). If 'auto', the head
        center will be calculated from the digitization points in info.
    head_radius : float | str | None
        If float, compute spherical shells for EEG using the given radius.
        If 'auto', estimate an appropriate radius from the dig points in Info,
        If None, exclude shells (single layer sphere model).
    %(info)s Only needed if ``r0`` or ``head_radius`` are ``'auto'``.
    relative_radii : array-like
        Relative radii for the spherical shells.
    sigmas : array-like
        Sigma values for the spherical shells.
    %(verbose)s

    Returns
    -------
    sphere : instance of ConductorModel
        The resulting spherical conductor model.

    See Also
    --------
    make_bem_model
    make_bem_solution

    Notes
    -----
    The default model has::

        relative_radii = (0.90, 0.92, 0.97, 1.0)
        sigmas = (0.33, 1.0, 0.004, 0.33)

    These correspond to compartments (with relative radii in ``m`` and
    conductivities σ in ``S/m``) for the brain, CSF, skull, and scalp,
    respectively.

    .. versionadded:: 0.9.0
    """
    for name in ('r0', 'head_radius'):
        param = locals()[name]
        if isinstance(param, str):
            if param != 'auto':
                raise ValueError('%s, if str, must be "auto" not "%s"'
                                 % (name, param))
    relative_radii = np.array(relative_radii, float).ravel()
    sigmas = np.array(sigmas, float).ravel()
    if len(relative_radii) != len(sigmas):
        raise ValueError('relative_radii length (%s) must match that of '
                         'sigmas (%s)' % (len(relative_radii),
                                          len(sigmas)))
    if len(sigmas) <= 1 and head_radius is not None:
        raise ValueError('at least 2 sigmas must be supplied if '
                         'head_radius is not None, got %s' % (len(sigmas),))
    if (isinstance(r0, str) and r0 == 'auto') or \
       (isinstance(head_radius, str) and head_radius == 'auto'):
        if info is None:
            raise ValueError('Info must not be None for auto mode')
        head_radius_fit, r0_fit = fit_sphere_to_headshape(info, units='m')[:2]
        if isinstance(r0, str):
            r0 = r0_fit
        if isinstance(head_radius, str):
            head_radius = head_radius_fit
    sphere = ConductorModel(is_sphere=True, r0=np.array(r0),
                            coord_frame=FIFF.FIFFV_COORD_HEAD)
    sphere['layers'] = list()
    if head_radius is not None:
        # Eventually these could be configurable...
        relative_radii = np.array(relative_radii, float)
        sigmas = np.array(sigmas, float)
        order = np.argsort(relative_radii)
        relative_radii = relative_radii[order]
        sigmas = sigmas[order]
        for rel_rad, sig in zip(relative_radii, sigmas):
            # sort layers by (relative) radius, and scale radii
            layer = dict(rad=rel_rad, sigma=sig)
            layer['rel_rad'] = layer['rad'] = rel_rad
            sphere['layers'].append(layer)

        # scale the radii
        R = sphere['layers'][-1]['rad']
        rR = sphere['layers'][-1]['rel_rad']
        for layer in sphere['layers']:
            layer['rad'] /= R
            layer['rel_rad'] /= rR

        #
        # Setup the EEG sphere model calculations
        #

        # Scale the relative radii
        for k in range(len(relative_radii)):
            sphere['layers'][k]['rad'] = (head_radius *
                                          sphere['layers'][k]['rel_rad'])
        rv = _fwd_eeg_fit_berg_scherg(sphere, 200, 3)
        logger.info('\nEquiv. model fitting -> RV = %g %%' % (100 * rv))
        for k in range(3):
            logger.info('mu%d = %g    lambda%d = %g'
                        % (k + 1, sphere['mu'][k], k + 1,
                           sphere['layers'][-1]['sigma'] *
                           sphere['lambda'][k]))
        logger.info('Set up EEG sphere model with scalp radius %7.1f mm\n'
                    % (1000 * head_radius,))
    return sphere


# #############################################################################
# Sphere fitting

@verbose
def fit_sphere_to_headshape(info, dig_kinds='auto', units='m', verbose=None):
    """Fit a sphere to the headshape points to determine head center.

    Parameters
    ----------
    %(info_not_none)s
    %(dig_kinds)s
    units : str
        Can be "m" (default) or "mm".

        .. versionadded:: 0.12
    %(verbose)s

    Returns
    -------
    radius : float
        Sphere radius.
    origin_head: ndarray, shape (3,)
        Head center in head coordinates.
    origin_device: ndarray, shape (3,)
        Head center in device coordinates.

    Notes
    -----
    This function excludes any points that are low and frontal
    (``z < 0 and y > 0``) to improve the fit.
    """
    if not isinstance(units, str) or units not in ('m', 'mm'):
        raise ValueError('units must be a "m" or "mm"')
    radius, origin_head, origin_device = _fit_sphere_to_headshape(
        info, dig_kinds)
    if units == 'mm':
        radius *= 1e3
        origin_head *= 1e3
        origin_device *= 1e3
    return radius, origin_head, origin_device


@verbose
def get_fitting_dig(info, dig_kinds='auto', exclude_frontal=True,
                    verbose=None):
    """Get digitization points suitable for sphere fitting.

    Parameters
    ----------
    %(info_not_none)s
    %(dig_kinds)s
    %(exclude_frontal)s
        Default is True.

        .. versionadded:: 0.19
    %(verbose)s

    Returns
    -------
    dig : array, shape (n_pts, 3)
        The digitization points (in head coordinates) to use for fitting.

    Notes
    -----
    This will exclude digitization locations that have ``z < 0 and y > 0``,
    i.e. points on the nose and below the nose on the face.

    .. versionadded:: 0.14
    """
    _validate_type(info, "info")
    if info['dig'] is None:
        raise RuntimeError('Cannot fit headshape without digitization '
                           ', info["dig"] is None')
    if isinstance(dig_kinds, str):
        if dig_kinds == 'auto':
            # try "extra" first
            try:
                return get_fitting_dig(info, 'extra')
            except ValueError:
                pass
            return get_fitting_dig(info, ('extra', 'eeg'))
        else:
            dig_kinds = (dig_kinds,)
    # convert string args to ints (first make dig_kinds mutable in case tuple)
    dig_kinds = list(dig_kinds)
    for di, d in enumerate(dig_kinds):
        dig_kinds[di] = _dig_kind_dict.get(d, d)
        if dig_kinds[di] not in _dig_kind_ints:
            raise ValueError('dig_kinds[#%d] (%s) must be one of %s'
                             % (di, d, sorted(list(_dig_kind_dict.keys()))))

    # get head digization points of the specified kind(s)
    dig = [p for p in info['dig'] if p['kind'] in dig_kinds]
    if len(dig) == 0:
        raise ValueError(
            f'No digitization points found for dig_kinds={dig_kinds}')
    if any(p['coord_frame'] != FIFF.FIFFV_COORD_HEAD for p in dig):
        raise RuntimeError(
            f'Digitization points dig_kinds={dig_kinds} not in head '
            'coordinates, contact mne-python developers')
    hsp = [p['r'] for p in dig]
    del dig

    # exclude some frontal points (nose etc.)
    if exclude_frontal:
        hsp = [p for p in hsp if not (p[2] < -1e-6 and p[1] > 1e-6)]
    hsp = np.array(hsp)

    if len(hsp) <= 10:
        kinds_str = ', '.join(['"%s"' % _dig_kind_rev[d]
                               for d in sorted(dig_kinds)])
        msg = ('Only %s head digitization points of the specified kind%s (%s,)'
               % (len(hsp), _pl(dig_kinds), kinds_str))
        if len(hsp) < 4:
            raise ValueError(msg + ', at least 4 required')
        else:
            warn(msg + ', fitting may be inaccurate')
    return hsp


@verbose
def _fit_sphere_to_headshape(info, dig_kinds, verbose=None):
    """Fit a sphere to the given head shape."""
    hsp = get_fitting_dig(info, dig_kinds)
    radius, origin_head = _fit_sphere(np.array(hsp), disp=False)
    # compute origin in device coordinates
    dev_head_t = info['dev_head_t']
    if dev_head_t is None:
        dev_head_t = Transform('meg', 'head')
    head_to_dev = _ensure_trans(dev_head_t, 'head', 'meg')
    origin_device = apply_trans(head_to_dev, origin_head)
    logger.info('Fitted sphere radius:'.ljust(30) + '%0.1f mm'
                % (radius * 1e3,))
    # 99th percentile on Wikipedia for Giabella to back of head is 21.7cm,
    # i.e. 108mm "radius", so let's go with 110mm
    # en.wikipedia.org/wiki/Human_head#/media/File:HeadAnthropometry.JPG
    if radius > 0.110:
        warn('Estimated head size (%0.1f mm) exceeded 99th '
             'percentile for adult head size' % (1e3 * radius,))
    # > 2 cm away from head center in X or Y is strange
    if np.linalg.norm(origin_head[:2]) > 0.02:
        warn('(X, Y) fit (%0.1f, %0.1f) more than 20 mm from '
             'head frame origin' % tuple(1e3 * origin_head[:2]))
    logger.info('Origin head coordinates:'.ljust(30) +
                '%0.1f %0.1f %0.1f mm' % tuple(1e3 * origin_head))
    logger.info('Origin device coordinates:'.ljust(30) +
                '%0.1f %0.1f %0.1f mm' % tuple(1e3 * origin_device))
    return radius, origin_head, origin_device


def _fit_sphere(points, disp='auto'):
    """Fit a sphere to an arbitrary set of points."""
    from scipy.optimize import fmin_cobyla
    if isinstance(disp, str) and disp == 'auto':
        disp = True if logger.level <= 20 else False
    # initial guess for center and radius
    radii = (np.max(points, axis=1) - np.min(points, axis=1)) / 2.
    radius_init = radii.mean()
    center_init = np.median(points, axis=0)

    # optimization
    x0 = np.concatenate([center_init, [radius_init]])

    def cost_fun(center_rad):
        d = np.linalg.norm(points - center_rad[:3], axis=1) - center_rad[3]
        d *= d
        return d.sum()

    def constraint(center_rad):
        return center_rad[3]  # radius must be >= 0

    x_opt = fmin_cobyla(cost_fun, x0, constraint, rhobeg=radius_init,
                        rhoend=radius_init * 1e-6, disp=disp)

    origin, radius = x_opt[:3], x_opt[3]
    return radius, origin


def _check_origin(origin, info, coord_frame='head', disp=False):
    """Check or auto-determine the origin."""
    if isinstance(origin, str):
        if origin != 'auto':
            raise ValueError('origin must be a numerical array, or "auto", '
                             'not %s' % (origin,))
        if coord_frame == 'head':
            R, origin = fit_sphere_to_headshape(
                info, verbose=_verbose_safe_false(), units='m')[:2]
            logger.info('    Automatic origin fit: head of radius %0.1f mm'
                        % (R * 1000.,))
            del R
        else:
            origin = (0., 0., 0.)
    origin = np.array(origin, float)
    if origin.shape != (3,):
        raise ValueError('origin must be a 3-element array')
    if disp:
        origin_str = ', '.join(['%0.1f' % (o * 1000) for o in origin])
        msg = ('    Using origin %s mm in the %s frame'
               % (origin_str, coord_frame))
        if coord_frame == 'meg' and info['dev_head_t'] is not None:
            o_dev = apply_trans(info['dev_head_t'], origin)
            origin_str = ', '.join('%0.1f' % (o * 1000,) for o in o_dev)
            msg += ' (%s mm in the head frame)' % (origin_str,)
        logger.info(msg)
    return origin


# ############################################################################
# Create BEM surfaces

@verbose
def make_watershed_bem(subject, subjects_dir=None, overwrite=False,
                       volume='T1', atlas=False, gcaatlas=False, preflood=None,
                       show=False, copy=True, T1=None, brainmask='ws.mgz',
                       verbose=None):
    """Create BEM surfaces using the FreeSurfer watershed algorithm.

    Parameters
    ----------
    subject : str
        Subject name.
    %(subjects_dir)s
    %(overwrite)s
    volume : str
        Defaults to T1.
    atlas : bool
        Specify the --atlas option for mri_watershed.
    gcaatlas : bool
        Specify the --brain_atlas option for mri_watershed.
    preflood : int
        Change the preflood height.
    show : bool
        Show surfaces to visually inspect all three BEM surfaces (recommended).

        .. versionadded:: 0.12

    copy : bool
        If True (default), use copies instead of symlinks for surfaces
        (if they do not already exist).

        .. versionadded:: 0.18
        .. versionchanged:: 1.1 Use copies instead of symlinks.
    T1 : bool | None
        If True, pass the ``-T1`` flag.
        By default (None), this takes the same value as ``gcaatlas``.

        .. versionadded:: 0.19
    brainmask : str
        The filename for the brainmask output file relative to the
        ``$SUBJECTS_DIR/$SUBJECT/bem/watershed/`` directory.
        Can be for example ``"../../mri/brainmask.mgz"`` to overwrite
        the brainmask obtained via ``recon-all -autorecon1``.

        .. versionadded:: 0.19
    %(verbose)s

    See Also
    --------
    mne.viz.plot_bem

    Notes
    -----
    If your BEM meshes do not look correct when viewed in
    :func:`mne.viz.plot_alignment` or :func:`mne.viz.plot_bem`, consider
    potential solutions from the :ref:`FAQ <faq_watershed_bem_meshes>`.

    .. versionadded:: 0.10
    """
    from .viz.misc import plot_bem
    env, mri_dir, bem_dir = _prepare_env(subject, subjects_dir)
    tempdir = _TempDir()  # fsl and Freesurfer create some random junk in CWD
    run_subprocess_env = partial(run_subprocess, env=env,
                                 cwd=tempdir)

    subjects_dir = env['SUBJECTS_DIR']  # Set by _prepare_env() above.
    subject_dir = op.join(subjects_dir, subject)
    ws_dir = op.join(bem_dir, 'watershed')
    T1_dir = op.join(mri_dir, volume)
    T1_mgz = T1_dir
    if not T1_dir.endswith('.mgz'):
        T1_mgz += '.mgz'

    if not op.isdir(bem_dir):
        os.makedirs(bem_dir)
    _check_fname(T1_mgz, overwrite='read', must_exist=True, name='MRI data')
    if op.isdir(ws_dir):
        if not overwrite:
            raise RuntimeError('%s already exists. Use the --overwrite option'
                               ' to recreate it.' % ws_dir)
        else:
            shutil.rmtree(ws_dir)

    # put together the command
    cmd = ['mri_watershed']
    if preflood:
        cmd += ["-h", "%s" % int(preflood)]

    if T1 is None:
        T1 = gcaatlas
    if T1:
        cmd += ['-T1']
    if gcaatlas:
        fname = op.join(env['FREESURFER_HOME'], 'average',
                        'RB_all_withskull_*.gca')
        fname = sorted(glob.glob(fname))[::-1][0]
        logger.info('Using GCA atlas: %s' % (fname,))
        cmd += ['-atlas', '-brain_atlas', fname,
                subject_dir + '/mri/transforms/talairach_with_skull.lta']
    elif atlas:
        cmd += ['-atlas']
    if op.exists(T1_mgz):
        cmd += ['-useSRAS', '-surf', op.join(ws_dir, subject), T1_mgz,
                op.join(ws_dir, brainmask)]
    else:
        cmd += ['-useSRAS', '-surf', op.join(ws_dir, subject), T1_dir,
                op.join(ws_dir, brainmask)]
    # report and run
    logger.info('\nRunning mri_watershed for BEM segmentation with the '
                'following parameters:\n\nResults dir = %s\nCommand = %s\n'
                % (ws_dir, ' '.join(cmd)))
    os.makedirs(op.join(ws_dir))
    run_subprocess_env(cmd)
    del tempdir  # clean up directory
    if op.isfile(T1_mgz):
        new_info = _extract_volume_info(T1_mgz) if has_nibabel() else dict()
        if not new_info:
            warn('nibabel is not available or the volume info is invalid.'
                 'Volume info not updated in the written surface.')
        surfs = ['brain', 'inner_skull', 'outer_skull', 'outer_skin']
        for s in surfs:
            surf_ws_out = op.join(ws_dir, '%s_%s_surface' % (subject, s))

            rr, tris, volume_info = read_surface(surf_ws_out,
                                                 read_metadata=True)
            # replace volume info, 'head' stays
            volume_info.update(new_info)
            write_surface(surf_ws_out, rr, tris, volume_info=volume_info,
                          overwrite=True)

            # Create symbolic links
            surf_out = op.join(bem_dir, '%s.surf' % s)
            if not overwrite and op.exists(surf_out):
                skip_symlink = True
            else:
                if op.exists(surf_out):
                    os.remove(surf_out)
                _symlink(surf_ws_out, surf_out, copy)
                skip_symlink = False

        if skip_symlink:
            logger.info("Unable to create all symbolic links to .surf files "
                        "in bem folder. Use --overwrite option to recreate "
                        "them.")
            dest = op.join(bem_dir, 'watershed')
        else:
            logger.info("Symbolic links to .surf files created in bem folder")
            dest = bem_dir

    logger.info("\nThank you for waiting.\nThe BEM triangulations for this "
                "subject are now available at:\n%s." % dest)

    # Write a head file for coregistration
    fname_head = op.join(bem_dir, subject + '-head.fif')
    if op.isfile(fname_head):
        os.remove(fname_head)

    surf = _surfaces_to_bem([op.join(ws_dir, subject + '_outer_skin_surface')],
                            [FIFF.FIFFV_BEM_SURF_ID_HEAD], sigmas=[1])
    write_bem_surfaces(fname_head, surf)

    # Show computed BEM surfaces
    if show:
        plot_bem(subject=subject, subjects_dir=subjects_dir,
                 orientation='coronal', slices=None, show=True)

    logger.info('Created %s\n\nComplete.' % (fname_head,))


def _extract_volume_info(mgz):
    """Extract volume info from a mgz file."""
    import nibabel
    header = nibabel.load(mgz).header
    version = header['version']
    vol_info = dict()
    if version == 1:
        version = '%s  # volume info valid' % version
        vol_info['valid'] = version
        vol_info['filename'] = mgz
        vol_info['volume'] = header['dims'][:3]
        vol_info['voxelsize'] = header['delta']
        vol_info['xras'], vol_info['yras'], vol_info['zras'] = header['Mdc']
        vol_info['cras'] = header['Pxyz_c']

    return vol_info


# ############################################################################
# Read

@verbose
def read_bem_surfaces(fname, patch_stats=False, s_id=None, on_defects='raise',
                      verbose=None):
    """Read the BEM surfaces from a FIF file.

    Parameters
    ----------
    fname : str
        The name of the file containing the surfaces.
    patch_stats : bool, optional (default False)
        Calculate and add cortical patch statistics to the surfaces.
    s_id : int | None
        If int, only read and return the surface with the given s_id.
        An error will be raised if it doesn't exist. If None, all
        surfaces are read and returned.
    %(on_defects)s

        .. versionadded:: 0.23
    %(verbose)s

    Returns
    -------
    surf: list | dict
        A list of dictionaries that each contain a surface. If s_id
        is not None, only the requested surface will be returned.

    See Also
    --------
    write_bem_surfaces, write_bem_solution, make_bem_model
    """
    # Open the file, create directory
    _validate_type(s_id, ('int-like', None), 's_id')
    fname = _check_fname(fname, 'read', True, 'fname')
    if fname.endswith('.h5'):
        surf = _read_bem_surfaces_h5(fname, s_id)
    else:
        surf = _read_bem_surfaces_fif(fname, s_id)
    if s_id is not None and len(surf) != 1:
        raise ValueError('surface with id %d not found' % s_id)
    for this in surf:
        if patch_stats or this['nn'] is None:
            _check_complete_surface(this, incomplete=on_defects)
    return surf[0] if s_id is not None else surf


def _read_bem_surfaces_h5(fname, s_id):
    read_hdf5, _ = _import_h5io_funcs()
    bem = read_hdf5(fname)
    try:
        [s['id'] for s in bem['surfs']]
    except Exception:  # not our format
        raise ValueError('BEM data not found')
    surf = bem['surfs']
    if s_id is not None:
        surf = [s for s in surf if s['id'] == s_id]
    return surf


def _read_bem_surfaces_fif(fname, s_id):
    # Default coordinate frame
    coord_frame = FIFF.FIFFV_COORD_MRI
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Find BEM
        bem = dir_tree_find(tree, FIFF.FIFFB_BEM)
        if bem is None or len(bem) == 0:
            raise ValueError('BEM data not found')

        bem = bem[0]
        # Locate all surfaces
        bemsurf = dir_tree_find(bem, FIFF.FIFFB_BEM_SURF)
        if bemsurf is None:
            raise ValueError('BEM surface data not found')

        logger.info('    %d BEM surfaces found' % len(bemsurf))
        # Coordinate frame possibly at the top level
        tag = find_tag(fid, bem, FIFF.FIFF_BEM_COORD_FRAME)
        if tag is not None:
            coord_frame = tag.data
        # Read all surfaces
        if s_id is not None:
            surf = [_read_bem_surface(fid, bsurf, coord_frame, s_id)
                    for bsurf in bemsurf]
            surf = [s for s in surf if s is not None]
        else:
            surf = list()
            for bsurf in bemsurf:
                logger.info('    Reading a surface...')
                this = _read_bem_surface(fid, bsurf, coord_frame)
                surf.append(this)
                logger.info('[done]')
            logger.info('    %d BEM surfaces read' % len(surf))
    return surf


def _read_bem_surface(fid, this, def_coord_frame, s_id=None):
    """Read one bem surface."""
    # fid should be open as a context manager here
    res = dict()
    # Read all the interesting stuff
    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_ID)

    if tag is None:
        res['id'] = FIFF.FIFFV_BEM_SURF_ID_UNKNOWN
    else:
        res['id'] = int(tag.data)

    if s_id is not None and res['id'] != s_id:
        return None

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SIGMA)
    res['sigma'] = 1.0 if tag is None else float(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NNODE)
    if tag is None:
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NTRI)
    if tag is None:
        raise ValueError('Number of triangles not found')
    res['ntri'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        tag = find_tag(fid, this, FIFF.FIFF_BEM_COORD_FRAME)
        if tag is None:
            res['coord_frame'] = def_coord_frame
        else:
            res['coord_frame'] = tag.data
    else:
        res['coord_frame'] = tag.data

    # Vertices, normals, and triangles
    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NODES)
    if tag is None:
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float64)
    if res['rr'].shape[0] != res['np']:
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NORMALS)
    if tag is None:
        res['nn'] = None
    else:
        res['nn'] = tag.data.astype(np.float64)
        if res['nn'].shape[0] != res['np']:
            raise ValueError('Vertex normal information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_TRIANGLES)
    if tag is None:
        raise ValueError('Triangulation not found')

    res['tris'] = tag.data - 1  # index start at 0 in Python
    if res['tris'].shape[0] != res['ntri']:
        raise ValueError('Triangulation information is incorrect')

    return res


@verbose
def read_bem_solution(fname, *, verbose=None):
    """Read the BEM solution from a file.

    Parameters
    ----------
    fname : str
        The file containing the BEM solution.
    %(verbose)s

    Returns
    -------
    bem : instance of ConductorModel
        The BEM solution.

    See Also
    --------
    read_bem_surfaces
    write_bem_surfaces
    make_bem_solution
    write_bem_solution
    """
    fname = _check_fname(fname, 'read', True, 'fname')
    # mirrors fwd_bem_load_surfaces from fwd_bem_model.c
    if fname.endswith('.h5'):
        read_hdf5, _ = _import_h5io_funcs()
        logger.info('Loading surfaces and solution...')
        bem = read_hdf5(fname)
        if 'solver' not in bem:
            bem['solver'] = 'mne'
    else:
        bem = _read_bem_solution_fif(fname)

    if len(bem['surfs']) == 3:
        logger.info('Three-layer model surfaces loaded.')
        needed = np.array([FIFF.FIFFV_BEM_SURF_ID_HEAD,
                           FIFF.FIFFV_BEM_SURF_ID_SKULL,
                           FIFF.FIFFV_BEM_SURF_ID_BRAIN])
        if not all(x['id'] in needed for x in bem['surfs']):
            raise RuntimeError('Could not find necessary BEM surfaces')
        # reorder surfaces as necessary (shouldn't need to?)
        reorder = [None] * 3
        for x in bem['surfs']:
            reorder[np.where(x['id'] == needed)[0][0]] = x
        bem['surfs'] = reorder
    elif len(bem['surfs']) == 1:
        if not bem['surfs'][0]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN:
            raise RuntimeError('BEM Surfaces not found')
        logger.info('Homogeneous model surface loaded.')

    assert set(bem.keys()) == set(
        ('surfs', 'solution', 'bem_method', 'solver'))
    bem = ConductorModel(bem)
    bem['is_sphere'] = False
    # sanity checks and conversions
    _check_option(
        'BEM approximation method', bem['bem_method'],
        (FIFF.FIFFV_BEM_APPROX_LINEAR,))  # CONSTANT not supported
    dim = 0
    solver = bem.get('solver', 'mne')
    _check_option('BEM solver', solver, ('mne', 'openmeeg'))
    for si, surf in enumerate(bem['surfs']):
        assert bem['bem_method'] == FIFF.FIFFV_BEM_APPROX_LINEAR
        dim += surf['np']
        if solver == 'openmeeg' and si != 0:
            dim += surf['ntri']
    dims = bem['solution'].shape
    if solver == "openmeeg":
        sz = (dim * (dim + 1)) // 2
        if len(dims) != 1 or dims[0] != sz:
            raise RuntimeError(
                'For the given BEM surfaces, OpenMEEG should produce a '
                f'solution matrix of shape ({sz},) but got {dims}')
        bem['nsol'] = dim
    else:
        if len(dims) != 2 and solver != "openmeeg":
            raise RuntimeError('Expected a two-dimensional solution matrix '
                               'instead of a %d dimensional one' % dims[0])
        if dims[0] != dim or dims[1] != dim:
            raise RuntimeError('Expected a %d x %d solution matrix instead of '
                               'a %d x %d one' % (dim, dim, dims[1], dims[0]))
        bem['nsol'] = bem['solution'].shape[0]
    # Gamma factors and multipliers
    _add_gamma_multipliers(bem)
    extra = f'made by {solver}' if solver != 'mne' else ''
    logger.info(f'Loaded linear collocation BEM solution{extra} from {fname}')
    return bem


def _read_bem_solution_fif(fname):
    logger.info('Loading surfaces...')
    surfs = read_bem_surfaces(
        fname, patch_stats=True, verbose=_verbose_safe_false())

    # convert from surfaces to solution
    logger.info('\nLoading the solution matrix...\n')
    solver = 'mne'
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Find the BEM data
        nodes = dir_tree_find(tree, FIFF.FIFFB_BEM)
        if len(nodes) == 0:
            raise RuntimeError('No BEM data in %s' % fname)
        bem_node = nodes[0]

        # Approximation method
        tag = find_tag(f, bem_node, FIFF.FIFF_DESCRIPTION)
        if tag is not None:
            tag = json.loads(tag.data)
            solver = tag['solver']
        tag = find_tag(f, bem_node, FIFF.FIFF_BEM_APPROX)
        if tag is None:
            raise RuntimeError('No BEM solution found in %s' % fname)
        method = tag.data[0]
        tag = find_tag(fid, bem_node, FIFF.FIFF_BEM_POT_SOLUTION)
        sol = tag.data

    return dict(solution=sol, bem_method=method, surfs=surfs, solver=solver)


def _add_gamma_multipliers(bem):
    """Add gamma and multipliers in-place."""
    bem['sigma'] = np.array([surf['sigma'] for surf in bem['surfs']])
    # Dirty trick for the zero conductivity outside
    sigma = np.r_[0.0, bem['sigma']]
    bem['source_mult'] = 2.0 / (sigma[1:] + sigma[:-1])
    bem['field_mult'] = sigma[1:] - sigma[:-1]
    # make sure subsequent "zip"s work correctly
    assert len(bem['surfs']) == len(bem['field_mult'])
    bem['gamma'] = ((sigma[1:] - sigma[:-1])[np.newaxis, :] /
                    (sigma[1:] + sigma[:-1])[:, np.newaxis])


# In our BEM code we do not model the CSF so we assign the innermost surface
# the id BRAIN. Our 4-layer sphere we model CSF (at least by default), so when
# searching for and referring to surfaces we need to keep track of this.
_sm_surf_dict = OrderedDict([
    ('brain', FIFF.FIFFV_BEM_SURF_ID_BRAIN),
    ('inner_skull', FIFF.FIFFV_BEM_SURF_ID_CSF),
    ('outer_skull', FIFF.FIFFV_BEM_SURF_ID_SKULL),
    ('head', FIFF.FIFFV_BEM_SURF_ID_HEAD),
])
_bem_surf_dict = {
    'inner_skull': FIFF.FIFFV_BEM_SURF_ID_BRAIN,
    'outer_skull': FIFF.FIFFV_BEM_SURF_ID_SKULL,
    'head': FIFF.FIFFV_BEM_SURF_ID_HEAD,
}
_bem_surf_name = {
    FIFF.FIFFV_BEM_SURF_ID_BRAIN: 'inner skull',
    FIFF.FIFFV_BEM_SURF_ID_SKULL: 'outer skull',
    FIFF.FIFFV_BEM_SURF_ID_HEAD: 'outer skin ',
    FIFF.FIFFV_BEM_SURF_ID_UNKNOWN: 'unknown    ',
}
_sm_surf_name = {
    FIFF.FIFFV_BEM_SURF_ID_BRAIN: 'brain',
    FIFF.FIFFV_BEM_SURF_ID_CSF: 'csf',
    FIFF.FIFFV_BEM_SURF_ID_SKULL: 'outer skull',
    FIFF.FIFFV_BEM_SURF_ID_HEAD: 'outer skin ',
    FIFF.FIFFV_BEM_SURF_ID_UNKNOWN: 'unknown    ',
}


def _bem_find_surface(bem, id_):
    """Find surface from already-loaded conductor model."""
    if bem['is_sphere']:
        _surf_dict = _sm_surf_dict
        _name_dict = _sm_surf_name
        kind = 'Sphere model'
        tri = 'boundary'
    else:
        _surf_dict = _bem_surf_dict
        _name_dict = _bem_surf_name
        kind = 'BEM'
        tri = 'triangulation'
    if isinstance(id_, str):
        name = id_
        id_ = _surf_dict[id_]
    else:
        name = _name_dict[id_]
    kind = 'Sphere model' if bem['is_sphere'] else 'BEM'
    idx = np.where(np.array([s['id'] for s in bem['surfs']]) == id_)[0]
    if len(idx) != 1:
        raise RuntimeError(f'{kind} does not have the {name} {tri}')
    return bem['surfs'][idx[0]]


# ############################################################################
# Write

@verbose
def write_bem_surfaces(fname, surfs, overwrite=False, *, verbose=None):
    """Write BEM surfaces to a fiff file.

    Parameters
    ----------
    fname : str
        Filename to write. Can end with ``.h5`` to write using HDF5.
    surfs : dict | list of dict
        The surfaces, or a single surface.
    %(overwrite)s
    %(verbose)s
    """
    if isinstance(surfs, dict):
        surfs = [surfs]
    fname = _check_fname(fname, overwrite=overwrite, name='fname')

    if fname.endswith('.h5'):
        _, write_hdf5 = _import_h5io_funcs()
        write_hdf5(fname, dict(surfs=surfs), overwrite=True)
    else:
        with start_and_end_file(fname) as fid:
            start_block(fid, FIFF.FIFFB_BEM)
            write_int(fid, FIFF.FIFF_BEM_COORD_FRAME, surfs[0]['coord_frame'])
            _write_bem_surfaces_block(fid, surfs)
            end_block(fid, FIFF.FIFFB_BEM)


@verbose
def write_head_bem(fname, rr, tris, on_defects='raise', overwrite=False,
                   *, verbose=None):
    """Write a head surface to a fiff file.

    Parameters
    ----------
    fname : str
        Filename to write.
    rr : array, shape (n_vertices, 3)
        Coordinate points in the MRI coordinate system.
    tris : ndarray of int, shape (n_tris, 3)
        Triangulation (each line contains indices for three points which
        together form a face).
    %(on_defects)s
    %(overwrite)s
    %(verbose)s
    """
    surf = _surfaces_to_bem([dict(rr=rr, tris=tris)],
                            [FIFF.FIFFV_BEM_SURF_ID_HEAD], [1], rescale=False,
                            incomplete=on_defects)
    write_bem_surfaces(fname, surf, overwrite=overwrite)


def _write_bem_surfaces_block(fid, surfs):
    """Write bem surfaces to open file handle."""
    for surf in surfs:
        start_block(fid, FIFF.FIFFB_BEM_SURF)
        write_float(fid, FIFF.FIFF_BEM_SIGMA, surf['sigma'])
        write_int(fid, FIFF.FIFF_BEM_SURF_ID, surf['id'])
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, surf['coord_frame'])
        write_int(fid, FIFF.FIFF_BEM_SURF_NNODE, surf['np'])
        write_int(fid, FIFF.FIFF_BEM_SURF_NTRI, surf['ntri'])
        write_float_matrix(fid, FIFF.FIFF_BEM_SURF_NODES, surf['rr'])
        # index start at 0 in Python
        write_int_matrix(fid, FIFF.FIFF_BEM_SURF_TRIANGLES,
                         surf['tris'] + 1)
        if 'nn' in surf and surf['nn'] is not None and len(surf['nn']) > 0:
            write_float_matrix(fid, FIFF.FIFF_BEM_SURF_NORMALS, surf['nn'])
        end_block(fid, FIFF.FIFFB_BEM_SURF)


@verbose
def write_bem_solution(fname, bem, overwrite=False, *, verbose=None):
    """Write a BEM model with solution.

    Parameters
    ----------
    fname : str
        The filename to use. Can end with ``.h5`` to write using HDF5.
    bem : instance of ConductorModel
        The BEM model with solution to save.
    %(overwrite)s
    %(verbose)s

    See Also
    --------
    read_bem_solution
    """
    fname = _check_fname(fname, overwrite=overwrite, name='fname')
    if fname.endswith('.h5'):
        _, write_hdf5 = _import_h5io_funcs()
        bem = {k: bem[k] for k in ('surfs', 'solution', 'bem_method')}
        write_hdf5(fname, bem, overwrite=True)
    else:
        _write_bem_solution_fif(fname, bem)


def _write_bem_solution_fif(fname, bem):
    _check_bem_size(bem['surfs'])
    with start_and_end_file(fname) as fid:
        start_block(fid, FIFF.FIFFB_BEM)
        # Coordinate frame (mainly for backward compatibility)
        write_int(fid, FIFF.FIFF_BEM_COORD_FRAME,
                  bem['surfs'][0]['coord_frame'])
        solver = bem.get('solver', 'mne')
        if solver != 'mne':
            write_string(
                fid, FIFF.FIFF_DESCRIPTION, json.dumps(dict(solver=solver)))
        # Surfaces
        _write_bem_surfaces_block(fid, bem['surfs'])
        # The potential solution
        if 'solution' in bem:
            _check_option(
                'bem_method', bem['bem_method'],
                (FIFF.FIFFV_BEM_APPROX_LINEAR,))
            write_int(fid, FIFF.FIFF_BEM_APPROX, FIFF.FIFFV_BEM_APPROX_LINEAR)
            write_float_matrix(fid, FIFF.FIFF_BEM_POT_SOLUTION,
                               bem['solution'])
        end_block(fid, FIFF.FIFFB_BEM)


# #############################################################################
# Create 3-Layers BEM model from Flash MRI images

def _prepare_env(subject, subjects_dir):
    """Prepare an env object for subprocess calls."""
    env = os.environ.copy()

    fs_home = _check_freesurfer_home()

    _validate_type(subject, "str")

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subjects_dir = op.abspath(subjects_dir)  # force use of an absolute path
    subjects_dir = op.expanduser(subjects_dir)
    if not op.isdir(subjects_dir):
        raise RuntimeError('Could not find the MRI data directory "%s"'
                           % subjects_dir)
    subject_dir = op.join(subjects_dir, subject)
    if not op.isdir(subject_dir):
        raise RuntimeError('Could not find the subject data directory "%s"'
                           % (subject_dir,))
    env.update(SUBJECT=subject, SUBJECTS_DIR=subjects_dir,
               FREESURFER_HOME=fs_home)
    mri_dir = op.join(subject_dir, 'mri')
    bem_dir = op.join(subject_dir, 'bem')
    return env, mri_dir, bem_dir


def _write_echos(mri_dir, flash_echos, angle):
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage
    if _path_like(flash_echos):
        flash_echos = nib.load(flash_echos)
    if isinstance(flash_echos, SpatialImage):
        flash_echo_imgs = []
        data = np.asanyarray(flash_echos.dataobj)
        affine = flash_echos.affine
        if data.ndim == 3:
            data = data[..., np.newaxis]
        for echo_idx in range(data.shape[3]):
            this_echo_img = flash_echos.__class__(
                data[..., echo_idx], affine=affine,
                header=deepcopy(flash_echos.header)
            )
            flash_echo_imgs.append(this_echo_img)
        flash_echos = flash_echo_imgs
        del flash_echo_imgs
    for idx, flash_echo in enumerate(flash_echos, 1):
        if _path_like(flash_echo):
            flash_echo = nib.load(flash_echo)
        nib.save(flash_echo,
                 op.join(mri_dir, 'flash', f'mef{angle}_{idx:03d}.mgz'))


@verbose
def convert_flash_mris(subject, flash30=True, unwarp=False,
                       subjects_dir=None, flash5=True, verbose=None):
    """Synthesize the flash 5 files for use with make_flash_bem.

    This function aims to produce a synthesized flash 5 MRI from
    multiecho flash (MEF) MRI data. This function can use MEF data
    with 5 or 30 flip angles. If flash5 (and flash30) images are not
    explicitly provided, it will assume that the different echos are available
    in the mri/flash folder of the subject with the following naming
    convention "mef<angle>_<echo>.mgz", e.g. "mef05_001.mgz"
    or "mef30_001.mgz".

    Parameters
    ----------
    subject : str
        Subject name.
    flash30 : bool | list of SpatialImage or path-like | SpatialImage | path-like
        If False do not use 30-degree flip angle data.
        The list of flash 5 echos to use. If True it will look for files
        named mef30_*.mgz in the subject's mri/flash directory and if not False
        the list of flash 5 echos images will be written to the mri/flash
        folder with convention mef05_<echo>.mgz. If a SpatialImage object
        each frame of the image will be interpreted as an echo.
    unwarp : bool
        Run grad_unwarp with -unwarp option on each of the converted
        data sets. It requires FreeSurfer's MATLAB toolbox to be properly
        installed.
    %(subjects_dir)s
    flash5 : list of SpatialImage or path-like | SpatialImage | path-like | True
        The list of flash 5 echos to use. If True it will look for files
        named mef05_*.mgz in the subject's mri/flash directory and if not None
        the list of flash 5 echos images will be written to the mri/flash
        folder with convention mef05_<echo>.mgz. If a SpatialImage object
        each frame of the image will be interpreted as an echo.
    %(verbose)s

    Returns
    -------
    flash5_img : path-like
        The path the synthesized flash 5 MRI.

    Notes
    -----
    This function assumes that the Freesurfer segmentation of the subject
    has been completed. In particular, the T1.mgz and brain.mgz MRI volumes
    should be, as usual, in the subject's mri directory.
    """  # noqa: E501
    env, mri_dir = _prepare_env(subject, subjects_dir)[:2]
    tempdir = _TempDir()  # fsl and Freesurfer create some random junk in CWD
    run_subprocess_env = partial(run_subprocess, env=env,
                                 cwd=tempdir)

    mri_dir = Path(mri_dir)
    # Step 1a : Data conversion to mgz format
    flash_dir = mri_dir / "flash"
    pm_dir = flash_dir / 'parameter_maps'
    pm_dir.mkdir(parents=True, exist_ok=True)
    echos_done = 0

    if not isinstance(flash5, bool):
        _write_echos(mri_dir, flash5, angle='05')
    if not isinstance(flash30, bool):
        _write_echos(mri_dir, flash30, angle='30')

    # Step 1b : Run grad_unwarp on converted files
    template = op.join(flash_dir, "mef*_*.mgz")
    files = sorted(glob.glob(template))
    if len(files) == 0:
        raise ValueError('No suitable source files found (%s)' % template)
    if unwarp:
        logger.info("\n---- Unwarp mgz data sets ----")
        for infile in files:
            outfile = infile.replace(".mgz", "u.mgz")
            cmd = ['grad_unwarp', '-i', infile, '-o', outfile, '-unwarp',
                   'true']
            run_subprocess_env(cmd)
    # Clear parameter maps if some of the data were reconverted
    if echos_done > 0 and pm_dir.exists():
        shutil.rmtree(pm_dir)
        logger.info("\nParameter maps directory cleared")
    if not pm_dir.exists():
        pm_dir.mkdir(parents=True, exist_ok=True)
    # Step 2 : Create the parameter maps
    if flash30:
        logger.info("\n---- Creating the parameter maps ----")
        if unwarp:
            files = sorted(glob.glob(op.join(flash_dir, "mef05_*u.mgz")))
        if len(os.listdir(pm_dir)) == 0:
            cmd = (['mri_ms_fitparms'] + files + [str(pm_dir)])
            run_subprocess_env(cmd)
        else:
            logger.info("Parameter maps were already computed")
        # Step 3 : Synthesize the flash 5 images
        logger.info("\n---- Synthesizing flash 5 images ----")
        if not (pm_dir / 'flash5.mgz').exists():
            cmd = ['mri_synthesize', '20', '5', '5',
                   (pm_dir / 'T1.mgz'),
                   (pm_dir / 'PD.mgz'),
                   (pm_dir / 'flash5.mgz')
                   ]
            run_subprocess_env(cmd)
            (pm_dir / 'flash5_reg.mgz').unlink()
        else:
            logger.info("Synthesized flash 5 volume is already there")
    else:
        logger.info("\n---- Averaging flash5 echoes ----")
        template = "mef05_*u.mgz" if unwarp else "mef05_*.mgz"
        files = sorted(flash_dir.glob(template))
        if len(files) == 0:
            raise ValueError('No suitable source files found (%s)' % template)
        cmd = (['mri_average', '-noconform'] + files + [pm_dir / 'flash5.mgz'])
        run_subprocess_env(cmd)
        (pm_dir / 'flash5_reg.mgz').unlink(missing_ok=True)
    del tempdir  # finally done running subprocesses
    assert (pm_dir / 'flash5.mgz').exists()
    return pm_dir / 'flash5.mgz'


@verbose
def make_flash_bem(subject, overwrite=False, show=True, subjects_dir=None,
                   copy=True, *, flash5_img=None, register=True, verbose=None):
    """Create 3-Layer BEM model from prepared flash MRI images.

    Parameters
    ----------
    subject : str
        Subject name.
    overwrite : bool
        Write over existing .surf files in bem folder.
    show : bool
        Show surfaces to visually inspect all three BEM surfaces (recommended).
    %(subjects_dir)s
    copy : bool
        If True (default), use copies instead of symlinks for surfaces
        (if they do not already exist).

        .. versionadded:: 0.18
        .. versionchanged:: 1.1 Use copies instead of symlinks.
    flash5_img : None | path-like | Nifti1Image
        The path to the synthesized flash 5 MRI image or the image itself. If
        None (default), the path defaults to
        ``mri/flash/parameter_maps/flash5.mgz`` within the subject
        reconstruction. If not present the image is copied or written to the
        default location.

        .. versionadded:: 1.1.0
    register : bool
        Register the flash 5 image with T1.mgz file. If False, we assume
        that the images are already coregistered.

        .. versionadded:: 1.1.0
    %(verbose)s

    See Also
    --------
    convert_flash_mris

    Notes
    -----
    This program assumes that FreeSurfer is installed and sourced properly.

    This function extracts the BEM surfaces (outer skull, inner skull, and
    outer skin) from a FLASH 5 MRI image synthesized from multiecho FLASH
    images acquired with spin angles of 5 and 30 degrees.
    """
    from .viz.misc import plot_bem

    env, mri_dir, bem_dir = _prepare_env(subject, subjects_dir)
    tempdir = _TempDir()  # fsl and Freesurfer create some random junk in CWD
    run_subprocess_env = partial(run_subprocess, env=env,
                                 cwd=tempdir)

    mri_dir = Path(mri_dir)
    bem_dir = Path(bem_dir)
    subjects_dir = env['SUBJECTS_DIR']
    flash_path = (mri_dir / 'flash' / 'parameter_maps').resolve()
    flash_path.mkdir(exist_ok=True, parents=True)

    logger.info('\nProcessing the flash MRI data to produce BEM meshes with '
                'the following parameters:\n'
                'SUBJECTS_DIR = %s\n'
                'SUBJECT = %s\n'
                'Result dir = %s\n' % (subjects_dir, subject,
                                       bem_dir / 'flash'))
    # Step 4 : Register with MPRAGE
    flash5 = flash_path / 'flash5.mgz'

    if _path_like(flash5_img):
        logger.info(f"Copying flash 5 image {flash5_img} to {flash5}")
        cmd = ['mri_convert', Path(flash5_img).resolve(), flash5]
        run_subprocess_env(cmd)
    elif flash5_img is None:
        if not flash5.exists():
            raise ValueError(f'Flash 5 image cannot be found at {flash5}.')
    else:
        logger.info(f"Writing flash 5 image at {flash5}")
        import nibabel as nib
        nib.save(flash5_img, flash5)

    if register:
        logger.info("\n---- Registering flash 5 with T1 MPRAGE ----")
        flash5_reg = flash_path / 'flash5_reg.mgz'
        if not flash5_reg.exists():
            if (mri_dir / 'T1.mgz').exists():
                ref_volume = mri_dir / 'T1.mgz'
            else:
                ref_volume = mri_dir / 'T1'
            cmd = ['fsl_rigid_register', '-r', str(ref_volume), '-i',
                   str(flash5), '-o', str(flash5_reg)]
            run_subprocess_env(cmd)
        else:
            logger.info("Registered flash 5 image is already there")
    else:
        flash5_reg = flash5

    # Step 5a : Convert flash5 into COR
    logger.info("\n---- Converting flash5 volume into COR format ----")
    flash5_dir = mri_dir / 'flash5'
    shutil.rmtree(flash5_dir, ignore_errors=True)
    flash5_dir.mkdir(exist_ok=True, parents=True)
    cmd = ['mri_convert', flash5_reg, flash5_dir]
    run_subprocess_env(cmd)
    # Step 5b and c : Convert the mgz volumes into COR
    convert_T1 = False
    T1_dir = mri_dir / 'T1'
    if not T1_dir.is_dir() or next(T1_dir.glob('COR*')) is None:
        convert_T1 = True
    convert_brain = False
    brain_dir = mri_dir / 'brain'
    if not brain_dir.is_dir() or next(brain_dir.glob('COR*')) is None:
        convert_brain = True
    logger.info("\n---- Converting T1 volume into COR format ----")
    if convert_T1:
        T1_fname = mri_dir / 'T1.mgz'
        if not T1_fname.is_file():
            raise RuntimeError("Both T1 mgz and T1 COR volumes missing.")
        T1_dir.mkdir(exist_ok=True, parents=True)
        cmd = ['mri_convert', T1_fname, T1_dir]
        run_subprocess_env(cmd)
    else:
        logger.info("T1 volume is already in COR format")
    logger.info("\n---- Converting brain volume into COR format ----")
    if convert_brain:
        brain_fname = mri_dir / 'brain.mgz'
        if not brain_fname.is_file():
            raise RuntimeError("Both brain mgz and brain COR volumes missing.")
        brain_dir.mkdir(exist_ok=True, parents=True)
        cmd = ['mri_convert', brain_fname, brain_dir]
        run_subprocess_env(cmd)
    else:
        logger.info("Brain volume is already in COR format")
    # Finally ready to go
    logger.info("\n---- Creating the BEM surfaces ----")
    cmd = ['mri_make_bem_surfaces', subject]
    run_subprocess_env(cmd)
    del tempdir  # ran our last subprocess; clean up directory

    logger.info("\n---- Converting the tri files into surf files ----")
    flash_bem_dir = bem_dir / 'flash'
    flash_bem_dir.mkdir(exist_ok=True, parents=True)
    surfs = ['inner_skull', 'outer_skull', 'outer_skin']
    for surf in surfs:
        out_fname = flash_bem_dir / (surf + '.tri')
        shutil.move(bem_dir / (surf + '.tri'), out_fname)
        nodes, tris = read_tri(out_fname, swap=True)
        # Do not write volume info here because the tris are already in
        # standard Freesurfer coords
        write_surface(op.splitext(out_fname)[0] + '.surf', nodes, tris,
                      overwrite=True)

    # Cleanup section
    logger.info("\n---- Cleaning up ----")
    (bem_dir / 'inner_skull_tmp.tri').unlink()
    if convert_T1:
        shutil.rmtree(T1_dir)
        logger.info("Deleted the T1 COR volume")
    if convert_brain:
        shutil.rmtree(brain_dir)
        logger.info("Deleted the brain COR volume")
    shutil.rmtree(flash5_dir)
    logger.info("Deleted the flash5 COR volume")
    # Create symbolic links to the .surf files in the bem folder
    logger.info("\n---- Creating symbolic links ----")
    # os.chdir(bem_dir)
    for surf in surfs:
        surf = bem_dir / (surf + '.surf')
        if not overwrite and surf.exists():
            skip_symlink = True
        else:
            if surf.exists():
                surf.unlink()
            _symlink(flash_bem_dir / surf.name, surf, copy)
            skip_symlink = False
    if skip_symlink:
        logger.info("Unable to create all symbolic links to .surf files "
                    "in bem folder. Use --overwrite option to recreate them.")
        dest = bem_dir / 'flash'
    else:
        logger.info("Symbolic links to .surf files created in bem folder")
        dest = bem_dir
    logger.info("\nThank you for waiting.\nThe BEM triangulations for this "
                "subject are now available at:\n%s.\nWe hope the BEM meshes "
                "created will facilitate your MEG and EEG data analyses."
                % dest)
    # Show computed BEM surfaces
    if show:
        plot_bem(subject=subject, subjects_dir=subjects_dir,
                 orientation='coronal', slices=None, show=True)


def _check_bem_size(surfs):
    """Check bem surface sizes."""
    if len(surfs) > 1 and surfs[0]['np'] > 10000:
        warn('The bem surfaces have %s data points. 5120 (ico grade=4) '
             'should be enough. Dense 3-layer bems may not save properly.' %
             surfs[0]['np'])


def _symlink(src, dest, copy=False):
    """Create a relative symlink (or just copy)."""
    if not copy:
        src_link = op.relpath(src, op.dirname(dest))
        try:
            os.symlink(src_link, dest)
        except OSError:
            warn('Could not create symbolic link %s. Check that your '
                 'partition handles symbolic links. The file will be copied '
                 'instead.' % dest)
            copy = True
    if copy:
        shutil.copy(src, dest)


def _ensure_bem_surfaces(bem, extra_allow=(), name='bem'):
    # by default only allow path-like and list, but handle None and
    # ConductorModel properly if need be. Always return a ConductorModel
    # even though it's incomplete (and might have is_sphere=True).
    assert all(extra in (None, ConductorModel) for extra in extra_allow)
    allowed = ('path-like', list) + extra_allow
    _validate_type(bem, allowed, name)
    if isinstance(bem, path_like):
        # Load the surfaces
        logger.info(f'Loading BEM surfaces from {str(bem)}...')
        bem = read_bem_surfaces(bem)
        bem = ConductorModel(is_sphere=False, surfs=bem)
    elif isinstance(bem, list):
        for ii, this_surf in enumerate(bem):
            _validate_type(this_surf, dict, f'{name}[{ii}]')
    if isinstance(bem, list):
        bem = ConductorModel(is_sphere=False, surfs=bem)
    # add surfaces in the spherical case
    if isinstance(bem, ConductorModel) and bem['is_sphere']:
        bem = bem.copy()
        bem['surfs'] = []
        if len(bem['layers']) == 4:
            for idx, id_ in enumerate(_sm_surf_dict.values()):
                bem['surfs'].append(_complete_sphere_surf(
                    bem, idx, 4, complete=False))
                bem['surfs'][-1]['id'] = id_

    return bem


def _check_file(fname, overwrite):
    """Prevent overwrites."""
    if op.isfile(fname) and not overwrite:
        raise IOError(f'File {fname} exists, use --overwrite to overwrite it')


_tri_levels = dict(
    medium=30000,
    sparse=2500,
)


@verbose
def make_scalp_surfaces(subject, subjects_dir=None, force=True,
                        overwrite=False, no_decimate=False, *,
                        threshold=20, mri='T1.mgz', verbose=None):
    """Create surfaces of the scalp and neck.

    The scalp surfaces are required for using the MNE coregistration GUI, and
    allow for a visualization of the alignment between anatomy and channel
    locations.

    Parameters
    ----------
    %(subject)s
    %(subjects_dir)s
    force : bool
        Force creation of the surface even if it has some topological defects.
        Defaults to ``True``. See :ref:`tut-fix-meshes` for ideas on how to
        fix problematic meshes.
    %(overwrite)s
    no_decimate : bool
        Disable the "medium" and "sparse" decimations. In this case, only
        a "dense" surface will be generated. Defaults to ``False``, i.e.,
        create surfaces for all three types of decimations.
    threshold : int
        The threshold to use with the MRI in the call to ``mkheadsurf``.
        The default is 20.

        .. versionadded:: 1.1
    mri : str
        The MRI to use. Should exist in ``$SUBJECTS_DIR/$SUBJECT/mri``.

        .. versionadded:: 1.1
    %(verbose)s
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    incomplete = 'warn' if force else 'raise'
    subj_path = op.join(subjects_dir, subject)
    if not op.exists(subj_path):
        raise RuntimeError('%s does not exist. Please check your subject '
                           'directory path.' % subj_path)

    # Backward compat for old FreeSurfer (?)
    _validate_type(mri, str, 'mri')
    if mri == 'T1.mgz':
        mri = mri if op.exists(op.join(subj_path, 'mri', mri)) else 'T1'

    logger.info('1. Creating a dense scalp tessellation with mkheadsurf...')

    def check_seghead(surf_path=op.join(subj_path, 'surf')):
        surf = None
        for k in ['lh.seghead', 'lh.smseghead']:
            this_surf = op.join(surf_path, k)
            if op.exists(this_surf):
                surf = this_surf
                break
        return surf

    my_seghead = check_seghead()
    threshold = _ensure_int(threshold, 'threshold')
    if my_seghead is None:
        this_env = deepcopy(os.environ)
        this_env['SUBJECTS_DIR'] = subjects_dir
        this_env['SUBJECT'] = subject
        this_env['subjdir'] = subjects_dir + '/' + subject
        if 'FREESURFER_HOME' not in this_env:
            raise RuntimeError(
                'The FreeSurfer environment needs to be set up to use '
                'make_scalp_surfaces to create the outer skin surface '
                'lh.seghead')
        run_subprocess([
            'mkheadsurf', '-subjid', subject, '-srcvol', mri,
            '-thresh1', str(threshold),
            '-thresh2', str(threshold)], env=this_env)

    surf = check_seghead()
    if surf is None:
        raise RuntimeError('mkheadsurf did not produce the standard output '
                           'file.')

    bem_dir = op.join(subjects_dir, subject, 'bem')
    if not op.isdir(bem_dir):
        os.mkdir(bem_dir)
    fname_template = op.join(bem_dir, '%s-head-{}.fif' % subject)
    dense_fname = fname_template.format('dense')
    logger.info('2. Creating %s ...' % dense_fname)
    _check_file(dense_fname, overwrite)
    # Helpful message if we get a topology error
    msg = ('\n\nConsider using pymeshfix directly to fix the mesh, or --force '
           'to ignore the problem.')
    surf = _surfaces_to_bem(
        [surf], [FIFF.FIFFV_BEM_SURF_ID_HEAD], [1],
        incomplete=incomplete, extra=msg)[0]
    write_bem_surfaces(dense_fname, surf, overwrite=overwrite)
    if os.getenv('_MNE_TESTING_SCALP', 'false') == 'true':
        tris = [len(surf['tris'])]  # don't actually decimate
    for ii, (level, n_tri) in enumerate(_tri_levels.items(), 3):
        if no_decimate:
            break
        logger.info(f'{ii}. Creating {level} tessellation...')
        logger.info(f'{ii}.1 Decimating the dense tessellation '
                    f'({len(surf["tris"])} -> {n_tri} triangles)...')
        points, tris = decimate_surface(points=surf['rr'],
                                        triangles=surf['tris'],
                                        n_triangles=n_tri)
        dec_fname = fname_template.format(level)
        logger.info('%i.2 Creating %s' % (ii, dec_fname))
        _check_file(dec_fname, overwrite)
        dec_surf = _surfaces_to_bem(
            [dict(rr=points, tris=tris)],
            [FIFF.FIFFV_BEM_SURF_ID_HEAD], [1], rescale=False,
            incomplete=incomplete, extra=msg)
        write_bem_surfaces(dec_fname, dec_surf, overwrite=overwrite)
    logger.info('[done]')


@verbose
def distance_to_bem(pos, bem, trans=None, verbose=None):
    """Calculate the distance of positions to inner skull surface.

    Parameters
    ----------
    pos : array, shape (..., 3)
        Position(s) in m, in head coordinates.
    bem : instance of ConductorModel
        Conductor model.
    %(trans)s If None (default), assumes bem is in head coordinates.

        .. versionchanged:: 0.19
            Support for 'fsaverage' argument.
    %(verbose)s

    Returns
    -------
    distances : float | array, shape (...)
        The computed distance(s). A float is returned if pos is
        an array of shape (3,) corresponding to a single position.

    Notes
    -----
    .. versionadded:: 1.1
    """
    ndim = pos.ndim
    if ndim == 1:
        pos = pos[np.newaxis, :]

    n = pos.shape[0]
    distance = np.zeros((n,))

    logger.info(
        'Computing distance to inner skull surface for ' +
        f'{n} position{_pl(n)}...'
    )

    if bem['is_sphere']:
        center = bem['r0']

        if trans:
            center = apply_trans(trans, center, move=True)
        radius = bem['layers'][0]['rad']

        distance = np.abs(radius - np.linalg.norm(
            pos - center, axis=1
        ))

    else:  # is BEM
        surface_points = bem['surfs'][0]['rr']

        if trans:
            surface_points = apply_trans(
                trans, surface_points, move=True
            )

        _, distance = _compute_nearest(surface_points, pos, return_dists=True)

    if ndim == 1:
        distance = distance[0]  # return just a float if one pos is passed

    return distance
