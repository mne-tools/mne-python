def fit_spheres_to_mri(subjects_dir, subject, bem_surf, trans, n_spheres,show_spheres=False):
    """Fits two spheres to MRI using BEM, such that spheres fit while brain but
    do not encroach on sensors. For use with Milti-SSS Maxwell Filtering

    Parameters
    ----------
    subjects dir: str
        director to Freesurfer subjects
    subject: str
        Subject ID
    bem_surf: list
        output of mne.make_bem_model(), must be three shell conductivity profiles
    trans: str
        path to trans file, mri_dev_t information
    n_spheres: int
        number of spheres to fit, recommended 2
    show_spheres: bool
        show pyvista plot of the origins and optimized spheres overlayed with the head

    Returns
    -------
    centers: np.ndarray
        2x3 array containing the two centers in HEAD coordinate space
        can be directly fed into:
            raw_msss = mne.preprocessing.maxwell_filter(raw, origin=origins, ...)
            for multi-SSS preprocessing


    Notes
    -----
    * Must have vedo and nibabel installed
    * Must have run mne watershed BEM using freesurfer segmentation
    """
    ## --- required imports

    import nibabel as nib
    import numpy as np
    import vedo
    from scipy.spatial import KDTree

    from .._fiff.constants import FIFF
    from ..surface import _CheckInside
    from ..transforms import (
        apply_trans,
        invert_transform,
        read_trans,
    )

    ## --- begin
    mindist = 2e-3
    assert bem_surf[0]['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD
    assert bem_surf[2]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN
    scalp, _, inner_skull = bem_surf
    inside_scalp = _CheckInside(scalp, mode='pyvista')
    inside_skull = _CheckInside(inner_skull, mode='pyvista')
    m3_to_cc = 100 ** 3
    assert inside_scalp(inner_skull['rr']).all()
    assert not inside_skull(scalp['rr']).any()
    b = vedo.Mesh([inner_skull['rr'], inner_skull['tris']])
    s = vedo.Mesh([scalp['rr'], scalp['tris']])
    s_tree = KDTree(scalp['rr'])
    brain_volume = b.volume()
    print(f'Brain vedo:     {brain_volume * m3_to_cc:8.2f} cc')
    brain_vol = nib.load(subjects_dir / subject / 'mri' / 'brainmask.mgz')
    brain_rr = np.array(np.where(brain_vol.get_fdata())).T
    brain_rr = apply_trans(brain_vol.header.get_vox2ras_tkr(), brain_rr) / 1000. #apply a transformation matrix
    del brain_vol #delete brain volume
    brain_rr = brain_rr[inside_skull(brain_rr)]
    vox_to_m3 = 1e-9
    brain_volume_vox = len(brain_rr) * vox_to_m3

    def _print_q(title, got, want):
        title = f'{title}:'.ljust(15)
        print(f'{title} {got * m3_to_cc:8.2f} cc ({(want - got) / want * 100:6.2f} %)')

    _print_q('Brain vox', brain_volume_vox, brain_volume_vox)

    # 1. Compute a naive sphere using the center of mass of brain surf verts
    naive_c = np.mean(inner_skull['rr'], axis=0)
    naive_r = np.min(np.linalg.norm(inner_skull['rr'] - naive_c, axis=1))
    naive_v = 4 / 3 * np.pi * naive_r ** 3
    _print_q('Naive sphere', naive_v, brain_volume)
    s1 = vedo.Sphere(naive_c, naive_r, res=100)
    _print_q('Naive vedo', s1.volume(), brain_volume)

    # 2. Now use the larger radius (to head) plus mesh arithmetic
    better_r = s_tree.query(naive_c)[0] - mindist
    s1 = vedo.Sphere(naive_c, better_r, res=24)
    _print_q('Better vedo', s1.boolean("intersect", b).volume(), brain_volume)
    v = np.sum(np.linalg.norm(brain_rr - naive_c, axis=1) <= better_r) * vox_to_m3
    _print_q('Better vox', v, brain_volume_vox)

    # 3. Now optimize one sphere
    from scipy.optimize import fmin_cobyla #constrained optimization by linear approximation

    def _cost(c):
        cs = c.reshape(-1, 3)
        rs = np.maximum(s_tree.query(cs)[0] - mindist, 0.)
        resid = brain_volume
        mask = None
        for c, r in zip(cs, rs):
            if not (r and s.contains(c)): #was is_inside
                continue
            m = np.linalg.norm(brain_rr - c, axis=1) <= r
            if mask is None:
                mask = m
            else:
                mask |= m
        resid = brain_volume_vox
        if mask is not None:
            resid = resid - np.sum(mask) * vox_to_m3
        return resid

    def _cons(c):
        cs = c.reshape(-1, 3)
        sign = np.array([2 * s.contains(c) - 1 for c in cs], float) #was "is_inside"
        cons = sign * s_tree.query(cs)[0] - mindist
        return cons

    x = naive_c
    c_opt_1 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
    v_opt_1 = brain_volume_vox - _cost(c_opt_1)
    _print_q('COBYLA 1', v_opt_1, brain_volume_vox)

    # 4. Now optimize two spheres
    x = np.concatenate([c_opt_1, naive_c])
    c_opt_2 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
    v_opt_2 = brain_volume_vox - _cost(c_opt_2)
    _print_q('COBYLA 2', v_opt_2, brain_volume_vox)

    # 4. Finally, three spheres (not perfect, not global opt)
    x = np.concatenate([c_opt_2, naive_c])
    c_opt_3 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
    v_opt_3 = brain_volume_vox - _cost(c_opt_3)
    _print_q('COBYLA 3', v_opt_3, brain_volume_vox)

    if show_spheres:
        import pyvista as pv
        import pyvistaqt
        import matplotlib
        plotter = pyvistaqt.BackgroundPlotter(
            shape=(1, 2), window_size=(1200, 300),
            editor=False, menu_bar=False, toolbar=False)
        plotter.background_color = 'w'
        brain_mesh = pv.make_tri_mesh(inner_skull['rr'], inner_skull['tris'])
        scalp_mesh = pv.make_tri_mesh(scalp['rr'], scalp['tris'])
        colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        mesh_kwargs = dict(render=False, reset_camera=False, smooth_shading=True)
        for ci, cs in enumerate((c_opt_1, c_opt_2, c_opt_3)):
            plotter.subplot(0, ci)
            plotter.camera.position = (0., -0.5, 0)
            plotter.camera.focal_point = (0., 0., 0.)
            plotter.camera.azimuth = 90
            plotter.camera.elevation = 0
            plotter.camera.up = (0., 0., 1.)
            plotter.add_mesh(brain_mesh, opacity=0.2, color='k', **mesh_kwargs)
            plotter.add_mesh(scalp_mesh, opacity=0.1, color='tan', **mesh_kwargs)
            for c, color in zip(cs.reshape(-1, 3), colors):
                sphere = pv.Sphere(s_tree.query(c)[0] - mindist, c)
                plotter.add_mesh(sphere, opacity=0.5, color=color, **mesh_kwargs)
        plotter.show()

    # Ready centers to output, transform into device space
    mri_head_t = invert_transform(read_trans(trans)) 
    if mri_head_t["from"] == FIFF.FIFFV_COORD_HEAD:
        mri_head_t = invert_transform(mri_head_t)
    assert mri_head_t['from'] == FIFF.FIFFV_COORD_MRI, mri_head_t['from']
    centers=[]
    for use in (c_opt_1,c_opt_2,c_opt_3):
        centers.append(apply_trans(mri_head_t, use.reshape(-1, 3)))
    if n_spheres==1:
        return centers[0]
    if n_spheres==2:
        return centers[1]
    if n_spheres==3:
        print("Warning: use of mSSS with three origins and expansions is not tested or recommended")
        return centers[2]

    
