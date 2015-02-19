# Authors: Denis Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from collections import defaultdict

import numpy as np
from scipy import stats

def find_outliers(X, threshold=3.0, max_iter=2):
    """Find outliers based on iterated Z-scoring

    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    my_mask = np.zeros(len(X), dtype=np.bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = np.abs(stats.zscore(X))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx


def corrmap(icas, template, threshold="auto", name="bads", plot=True, inplace=False):
    
    """Corrmap (Viola et al. 2009 Clin Neurophysiol) identifies the best group
    match to a supplied template. Typically, feed it a list of fitted ICAs and
    a template IC, for example, the blink for the first subject, to identify
    specific ICs across subjects.
    
    The specific procedure consists of two iterations. In a first step, the maps 
    best correlating with the template are identified. In the step, the analysis
    is repeated with the mean of the maps identified in the first stage.
    
    Outputs a list of fitted ICAs with the indices of the marked ICs in a 
    specified field.
    
    The original Corrmap website: www.debener.de/corrmap/corrmapplugin1.html
    
    Parameters
    ----------
    icas : list
        A list of fitted ICAs.
    template : (int, int) tuple
        Index of the list member from which the template should be obtained, and
        the index of the IC.
    threshold : "auto" | list of floats > 0 < 1 | float > 0 < 1 | float > 1
        Correlation threshold for identifying ICs
        If "auto": search for the best map by trying all correlations between 
        0.6 and 0.95. In the original proposal, lower values are considered, but
        this is not yet implemented.
        If list of floats: search for the best map in the specified range of 
        correlation strengths.
        If float > 0: select ICs correlating better than this.
        If float > 1: use find_outliers to identify ICs within subjects (not in 
        original Corrmap)
        Defaults to "auto".
    name : str
        Categorised ICs are stored in a defautl dictionary "categorised_ics". This
        paramrter gives the key under which found ICs will be stored.
        Defaults to "bads".
    plot : bool
        Should constructed template and selected maps be plotted?
    
    Returns 
    -------
    new_icas  :  list 
        Returns a list of fitted ICs, enrichened with the indices of the selected
        maps in the field specified by the name keyword.   
    """


    def vcorrcoef(X,y): # vectorised correlation, by Jon Herman
        Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
        ym = np.mean(y)
        r_num = np.sum((X - Xm) * (y - ym), axis=1)
        r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
        r = r_num / r_den
        return r

    def get_ica_map(ica, components=None):
        if components is None:
            components = range(ica.n_components_)
        maps = np.dot(ica.mixing_matrix_.T, ica.pca_components_)[components]
        return maps

    def find_max_corrs(all_maps, target, threshold):
        all_corrs = [vcorrcoef(subj, target) for subj in all_maps]
        abs_corrs = [np.abs(a) for a in all_corrs]
        corr_polarities = [np.sign(a) for a in all_corrs]

        if threshold < 1:
            max_corrs = [list(np.nonzero(s_corr > threshold))
                         for s_corr in abs_corrs]
        else:     
            max_corrs = [list(find_outliers(s_corr, threshold = threshold))
                         for s_corr in abs_corrs]

        am = [sub[i] for l, i_s in zip(abs_corrs, max_corrs) 
                                   for sub, i in zip(l, i_s)]
        median_corr_with_target = np.median(am)

        polarities = [sub[i] for l, i_s in zip(corr_polarities, max_corrs) 
                                   for sub, i in zip(l, i_s)]

        maxmaps = [sub[i] for l, i_s in zip(all_maps, max_corrs) 
                                   for sub, i in zip(l, i_s)]

        try:
            newtarget = np.zeros(maxmaps[0].size)
            for maxmap, polarity in zip(maxmaps, polarities):
                newtarget += maxmap * polarity
    
            newtarget /= len(maxmaps)

            similarity_i_o = np.abs(np.corrcoef(target, newtarget)[1, 0])

            return newtarget, median_corr_with_target, similarity_i_o, max_corrs
        except:
            return [], 0, 0, []    

    if threshold == "auto":
        threshold = np.arange(60, 95) / 100

    all_maps = [get_ica_map(ica) for ica in icas]        
        
    target = all_maps[template[0]][template[1]]
    
    # first run: use user-selected map
    if isinstance(threshold, (int, float)):
        nt, mt, s, mx = find_max_corrs(all_maps, target, threshold)
    elif len(threshold) > 1:
        paths = [find_max_corrs(all_maps, target, t) for t in threshold]
        # find iteration with highest avg correlation with target
        nt, mt, s, mx = paths[np.argmax([path[2] for path in paths])]

    # second run: use output from first run
    if isinstance(threshold, (int, float)):
        nt, mt, s, mx = find_max_corrs(all_maps, nt, threshold)
    elif len(threshold) > 1:
        paths = [find_max_corrs(all_maps, nt, t) for t in threshold]
        # find iteration with highest avg correlation with target
        nt, mt, s, mx = paths[np.argmax([path[1] for path in paths])]

    nones = []
    new_icas = []
    logger.info("Median correlation with constructed map: " + str(mt))
    if plot: logger.info("Displying selected ICs per subject.")

    for i, (ica, max_corr) in enumerate(zip(icas, mx)):
        if inplace == False:
            ica = deepcopy(ica)
        if not hasattr(ica, 'categorised_ics'):
            ica.categorised_ics = defaultdict(lambda:[])
        try:
            if isinstance(max_corr[0], np.ndarray): max_corr = max_corr[0]
            ica.categorised_ics[name] = list(set(list(max_corr) + 
                                                 ica.categorised_ics[name]))
            if plot: 
                logger.info("Subject " + str(i))
                ica.plot_components(max_corr, ch_type="eeg")
        except IndexError:
            nones.append(i)
            if plot: 
                logger.info("No map selected for subject " + str(i) + 
                             ", consider a more liberal threshold.")
        if inplace == False:
            new_icas.append(ica)
    if nones: logger.info("Subjects without any IC selected: ", nones)
    else: logger.info("At least 1 IC detected for each subject.")


    if inplace == True:
        return
    else:
        return new_icas

