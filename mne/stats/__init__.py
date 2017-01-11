"""Functions for statistical analysis."""

from .parametric import f_threshold_mway_rm, f_mway_rm, f_oneway
from .permutations import permutation_t_test
from .cluster_level import (permutation_cluster_test,
                            permutation_cluster_1samp_test,
                            spatio_temporal_cluster_1samp_test,
                            spatio_temporal_cluster_test,
                            _st_mask_from_s_inds,
                            ttest_1samp_no_p,
                            summarize_clusters_stc)
from .multi_comp import fdr_correction, bonferroni_correction
from .regression import linear_regression, linear_regression_raw
