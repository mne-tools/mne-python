"""Functions for statistical analysis."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "parametric": [
            "f_threshold_mway_rm",
            "f_mway_rm",
            "f_oneway",
            "_parametric_ci",
            "ttest_1samp_no_p",
            "ttest_ind_no_p",
        ],
        "permutations": [
            "permutation_t_test",
            "_ci",
            "bootstrap_confidence_interval",
        ],
        "cluster_level": [
            "permutation_cluster_test",
            "permutation_cluster_1samp_test",
            "spatio_temporal_cluster_test",
            "spatio_temporal_cluster_1samp_test",
            "_st_mask_from_s_inds",
            "summarize_clusters_stc",
        ],
        "multi_comp": ["fdr_correction", "bonferroni_correction"],
        "regression": ["linear_regression", "linear_regression_raw"],
        "_adjacency": ["combine_adjacency"],
    },
)
