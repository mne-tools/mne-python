"""Functions for statistical analysis"""

from .permutations import permutation_t_test
from .cluster_level import permutation_cluster_test, \
                           permutation_cluster_1samp_test
from .multi_comp import fdr_correction, bonferroni_correction
