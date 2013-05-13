from ..utils import deprecated
from ..inverse_sparse import inverse

dec = deprecated('Use the function from mne.inverse_sparse')

mixed_norm = dec(inverse.mixed_norm)
tf_mixed_norm = dec(inverse.tf_mixed_norm)
