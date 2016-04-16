import numpy as np
import scipy as sp
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.connectivity import fcn_pearson, fcn_cosine, fcn_mutual_info

def test_fcn_pearson():
	"""
	Test 'fcn_pearson' function
	"""
	np.random.seed(3636)
	data = np.random.rand(4,1000)
	
	# test normal functionality
	fcn = fcn_pearson(data)
	assert_true(round(fcn[1,2],3) == 0.535)

	# test normalization functionality
	fcn_round = fcn_pearson(data, norm=[10,12])
	
	# test pval functionality
	fcn_pval = fcn_pearson(data, pval=True)

def test_fcn_cosine():
	"""
	Test 'fcn_cosine' function
	"""
	np.random.seed(3636)
	data = np.random.rand(4,1000)
	
	# test normal functionality
	fcn = fcn_pearson(data)
	assert_true(round(fcn[1,2],3) == 0.535)

	# test normalization functionality
	fcn_round = fcn_pearson(data, norm=[10,12])
	
	# test pval functionality
	fcn_pval = fcn_pearson(data, pval=True)

def test_fcn_mutual_info():
	"""
	Test 'fcn_mutual_info' function
	"""
	np.random.seed(3636)
	data = np.random.rand(4,1000)
	
	# test normal functionality
	fcn = fcn_pearson(data)
	assert_true(round(fcn[1,2],3) == 0.535)

	# test normalization functionality
	fcn_round = fcn_pearson(data, norm=[10,12])
	
	# test pval functionality
	fcn_pval = fcn_pearson(data, pval=True)

















