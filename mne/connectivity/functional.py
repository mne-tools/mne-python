# Authors: Nicholas Cullen <ncullen.th@dartmouth.edu>
#
# License: BSD (3-clause)

"""
Code for Static Functional Connectivity Networks in Source Space.

Functonal Connectivity is the 'temporal coincidence of spatially
distinct neurophysiological events' [1] It is based 
on correlation or mutual information between time series and 
does NOT imply causality. While FCNs are most commonly studied in fMRI, 
the strong temporal resolution of EEG data has certain advantages for 
mapping FCNs [11, 12]. Also, nearly all common FC metrics used 
in the fMRI literature are valid for EEG.

Wang et al [7] propose and systematically compare metrics 
from 7 families of functional connectivity measures:

		1) correlation
		2) h^2
		3) mutual information
		4) coherence
		5) Granger
		6) transfer entropy
		7) MVAR-frequency domain-based techniques -> A(H)

References
----------
[1] Eickhoff and Grefkes, (2011), "Approaches for the integrated 
	analysis of structure, function and connectivity of the human brain."
[2] Olaf Sporns, (2013), "Structure and function of complex brain networks."
[3] Iglesia-Vaya et al, (2013), "Brain Connections - Resting State fMRI
	Functional Connectivity."
[4] Karl Friston, (2011), "Functional and Effective Connectivity: A Review."
[5] Di Lorenzo et al, (2015), "Altered resting-state EEG source functional
	connectivity in schizophrenia: the effect of illness duration."
[6] Iyer, Egan, et al, (2015), "Functional Connectivity Changes in Resting-
	state EEG as Potential Biomarker for Amyotrophic Lateral Sclerosis."
[7] Wange, Benar, et al. (2014), "A sytematic framework for functional
	connectivity measures."
[8] David, Cosmelli, Friston,, (2003), "Evaluation of different measures of
	functional connectivity using a neural mass model."
[9] Xu, Kroupi, Ebrahimi, (2015), "Functional Connectivity from EEG Signals
	during Perceiving Pleasant and Unpleasant Odors."
[10] Sargolzaei et al, (2015), "Scalp EEG brain functional connectivity
	networks in pediatric epilepsy."
[11] Olaf Sporns, (2012), "Discovering the Human Connectome."
[12] Olaf Sporns, (2010), "Networks of the Brain."
[13] Lee, Kim, Jung, (2006), "Classification of epilepsy types through
	global network analysis of scalp electroencephalograms."
[14] Xu, Bakardjian, et al, (2007), "A new nonlinear similarity measure
	for multichannel biological signals."
[15] Jovanovic et al. (2013), "Brain Connectivity measures: computation
	and comparison."

"""
import numpy as np
import scipy as sp
import scipy.stats
import scipy.spatial


######### CORRELATION METRICS #########

def fcn_pearson(data, pval=False, norm=[], fill=True):
	"""
	Pearson correlation coefficeint for FCN as described in [1,2,3,3].

	The range for the p-value is [0,1], where small values imply
	signficant correlation. The range of the correlation coefficent
	(pval=False) is [-1,1] where values near -1 imply strong negative
	correlation and values near 1 imply strong positive correlation.

	This function uses the Scipy implementation of pearsonr. It's
	possible that we will allow for delay-based correlation as in [15].

	References
	----------
	[1] Wange, Benar, et al. (2014), "A sytematic framework for functional
	connectivity measures."
	[2] Sargolzaei et al, (2015), "Scalp EEG brain functional connectivity
	networks in pediatric epilepsy."
	[3] Olaf Sporns, (2012), "Discovering the Human Connectome."
	[4] Jovanovic et al. (2013), "Brain Connectivity measures: computation
	and comparison."

	Parameters
	----------
	data : a 2-d numpy array
		The data from which the FCN will be learned

	pval : a boolean (default=False)
		Whether to use the p-value from the pearson calculation, or
		simply the correlation coefficient.

	norm: a list of two values (default=[])
		Users can supply a 2 value range between which the pearson
		correlation output values will be normalized.
			e.g. setting 'normalize=[0,20]' means that the adj matrix values
			will be normalized between 0 and 20.

	fill : a boolean (default=True)
		Whether to copy the Upper Triangle of the resulting adjacency matrix
		into the Lower Triangle, thereby making the matrix symmetric. 
		If fill_lower=False, then the bottom triangle indices of 
		the returned adj matrix will be 0.

	Returns
	-------
	fcn_mat : a numpy ndarray w/ shape=(n_signals,n_signals)
		The functional connectivity network Adjacency Matrix, where
		fcn_mat[i,j] represents the edge strengt between 
		signal 'i' and signal 'j' according to the metric used.
	"""
	N_SIG, N_OBS = data.shape
	fcn_mat = np.zeros((N_SIG, N_SIG)) # pre-allocate fcn matrix

	min_val = 1e9
	max_val = -1e9
	for i in xrange(N_SIG):
		for j in xrange(i+1, N_SIG):
			# calculate pearson valuess
			rho, p_val = sp.stats.pearsonr(data[i,:], data[j,:])
			# get appropriate statistic
			if pval:
				fcn_mat[i,j] = p_val
			else:
				fcn_mat[i,j] = rho

	# normalize between norm[0] and norm[1]
	if norm:
		fcn_mat = _normalize(fcn_mat, norm)

	# make the matrix symmetric
	if fill:
		fcn_mat += fcn_mat.T

	return fcn_mat

def fcn_cosine(data, norm=[], fill=True):
	"""
	Cosine similarity for FCN. See [1,2].

	It is common to normalize distances between [0, \pi/2], where
	distance of 0 rad corresponds to maximum correlation among two
	time series vectors while \pi/2 rad implies the vectors are
	orthogonal and thus uncorrelated.

	We use the 'scipy.spatial.distance.cosine' function.

	\Theta_ij = \pi - cos^{-1}( (x_i * x_j) / (||x_i|| * ||x_j||) )
		for i,j = 1, ... , m

	References
	----------
	[1] Sargolzaei et al, (2015), "Scalp EEG brain functional connectivity
	networks in pediatric epilepsy."
	[2] Xu, Bakardjian, et al, (2007), "A new nonlinear similarity measure
	for multichannel biological signals."

	Parameters
	----------
	data : a 2-d numpy array
		The data from which the FCN will be learned

	norm: a list of two values (default=[])
		Users can supply a 2 value range between which the pearson
		correlation output values will be normalized.
			e.g. setting 'normalize=[0,20]' means that the adj matrix values
			will be normalized between 0 and 20.

	fill : a boolean (default=True)
		Whether to copy the Upper Triangle of the resulting adjacency matrix
		into the Lower Triangle, thereby making the matrix symmetric. 
		If fill_lower=False, then the bottom triangle indices of 
		the returned adj matrix will be 0.

	Returns
	-------
	fcn_mat : a numpy ndarray w/ shape=(n_signals,n_signals)
		The functional connectivity network Adjacency Matrix, where
		fcn_mat[i,j] = the edge strength (some correlation metric)
		between signal 'i' and signal 'j'.

	"""
	N_SIG, N_OBS = data.shape
	fcn_mat = np.zeros((N_SIG, N_SIG)) # pre-allocate fcn matrix

	min_val = 1e9
	max_val = -1e9
	for i in xrange(N_SIG):
		for j in xrange(i+1, N_SIG):
			# calculate value
			fcn_mat[i,j] = sp.spatial.distance.cosine(data[i,:],data[j,:])
	
	# normalize between norm[0] and norm[1]
	if norm:
		fcn_mat = _normalize(fcn_mat, norm)

	# make the matrix symmetric
	if fill_lower:
		fcn_mat += fcn_mat.T

	return fcn_mat

######### H2 METRICS ##########


######### MUTUAL INFORMATION METRICS #########

def fcn_mutual_info(data, bins=[], norm=[], fill=True):
	"""
	Mutual Information for FCNs.

	As described in [1], you partition the amplitude x_i
	into L bins where each bin value has probability
	p_l. Then, the shannon entropy 'H' is defined as:

		H(x) = - \Sigma_{l=1}^{L} p_l * ln(p_l) 

	and for	a pair of variables x_i and x_j, the joint entropy
	is defined as:

		H(x_i,x_j)= - \Sigma_{i,j}^{L} p_{i,j}*ln(p_{i}/p_{j})

	and finally the mutual information is calculated as:

		MI_{i,j} = H(x_i) + H(x_j) - H(x_i, x_j)

	Additionally, mutual information can be calculated from the joint
	and marginal probability distribution as follows:

		MI_{i,j} = \Sigma_{i,j} P(x_i,x_j) * log( P(x_i,x_j) / P(x_i)*P(x_j) )

	NOTE: This function relies on '_bin_data' function, which
	should probably be added to 'utils.py' in the 'connectivity' folder.
	If MNE already has a data binning function, then I can use that instead.

	References
	----------
	[1] Wange, Benar, et al. (2014), "A sytematic framework for functional
	connectivity measures."

	Parameters
	----------
	data : a 2-d numpy array
		The data from which the FCN will be learned

	bins : a list of integers
		The number of bins into which each row (source)
		array will be split - defaults to 5 for
		all columns

	pval : a boolean (default=False)
		Whether to use the pval for mutual information, which 
		is derived from the fact that 2*N*Mutual_information can
		approximate the chi-square distribution.

	norm: a list of two values (default=[])
		Users can supply a 2 value range between which the pearson
		correlation output values will be normalized.
			e.g. setting 'normalize=[0,20]' means that the adj matrix values
			will be normalized between 0 and 20.

	fill : a boolean (default=True)
		Whether to copy the Upper Triangle of the resulting adjacency matrix
		into the Lower Triangle, thereby making the matrix symmetric. 
		If fill_lower=False, then the bottom triangle indices of 
		the returned adj matrix will be 0.

	Returns
	-------
	fcn_mat : a numpy ndarray w/ shape=(n_signals,n_signals)
		The functional connectivity network Adjacency Matrix, where
		fcn_mat[i,j] = the edge strength (some correlation metric)
		between signal 'i' and signal 'j'.

	Notes
	-----
	- I am not getting the correct results from the hypothesis test,
		so I will have to work on that and add it in later. (Nick)

	"""
	N_SIG, N_OBS = data.shape
	fcn_mat = np.zeros((N_SIG, N_SIG)) # pre-allocate fcn matrix


	if not bins:
		bins = [5]*N_SIG

	bin_data = _bin_data(data, bins=bins)
	# get entire joint frequency distribution
	joint_hist,_ = np.histogramdd(bin_data.T, bins=bins)
	# turn into joint probability distribution
	joint_pdf = joint_hist / joint_hist.sum()

	# calculate marginal distributions once to save repeated calculations in the loop
	marg_dict = {}
	for i in xrange(N_SIG):
		marg_dict[i] = np.sum(joint_pdf, axis=tuple([idx for idx in range(N_SIG) if idx!=i]))

	for i in xrange(N_SIG):
		for j in xrange(i+1, N_SIG):
			axis_indices = tuple([idx for idx in range(N_SIG) if idx not in [i,j]])
			Pxy = np.sum(joint_pdf, axis=axis_indices) # joint pdf over x_i, x_j
			Px = marg_dict[i]
			Py = marg_dict[j]

			PxPy = np.outer(Px,Py)
			Pxy += 1e-7 # in case of zero
			PxPy += 1e-7 # in case of zero
			MI = np.sum(Pxy * np.log(Pxy / PxPy))

			#if pval:
			#	# This is the "G-test"
			#	chi2_statistic = 2 * N_OBS * MI
			#	ddof = (bins[i] - 1) * (bins[j] - 1)
			#	p_val = 2*sp.stats.chi2.pdf(chi2_statistic, ddof) 
			#	fcn_mat[i,j] = round(p_val,5)
			#else:
			fcn_mat[i,j] = round(MI,3)

	# normalize between norm[0] and norm[1]
	if norm:
		fcn_mat = _normalize(fcn_mat, norm)

	# make the matrix symmetric
	if fill:
		fcn_mat += fcn_mat.T

	return fcn_mat



########## COHERENCE METRICS ##########


########## UTILS ##########

def _normalize(fcn_mat, norm):
	"""
	Normalize a fcn matrix.

	Follows the formula:

		y = norm[0] + (x-min(x))*(norm[1]-norm[0])/(max(x)-min(x))

	Parameters
	----------
	fcn_mat : a numpy nd-array
		The data to be normalized

	norm : a list of two values
		The min and max value between which the
		data will be normalized

	Returns
	-------
	fcn_mat : a numpy nd-array
		A normalized version of the passed-in data
	"""
	min_val = np.min(fcn_mat)
	max_val = np.max(fcn_mat)
	data_range = max_val-min_val
	norm_range = norm[1]-norm[0]
	for i in xrange(N_SIG):
		for j in xrange(i+1, N_SIG):
			fcn_mat[i,j] = norm[0]+(fcn_mat[i,j]-min_val)*norm_range/data_range
	return fcn_mat


def _bin_data(data, bins=[]):
	"""
	Discretize/Bin the passed-in dataset -- effectively assigning
	real valued amplitudes into integer bins. Keep in mind 
	this is binning over the ROWS of the array, so
	each source should be a row in the array.

	The default is to have TEN (10) discrete bins for
	all rows. See 'numpy.digitize' for more information
	on how the binning actually occurs.

	Parameters
	----------
	data : a numpy nd-array
		The data to be binned

	bins : a list of integers (optional)
		The number of bins into which each row (source)
		array will be split - defaults to 5 for
		all columns. Note that if you supply a value to the
		'bins' param then you need to supply a bin size
		for EACH row (source) in the array.

	Returns
	-------
	bin_data : a numpy nd-array
		A discretized/binned copy of original data, with
		the same shape as the passed-in data.

	"""
	NROWS,NCOLS = data.shape
	bin_data = np.empty((NROWS,NCOLS))

	if not bins:
		bins = [10]*NROWS

	minmax = zip(np.amin(data,axis=1),np.amax(data,axis=1))
	for c in range(NROWS):
		# get min and max of each row
		_min, _max = minmax[c]
		# create the bins from np.linspace
		_bins = np.linspace(_min,_max,bins[c])
		# discretize with np.digitize(col,bins)
		bin_data[c,:] = np.digitize(data[c,:],_bins)

	bin_data = np.array(bin_data,dtype=np.int32,copy=False)
	return bin_data















