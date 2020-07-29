'''
	Python module to compute information theoretical quantities
'''

import numpy             as     np
import os
from   ..utils.tools      import *
#from   jpype             import *
from   ..general.measures import EntropyFromDataProbabilities

def KernelEstimatorEntropy(x, bandwidth = 0.3, kernel = 'tophat', metric = 'euclidean', algorithm = 'auto', norm = False):
	#####################################################################################################
	# Description: Computes the entropy of continuous time series 
	# > Inputs:
	# x: Continuous time series should be size [N_variables, N_observations].
	# bandwidth: Kernel bandwidth
	# kernel: Kernel shape [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
	# metric: Distance metric to use [‘euclidean’|‘manhattan’|‘chebyshev’|‘minkowski’|]
	# norm: Weather to normalize or not the data (default "False")
	# > Outputs:
	# H: Entropy of x if N_variables = 1, joint entropy if N_variables > 1
	#####################################################################################################


	# Normalizing data
	if norm == True:
		x = normalize_data(x)

	prob  = KernelDensityEstimator(x, bandwidth, kernel, metric, algorithm)

	H     = EntropyFromDataProbabilities(prob)

	return H

def KernelEstimatorMutualInformation(x, bandwidth = 0.3, kernel = 'tophat', metric = 'euclidean', algorithm = 'auto', norm = False):
	#####################################################################################################
	# Description: Computes the mutual information between continuous variables
	# > Inputs:
	# x: Continuous time series should be size [N_variables, N_observations].
	# bandwidth: Kernel bandwidth
	# kernel: Kernel shape [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
	# metric: Distance metric to use [‘euclidean’|‘manhattan’|‘chebyshev’|‘minkowski’|]
	# norm: Weather to normalize or not the data (default "False")
	# > Outputs:
	# MI: The mutual information between the continuous time series.
	#####################################################################################################

	# Check array shape
	if x.shape[0] == 1 or len(x.shape) == 1:
		print('To compute the mutual information at leat two variables should be provided!')

	# Get the number of variables
	N_var = x.shape[0]

	# Compute the entropy of each variable
	H = np.zeros(N_var)
	for i in range(N_var):
		H[i] = KernelEstimatorEntropy(x[i], bandwidth, kernel, metric, algorithm,  norm)

	# Compute the joint entropy
	H_joint = KernelEstimatorEntropy(x, bandwidth, kernel, metric, algorithm, norm)

	# Compute the mutual information
	MI = np.sum(H) - H_joint

	return MI

def KernelEstimatorLaggedMutualInformation(x, y, bandwidth = 0.3, kernel = 'tophat', metric = 'euclidean', lag = 0, algorithm = 'auto',  norm = False):
	#####################################################################################################
	# Description: Computes the mutual lagged information between two continuous variables
	# > Inputs:
	# x: Continuous time series should be size [N_observations].
	# y: Continuous time series should be size [N_observations].
	# bandwidth: Kernel bandwidth
	# kernel: Kernel shape [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
	# metric: Distance metric to use [‘euclidean’|‘manhattan’|‘chebyshev’|‘minkowski’|]
	# lag:  lag applied between variable
	# norm: Weather to normalize or not the data (default "False")
	# > Outputs:
	# Returns the lagged MI between time series if lag > 0 [MI(x_i, y_{i-lag})], and regular MI otherwise.
	#####################################################################################################

	# If lag is zero then compute the regular bicariate MI 
	if lag == 0:
		return KernelEstimatorMutualInformation(np.vstack([x,y]), bandwidth, kernel, metric, algorithm, norm)
	else:
		# Otherwise we apply the lag to y
		x = x[lag:]
		y = y[0:-lag]
	return KernelEstimatorMutualInformation(np.vstack([x,y]), bandwidth, kernel, metric, algorithm, norm)

def KernelEstimatorConditionalMutualInformation(x, y, z, bandwidth = 0.3, kernel = 'tophat', metric = 'euclidean', algorithm = 'auto', norm = False):
	#####################################################################################################
	# Description: Computes the conditional mutual information between two continuous variables conditioned
	# to a third one.
	# > Inputs:
	# x: Continuous time series should be size [N_observations].
	# y: Continuous time series should be size [N_observations].
	# z: Continuous time series should be size [N_observations].
	# bandwidth: Kernel bandwidth
	# kernel: Kernel shape [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
	# metric: Distance metric to use [‘euclidean’|‘manhattan’|‘chebyshev’|‘minkowski’|]
	# norm: Weather to normalize or not the data (default "False")
	# > Outputs:
	# cMI: The mutual information between x and y conditioned to z.
	#####################################################################################################

	cMI = KernelEstimatorMutualInformation(np.vstack([x,y,z]), bandwidth, kernel, metric, algorithm, norm) - \
		  KernelEstimatorMutualInformation(np.vstack([x,z]), bandwidth, kernel, metric, algorithm, norm ) 

	return cMI

def KernelEstimatorTransferEntropy(x, y, bandwidth = 0.3, kernel = 'tophat', metric = 'euclidean', lag = 0, algorithm = 'auto', norm = False):
	#####################################################################################################
	# Description: Computes the transfer entropy between two continuous variables
	# to a third one.
	# > Inputs:
	# x: Continuous time series should be size [N_observations].
	# y: Continuous time series should be size [N_observations].
	# bandwidth: Kernel bandwidth
	# kernel: Kernel shape [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
	# metric: Distance metric to use [‘euclidean’|‘manhattan’|‘chebyshev’|‘minkowski’|]
	# lag:  lag applied between variable
	# norm: Weather to normalize or not the data (default "False")
	# > Outputs:
	# TE: The transfer entropy x->y
	#####################################################################################################

	# Applying lags, by definition the lag for TE must be at least 1
	y_c = y[1+lag:].copy()         # x in the current time step
	y_l = y[0:-(1+lag)].copy()	   # lagged x
	x_l = x[0:-(1+lag)].copy()	   # lagged y
	
	# Computing TE x->y
	TE_xy = KernelEstimatorConditionalMutualInformation(y_c, x_l, y_l, bandwidth, kernel, metric, algorithm, norm)

	return TE_xy
