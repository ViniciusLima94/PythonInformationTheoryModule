import numpy as np


def silverman(Nvar, Nobs):

	return (Nobs * (Nvar + 2) / 4.)**(-1. / (Nvar + 4))

def normalize_data(x):
	#####################################################################################################
	# Description: Normalize each column of the data matrix X
	# > Inputs:
	# x: Data matrix must have size [N_variables, N_observations].
	# > Outputs:
	# Normalized data
	#####################################################################################################
	from sklearn.preprocessing import StandardScaler

	# Instantiate scaler object
	scaler = StandardScaler()
	# Fit on data
	scaler.fit(x.T)
	# Transform dada
	x_norm = scaler.transform(x.T)

	return x_norm.T

def KernelDensityEstimator(x, bandwidth, kernel = 'tophat', metric = 'euclidean', algorithm = 'auto'):
	#####################################################################################################
	# Description: Uses kernel estimaton to compute probabiliry distribution
	# > Inputs:
	# x: Data matrix must have size [N_variables, N_observations].
	# bandwidth: Kernel bandwidth
	# kernel: Kernel shape [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
	# metric: Distance metric to use [‘euclidean’|‘manhattan’|‘chebyshev’|‘minkowski’|]
	# for more see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
	# > Outputs:
	# Probability distribution of the data obtained with kernel density estimation
	#####################################################################################################
	from sklearn.neighbors import KernelDensity

	# Checking data shape
	if x.shape[0] >= 1:
		x = x.T
	if len(x.shape) == 1:
		x = x[np.newaxis, :].T

	d = x.shape[0]

	kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric=metric, algorithm=algorithm)

	if d == 1:
		kde.fit(x)
		p = kde.score_samples(x)

	else:
		kde.fit(x)
		p = kde.score_samples(x)
	
	return np.exp(p)