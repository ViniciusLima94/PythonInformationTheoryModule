import numpy as np

def EntropyNormalDistribution(sigma):
	#####################################################################################################
	# Description: Computes the analytical entropy of a normal distribution
	# > Inputs:
	# sigma: Standard deviation of the distribution
	# > Outputs:
	# The entropy of the distribution.
	#####################################################################################################

	return 0.5 * np.log(2 * np.pi * np.exp(1) * (sigma**2) ) / np.log(2)

def JointEntropyNormalDistribution(cov):
	#####################################################################################################
	# Description: Computes the analytical joint entropy of a multivariate normal distribution
	# > Inputs:
	# cov: Covariance matrix
	# > Outputs:
	# The joint entropy of the distribution.
	#####################################################################################################

	return 0.5 * np.log( np.linalg.det( 2 * np.pi * np.exp(1) * cov ) ) / np.log(2)


def MutualInformationNormalDistribution(rho):
	#####################################################################################################
	# Description: Computes the analytical mutual information between two gaussian processes
	# > Inputs:
	# rho: correlation coefficient between two gaussian processes.
	# > Outputs:
	# The mutual information of the distribution.
	#####################################################################################################

	return -0.5 * np.log(1 - rho**2) / np.log(2) 

def ConditionalutualInformationNormalDistribution(sigma, cov1, cov2):

	MI_xyz = 0
	for s in sigma:
		MI_xyz += EntropyNormalDistribution(s) 
	MI_xyz -= JointEntropyNormalDistribution(cov1)

	MI_xz = 0
	for s in [sigma[0], sigma[2]]:
		MI_xz  += EntropyNormalDistribution(s) 
	MI_xz -= JointEntropyNormalDistribution(cov2)

	return MI_xyz - MI_xz