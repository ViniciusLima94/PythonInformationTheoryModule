'''
Python module to compute information theoretical quantities
'''

import numpy             as     np
import pandas            as     pd
from   .general.measures import EntropyFromProbabilities

def BinEntropy(x):
	#####################################################################################################
	# Description: Computes the entropy of a binary discrete time seres.
	# > Inputs:
	# x: Binary discrete time series (series of 0s and 1s), should be size [N_variables, N_observations].
	# > Outputs:
	# H: Vector (size [N_variables]) containing the entropy of each variable.
	#####################################################################################################

	# Check array shape
	if len(x.shape) == 1:
		# If has only one dimension we considerer that only one variable was provided
		N_var = 1
		N_obs = x.shape[0]
	else:
		N_var = x.shape[0]
		N_obs = x.shape[1]

	# Convert to data frame 
	X    = pd.DataFrame(x.T)
	# Computing probabilities of 0s and 1s for each variable (column) in the data frame
	prob = X.apply(pd.value_counts) / N_obs
	# Computing entropy for each variable (column) in the data frame
	H    = prob.apply(EntropyFromProbabilities).values

	return H

def BinJointEntropy(x, return_entropies = False):
	#####################################################################################################
	# Description: Computes the entropy and joint entropy from binary discrete time series.
	# Inputs:
	# x: Binary discrete time series (series of 0s and 1s), should be size [N_variables, N_observations]
	# return_entropies: If "True", also return entropy of each variable
	# Outputs:
	# H: Vector (size [N_variables]) containing the entropy of each variable (if return_entropies is "True").
	# H_joint: Float with the joint entropy.
	#####################################################################################################

	# Check array shape
	if x.shape[0] == 1 or len(x.shape) == 1:
		print('To compute the joint entropy at leat two variables should be provided!')
	else:
		N_var = x.shape[0]
		N_obs = x.shape[1]

	# In case return_entropies is true, single entropies will also be returned.
	if return_entropies  == True:
		H = BinEntropy(x)

	# Here we first convert the data to string before creating the data frame
	# because it is easier to count the symbols later.
	X = pd.DataFrame(x.T.astype('str'))
	# Concatenate rows to form "joint" symbols
	X = X.T.apply(''.join) 
	# Computing probabilities of joint symbols "0", "1"
	prob = X.value_counts() / N_obs
	# Computing joint entropy
	H_joint =  EntropyFromProbabilities(prob.values)

	if return_entropies  == True:
		return H, H_joint
	else:
		return H_joint


def BinMutualInformation(x):
	#####################################################################################################
	# Description: Computes the mutual information between two time series.
	# Inputs:
	# x: Binary discrete time series (series of 0s and 1s), should be size [N_variables, N_observations].
	# lag: lag applyed bewteen time series.
	# Outputs:
	# MI: The mutual information between the discrete time series.
	#####################################################################################################

	# Check array shape
	if x.shape[0] == 1 or len(x.shape) == 1:
		print('To compute the mutual information at leat two variables should be provided!')

	else:
		# Compute entropies and joint entropy
		H, H_joint = BinJointEntropy(x, return_entropies = True)

		MI = np.sum(H) - H_joint

		return MI

def BinLaggedMutualInformation(x, y, lag = 0):
	#####################################################################################################
	# Description: Computes the lagged mutual information between two time series.
	# Inputs:
	# x: Binary discrete time series (series of 0s and 1s), should be size [N_observations].
	# y: Binary discrete time series (series of 0s and 1s), should be size [N_observations].
	# lag: lag applyed bewteen time series.
	# Outputs:
	# MI: The lagged mutual information between the discrete time series MI(x_i, y_{i-lag}).
	#####################################################################################################

	# If lag is zero then compute the regular bicariate MI 
	if lag == 0:
		return BinMutualInformation(np.vstack([x,y]))
	else:
		# Otherwise we apply the lag to y
		x = x[lag:]
		y = y[0:-lag]
		return BinMutualInformation(np.vstack([x,y]))


def BinConditionalMutualInformation(x, y, z):
	#####################################################################################################
	# Description: Computes the mutual information between two time series conditioned in a third one.
	# Inputs:
	# x: Time series of shape [N_observations].
	# y: Time series of shape [N_observations].
	# z: Time series of shape [N_observations].
	# lag: lag applyed bewteen time series.
	# Outputs:
	# cMI: The mutual information between x and y conditioned to z.
	#####################################################################################################

	# Computing conditional MI
	cMI   = BinMutualInformation( np.vstack([x,y,z]) ) - BinMutualInformation( np.vstack([x,z]) ) 

	return cMI

def BinTransferEntropy(x, y, lag = 0):
	#####################################################################################################
	# Description: Computes the transfer entropy between two time series.
	# Inputs:
	# x: Time series of shape [N_observations].
	# y: Time series of shape [N_observations].
	# lag: lag applyed bewteen time series.
	# Outputs:
	# cMI: The mutual information between x and y conditioned to z.
	#####################################################################################################

	# Applying lags, by definition the lag for TE must be at least 1
	y_c = y[1+lag:].copy()         # x in the current time step
	y_l = y[0:-(1+lag)].copy()	   # lagged x
	x_l = x[0:-(1+lag)].copy()	   # lagged y
	# Computing TE x->y
	TE_xy = BinConditionalMutualInformation(y_c, x_l, y_l)
	
	return TE_xy