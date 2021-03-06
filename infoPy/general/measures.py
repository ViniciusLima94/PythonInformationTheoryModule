import numpy as np

def EntropyFromProbabilities(prob):
	#####################################################################################################
	# Description: Computes the entropy given a discrete probability distribution.
	# > Inputs:
	# prob: A discrete probability distribution.
	# > Outputs:
	# H: The entropy of the distribution.
	#####################################################################################################

	# Drop small values from the probability distribution to avoid log2(0)
	prob = prob[prob > 1e-10]
	H = -np.sum(prob * np.log2(prob))

	return H

def EntropyFromDataProbabilities(prob):
	#####################################################################################################
	# Description: Computes the entropy given the probability of each observation in the data
	# > Inputs:
	# prob: A discrete probability distribution.
	# > Outputs:
	# H: The entropy of the data.
	#####################################################################################################
	# Drop small values from the probability distribution to avoid log2(0)
	prob = prob[prob > 1e-10]
	H    = -np.mean(np.log2(prob) )

	return H
