#################
## AAYUSH JAIN ##
#################

###########################
## MSE and Cross Entropy ##
###########################


import numpy as np
from sklearn.metrics import log_loss

true_labels = np.array([1, 0, 1])
predicted_probs = np.array([0.7, 0.3, 0.5])

mse_loss = np.mean((true_labels - predicted_probs) ** 2)
cross_entropy_loss = log_loss(true_labels, predicted_probs)

print("MSE Loss:", mse_loss)
print("Cross-Entropy Loss:", cross_entropy_loss)