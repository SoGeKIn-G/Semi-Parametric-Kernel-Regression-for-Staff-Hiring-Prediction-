import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

def my_kernel( X1, Z1, X2, Z2 ):
	# Use this method to compute Gram matrices for your proposed kernel
	# Your kernel matrix will be used to train a kernel ridge regressor
    """
    Computes:
        x1 * x2 * ( (z1·z2 + c)^d ) + 1

    Uses optimized paths for d = 3.
    """
    degree = 3
    coef = 0.5

    # Convert all inputs to clean arrays
    X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=np.float64).reshape(-1, 1)
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)

    dot_z = Z1 @ Z2.T

    # Adds coef and then raises to the corresponding degree
    base = dot_z + coef

    # Explicit exponentiation for degree 3 decreases the runtime
    if degree == 3:
        K_z = base * base * base
    else:
        K_z = np.power(base, degree)

    K_x = X1 @ X2.T

    G= K_x * K_z + 1.0
    return G