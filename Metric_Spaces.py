
""" Metric spaces: creating, testing for being valid, Euclidean, non-Euclidean, etc.  """

import sys 
import numpy as np
from numpy import linalg as LA
import math


#Retruns a randomly generated vector space, of size and dimesnion dim, as numpy 2D array
def get_random_space(size, dim):
    space=np.random.randn(size, dim)
    for i in range(size):
        sdv=np.random.randint(1,30)
        space[i]=sdv*space[i]
    return(space);