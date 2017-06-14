import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv = False):
    if deriv:
        return x*(1-x)
    return 1/(1 + np.exp(-x))
    




