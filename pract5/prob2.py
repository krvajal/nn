import numpy as np


def sigmoid(x, deriv = False):
    if deriv:
        return x*(1-x)
    return 1/(1 + np.exp(-x))

N1 = 50
N2  = 50

p = 20 # number of test cases
np.random.seed(1)


# input data, first and last are the bias
D_in  = 2*np.random.randint(0,2,(p,N1)) - 1
D_out = np.empty((p,1))


for i in range(p):
    D_out[i] = (np.prod(D_in[i,:]) + 1)/2






syn0 = 2*np.random.random((N1,N2)) - 1
syn1 = 2*np.random.random((N2,1)) - 1

# training step
for j in range(3000):
    l0 = D_in
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    l2_error = D_out - l2

    if(j % 100) == 0:
        print "Error: " + str(np.mean(np.abs(l2_error)))
    l2_delta = l2_error * sigmoid(l2, deriv = True)

    l1_error = l2_delta.dot(syn1.T)


    l1_delta = l1_error * sigmoid(l1, deriv = True)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(l2)





