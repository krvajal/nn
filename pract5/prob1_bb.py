import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv = False):
    if deriv:
        return x*(1-x)
    return 1/(1 + np.exp(-x))


# input data
D_in  = np.array([[0,1,0],
                 [0,1,1],
                 [1,1,0],
                 [1,1,1]])
D_out = np.array([[0],[1],[1],[0]])

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1



errors = []
# training step
for j in range(300000):
    l0 = D_in
    l1 = sigmoid(np.dot(l0, syn0))
    l1[:,0] = l0[:,0]
    l1[:,2] = l0[:,2]
    l1[:,3] =  1 # bias
    l2 = sigmoid(np.dot(l1, syn1))
    l2_error = D_out - l2

    if(j % 2000) == 0:
        err = np.mean(np.abs(l2_error))
        print "Error: " + str(err) 
        print syn0[1,1]
        errors.append(err)
    l2_delta = l2_error * sigmoid(l2, deriv = True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error* sigmoid(l1, deriv = True)


    # update weights
    syn1 += l1.T.dot(l2_delta)
    #print "l1", l1.shape
    # print "l1_error",l1_error[:,2:3].shape
    # quit()
    #print "sig", sigmoid(l1,de`riv = True).shape
    # print "l1_delta", l1_delta.shape
    #print "shape",l0.shape
    syn0 += l0.T.dot(l1_delta)

print "l1",l1
print "l2",l2

print "weights"
print "Layer1",syn0[:,2] # only neuron 2 is real
print "Layer2",syn1
n = len(errors)
# plt.plot(np.linspace(0,1,n)*n, errors)
# plt.show()

