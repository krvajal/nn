import numpy as np
import matplotlib.pyplot as plt

# defino los parametros
N = 4000
p = 40
print ("Generation  %d input patterns"%(p))
chi_in = 2*np.random.randint(0,2,size=(N,p)) - 1

def sign(x):
    """ basic sign function, returns the sign of the argument"""
    if x < 0: return -1
    return 1

temperatures = np.linspace(0.1,2,20)

def connection_matrix(inputs):
    """  Compute the connection matrix Jij"""
    N, p = chi_in.shape

    retval = np.zeros((N,N))
    
    for i in range(0,N):
        for j in range(0,i):
            retval[i,j] = np.dot(inputs[i,:],inputs[j,:])/float(N)
            retval[j,i] = retval[i,j]
    return retval



def compute_h(weight_matrix,pattern):
    return np.dot(weight_matrix,pattern)


def pattern_step(weight_matrix, pattern):
    """ iterate the pattern """
    outpattern = np.array(map(sign, compute_h(weight_matrix, pattern)))
    return outpattern


print("Compute connection matrix")
conn_mat =  connection_matrix(chi_in)
print conn_mat


def prob(h, beta):

    def noise_dist(mat, pat):
        hh = np.array(h(mat, pat))
        return np.exp(beta * hh) / (np.exp(beta * hh) + np.exp(- beta * hh))

    def sign_prob(mat,pat):
        trial = np.random.uniform(0, 1, size=len(pat))
        return np.array(map(sign, trial - noise_dist(mat, pat) ))

    return sign_prob

def compute_overlaps(beta):
        """ Compute overlap between output pattern and input """
        overlap = np.zeros(p)
        noise_pattern_step = prob(compute_h, beta)
        convergences = np.zeros((N, p))    
        for pat_idx in range(0, p):
            # go over each pattern
            pattern = chi_in[:, pat_idx]
            for i in range(0, 10):
                
                pattern = noise_pattern_step(conn_mat, pattern)
            overlap[pat_idx] = np.dot(pattern, chi_in[:, pat_idx])
            overlap[pat_idx] = overlap[pat_idx]/float(N)
            
        return overlap

overlaps_points = []

for temp in temperatures:
    beta = 1.0 / temp
    overlap = compute_overlaps(beta)
    print ("overlap", overlap)
    mean_overlap = np.mean(overlap)
    print ([temp, mean_overlap])    
    overlaps_points.append( mean_overlap)
    
plt.plot(temperatures, overlaps_points,'o-')
plt.xlabel("Temperatura")
plt.ylabel(r"$m^{\mu}$")
plt.savefig('fig1.eps', format='eps', dpi=1000)
