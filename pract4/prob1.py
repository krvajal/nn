import numpy as np
import matplotlib.pyplot as plt

# defino los parametros
ii = -1
f, axarr = plt.subplots(4, 4)
for N in [500,1000,2000,4000]:
	jj = -1
	ii = ii + 1
	for alpha in [0.12, 0.14, 0.16, 0.18]:
		jj = jj + 1
		
		p = int(alpha * N)
		chi_in = 2*np.random.randint(0,2,size=(N,p)) - 1
		print chi_in

		def sign(x):
			if x < 0: return -1
			return 1
		def connection_matrix(chi_in):
			N, p = chi_in.shape
			retval = np.zeros((N,N))
			
			for i in range(0,N):
				for j in range(0,i):
					retval[i,j] = np.dot(chi_in[i,:],chi_in[j,:])/float(N)
					retval[j,i] = retval[i,j]
			return retval

		def pattern_step(weight_matrix, pattern):
			outpattern = np.array(map(sign,np.dot(weight_matrix,pattern)))
			return outpattern
		conn_mat =  connection_matrix(chi_in)

		def compute_overlap(pat1, pat2):
			N = len(pat1)
			retval  =  np.dot(pat1,pat2)/ float(N)
			return retval

		def main():
			overlap= np.zeros(p)
			convergences  = np.zeros((N,p))
			for pat_idx in range(0,p):
				print "computing pattern " , pat_idx
				pattern = chi_in[:,pat_idx]
				# iterate the pattern to convergence
				old_pattern = pattern
				for i in range(0,100):
		 			pattern = pattern_step(conn_mat, pattern)
		 			if( np.sum(np.abs(old_pattern - pattern)) == 0): break
		 			old_pattern = pattern
		 			# print "Iteration ", i, "pattern: ",  pattern

		 		convergences[:,pat_idx] = pattern		
			print "Overlaps"
			for i in range(0,p):
				overlap[i] = compute_overlap(convergences[:,i], chi_in[:,i])
				print "Overlap with pattern %d:" % (i), overlap[i]
			print (ii,jj)
			axarr[ii,jj].hist(overlap, bins = p/10, normed=True)
			axarr[ii,jj].set_xlim([0,1])
			
		main()


plt.show()
plt.savefig('fig1.eps', format='eps', dpi=1000)







