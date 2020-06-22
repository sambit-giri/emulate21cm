import numpy as np
from pyDOE import lhs
​
def LH_sampling(n_params=2, samples=10, mins=0, maxs=1, outfile=None):
	"""
	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	samples (int) : Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	
	Return:
	-------
	An array containing the parameter values.
	"""
	#lhs(n, [samples, criterion, iterations])
	lhd = lhs(n_params, samples=samples)
	if np.array(mins).size==1: mins = [mins for i in range(n_params)]
	if np.array(maxs).size==1: maxs = [maxs for i in range(n_params)]
	for i,[mn,mx] in enumerate(zip(mins,maxs)): lhd[:,i] = mn + (mx-mn)*lhd[:,i]
	if outfile is not None:	np.savetxt(outfile.split('.txt')[0]+'.txt', lhd)
	return lhd
	
if __name__ == '__main__':
	# Here we run a simple example with two parameters.
	# Two parameters.
	n_params = 2
	samples  = 20
	mins, maxs = [0,-1], [10,1]
	lhd = LH_sampling(n_params=n_params, samples=samples, mins=mins, maxs=maxs, outfile='lhs_params')
	import matplotlib.pyplot as plt
	plt.scatter(lhd[:,0], lhd[:,1])
	plt.title('See that the horizontal and vertical line \n from any point will not intersect any other point.')
	#plt.grid(True)
	dummy = np.random.randint(samples)
	plt.plot(np.linspace(mins[0]-100, maxs[0]+100,10), np.ones(10)*lhd[dummy,1], '--', c='C1')
	plt.plot(np.ones(10)*lhd[dummy,0], np.linspace(mins[1]-100, maxs[1]+100,10), '--', c='C1')
	plt.xlim(mins[0], maxs[0])
	plt.ylim(mins[1], maxs[1])
	plt.show()
	
	print('You can use the create function above in your code by')
	print("\"import create_LHS\" and then")
	print("\"lhd = create(n_params=n_params, samples=samples, mins=mins, maxs=maxs, outfile='lhs_params')\"")