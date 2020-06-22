import numpy as np
#from sklearn.model_selection import KFold
from scipy.integrate import simps
import warnings 
warnings.filterwarnings("ignore")
from . import distances
from . import helper_functions as hf
from . import bayesian_optimisation as bopt
from . import sampling_space as smp 
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GPRemul:
	def __init__(self, simulator, prior, bounds, gpr=None, verbose=True, N=100, sampling='LHS', param_file=None, output_file=None):
		#self.N_init  = N_init
		#self.N_max  = N_max
		self.N = N
		self.simulator = simulator
		#self.distance  = distance
		self.verbose = verbose
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.sampling = sampling

		self.setup_gpr(gpr=gpr)
		self.param_file  = param_file
		self.output_file = output_file

	def setup_gpr(self, gpr=None, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5):
		if gpr is not None:
			self.gpr = gpr
		else:
			if kernel is None: kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
			self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer)

	def create_params(self, samples):
		n_params = self.bounds.shape[0]
		#samples  = self.N
		mins, maxs = self.bounds.min(axis=1), self.bounds.max(axis=1)
		params = smp.LH_sampling(n_params=n_params, samples=samples, mins=mins, maxs=maxs, outfile=None)
		return params

	def get_params(self, param_file=None, save_file=None):
		if param_file is not None: self.param_file = param_file
		if self.param_file is not None: 
			if isinstance(self.param_file, (str)):
				self.params = np.loadtxt(self.param_file)
			elif isinstance(self.param_file, (np.ndarray)):
				self.params = self.param_file
			else:
				print('Please provide the params as a text file or numpy array.')
		else:
			params = self.create_params(self.N)
			self.params = params
		if save_file is not None:
			np.savetxt(save_file.split('.txt')[0]+'.txt', self.params)

	def get_training_set(self, train_file=None, save_file=None):
		if train_file is None: self.train_file = train_file
		if self.train_file is not None: 
			if isinstance(self.train_file, (str)):
				self.train_out = np.loadtxt(self.train_file)
			elif isinstance(self.train_file, (np.ndarray)):
				self.train_out = self.train_file
			else:
				print('Please provide the training set outputs as a text file or numpy array.')
		else:
			self.train_out = np.array([self.simulator(i) for i in self.params])
		if save_file is not None:
			np.savetxt(save_file.split('.txt')[0]+'.txt', self.train_out)

	def get_testing_set(self, test_file=None, test_param_file=None, save_file=None, N=50):
		if test_param_file is None: self.test_param_file = test_param_file
		if self.test_param_file is not None: 
			if isinstance(self.test_param_file, (str)):
				self.test_params = np.loadtxt(self.test_param_file)
			elif isinstance(self.test_file, (np.ndarray)):
				self.test_params = self.test_param_file
			else:
				print('Please provide the testing set outputs as a text file or numpy array.')
		else:
			self.test_params = self.create_params(N)
		
		if test_file is None: self.test_file = test_file
		if self.test_file is not None: 
			if isinstance(self.test_file, (str)):
				self.test_out = np.loadtxt(self.test_file)
			elif isinstance(self.test_file, (np.ndarray)):
				self.test_out = self.test_file
			else:
				print('Please provide the testing set outputs as a text file or numpy array.')
		else:
			self.test_out = np.array([self.simulator(i) for i in self.test_params])
		if save_file is not None:
			np.savetxt(test_file.split('.txt')[0]+'.txt', self.test_out)
			np.savetxt(test_param_file.split('.txt')[0]+'.txt', self.test_params)

	def run(self):
		## Setup the training set.
		self.get_params()
		self.get_training_set()
		## Training with GPR
		self.gpr(self.params, self.train_out)
		scr = gpr.score(params, outdata)
		print('Score: {0:.3f}'.format(scr))
		## Testing
		# Update this

