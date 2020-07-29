import numpy as np
from sklearn.metrics import r2_score

try: import GPy
except: print('Install GPy to use GPR_GPy and SparseGPR_GPy.')

try: import torch
except: print('Install PyTorch.')

try:
	import pyro
	import pyro.contrib.gp as gp
	import pyro.distributions as dist
except:
	print('Install Pyro to use GPR_pyro.')

try:
	import gpytorch
except:
	print('Install Pyro to use GPR_GPyTorch.')


class GPR_GPy:
    def __init__(self, max_iter=1000, max_f_eval=1000, kernel=None, verbose=True, n_restarts_optimizer=5, n_jobs=0):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.max_f_eval = max_f_eval
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
    
    def fit(self, X_train, y_train):
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            input_dim = X_train.shape[1]
            # self.kernel = GPy.kern.Matern52(input_dim,ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim,ARD=True)
        # create simple GP model
        self.m = GPy.models.GPRegression(X_train,y_train,self.kernel)
        # optimize
        if self.n_restarts_optimizer:
            self.m.optimize_restarts(
                num_restarts=self.n_restarts_optimizer,
                robust=False,
                #verbose=self.verbose,
                messages=self.verbose,
                parallel=True if self.n_jobs else False,
                num_processes=self.n_jobs if self.n_jobs else None,
                max_f_eval=self.max_f_eval,
                max_iters=self.max_iter,
                )
        else:
            self.m.optimize(messages=self.verbose, max_f_eval=self.max_f_eval)
        
    def predict(self, X_test, return_std=False):
        y_pred, y_var = self.m.predict(X_test)
        if return_std: return y_pred, np.sqrt(y_var)
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr


class SparseGPR_GPy:
    def __init__(self, max_iter=1000, max_f_eval=1000, kernel=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, num_inducing=10):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.max_f_eval = max_f_eval
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
        self.num_inducing = num_inducing
    
    def fit(self, X_train, y_train):
        input_dim = X_train.shape[1]
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            # self.kernel = GPy.kern.Matern52(input_dim,ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim,ARD=True)

        # define inducing points
        # self.Z = np.random.rand(self.num_inducing,input_dim)*(X_train.max(axis=0)-X_train.min(axis=0))+X_train.min(axis=0)

        # create simple GP model
        # self.m = GPy.models.SparseGPRegression(X_train,y_train,Z=self.Z,kernel=self.kernel)
        self.m = GPy.models.SparseGPRegression(X_train,y_train,num_inducing=self.num_inducing,kernel=self.kernel)

        # optimize
        if self.n_restarts_optimizer:
            self.m.optimize_restarts(
                num_restarts=self.n_restarts_optimizer,
                robust=False,
                #verbose=self.verbose,
                messages=self.verbose,
                parallel=True if self.n_jobs else False,
                num_processes=self.n_jobs if self.n_jobs else None,
                max_f_eval=self.max_f_eval,
                max_iters=self.max_iter,
                )
        else:
            self.m.optimize(messages=self.verbose, max_f_eval=self.max_f_eval)
        
    def predict(self, X_test, return_std=False):
        y_pred, y_std = self.m.predict(X_test)
        if return_std: return y_pred, y_std
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr

class SVGPR_GPy:
    def __init__(self, max_iter=1000, max_f_eval=1000, kernel=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, num_inducing=10):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.max_f_eval = max_f_eval
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
        self.num_inducing = num_inducing
    
    def fit(self, X_train, y_train):
        input_dim = X_train.shape[1]
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            # self.kernel = GPy.kern.Matern52(input_dim,ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim,ARD=True)

        # define inducing points
        #self.Z = np.random.rand(self.num_inducing,input_dim)*(X_train.max(axis=0)-X_train.min(axis=0))+X_train.min(axis=0)

        # create simple GP model
        self.m = GPy.models.SparseGPRegression(X,y,num_inducing=self.num_inducing,kernel=self.kernel)

        # optimize
        if self.n_restarts_optimizer:
            self.m.optimize_restarts(
                num_restarts=self.n_restarts_optimizer,
                robust=False,
                #verbose=self.verbose,
                messages=self.verbose,
                parallel=True if self.n_jobs else False,
                num_processes=self.n_jobs if self.n_jobs else None,
                max_f_eval=self.max_f_eval,
                max_iters=self.max_iter,
                )
        else:
            self.m.optimize(messages=self.verbose, max_f_eval=self.max_f_eval)
        
    def predict(self, X_test, return_std=False):
        y_pred, y_std = self.m.predict(X_test)
        if return_std: return y_pred, y_std
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr

class GPR_pyro:
	def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, estimate_method='MLE', learning_rate=1e-3):
		# define kernel
		self.kernel     = kernel
		self.max_iter   = max_iter
		self.verbose    = verbose
		self.n_jobs     = n_jobs
		self.n_restarts_optimizer = n_restarts_optimizer
		self.estimate_method = estimate_method
		self.learning_rate   = learning_rate
		self.loss_fn = loss_fn
		self.tol     = tol

	def fit(self, train_x, train_y):
		if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
		if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
		# check kernel
		if self.kernel is None:
			print('Setting kernel to Matern32.')
			input_dim = train_x.shape[1]
			self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

		# create simple GP model
		self.model = gp.models.GPRegression(train_x, train_y, kernel)

		# optimize
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		if self.loss_fn is None: self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
		self.losses = np.array([])
		n_wait, max_wait = 0, 5

		for i in range(self.max_iter):
			self.optimizer.zero_grad()
			loss = self.loss_fn(self.model.model, self.model.guide)
			loss.backward()
			self.optimizer.step()
			self.losses = np.append(self.losses,loss.item()) 
			print(i+1, loss.item())
			dloss = self.losses[-1]-self.losses[-2]    			
			if 0<=dloss and dloss<self.tol: n_wait += 1
			else: n_wait = 0
			if self.n_wait>=self.max_wait: break

	def predict(self, X_test, return_std=True, return_cov=False):
		y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

		if return_std: 
			y_std = cov.diag().sqrt()
			return y_pred, y_std
		if return_cov: return y_pred, y_cov
		return y_pred

	def score(self, X_test, y_test):
		y_pred = self.predict(X_test, return_std=False, return_cov=False)
		scr = r2_score(y_test, y_pred)
		return scr

class SparseGPR_pyro:
	def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, estimate_method='MLE', learning_rate=1e-3):
		# define kernel
		self.kernel     = kernel
		self.max_iter   = max_iter
		self.verbose    = verbose
		self.n_jobs     = n_jobs
		self.n_restarts_optimizer = n_restarts_optimizer
		self.estimate_method = estimate_method
		self.learning_rate   = learning_rate
		self.loss_fn = loss_fn
		self.tol     = tol

	def fit(self, train_x, train_y, n_Xu=10):
		if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
		if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
		# check kernel
		if self.kernel is None:
			print('Setting kernel to Matern32.')
			input_dim = train_x.shape[1]
			self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

		self.Xu = np.linspace(train_x.min(axis=0)[0].data.numpy(), train_x.max(axis=0)[0].data.numpy(), n_Xu)
		self.Xu = torch.from_numpy(self.Xu)

		# create simple GP model
		self.model = gp.models.SparseGPRegression(train_x, train_y, self.kernel, Xu=self.Xu, jitter=1.0e-5)

		# optimize
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		if self.loss_fn is None: self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
		self.losses = np.array([])
		n_wait, max_wait = 0, 5

		for i in range(self.max_iter):
			self.optimizer.zero_grad()
			loss = self.loss_fn(self.model.model, self.model.guide)
			loss.backward()
			self.optimizer.step()
			self.losses = np.append(self.losses,loss.item()) 
			print(i+1, loss.item())
			dloss = self.losses[-1]-self.losses[-2]    			
			if 0<=dloss and dloss<self.tol: n_wait += 1
			else: n_wait = 0
			if self.n_wait>=self.max_wait: break

	def predict(self, X_test, return_std=True, return_cov=False):
		y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

		if return_std: 
			y_std = cov.diag().sqrt()
			return y_pred, y_std
		if return_cov: return y_pred, y_cov
		return y_pred

	def score(self, X_test, y_test):
		y_pred = self.predict(X_test, return_std=False, return_cov=False)
		scr = r2_score(y_test, y_pred)
		return scr


class GPR_GPyTorch:
	def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, estimate_method='MLE', learning_rate=1e-3):
		# define kernel
		self.kernel     = kernel
		self.max_iter   = max_iter
		self.verbose    = verbose
		self.n_jobs     = n_jobs
		self.n_restarts_optimizer = n_restarts_optimizer
		self.estimate_method = estimate_method
		self.learning_rate   = learning_rate
		self.loss_fn = loss_fn
		self.tol     = tol

	def fit(self, train_x, train_y, n_Xu=10):
		if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
		if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
		# check kernel
		if self.kernel is None:
			print('Setting kernel to Matern32.')
			input_dim = train_x.shape[1]
			self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

		self.Xu = np.linspace(train_x.min(axis=0)[0].data.numpy(), train_x.max(axis=0)[0].data.numpy(), n_Xu)
		self.Xu = torch.from_numpy(self.Xu)

		# create simple GP model
		self.model = gp.models.SparseGPRegression(train_x, train_y, self.kernel, Xu=self.Xu, jitter=1.0e-5)

		# optimize
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		if self.loss_fn is None: self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
		self.losses = np.array([])
		n_wait, max_wait = 0, 5

		for i in range(self.max_iter):
			self.optimizer.zero_grad()
			loss = self.loss_fn(self.model.model, self.model.guide)
			loss.backward()
			self.optimizer.step()
			self.losses = np.append(self.losses,loss.item()) 
			print(i+1, loss.item())
			dloss = self.losses[-1]-self.losses[-2]    			
			if 0<=dloss and dloss<self.tol: n_wait += 1
			else: n_wait = 0
			if self.n_wait>=self.max_wait: break

	def predict(self, X_test, return_std=True, return_cov=False):
		y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

		if return_std: 
			y_std = cov.diag().sqrt()
			return y_pred, y_std
		if return_cov: return y_pred, y_cov
		return y_pred

	def score(self, X_test, y_test):
		y_pred = self.predict(X_test, return_std=False, return_cov=False)
		scr = r2_score(y_test, y_pred)
		return scr
