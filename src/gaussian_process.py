import numpy as np
from sklearn.metrics import r2_score
import pickle
from . import helper_functions as hf

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
	print('Install gpytorch to use GPR_GPyTorch.')


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

    def save_model(self, filename, save_trainset=True):
        # np.save(filename, self.m.param_array)
        save_dict = {'kernel': self.m.kern.to_dict(), 'param_array': self.m.param_array}
        if save_trainset:
            save_dict['X'] = np.array(self.m.X)
            save_dict['Y'] = np.array(self.m.Y)
        pickle.dump(save_dict, open(filename, 'wb'))
        print('Model parameters are saved.')

    def load_model(self, filename, X=None, Y=None):
        load_dict = pickle.load(open(filename, 'rb'))
        self.kernel = GPy.kern.Kern.from_dict(load_dict['kernel'])
        # self.num_inducing = load_dict['num_inducing']
        if 'X' in load_dict.keys() and 'Y' in load_dict.keys():
            X = load_dict['X']
            Y = load_dict['Y']
        else:
            print('The file does not contain the training data.')
            print('Please provide it to the load_model through X and Y parameters.')
            return None
        
        m_load = GPy.models.GPRegression(X, Y, initialize=False, kernel=self.kernel)
        m_load.update_model(False)
        m_load.initialize_parameter()
        m_load[:] = load_dict['param_array']
        m_load.update_model(True)
        self.m = m_load
        return m_load


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

    def setup_model(self, X_train, y_train):
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

    def fit(self, X_train, y_train):
        self.setup_model(X_train, y_train)
        
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
        # if self.verbose:
        #     print(self.m)
        return self.m
        
    def predict(self, X_test, return_std=False):
        y_pred, y_std = self.m.predict(X_test)
        if return_std: return y_pred, y_std
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr

    def save_model(self, filename, save_trainset=True):
    	# np.save(filename, self.m.param_array)
    	save_dict = {'kernel': self.m.kern.to_dict(), 'param_array': self.m.param_array, 'num_inducing': self.num_inducing}
    	if save_trainset:
    		save_dict['X'] = np.array(self.m.X)
    		save_dict['Y'] = np.array(self.m.Y)
    	pickle.dump(save_dict, open(filename, 'wb'))
    	print('Model parameters are saved.')

    def load_model(self, filename, X=None, Y=None):
    	load_dict = pickle.load(open(filename, 'rb'))
    	self.kernel = GPy.kern.Kern.from_dict(load_dict['kernel'])
    	self.num_inducing = load_dict['num_inducing']
    	if 'X' in load_dict.keys() and 'Y' in load_dict.keys():
    		X = load_dict['X']
    		Y = load_dict['Y']
    	else:
    		print('The file does not contain the training data.')
    		print('Please provide it to the load_model through X and Y parameters.')
    		return None
    	
    	m_load = GPy.models.SparseGPRegression(X, Y, initialize=False, num_inducing=self.num_inducing, kernel=self.kernel)
    	m_load.update_model(False)
    	m_load.initialize_parameter()
    	m_load[:] = load_dict['param_array']
    	m_load.update_model(True)
    	self.m = m_load
    	return m_load
    	

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
    def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, n_restarts_optimizer=5, n_Xu=10, n_jobs=0, estimate_method='MLE', learning_rate=1e-3, method='VFE'):
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
        self.n_Xu    = n_Xu
        self.method  = method

    def fit_1out(self, train_x, train_y, n_Xu=None):
        if n_Xu is not None: self.n_Xu = n_Xu
        if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
        if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            input_dim = train_x.shape[1]
            self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

        self.Xu = np.linspace(train_x.min(axis=0)[0].data.numpy(), train_x.max(axis=0)[0].data.numpy(), self.n_Xu)
        self.Xu = torch.from_numpy(self.Xu)

        # create simple GP model
        model = gp.models.SparseGPRegression(train_x, train_y, self.kernel, Xu=self.Xu, jitter=1.0e-5, approx=self.method)

        # optimize
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.loss_fn is None: self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = np.array([])
        n_wait, max_wait = 0, 5

        for i in range(self.max_iter):
            optimizer.zero_grad()
            loss = self.loss_fn(model.model, model.guide)
            loss.backward()
            optimizer.step()
            losses = np.append(losses,loss.item()) 
            if self.verbose: 
                hf.loading_verbose('                                                       ')
                hf.loading_verbose('{0} {1:.2f}'.format(i+1, loss.item()))
            dloss = losses[-1]-losses[-2] if len(losses)>2 else self.tol*1000			
            if 0<=dloss and dloss<self.tol: n_wait += 1
            else: n_wait = 0
            if n_wait>=max_wait: break

        return model, optimizer, losses

    def fit(self, train_x, train_y, n_Xu=None):
        if n_Xu is not None: self.n_Xu = n_Xu
        if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
        if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            input_dim = train_x.shape[1]
            self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

        if train_y.ndim==1:
            model, optimizer, losses = self.fit_1out(train_x, train_y)
            self.model, self.optimizer, self.losses = model, optimizer, losses
        else:
            self.model, self.optimizer, self.losses = {}, {}, {}
            for i in range(train_y.shape[1]):
                print('Regressing output variable {}'.format(i))
                model, optimizer, losses = self.fit_1out(train_x, train_y)
                self.model[i], self.optimizer[i], self.losses[i] = model, optimizer, losses
                print('...done')

    def predict(self, X_test, return_std=True, return_cov=False):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        
        y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

        if return_std: 
            y_std = y_cov.diag().sqrt()
            return y_mean.detach().numpy(), y_std.detach().numpy()
        if return_cov: return y_mean.detach().numpy(), y_cov.detach().numpy()
        return y_mean.detach().numpy()

    def score(self, X_test, y_test):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        if type(y_test)==torch.Tensor: y_test = y_test.detach().numpy()

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
            if self.verbose: print(i+1, loss.item())
            dloss = self.losses[-1]-self.losses[-2]    			
            if 0<=dloss and dloss<self.tol: n_wait += 1
            else: n_wait = 0
            if self.n_wait>=self.max_wait: break

    def predict(self, X_test, return_std=True, return_cov=False):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        
        y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

        if return_std: 
            y_std = y_cov.diag().sqrt()
            return y_mean.detach().numpy(), y_std.detach().numpy()
        if return_cov: return y_mean.detach().numpy(), y_cov.detach().numpy()
        return y_mean.detach().numpy()

    def score(self, X_test, y_test):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        if type(y_test)==torch.Tensor: y_test = y_test.detach().numpy()

        y_pred = self.predict(X_test, return_std=False, return_cov=False)
        scr = r2_score(y_test, y_pred)
        return scr
