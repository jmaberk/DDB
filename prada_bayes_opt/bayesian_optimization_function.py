# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""

from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
#from visualization import Visualization
from prada_gaussian_process import PradaGaussianProcess
#from prada_gaussian_process import PradaMultipleGaussianProcess

from acquisition_maximization import acq_max
from acquisition_maximization import acq_max_global
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import optimize
from scipy import stats
from pyDOE import lhs
import matplotlib.pyplot as plt
from cycler import cycler
import time
import math


#@author: Julian

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
counter = 0

###############################################################################
'''
The max_bound_size variable below contols the size of the maximum allowed
bound. This is set at [0,max_bound_size] in each dimention.

###IMPORTANT###
This variable must be consistant in all of the following files:
1) acquisition_functions.py
2) bayesian_optimization_function.py
3) function.py
4) real_experiment_functon.py
'''
max_bound_size=10
###############################################################################
class PradaBayOptFn(object):

    def __init__(self, gp_params, func_params, acq_params, experiment_num, seed):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.theta:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
        
        experiment_num: the interation of the GP method. Used to make sure each 
                        independant stage of the experiment uses different 
                        initial conditions
        seed: Variable used as part of a seed to generate random initial points
                            
        Returns
        -------
        dim:            dimension
        scalebounds:    bound used thoughout the BO algorithm
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.experiment_num=experiment_num
        self.seed=seed
        np.random.seed(self.experiment_num*self.seed)
        
        # Prior distribution paramaters for the DDB method
        self.alpha=2
        self.beta=4
        
        # Find number of parameters
        bounds=func_params['bounds']
        if 'init_bounds' not in func_params:
            init_bounds=bounds
        else:
            init_bounds=func_params['init_bounds']
        # Find input dimention
        self.dim = len(bounds)
        self.radius=np.ones([self.dim,1])

        # Generate bound array
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        # find function to be optimized
        self.f = func_params['f']

        # acquisition function type
        
        self.acq=acq_params['acq_func']
        
        # Check if the search space is to be modified
        self.bb_function=acq_params["bb_function"]
        if 'expandSS' not in acq_params:
            self.expandSS=0
        else:                
            self.expandSS=acq_params['expandSS']
        # Check if the bound is to be set randomly. If so, shift the bound by a random amount
        if (acq_params['random_initial_bound']==1):
            randomizer=np.random.rand(self.dim)*max_bound_size
            for d in range(0,self.dim):
                self.scalebounds[d]=self.scalebounds[d]+randomizer[d]
        # Other checks
        if 'debug' not in self.acq:
            self.acq['debug']=0           
        if 'stopping' not in acq_params:
            self.stopping_criteria=0
        else:
            self.stopping_criteria=acq_params['stopping']
        if 'optimize_gp' not in acq_params:
            self.optimize_gp=0
        else:                
            self.optimize_gp=acq_params['optimize_gp']
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
        
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            if self.acq['name']=='ei_reg':
                self.opt_toolbox='unbounded'
            else:
                self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
        self.iteration_factor=acq_params['iteration_factor']
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        
        # value of the acquisition function at the selected point
        self.alpha_Xt=None
        self.Tau_Xt=None
        
        self.time_opt=0

        self.k_Neighbor=2
        
        # Gaussian Process class
        self.gp=PradaGaussianProcess(gp_params)
        self.gp_params=gp_params

        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0
        self.logmarginal=0
        
        # xt_suggestion, caching for Consensus
        self.xstars=[]
        self.ystars=np.zeros((2,1))
        
        # theta vector for marginalization GP
        self.theta_vector =[]
    
    def init(self,gp_params, n_init_points=3):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """
        # set seed to allow for reproducible results
        np.random.seed(self.experiment_num*self.seed)
        print(self.experiment_num)
        #Generate initial points on grid
        l=np.zeros([n_init_points,self.dim])
        bound_length=self.scalebounds[0,1]-self.scalebounds[0,0]
        for d in range(0,self.dim):
            l[:,d]=lhs(n_init_points)[:,0]
        self.X=np.asarray(l)+self.scalebounds[:,0]         
        self.X=self.X*bound_length #initial inouts
        print("starting points={}".format(self.X))
        print("starting bounds={}".format(self.scalebounds))
        y_init=self.f(self.X)
        y_init=np.reshape(y_init,(n_init_points,1))
        self.Y_original = np.asarray(y_init)     #initial outputs   
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original) #outputs normalised
        print("starting Y values={}".format(self.Y))

            
        #############Rename#############
    def radiusPDF(self,r,alpha,beta,b,ymax,a):
        """
        Description: Evaluates the posterior distribution for our DDB method
        Input parameters
        ----------
        r:            radius to be evaluated   
        alpha:        # gamma distribution shape paramater
        beta:         # gamma distribution rate paramater
        a:            # log-logistic distribution scale paramater
        b:            # log-logistic distribution rate paramater with y_max
        y_max:            # log-logistic distribution rate paramater with b

        Output: posterior distribution evaluated at r
        """
        gamma=stats.gamma.pdf(r,alpha,scale=1/beta)
        loglog=stats.fisk.pdf(r,ymax/b,scale=a)
        P=gamma*loglog
        return -P
    def sufficientBoundPDF(self,r,bDivYmax,a):
        """
        Description: Evaluates the likelihood distribution for our DDB method
        Input parameters
        ----------
        r:            radius to be evaluated   
        a:            # log-logistic distribution scale paramater
        bDivYmax:            # log-logistic distribution rate paramater
       
        Output: likelihood distribution evaluated at r
        """
        P=stats.fisk.cdf(r,bDivYmax,scale=a)
        return P
    
    def expandBoundsDDB_MAP(self):
        """
        Description: Expands the search space with the MAP implementation of
        our DDB method

        """
        
        print('Attempting to expand search space with DDB-MAP method')
        alpha=self.alpha
        beta=self.beta
        bound_samples=100    # Number of radius sample to fit the log-logistic distribution
        # Find y^+ and x^+
        ymax=np.max(self.Y)
        # Generate test radii
        max_loc=np.argmax(self.Y)
        xmax=self.X[max_loc]
        test_bound=np.zeros(self.scalebounds.shape)
        bound_dist=np.zeros(bound_samples)
        bound_center=xmax
        test_bound[:,1]=bound_center+0.5
        test_bound[:,0]=bound_center-0.5
        max_radius=np.max(np.array([np.max(max_bound_size-test_bound[:,1]),np.max(test_bound[:,0])]))
        step=max_radius/bound_samples
        packing_number=np.zeros(bound_samples)
        # Generate a Thompson sample maxima to estimate internal maxima
        TS=AcquisitionFunction.ThompsonSampling(self.gp)
        tsb_x,tsb_y=acq_max_global(TS, self.gp, bounds=self.scalebounds)
        # Generate Gumbel samples to estimate the external maxima
        for i in range(0,bound_samples):
            bound_length=test_bound[:,1]-test_bound[:,0]
            volume=np.power(max_bound_size,self.dim)-np.prod(bound_length)
            packing_number[i]=round(volume/(5*self.gp.lengthscale))
            mu=stats.norm.ppf(1.0-1.0/packing_number[i])
            sigma=stats.norm.ppf(1.0-(1.0/packing_number[i])*np.exp(-1.0))-stats.norm.ppf(1.0-(1.0/(packing_number[i])))
            bound_dist[i]=np.exp(-np.exp(-(-tsb_y-mu)/sigma))
            test_bound[:,1]=test_bound[:,1]+step
            test_bound[:,0]=test_bound[:,0]-step
        bound_dist[np.isnan(bound_dist)]=1
        # Fit the log-logistic paramaters to the Gumbel samples
        xfit=np.arange(0,max_radius,max_radius/100)
        popt,pcov=optimize.curve_fit(self.sufficientBoundPDF,xfit[0:100],bound_dist,bounds=np.array([[5,1.1],[20,5]]))
        print("popt={}".format(popt))
        b=ymax/popt[0]
        a=popt[1]
        print("b={}, ymax={}".format(b,ymax))
        # Find the gamma and log-logistic modes to determine the optimisation bound
        c=ymax/b
        loglog_mode=a*np.power((c-1.0)/(c+1.0),(1/c))
        gamma_mode=(alpha-1)/beta
        opt_bound=np.ones([2])
        opt_bound[0]=min(loglog_mode,gamma_mode)
        opt_bound[1]=max(loglog_mode,gamma_mode)
        bound_range=(opt_bound[1]-opt_bound[0])
        # Find MAP Estimate of radius r
        for d in range(0,self.dim):
            r_max=0
            p_max=0
            for x0 in np.arange(opt_bound[0],opt_bound[1],bound_range/10):
                res=optimize.minimize(lambda x: self.radiusPDF(x,alpha,beta,b,ymax,a),x0=x0, bounds=np.array([opt_bound]), method='L-BFGS-B')
                if -res.fun>p_max:
                    r_max=res.x
                    p_max=-res.fun
            if r_max>opt_bound[1]:
                r_max=opt_bound[1]
            xplot=np.arange(0,10,0.01)
            yplot=-self.radiusPDF(xplot,alpha,beta,b,ymax,a)
            max_loc=np.argmax(yplot)

            print("optimal radius of {} with unscaled probability of {}".format(r_max,p_max))
            self.scalebounds[d,1]=xmax[d]+r_max
            self.scalebounds[d,0]=xmax[d]-r_max
        print("seach space extended to {} with DDB".format(self.scalebounds))
        
    def expandBoundsDDB_FB(self):
        """
        Description: Expands the search space with the full Bayesian 
        implementation of our DDB method

        """
        print('Attempting to expand search space with DDB-FB method')
        alpha=self.alpha
        beta=self.beta
        bound_samples=100    # Number of radius sample to fit the log-logistic distribution
        # Find y^+ and x^+
        ymax=np.max(self.Y)
        # Generate test radii
        max_loc=np.argmax(self.Y)
        xmax=self.X[max_loc]
        test_bound=np.zeros(self.scalebounds.shape)
        bound_dist=np.zeros(bound_samples)
        bound_center=xmax
        test_bound[:,1]=bound_center+0.5
        test_bound[:,0]=bound_center-0.5
        max_radius=np.max(np.array([np.max(max_bound_size-test_bound[:,1]),np.max(test_bound[:,0])]))
        step=max_radius/bound_samples
        packing_number=np.zeros(bound_samples)
        # Generate a Thompson sample maxima to estimate internal maxima
        TS=AcquisitionFunction.ThompsonSampling(self.gp)
        tsb_x,tsb_y=acq_max_global(TS, self.gp, bounds=self.scalebounds)
        # Generate Gumbel samples to estimate the external maxima
        for i in range(0,bound_samples):
            bound_length=test_bound[:,1]-test_bound[:,0]
            volume=np.power(max_bound_size,self.dim)-np.prod(bound_length)
            packing_number[i]=round(volume/(5*self.gp.lengthscale))
            mu=stats.norm.ppf(1.0-1.0/packing_number[i])
            sigma=stats.norm.ppf(1.0-(1.0/packing_number[i])*np.exp(-1.0))-stats.norm.ppf(1.0-(1.0/(packing_number[i])))
            bound_dist[i]=np.exp(-np.exp(-(-tsb_y-mu)/sigma))
            test_bound[:,1]=test_bound[:,1]+step
            test_bound[:,0]=test_bound[:,0]-step
        bound_dist[np.isnan(bound_dist)]=1
        # Fit the log-logistic paramaters to the Gumbel samples
        xfit=np.arange(0,max_radius,max_radius/100)
        popt,pcov=optimize.curve_fit(self.sufficientBoundPDF,xfit[0:100],bound_dist,bounds=np.array([[5,1.1],[20,5]]))
        print("popt={}".format(popt))
        b=ymax/popt[0]
        a=popt[1]
        print("b={}, ymax={}".format(b,ymax))
        # Sample for the optimal radius
        for d in range(0,self.dim):
            gamma=np.random.gamma(shape=alpha,scale=1/beta,size=100)
            loglog=stats.fisk.pdf(gamma,ymax/b,scale=a)
            scaled_weights=loglog/np.sum(loglog)
            multi=np.random.multinomial(1,scaled_weights)
            r_index=np.argmax(multi)
            print("Radius of {} selected".format(gamma[r_index]))
            self.scalebounds[d,1]=xmax[d]+gamma[r_index]
            self.scalebounds[d,0]=xmax[d]-gamma[r_index]

        print("seach space extended to {} with DDB".format(self.scalebounds))
        
                            
    def lcb(self,x, gp):
        """
        Calculates the GP-LCB acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0 #prevents negative variances obtained through comp errors
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        beta=2*np.log(len(gp.Y)*np.square((self.experiment_num+1)*math.pi)/(6*0.9))  
        return mean - np.sqrt(beta) * np.sqrt(var) 
    
    def ucb(self,x, gp):
        """
        Calculates the GP-UCB acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0 #prevents negative variances obtained through comp errors
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T   
        beta=2*np.log(len(gp.Y)*np.square(self.experiment_num*math.pi)/(6*0.9))         
        return mean + np.sqrt(beta) * np.sqrt(var)
                      
    def expandBoundsFiltering(self):
        step=0.1*self.gp.lengthscale
        print('Attempting to expand search space with FBO method')
        myopts ={'maxiter':3*self.dim,'maxfun':5*self.dim}
        
        #print("Initial search space: {}".format(self.scalebounds))
        extended_bound=np.copy(self.scalebounds)
        extention=math.pow(self.iteration_factor/(max([self.experiment_num,1])),(1/self.dim))
       # print("iteration_factor={}, experiment_num={}, extention={}".format(self.iteration_factor,self.experiment_num, extention))
        old_radius=(extended_bound[:,1]-extended_bound[:,0])/2
        mid_point=extended_bound[:,0]+old_radius
        new_radius=old_radius*extention
        #print("radius={}".format(new_radius))
        extended_bound[:,1]=mid_point+new_radius
        extended_bound[:,0]=mid_point-new_radius
#        if (self.bb_function.name=="SVR_function")|(self.bb_function.name=="BayesNonMultilabelClassification")|(self.bb_function.name=="AlloyCooking_Profiling"):
#            for d in range(0,self.dim):
#                extended_bound[d,0]=max(extended_bound[d,0],self.bb_function.maxbounds[d,0])
#                extended_bound[d,1]=min(extended_bound[d,1],self.bb_function.maxbounds[d,1])
        print("bounds.shape={}".format(extended_bound.shape))
        lcb_x,lcb_y=acq_max_global(self.lcb, self.gp, extended_bound)
        for d in range(0,self.dim):
            #Upper bound
            x_boundry=np.max(self.X[d],axis=0)
            x_boundry_index=np.argmax(self.X[d],axis=0)
            ucb_y=self.ucb(self.X[x_boundry_index],self.gp)
            while((ucb_y>lcb_y)&(x_boundry<extended_bound[d,1])):
                x_boundry=x_boundry+step
                ucb_y=self.ucb(self.X[x_boundry_index],self.gp)
            extended_bound[d,1]=x_boundry
            #Lower bound
            x_boundry=np.min(self.X[d],axis=0)
            ucb_y=self.ucb(self.X[x_boundry_index],self.gp)
            while((ucb_y>lcb_y)&(x_boundry>extended_bound[d,0])):
                x_boundry=x_boundry-step
                ucb_y=self.ucb(self.X[x_boundry_index],self.gp)
            extended_bound[d,0]=x_boundry
                
            self.scalebounds=extended_bound
#        if (self.bb_function.name=="SVR_function")|(self.bb_function.name=="BayesNonMultilabelClassification")|(self.bb_function.name=="AlloyCooking_Profiling"):
#            for d in range(0,self.dim):
#                self.scalebounds[d,0]=max(self.scalebounds[d,0],self.bb_function.maxbounds[d,0])
#                self.scalebounds[d,1]=min(self.scalebounds[d,1],self.bb_function.maxbounds[d,1])
        print("seach space extended to {}".format(self.scalebounds))
        
    def volumeDoubling(self):
        print('Attempting to expand search space with volume doubling method')
        extended_bound=np.copy(self.scalebounds)
        old_radius=(extended_bound[:,1]-extended_bound[:,0])/2
        volume=np.power(2*old_radius,self.dim)
        mid_point=extended_bound[:,0]+old_radius
        new_radius=np.power(2*volume,1/self.dim)/2      
        extended_bound[:,0]=mid_point-new_radius
        extended_bound[:,1]=mid_point+new_radius
        self.scalebounds=extended_bound

        print("seach space extended to {}".format(self.scalebounds))
        
    def maximize(self,gp_params):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.scalebounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

 
        acq=self.acq

        if acq['debug']==1:
            logmarginal=self.gp.log_marginal_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            print(gp_params['theta'])
            print("log marginal before optimizing ={:.4f}".format(logmarginal))
            self.logmarginal=logmarginal
                
            if logmarginal<-999999:
                logmarginal=self.gp.log_marginal_lengthscale(gp_params['theta'],gp_params['noise_delta'])

        if self.optimize_gp==1 and len(self.Y)%2*self.dim==0 and len(self.Y)>5*self.dim:

            print("Initial length scale={}".format(gp_params['theta']))
            newtheta = self.gp.optimize_lengthscale(gp_params['theta'],gp_params['noise_delta'],self.scalebounds)
            gp_params['theta']=newtheta
            print("New length scale={}".format(gp_params['theta']))

            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

            
        # Modify search space based on selected method
        if self.expandSS=='expandBoundsDDB_MAP':
			self.expandBoundsDDB_MAP()
        if self.expandSS=='expandBoundsDDB_FB':
			self.expandBoundsDDB_FB()            
        if self.expandSS=='expandBoundsFiltering':
            self.expandBoundsFiltering()
        if self.expandSS=='volumeDoubling' and len(self.Y)%3*self.dim==0:
            self.volumeDoubling()
        # Prevent bounds from breaching maximum limit
        for d in range(0,self.dim):
            if self.scalebounds[d,0]<0:
                print('Lower bound of {} in dimention {} exceeded minimum bound of {}. Scaling up.'.format(self.scalebounds[d,0],d,0))
                self.scalebounds[d,0]=0
                print('bound set to {}'.format(self.scalebounds))
            if self.scalebounds[d,1]>max_bound_size:
                print('Upper bound of {} in dimention {} exceeded maximum bound of {}. Scaling down.'.format(self.scalebounds[d,1],d,max_bound_size))
                self.scalebounds[d,1]=max_bound_size
                self.scalebounds[d,0]=min(self.scalebounds[d,0],self.scalebounds[d,1]-np.sqrt(3*self.gp.lengthscale))
                print('bound set to {}'.format(self.scalebounds))
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()
        
        if acq['name'] in ['consensus','mes']: 
            ucb_acq_func={}
            ucb_acq_func['name']='ucb'
            ucb_acq_func['kappa']=np.log(len(self.Y))
            ucb_acq_func['dim']=self.dim
            ucb_acq_func['scalebounds']=self.scalebounds
        
            myacq=AcquisitionFunction(ucb_acq_func)
            xt_ucb = acq_max(ac=myacq.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
            
            xstars=[]
            xstars.append(xt_ucb)
            
            ei_acq_func={}
            ei_acq_func['name']='ei'
            ei_acq_func['dim']=self.dim
            ei_acq_func['scalebounds']=self.scalebounds
        
            myacq=AcquisitionFunction(ei_acq_func)
            xt_ei = acq_max(ac=myacq.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
            xstars.append(xt_ei)
                 
            
            pes_acq_func={}
            pes_acq_func['name']='pes'
            pes_acq_func['dim']=self.dim
            pes_acq_func['scalebounds']=self.scalebounds
        
            myacq=AcquisitionFunction(pes_acq_func)
            xt_pes = acq_max(ac=myacq.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
            xstars.append(xt_pes)
            
            
            self.xstars=xstars            
            
        if acq['name']=='vrs':
            print("please call the maximize_vrs function")
            return
                      
        if 'xstars' not in globals():
            xstars=[]
            
        self.xstars=xstars

        self.acq['xstars']=xstars
        self.acq['WW']=False
        self.acq['WW_dim']=False
        self.acq_func = AcquisitionFunction(self.acq,self.bb_function)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean
            mu_acq={}
            mu_acq['name']='mu'
            mu_acq['dim']=self.dim
            acq_mu=AcquisitionFunction(mu_acq)
            x_mu_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
            # set y_max = mu_max
            y_max=acq_mu.acq_kind(x_mu_max,gp=self.gp, y_max=y_max)

        
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)

        if acq['name']=='consensus' and acq['debug']==1: # plot the x_max and xstars
            fig=plt.figure(figsize=(5, 5))

            plt.scatter(xt_ucb[0],xt_ucb[1],marker='s',color='g',s=200,label='Peak')
            plt.scatter(xt_ei[0],xt_ei[1],marker='s',color='k',s=200,label='Peak')
            plt.scatter(x_max[0],x_max[1],marker='*',color='r',s=300,label='Peak')
            plt.xlim(0,1)
            plt.ylim(0,1)
            strFileName="acquisition_functions_debug.eps"
            fig.savefig(strFileName, bbox_inches='tight')

        if acq['name']=='vrs' and acq['debug']==1: # plot the x_max and xstars
            fig=plt.figure(figsize=(5, 5))

            plt.scatter(xt_ucb[0],xt_ucb[1],marker='s',color='g',s=200,label='Peak')
            plt.scatter(xt_ei[0],xt_ei[1],marker='s',color='k',s=200,label='Peak')
            plt.scatter(x_max[0],x_max[1],marker='*',color='r',s=300,label='Peak')
            plt.xlim(0,1)
            plt.ylim(0,1)
            strFileName="vrs_acquisition_functions_debug.eps"
            #fig.savefig(strFileName, bbox_inches='tight')
            
            
        val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
        #print x_max
        #print val_acq
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)

            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # evaluate Y using original X
        self.Y_original = np.append(self.Y_original, self.f(x_max))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
        if self.gp.flagIncremental==1:
            self.gp.fit_incremental(x_max,self.Y[-1])
#        if (self.acq['name']=='ei_regularizerH') or (self.acq['name']=='ei_regularizerQ'):
#            self.scalebounds[:,0]=self.scalebounds[:,0]+1
#            self.scalebounds[:,1]=self.scalebounds[:,1]-1
#        self.acq['scalebounds']=self.scalebounds
        self.experiment_num=self.experiment_num+1

   