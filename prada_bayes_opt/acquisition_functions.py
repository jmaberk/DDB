from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from acquisition_maximization import acq_max


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
class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq, bb_function=False):
        self.acq=acq
        acq_name=acq['name']
        
        if bb_function:
            self.bb_function=bb_function

        if 'WW' not in acq:
            self.WW=False
        else:
            self.WW=acq['WW']
        if 'WW_dim' not in acq:
            self.WW_dim=False
        else:
            self.WW_dim=acq['WW_dim']
        ListAcq=['ei']
        
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        self.dim=acq['dim']
        
        if 'scalebounds' not in acq:
            self.scalebounds=[0,1]*self.dim
            
        else:
            self.scalebounds=acq['scalebounds']
        self.initialized_flag=0
        self.objects=[]

    def acq_kind(self, x, gp, y_max):

        #print self.kind
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'ei_regularizerH':
            bound=np.array(self.scalebounds).reshape(self.dim,-1)
            length=bound[:,1]-bound[:,0]
            x_bar=bound[:,0]+length/2
            return self._ei_regularizerH(x, gp, y_max,x_bar=x_bar,R=0.5)
    
    @staticmethod
    def _ei(x, gp, y_max):
        """
        Calculates the EI acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
                y_max: The maxima of the found y values
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-8 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-8]=0
        return out
    
    def _ei_regularizerH(self,x, gp, y_max,x_bar,R=0.5):
        """
        Calculates the EI acquisition function values with a hinge regulariser
        Inputs: gp: The Gaussian process, also contains all data
                y_max: The maxima of the found y values
                x:The point at which to evaluate the acquisition function 
                x_bar: Centroid for the regulariser
                R: Radius for the regulariser
        Output: acq_value: The value of the aquisition function at point x
        """
        mean, var = gp.predict(x, eval_MSE=True)
        extended_bound=np.array(self.scalebounds).copy()
        extended_bound=extended_bound.reshape(self.dim,-1)
        extended_bound[:,0]=extended_bound[:,0]-2
        extended_bound[:,1]=extended_bound[:,1]+2
        for d in range(0,self.dim):
            extended_bound[d,0]=max(extended_bound[d,0],0)
            extended_bound[d,1]=min(extended_bound[d,1],max_bound_size)
        #compute regularizer xi
        dist= np.linalg.norm(x - x_bar)
        if dist>R:
            xi=dist/R-1
        else:
            xi=0
        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
            z = (mean - y_max-y_max*xi)/np.sqrt(var)        
            out=(mean - y_max-y_max*xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)        
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])           
                z[idx] = (mean[idx] - y_max-y_max*xi)/np.sqrt(var[idx])
                out[idx]=(mean[idx] - y_max-y_max*xi) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000: 

                return out
            else:
                return np.mean(out)# get mean over acquisition functions
                return np.prod(out,axis=0) # get product over acquisition functions
    
    
    class ThompsonSampling(object):
        """
        Class used for calulating Thompson samples. Re-usable calculations are
        done in __init__ to reduce compuational cost.
        """
        #Calculates the thompson sample paramers 
        def __init__(self,gp,seed=False):
            var_mag=1
            ls_mag=1
            if seed!=False:
                np.random.seed(seed)
            dim=gp.X.shape[1]
            # used for Thompson Sampling
            self.WW_dim=200 # dimension of random feature
            self.WW=np.random.multivariate_normal([0]*self.WW_dim,np.eye(self.WW_dim),dim)/(gp.lengthscale*ls_mag)
            self.bias=np.random.uniform(0,2*3.14,self.WW_dim)

            # computing Phi(X)^T=[phi(x_1)....phi(x_n)]
            Phi_X=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(gp.X,self.WW)+self.bias), np.cos(np.dot(gp.X,self.WW)+self.bias)]) # [N x M]
            
            # computing A^-1
            A=np.dot(Phi_X.T,Phi_X)+np.eye(2*self.WW_dim)*gp.noise_delta*var_mag
            gx=np.dot(Phi_X.T,gp.Y)
            self.mean_theta_TS=np.linalg.solve(A,gx)
        #Calculates the thompson sample value at the point x    
        def __call__(self,x,gp):
            phi_x=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(x,self.WW)+self.bias), np.cos(np.dot(x,self.WW)+self.bias)])
            
            # compute the TS value
            gx=np.dot(phi_x,self.mean_theta_TS)    
            return gx
  
    
def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
