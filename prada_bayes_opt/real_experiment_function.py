# -*- coding: utf-8 -*-
"""
Name: real_experiment_functions.py
Authors: Julian Berk and Vu Nguyen
Publication date:08/04/2019
Description: These classes run real-world experiments that can be used to test
our acquisition functions

###############################IMPORTANT#######################################
The classes here all have file paths that need to be set correctlt for them to
work. Please make sure you change all paths before using a class
"""

import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVR
import math

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
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
        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class functions:
    def plot(self):
        print "not implemented"
        
    
class SVR_function:
    '''
    SVR_function: function to run SVR for tetsing the our method. The default
    dataset is the Space GA but othe datasets can be used.
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        self.maxbounds = np.array([[0.1,1000*20],[0.000001,1*20],[0.00001,5*20]])
        if bounds == None: 
            self.bounds = OrderedDict([('C',(0.1,1000)),('epsilon',(0.000001,1)),('gamma',(0.00001,5))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='SVR on Space GA'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_SVR(self,X,X_train,y_train,X_test,y_test):
        Xc=np.copy(X)/max_bound_size
        x1=Xc[0]*1000+0.1
        x2=Xc[1]+0.000001
        x3=Xc[2]*5+0.00001
        if x1<0.1:
            x1=0.1
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        # Fit regression model
        if(x1<=0):
            print("x1={}".format(x1))

        svr_model = SVR(kernel='rbf', C=x1, epsilon=x2,gamma=x3)
        svr_model.fit(X_train, y_train).predict(X_test)
        y_pred = svr_model.predict(X_test)
        
        squared_error=y_pred-y_test
        squared_error=np.mean(squared_error**2)
        
        RMSE=np.sqrt(squared_error)
        return RMSE
        
    def func(self,X):
        X=np.asarray(X)
        ##########################CHANGE PATH##################################    
        Xdata, ydata = self.get_data("C:\your_path\\real_experiment\\space_ga_scale")

        nTrain=np.int(0.7*len(ydata))
        X_train, y_train = Xdata[:nTrain], ydata[:nTrain]
        X_test, y_test = Xdata[nTrain+1:], ydata[nTrain+1:]
        ###############################################################################
        # Generate sample data

        #y_train=np.reshape(y_train,(nTrain,-1))
        #y_test=np.reshape(y_test,(nTest,-1))
        ###############################################################################

        
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_SVR(X,X_train,y_train,X_test,y_test)
        else:
            RMSE=np.zeros(X.shape[0])
            for i in range(0,np.shape(X)[0]):
                RMSE[i]=self.run_SVR(X[i],X_train,y_train,X_test,y_test)

        #print RMSE    
        return RMSE*self.ismax
        
class AlloyCooking_Profiling:
    '''
    Simulation for the cooking of an Aluminium Scandium alloy with two cooking stages
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,3*3600)),('Time2',(1*3600,3*3600)),('Temp1',(200,300)),('Temp2',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        if X.ndim==1:
            x1=X[0]
            x2=X[1]
            x3=X[2]
            x4=X[3]

        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.0004056486;# dataset1
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'C:\your_path\PradaBayesianOptimization\real_experiment\KWN_Heat_Treatment',nargout=0)

        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc=Xc/max_bound_size
        if X.ndim==1:
            Xc[0]=Xc[0]*3*3600+3600
            Xc[1]=Xc[1]*3*3600+3600
            Xc[2]=Xc[2]*100+200
            Xc[3]=Xc[3]*100+200
            
        else:
            Xc[:,0]=Xc[:,0]*3*3600+3600
            Xc[:,1]=Xc[:,1]*3*3600+3600
            Xc[:,2]=Xc[:,2]*100+200
            Xc[:,3]=Xc[:,3]*100+200
        if len(Xc.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(Xc)

        else:

            temp=np.apply_along_axis( self.run_Profiling,1,Xc)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]



        #utility_score=-AveRad/13+PhaseFraction/2
        utility_score=Strength
        
        return utility_score    

class AlloyCooking_Profiling_3Steps:
    '''
    Simulation for the cooking of an Aluminium Scandium alloy with three cooking stages
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        self.maxbounds = np.array([[0,20],[0,20],[0,20],[0,20],[0,20],[0,20]])
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,3*3600)),('Time2',(1*3600,3*3600)),('Time3',(1*3600,3*3600)),
                                       ('Temp1',(200,300)),('Temp2',(300,400)),('Temp3',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        if X.ndim==1:
            x1=X[0]
            x2=X[1]
            x3=X[2]
            x4=X[3]
            x5=X[4]
            x6=X[5]
            
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.0004056486;# dataset1
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'C:\your_path\PradaBayesianOptimization\real_experiment\KWN_Heat_Treatment',nargout=0)

        myCookTemp=matlab.double([x4,x5,x6])
        myCookTime=matlab.double([x1,x2,x3])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc=Xc/max_bound_size
        if X.ndim==1:
            Xc[0]=Xc[0]*3*3600+3600
            Xc[1]=Xc[1]*3*3600+3600
            Xc[2]=Xc[2]*3*3600+3600
            Xc[3]=Xc[3]*100+200
            Xc[4]=Xc[4]*100+300
            Xc[5]=Xc[5]*100+300
            
        else:
            Xc[:,0]=Xc[:,0]*3*3600+3600
            Xc[:,1]=Xc[:,1]*3*3600+3600
            Xc[:,2]=Xc[:,2]*3*3600+3600
            Xc[:,3]=Xc[:,3]*100+200
            Xc[:,4]=Xc[:,4]*100+300
            Xc[:,5]=Xc[:,5]*100+300
        if len(Xc.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(Xc)

        else:

            temp=np.apply_along_axis( self.run_Profiling,1,Xc)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]


        utility_score=Strength
        
        return utility_score  
 