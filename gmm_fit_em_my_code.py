import sys
sys.path.append('/home/prakash/gps/python/gps/Linear EM with GPS')
import numpy as np
from scipy.stats import multivariate_normal
from expectation_maximization_with_kalman import *
import scipy.linalg
import numpy as np
import scipy.io as sio
import h5py 
from numpy import *
import math
import pandas as pd
import csv
import pprint, pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
xux_shape=14

class GMM:
    
    def __init__(self,X, k = 3, eps = 0.0001,strength=0.9):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`
        self._strength= strength
        # All parameters from fitting/learning are kept in a named tuple
        from collections import namedtuple
        self.X=X
    

        

    def fit_EM(self, X, max_iters = 1000):
        
        # n = number of data-points, d = dimension of data points        
        n, d = X.shape
        
        # randomly choose the starting centroids/means 
        ## as 3 of the points from datasets        
        mu = X[np.random.choice(n, self.k, False), :]
        
        # initialize the covariance matrices for each gaussians
        Sigma = [np.eye(d)] * self.k
        
        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k
        
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))
        
        ### log_likelihoods
        log_likelihoods = []
        

                        

        # Iterate till max_iters iterations        
        while len(log_likelihoods) < max_iters:
            
            # E - Step
            
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            #print np.linalg.det(Sigma[0])
            
            for k in range(self.k):
            
                R[:, k] = w[k] * estep_calc(X,mu[k], Sigma[k]) + 1* 2e-9
            
            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            #print R.shape
            #print R
            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T 
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            # M Step
            ## calculate the new mean and covariance for each gaussian by 
            ## utilizing the new responsibilities
            for k in range(self.k):
                
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])
                
                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps:
                print "GMM iterations are converged now"
                break
        
        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)       
        return self.params
    def extract_samples():
        return self.X
    def predict(self, x):
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in \
            zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs/np.sum(probs)
        
def estep_calc(X,mu,s):

    """         P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
            * np.exp(-.5 * np.einsum('ij, ij -> i',\
                    X - mu, np.dot(    np.linalg.inv(s) , (X - mu).T).T ) )  """
    #logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)
    s=0.5 * (s + np.transpose(s)) + 1e-6 * np.eye(X.shape[1])
    if np.isnan(s).any():
        print "nans appearing in the s matrix inside estep_calc"

    L = scipy.linalg.cholesky(s, lower=True)
    #logobs[:, i] -= np.sum(np.log(np.diag(L)))
    diff = (X - mu).T
    soln = scipy.linalg.solve_triangular(L, diff, lower=True)
    inverse_mat= np.exp (-.5 * np.sum(soln**2,axis=0))
    P1=np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) * inverse_mat
    

    """     P = np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
        * np.exp(-.5 * np.einsum('ij, ij -> i',\
                X - mu, np.dot(    np.linalg.inv(np.linalg.cholesky(s))**2 , (X - mu).T).T ) ) 
    print P.shape """
    return P1

##du is 6 and dx is 8 below...change of terms now
def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
    """ Perform Gaussian fit to data with a prior. """
    # Build weights matrix.
    D = np.diag(dwts)
    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    
    empsig = 0.5 * (empsig + empsig.T)
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) *
             np.outer(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T) + 1e-6 * np.eye(pts.shape[1])
    
    #print mu.shape
    # Add sigma regularization.
    sigma += sig_reg 
    #print "Conditioning the gaussian now::"
    # Conditioning to get dynamics.
    ##du is 6 and dx is 8
   # dynsig_dyn=sigma[dX+dU:, dX+dU:]-(np.linalg.solve(sigma[:dX+dU,:dX+dU],sigma[:dX+dU,dX+dU:]).T).dot(sigma[:dX+dU,dX+dU:])
    
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
    #print "fd shape is ",fd.shape
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    #print "fc shape is ",fc.shape
    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    #print "dyn_sigma shape is ",dynsig.shape
    return fd, fc, dynsig


def higher_dimensions_dynamics_learn(X):
    
    gmm = GMM(X,k=20, eps=0.0001)
    if np.isnan(X).any():
        print "nans appearing in the Sigmadyn"
    params = gmm.fit_EM(X, max_iters= 300)
    
    dX=6
    dU=2
    k=20
    mu_param=np.zeros((k,14)) 
    sigma_param=np.zeros((k,14,14)) 
    wghts=np.array(params.w).reshape(k,1)
    for iter in range (20):
        mu_param[iter,:]=params.mu[iter,:]
        sigma_param[iter,:,:]=params.Sigma[iter]
    #print mu_param.shape
    #print sigma_param.shape
    #print wghts.shape
    mu_moment = np.sum(mu_param * wghts, axis=0)
    diff = mu_param - np.expand_dims(mu_moment, axis=0)
    diff_expand = np.expand_dims(mu_param, axis=1) * \
        np.expand_dims(diff, axis=2)
    wts_expand = np.expand_dims(wghts, axis=2)
    sigma_moment = np.sum((sigma_param + diff_expand) * wts_expand, axis=0)
    m = X.shape[0]
    dwts = (1.0 / X.shape[0]) * np.ones(X.shape[0])
    n0 = m - 2 - mu_moment.shape[0]
    # Normalize.
    m = float(m) / X.shape[0]
    n0 = float(n0) / X.shape[0]
        # Factor in multiplier.
    """     n0 = n0 * self._strength
    m = m * self._strength """
    # Multiply Phi by m (since it was normalized before).
    sigma_moment *= m 
    it = slice(dX+dU)
    sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
    sig_reg[it, it] = 1e-6
    #print sig_reg[it,it]
    Ys=np.c_[X[:, :dX], X[:,dX:dX+dU], X[:, dX+dU:dX+dU+dX]]
    #print sigma_moment
    Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,mu_moment, sigma_moment, m, n0, dwts, dX+dU, dX, sig_reg)

    return Fm,fv,dyn_covar,params.log_likelihoods



def reward_learn(X):

    gmm = GMM(X,k=20, eps=0.0001)
    params = gmm.fit_EM(X, max_iters= 200)
    dX=6
    dU=2
    dR=1
    k=20
    mu_param=np.zeros((k,dX+dU+dR)) 
    sigma_param=np.zeros((k,dX+dU+dR,dX+dU+dR)) 
    wghts=np.array(params.w).reshape(k,1)
    for iter in range (20):
        mu_param[iter,:]=params.mu[iter,:]
        sigma_param[iter,:,:]=params.Sigma[iter]

    mu_moment = np.sum(mu_param * wghts, axis=0)
    diff = mu_param - np.expand_dims(mu_moment, axis=0)
    diff_expand = np.expand_dims(mu_param, axis=1) * \
        np.expand_dims(diff, axis=2)
    wts_expand = np.expand_dims(wghts, axis=2)
    sigma_moment = np.sum((sigma_param + diff_expand) * wts_expand, axis=0)
    m = X.shape[0]
    dwts = (1.0 / X.shape[0]) * np.ones(X.shape[0])
    n0 = m - 2 - mu_moment.shape[0]
    # Normalize.
    m = float(m) / X.shape[0]
    n0 = float(n0) / X.shape[0]
        # Factor in multiplier.
    """     n0 = n0 * self._strength
    m = m * self._strength """
    # Multiply Phi by m (since it was normalized before).
    sigma_moment *= m
    it = slice(dX+dU)
    sig_reg = np.zeros((dX+dU+dR, dX+dU+dR))
    sig_reg[it, it] = 1e-6
    #print sig_reg[it,it]
    Ys=np.c_[X[:, :dX], X[:,dX:dX+dU], X[:, dX+dU]]
    Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,mu_moment, sigma_moment, m, n0, dwts, dX+dU, dX, sig_reg)
    return Fm,fv,dyn_covar,params.log_likelihoods



def main_call(X,K,k,sig):
    A_dyn,B_dyn,sigma_dyn,log_likelihood_dynamics_fitting = higher_dimensions_dynamics_learn(X)
    beta=1  
    Q_dim=6
    gamma=1
    q_dim=2
    T=50
    small_fac=1e-15
    Q = small_fac* np.identity(Q_dim)
    q = small_fac* np.identity(q_dim)

    reward=np.zeros((T,1))
    R_t=np.zeros((T,1))
    target_state= np.array([ 0.0, 40.0, 0.0, 0.0,  0.0, 0.0]).reshape(1,6)
    ##################################################################################################
    #### Simulate the equations to produce the reward values (beta* ||(x-x)||^2 + ||(u-u)||^2)
    ###################################################################################################
    for iter in range (T):
        reward[iter]= gamma * ( np.dot((X[iter,:6]-target_state),np.dot(Q,np.transpose((X[iter,:6]-target_state)))) + np.dot(X[iter,6:8],np.dot(q, np.transpose(X[iter,6:8])) ) )
    #R_t=(reward)
    R_t=np.cumsum(reward)

    sum_discounted_reward =  (beta*np.exp(-beta*R_t)).reshape(T,1)
    reward_with_data_points = (np.hstack( (X[:T,:8],sum_discounted_reward )))
    A_rew,B_rew,sig_reward,log_likelihood_reward_fitting = reward_learn(reward_with_data_points)
    
    #############################################################
    #### PLot the log likelihood values
    #############################################################    
    #log_likelihood_plot(log_likelihood_dynamics_fitting,log_likelihood_reward_fitting)
   
    
    EM_kalman(T,K,k,sig,X,reward_with_data_points,A_dyn,B_dyn,sigma_dyn,A_rew,B_rew,sig_reward,target_state) 





if __name__ == "__main__":

    ####
    # Policy K
    ####
    #pkl_file_K = open('/home/prakash/gps/python/gps/dataset and pkl file/T_S_20_4_0/Till 4 experiments/Till_4_data_K.pkl', 'rb')
    pkl_file_K = open('/home/prakash/gps/python/gps/dataset and pkl file/t_s_0_40_0/till_4_iterations/Till_4_data_K.pkl', 'rb')
    #pkl_file_K = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 1 Experiment/Till_1_data_K.pkl', 'rb')
    #pkl_file_K = open('/home/prakash/gps/python/gps/dataset and pkl file/Till the end of experiment 10 iterations/complete_data_K.pkl', 'rb')                    
    data_K = pickle.load(pkl_file_K)
    print np.array(data_K).shape
    pkl_file_K.close()

    ####
    # Policy k
    ####
    pkl_file_k = open('/home/prakash/gps/python/gps/dataset and pkl file/t_s_0_40_0/till_4_iterations/Till_4_data_k.pkl', 'rb')
    #pkl_file_k = open('/home/prakash/gps/python/gps/dataset and pkl file/Till the end of experiment 10 iterations/complete_data_k.pkl', 'rb')
    #pkl_file_k = open('/home/prakash/gps/python/gps/dataset and pkl file/T_S_20_4_0/Till 4 experiments/Till_4_data_k.pkl', 'rb')
    #pkl_file_k = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 1 Experiment/Till_1_data_k.pkl', 'rb')

    data_k = pickle.load(pkl_file_k)
    print np.array(data_k).shape
    pkl_file_k.close()

    ####
    # Policy Sigma
    ####
    #pkl_file_sig = open('/home/prakash/gps/python/gps/dataset and pkl file/Till the end of experiment 10 iterations/complte_data_sig.pkl', 'rb')
    pkl_file_sig = open('/home/prakash/gps/python/gps/dataset and pkl file/t_s_0_40_0/till_4_iterations/Till_4_data_sig.pkl', 'rb')
    #pkl_file_sig = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 1 Experiment/Till_1_data_sig.pkl', 'rb')
    #pkl_file_sig = open('/home/prakash/gps/python/gps/dataset and pkl file/T_S_20_4_0/Till 4 experiments/Till_4_data_sig.pkl', 'rb')
    data_sig = pickle.load(pkl_file_sig)
    print np.array(data_sig).shape
    pkl_file_sig.close()

    ####
    # Data Set of X_t,U_t,X_{t+1} for the last iteration of the LQR
    ####
    #pkl_file_X = open('/home/prakash/gps/python/gps/dataset and pkl file/Till the end of experiment 10 iterations/complete_data_X.pkl', 'rb')
    pkl_file_X = open('/home/prakash/gps/python/gps/dataset and pkl file/T_S_20_4_0/Till 1 Experiment/Till_1_data_X.pkl', 'rb')
    #pkl_file_X = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 1 Experiment/Till_1_data_X.pkl', 'rb')

    data_X = pickle.load(pkl_file_X)
    print np.array(data_X).shape
    pkl_file_X.close()
    
    #print data_X[0,:]
    main_call(data_X,data_K,data_k,data_sig)

def log_likelihood_plot(log_likelihood_dynamics_fitting,log_likelihood_reward_fitting):
    fig = plt.figure(figsize = (13, 9))
    ax_log_likelihood = fig.add_subplot(111)
    ax_log_likelihood.plot(np.array(log_likelihood_dynamics_fitting),color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=3)
    ax_log_likelihood.plot(np.array(log_likelihood_reward_fitting),color='red', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=3)
    plt.xlabel('Number of iterations for the Gaussian clusters to converge') 
    # naming the y axis 
    plt.ylabel('Loglikelihood values of the EM iteration for both Dynamics learning as well as reward learning ') 
    # giving a title to my graph 
    plt.title('Log Likelihood Plots') 
    plt.show()

def is_pd(K):
    try:
        np.linalg.cholesky(K)
        return 1 
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0
