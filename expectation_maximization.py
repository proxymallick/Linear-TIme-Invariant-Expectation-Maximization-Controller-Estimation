import csv
import numpy as np
import pandas as pd
from scipy import random, linalg
from scipy.stats import multivariate_normal
from gmm_fit_em import GMM
import math
from gmm import GMM
#############################################################
### Generate a Positive semidefinite matrix
#############################################################
state_dim=6
matrixSize = state_dim 
A = random.rand(matrixSize,matrixSize)
psd_mat_w = np.dot(A,A.transpose())


reward_dim=1
A = random.rand(reward_dim,reward_dim)
psd_mat_v = np.dot(A,A.transpose())

#############################################################
### SAmple from a multivariate gaussian density
#############################################################
def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))

#############################################################
### Define the policy parameters compatible with the GPS
#############################################################
k1=np.random.randn(2,state_dim)
k2=(np.arange(2)+0.5)
policy_sigma=np.eye(2)

#############################################################
### Define the Dynamics parameters initiallya s a random values
#############################################################
gamma=0.99
A_x_dyn= Fm#np.random.randn(state_dim,state_dim)
B_u_dyn= np.random.randn(state_dim,2)
covdyn=np.dot(np.random.rand(state_dim,state_dim),np.random.rand(state_dim,state_dim).transpose())
mean_noise=np.zeros((state_dim,))
w_t = np.random.multivariate_normal(mean_noise,psd_mat_w)

v_t = np.random.multivariate_normal(np.zeros((1,)),psd_mat_v)
T=100

####Initial values of the states and the actions defined or known
x=np.zeros((T+1,state_dim))
u=np.zeros((T+1,2))
reward=np.zeros((T,))

x[0,:]=np.random.randn(state_dim).reshape(1,state_dim)
u[0,:]=np.random.multivariate_normal((np.dot(k1,x[0,:])+k2).reshape(2,),policy_sigma,1)

print x[0,:].shape
print u[0,:].shape

#############################################################
### Define the reward equation
#############################################################
Reward_Q=np.absolute(np.random.randn(state_dim,state_dim))
reward_x_coef=np.random.randn(1,state_dim)
reward_u_coef=np.random.randn(1,2)

#############################################################
### Simulate the state space
#############################################################
for t in range(T-1):
    if t!=0:
        u[t,:]=np.random.multivariate_normal((np.dot(k1,x[t,:])+k2).reshape(2,),policy_sigma,1)    
    x[t+1,:]=np.dot(A_x_dyn,x[t,:])+np.dot(B_u_dyn,u[t,:]) + w_t
    reward[t]= np.dot(reward_x_coef,x[t,:])+ np.dot(reward_u_coef,u[t,:]) +v_t 



#############################################################
### Kalman filter and smoother
#############################################################
s_hat=np.absolute(np.random.randn(state_dim,1))
A_kal=A_x_dyn-np.dot(s_hat,(np.linalg.solve(psd_mat_v,reward_x_coef)))
B_kal=B_u_dyn-np.dot(s_hat,(np.linalg.solve(psd_mat_v,reward_u_coef)))
psd_mat_w_kal=psd_mat_w-np.dot(s_hat,np.dot(np.linalg.inv(psd_mat_v),np.transpose(s_hat)))


w_t_kal = np.random.multivariate_normal(mean_noise,np.dot(psd_mat_w_kal,np.transpose(psd_mat_w_kal)))
v_t_kal = np.random.multivariate_normal(np.zeros((1,)),psd_mat_v)
x_est=np.zeros((T+1,state_dim))
x_est[0,:]=x[0,:]
cov=np.zeros((T+1,state_dim,state_dim))
cov[0,:,:]=psd_mat_w
kalman_gain=np.zeros((T+1,state_dim))
for t in range(0,T):
    x_est[t+1,:]=np.dot(A_kal,x_est[t,:])+np.dot(B_kal,np.dot(k1,x_est[t,:])+k2)+np.dot(s_hat, np.dot(np.linalg.inv(psd_mat_v+1e-6),reward[t-1])).reshape(1,state_dim)
    cov[t+1,:,:]= np.dot(A_kal,np.dot(cov[t,:,:],np.transpose(cov[t,:,:])))+psd_mat_w_kal
  #  kalman_gain[t+1,:] = np.dot(cov[t+1,:,:],np.dot(np.transpose(reward_x_coef),(np.linalg.inv(np.dot(reward_x_coef,np.dot(cov[t+1,:,:],np.transpose(reward_x_coef)  )) + psd_mat_v+1e-6)))).reshape(1,state_dim)
  #  cov[t+1,:,:]=cov[t+1,:,:] - np.dot(kalman_gain[t+1,:].reshape(state_dim,1),np.dot(reward_x_coef,cov[t+1,:,:]))
  #  x_est[t+1,:]=x_est[t+1,:]+ np.dot(kalman_gain[t+1].reshape(state_dim,1),(reward[t]-np.dot(reward_x_coef,x_est[t+1,:])-np.dot(reward_u_coef,(np.dot(k1,x_est[t+1,:])+k2))) ) 


print cov


""" J=np.zeros((T,state_dim,state_dim))
x_est_smooth=np.zeros((T+1,state_dim))
x_est_smooth[T,:]=x_est[T,:]


cov_smooth=np.zeros((T+1,state_dim,state_dim))
cov_smooth[T,:,:]=cov[T,:,:]

for t in range(T-1,-1,-1):
    J[t,:,:]=np.dot(cov[t,:,:],np.dot(np.transpose(A_kal), np.linalg.inv(cov[t+1,:,:])))
    x_est_smooth[t,:] = (x_est[t,:].reshape(state_dim,1)+ np.dot(J[t,:,:], (x_est_smooth[t+1,:].reshape(state_dim,1)-   np.dot(A_kal,x_est[t,:].reshape(state_dim,1)) \
     - np.dot(B_kal,(np.dot(k1,x_est[t,:].reshape(state_dim,1))+k2.reshape(2,1))) - np.dot(s_hat,np.dot(np.linalg.inv(psd_mat_v),reward[t])))) ).reshape(1,state_dim)
    cov_smooth[t,:,:]=cov[t,:,:]+np.dot(J[t,:] , np.dot(cov_smooth[t+1,:,:]-cov[t+1,:,:],np.transpose(J[t,:])))


M=np.zeros((T+1,state_dim))

print np.dot((np.identity(state_dim)-  np.dot( kalman_gain[T,:].reshape(state_dim,1),reward_x_coef)) , np.dot( A_kal,cov_smooth[T-1,:,:] ))
 
for t in range(T-1,-1,-1):
    M[t,:]=np.dot(cov[t,:],J[t-1,:])+np.dot(J[t,:],np.dot((M[t+1,:]-np.dot (A_kal,cov[t,:])),np.transpose(J[t-1,:])))   """




