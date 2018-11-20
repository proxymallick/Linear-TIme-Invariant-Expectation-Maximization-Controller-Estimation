from __future__ import division
__author__ = "Prakash Mallick"
__copyright__ = "Copyright 2018, Learning based Control for Quadrotor"
__email__ = "prakash.mallick@uon.edu.au"
__status__ = "In process"
__references__= " Papers by the Guided policy search finn et al (2016) and Higham (2000)/Rebonato and Jackel (1999)  "
from scipy import optimize
import csv
import numpy as np
import pandas as pd
from scipy import random, linalg
from scipy.stats import multivariate_normal
from numpy.linalg import multi_dot
import math
import time,timeit
Nfeval = 1
import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def write_csv(data):
    with open('/home/prakash/gps/python/gps/em_parameters.csv', 'a') as outfile:
        writer = csv.writer(outfile, lineterminator='\n')
        writer.writerow(data)

#############################################################
### Nearest correlation amtrix approximation by Rebonato and Jackel (1999)
#############################################################

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

#############################################################
### Nearest correlation amtrix approximation by Higham (2000)
#############################################################

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def is_pd(K):
    try:
        np.linalg.cholesky(K)
        #print "Matrix is positive definite ------ Can be used"
        return 1 
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0

def robust_cholesky(A):
    A = np.triu(A); n = np.shape(A)[0]; tol = n*np.spacing(1)
    if A[0,0]<=tol :
        A[0,0:n]=0
    else:
        A[0,0:n] = A[0,0:n]/ np.sqrt(A[0,0])
    for k in range(1,n):
        """         if k==1:
            A[1,1:n]=A[1,1:n]-  np.dot(np.transpose (A[0,1]) , A[0,1:n] )
        if k==2:
            A[2,2:n]=A[2,2:n]-  np.dot(np.transpose (A[1,2]) , A[1,2:n] )
        else: """
        #A[k,k:n]=A[k,k:n] -  np.dot(np.transpose(A[k-1,k]) ,  A[k-1,k:n])
        A[k,k:n]=A[k,k:n] -  np.dot(np.transpose(A[ :k,k]) ,  A[:k,k:n])
        if A[k,k]<=tol:
            A[k,k:n]=0
        else:
            A[k,k:n]=A[k,k:n]/np.sqrt(A[k,k])
    return A


def EM_kalman(T,data_K,data_k,data_sig,data_X,r,Adyn,Bdyn,Sigmadyn,A_rew,B_rew,Sig_rew):
    #############################################################
    ### Define the Dynamics parameters initialized after learning the transition dynamics
    #############################################################
    """ A=np.matrix('2 4.0 7;10 5 7;3 ,2 5')
    print Robust_CHolesky(A)
    ff """
    if np.isnan(Sigmadyn).any():
        print "nans appearing in the Sigmadyn"

    if np.isnan(Sig_rew).any():
        print "nans appearing in the Sig_rew"

    print data_K.shape,data_k.shape,data_sig.shape

    
    
    state_dim=6
    gamma=1
    A_x_dyn=Adyn[:,:6]
    B_u_dyn=Adyn[:,6:8] # np.random.randn(state_dim,2)
    covdyn= Sigmadyn # np.dot(np.random.rand(state_dim,state_dim),np.random.rand(state_dim,state_dim).transpose())
    print "shape of the A_xdyn, B_u_dyn, cov_dyn reward is ",A_x_dyn.shape,B_u_dyn.shape,covdyn.shape


    mean_noise=np.zeros((state_dim,))
    w_t = np.random.multivariate_normal(mean_noise,covdyn)
 
    #############################################################
    ### Define the Reward parameters initialized after learning the reward dynamics
    #############################################################

    reward_dim=1
    action_dim=2
    R_x_dyn=A_rew[:,:6]
    R_u_dyn=A_rew[:,6:8] # np.random.randn(state_dim,2)  

    
    #####
    ## Check if the matrix is posive definite or not
    #####
    if is_pd(Sig_rew):
        covrew=(Sig_rew)
    else:
        print "cov reward entering into the nearPD function"
        covrew=(Sig_rew)  

    
    mean_noise_reward=np.zeros((reward_dim,))
    print "shape of the R_xdyn, R_u_dyn, cov_rew reward is ",R_x_dyn.shape,R_u_dyn.shape,covrew.shape
    v_t = np.random.normal(mean_noise_reward,covrew)



    #time=100
    #############################################################
    ### Define the policy parameters compatible with the GPS
    #############################################################
    
    k1= data_K[0,:,:]#np.array([[-1.06931210e-02,  6.57308733e-08,  5.17954453e-03, -5.72602762e-09,1.33445090e-09,  5.17954453e-03], [ 4.41549804e-08, -1.06932109e-02,  5.17959949e-03, -4.21970228e-09,  8.07066476e-10,  5.17959949e-03]])#np.random.randn(2,state_dim)
    k2= data_k[0,:].reshape(2,)#np.array([ 1.06931360e-02, -5.77100907e-08])
    policy_sigma= data_sig[0,:,:].reshape(2,2)#np.array(([4.98151299e+00,1.97741607e-08,1.97741607e-08,4.98151315e+00])).reshape(2,2) #np.eye(2)

    print "k1 shape and k2 shape is ",k1.shape,k2.shape
    ####  DEFINE X U AND R AND ASSIGN SPACE
    x=np.zeros((T,state_dim))
    u=np.zeros((T,action_dim))
    reward=np.zeros((T,))

    #### INITIALIZE THE STATE AND ACTION
    initial_state_mu=np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])
    p_1_n=1.e-6*np.identity(6)


    x[0,:]=  np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
    u[0,:]=np.random.multivariate_normal((np.dot(k1,x[0,:])+k2).reshape(2,),policy_sigma,1)

    #############################################################
    ### Simulate the state space the trick is that use the rewards but d
    ### dont use the true states or the true actions which is dependent 
    ### on the true states-- 
    ### Note: - Also the reward calculated after simulating the state space
    ### are the approxmation of the reward distribution
    #############################################################
    for t in range(1,T):
        x[t,:]=np.dot(A_x_dyn,x[t-1,:])+np.dot(B_u_dyn,u[t-1,:]) + w_t + Bdyn
        u[t,:]=np.random.multivariate_normal((np.dot(data_K[t,:,:],x[t,:])+data_k[t,:].reshape(2,)),data_sig[t,:,:],1)    
        reward[t] =  (np.dot(R_x_dyn,x[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn,u[t,:].reshape(2,1)) +v_t +B_rew ).reshape(1,)
        #reward[t]=r[t,-1]
    
    #print "x is", x
    #print "rewards are",reward

    
    
    
    #############################################################
    ### Kalman filter and smoother definations
    #############################################################
    s_hat=np.absolute(np.random.randn(state_dim,1))

    """ R_S_trans_S_Q=  np.vstack((np.hstack((covrew, np.transpose(s_hat)))  ,  np.hstack( ( s_hat , covdyn) ) ) )
    temp_mat=(robust_cholesky(R_S_trans_S_Q))
    print  """

    
    A_kal=A_x_dyn-np.dot(s_hat,(np.linalg.solve(covrew,R_x_dyn)))
    B_kal=B_u_dyn-np.dot(s_hat,(np.linalg.solve(covrew,R_u_dyn)))


    covdyn_kal=covdyn-np.dot(s_hat,np.dot(np.linalg.inv(covrew),np.transpose(s_hat)))
    
    #############################################################
    ### Approximating the covariance matrix to the 
    ### nearest positive def. Matrix if matrix is not positive deifnite
    #############################################################
    if is_pd(Sig_rew) and np.all(np.linalg.eigvals(covdyn_kal) >= 0) :
        covdyn_kal=(covdyn_kal)
    else:
        print "covdyn_kal entering into the nearPSD function"
        covdyn_kal=nearPD(covdyn_kal)  
        
    

    
    
    w_t_kal = np.random.multivariate_normal(mean_noise,covdyn_kal)
    v_t_kal = np.random.multivariate_normal(np.zeros((1,)),covrew)
    x_est=np.zeros((T,state_dim))
    x_est[0,:]=x[0,:]
    cov=np.zeros((T,state_dim,state_dim))
    cov[0,:,:]= 0.5*(p_1_n+np.transpose(p_1_n)) 
    
    kalman_gain=np.zeros((T,state_dim))

    
    
    it = slice(state_dim-2)
    sig_reg = np.zeros((state_dim,state_dim))
    sig_reg[it, it] = 1e-6

    for t in range(1,T):
        x_est[t,:]=np.dot(A_kal,x_est[t-1,:])+np.dot(B_kal,np.dot(data_K[t,:,:],x_est[t-1,:])+data_k[t,:].reshape(2,))+np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim)
        cov[t,:,:]= np.dot(A_kal,np.dot(cov[t-1,:,:],np.transpose(A_kal)))+covdyn_kal #+sig_reg
        #cov[t+1,:,:]=nearPD(cov[t+1,:,:])
        kalman_gain[t,:] = np.dot(cov[t,:,:],np.dot(np.transpose(R_x_dyn),(np.linalg.inv(np.dot(R_x_dyn,np.dot(cov[t,:,:],np.transpose(R_x_dyn)  )) + covrew)))).reshape(1,state_dim)
        cov[t,:,:]=cov[t,:,:] - np.dot(kalman_gain[t,:].reshape(state_dim,1),np.dot(R_x_dyn,cov[t,:,:]))
        x_est[t,:]=x_est[t,:]+ np.dot(kalman_gain[t].reshape(state_dim,1),(reward[t]-np.dot(R_x_dyn,x_est[t,:])-np.dot(R_u_dyn,(np.dot(data_K[t,:,:],x_est[t,:])+data_k[t,:].reshape(2,)))) ) 
        if np.isnan(x_est[t,:]).any():
            print "nans appearing in the covariance matrix just after the kalman filter at time",t
        
    if np.isnan(cov).any()  :
        print "nans appearing in the covariance matrix just after the kalman filter"

    if  np.isnan(kalman_gain).any() :
        print "nans appearing in the kalman gain matrix just after the kalman filter"
    
    

    ####                                             ####
    ####   Initialisation of the smoother equations  ####
    ####                                             ####
    J=np.zeros((T,state_dim,state_dim))
    x_est_smooth=np.zeros((T,state_dim))
    x_est_smooth[T-1,:]=x_est[T-1,:]
    cov_smooth=np.zeros((T,state_dim,state_dim))
    cov_smooth[T-1,:,:]=cov[T-1,:,:]

    for t in range(T-2,-1,-1): #from T to 0 with 1 step down
        
        J[t,:,:]=np.dot(cov[t,:,:],np.dot(np.transpose(A_kal), np.linalg.inv(cov[t+1,:,:])))
        x_est_smooth[t,:] = (x_est[t,:].reshape(state_dim,1)+ np.dot(J[t,:,:], (x_est_smooth[t+1,:].reshape(state_dim,1)-   np.dot(A_kal,x_est[t,:].reshape(state_dim,1)) \
        - np.dot(B_kal,(np.dot(data_K[t,:,:],x_est[t,:].reshape(state_dim,1))+data_k[t,:].reshape(2,1))) - np.dot(s_hat,np.dot(np.linalg.inv(covrew),reward[t])))) ).reshape(1,state_dim)
        cov_smooth[t,:,:]=cov[t,:,:]+np.dot(J[t,:] , np.dot(cov_smooth[t+1,:,:]-cov[t+1,:,:],np.transpose(J[t,:])))



    M=np.zeros((T,state_dim,state_dim))

    M[T-1,:,:]= (np.dot((np.identity(state_dim)-  np.dot( kalman_gain[T-1,:].reshape(state_dim,1),R_x_dyn)) , np.dot( A_kal,cov_smooth[T-2,:,:] )))
  #  
    for t in range(T-2,-1,-1):
        M[t,:,:]=np.dot(cov[t,:], np.transpose( J[t-1,:]))+np.dot(J[t,:],np.dot((M[t+1,:]-np.dot (A_kal,cov[t,:])),np.transpose(J[t-1,:])))   


    if np.isnan(M).any() or  np.isnan(cov_smooth).any() :
        print "nans appearing in the covariance matrices or the kalman gains of the smoother evaluations"
 
    

    
        
    x_1_n=x_est_smooth[0,:].reshape(6,1)
    
    mu_1=initial_state_mu.reshape(6,1)

    #############################################################
    ### Expectation expression for the log likelihood
    #############################################################
    temp = np.dot(x_1_n, np.transpose(x_1_n) ) + p_1_n - np.dot(x_1_n,np.transpose(mu_1)) -  np.dot( mu_1,np.transpose(x_1_n) ) \
        - np.dot( mu_1,np.transpose(mu_1) )
    Expec_log_joint_1st_term=-.5 *( np.trace(np.dot( np.linalg.inv( p_1_n ), temp )))  -.5 * np.linalg.det(p_1_n)
    
    
    sigma_total=  np.vstack((np.hstack (( Sigmadyn- np.dot(s_hat,np.dot(np.linalg.inv(covrew),np.transpose(s_hat) )) ,np.zeros((6,1)))) ,np.hstack( (np.zeros((1,6)) ,covrew))))  
    print "shape of the sigma total is ",sigma_total.shape
    A_total= np.vstack((np.hstack((A_x_dyn,B_u_dyn)) , np.hstack((R_x_dyn,R_u_dyn)) ))
    print "shape of the A_total is",A_total.shape

    #Expec_log_joint_2nd_term=np.zeros((T,7,7))

    

    #  Expec_log_joint_2nd_term=
    """ def write_csv(data):
        with open('/home/prakash/gps/python/gps/csv_em_updated_params.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data) """
    
    def callbackF(Xi):
        global Nfeval

        #print "value of Xi is ",Xi
        print "value of the function eval while the optimization is being carried out is",f(Xi)

        #print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[:12], Xi[12:14], Xi[14:16], f(Xi))
        Nfeval += 1

    def f(k):  
        Expec_log_joint_sum=0 
        for t in range (T-1):
            
            x_t_smooth = x_est_smooth[t,:].reshape(6,1)
            x_t_plus_1_smooth = x_est_smooth[t+1,:].reshape(6,1)
            x_t_x_t=np.dot(x_t_smooth,np.transpose(x_t_smooth))+cov_smooth[t,:,:]
            x_t_plus_1_r_t_transpose= np.dot(np.dot(x_t_plus_1_smooth, np.transpose( x_t_smooth ) ) + M[t+1,:,:] , np.transpose(R_x_dyn)  + np.dot( np.transpose(k[:12].reshape(2,6)) , np.transpose(R_u_dyn) ) \
            ) + np.dot( x_t_plus_1_smooth ,   np.dot( np.transpose(k[12:14].reshape(2,1)) , np.transpose(R_u_dyn) )  )
            u_t_x_t_transpose = np.transpose  (np.dot(x_t_x_t , np.transpose(k[:12].reshape(2,6)) )   +\
                           np.dot( x_t_smooth,np.transpose(k[12:14].reshape(2,).reshape (2,1) ) ) )
            r_t_r_t_1st_term =  multi_dot([ R_x_dyn, x_t_x_t , np.transpose (R_x_dyn) ]) 
            r_t_r_t_2nd_term =  multi_dot([R_x_dyn ,np.transpose(u_t_x_t_transpose) , np.transpose (R_u_dyn) ]  )   
            r_t_r_t_3rd_term= np.transpose(r_t_r_t_2nd_term)
            r_t_r_t_4th_term = covrew
            u_t_u_t_transpose= ( multi_dot( [k[:12].reshape(2,6) ,  np.dot( x_t_smooth, np.transpose(x_t_smooth) ) +cov_smooth[t,:,:] , np.transpose(k[:12].reshape(2,6))] ) \
            +multi_dot([k[:12].reshape(2,6),x_t_smooth,np.transpose(k[12:14].reshape(2,).reshape(2,1))]) +  np.transpose (multi_dot([k[:12].reshape(2,6),x_t_smooth,np.transpose(k[12:14].reshape(2,).reshape(2,1))]) )  \
            + np.dot(k[12:14].reshape(2,).reshape(2,1) ,np.transpose(k[12:14].reshape(2,).reshape(2,1))  ) + np.dot( k[14:18].reshape(2,2) , np.transpose(k[14:18].reshape(2,2)) )   )

            r_t_r_t_5th_term = multi_dot ([ R_u_dyn , u_t_u_t_transpose , np.transpose(R_u_dyn) ])

            r_t_r_t=r_t_r_t_1st_term+r_t_r_t_2nd_term+r_t_r_t_3rd_term+ r_t_r_t_4th_term +r_t_r_t_5th_term

            zeta_zeta=  np.vstack( ( np.hstack(  (    np.dot(x_t_plus_1_smooth, np.transpose(x_t_plus_1_smooth)) + cov_smooth[t+1,:,:]  \
            , x_t_plus_1_r_t_transpose   ) )  ,   np.hstack(  (   np.transpose(x_t_plus_1_r_t_transpose)  \
            ,  r_t_r_t  ) )   )  ) 

            #print  "shape of zeta zeta is",zeta_zeta.shape
            
            zeta_z_1 = np.hstack(( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:]  \
            , np.dot( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:], np.transpose(k[:12].reshape(2,6))  ) \
            + multi_dot([x_t_plus_1_smooth,np.transpose (k[12:14].reshape(2,1))])) )

            

            r_t_x_t_transpose=   np.dot(R_x_dyn,x_t_x_t ) + np.dot(R_u_dyn , \
            u_t_x_t_transpose )  

            r_t_u_t_transpose = multi_dot([ R_x_dyn, x_t_x_t , np.transpose (k[:12].reshape(2,6)) ])  + \
                multi_dot([ R_u_dyn ,  u_t_x_t_transpose ,  np.transpose(k[:12].reshape(2,6)) ]) \
                    + multi_dot([R_x_dyn ,x_t_smooth, np.transpose(k[12:14].reshape(2,).reshape(2,1)) ])+ multi_dot([ R_u_dyn, np.dot(k[:12].reshape(2,6),x_t_smooth)+k[12:14].reshape(2,1) , np.transpose(k[12:14].reshape(2,1))])

            zeta_z_2= np.hstack((  r_t_x_t_transpose,r_t_u_t_transpose ))
            zeta_z = np.vstack (( zeta_z_1,zeta_z_2 ))

            #print "shape of zeta z is",zeta_z.shape


            z_z_1=  np.hstack  ((x_t_x_t, np.transpose(u_t_x_t_transpose)))
            z_z_2=  np.hstack(( u_t_x_t_transpose, u_t_u_t_transpose  )) 

            #print "z_z_1 shape and z_z2_shapes are",z_z_1.shape,z_z_2.shape

            z_z = np.vstack((z_z_1,z_z_2))
            #print "z_z shape is ",z_z.shape
    
            Expec_log_joint_2nd_term= -.5 * ( np.trace( np.dot( np.linalg.inv(sigma_total) , (zeta_zeta - np.dot( zeta_z, np.transpose(A_total)) - np.transpose(np.dot( zeta_z, np.transpose(A_total)) )  \

            + multi_dot ([A_total , z_z , np.transpose(A_total)])  )  )) )  -.5 * np.linalg.det(sigma_total)
            Expec_log_joint_sum  = Expec_log_joint_sum +Expec_log_joint_2nd_term
        Expec_log_joint=-1* (Expec_log_joint_1st_term + Expec_log_joint_sum)

        return Expec_log_joint

    data_sig_chol=np.zeros((T,2,2))
    for t in range(T):
        if is_pd(data_sig[t,:,:].reshape(2,2)) :
            data_sig_chol[t,:,:] =  np.linalg.cholesky(data_sig[t,:,:])
        else: 
            print "Breaking.....at line 401" 
            break

    
    #############################################################
    ### LBFGS Optimization of the Joint Complete Likelihood 
    ### of the rewards and latent variables 
    #############################################################
    #k_param=np.hstack ((k1.reshape(1,12),k2.reshape(1,2),policy_sigma.reshape(1,4)))
    #f(k_param.reshape((18,)))
    #print "k_param is ",k_param
    #print "the original value fo the function with the initial params are",f(k_param.reshape((18,)))
    """ updated_params_fvalue=np.zeros((T,1))
    updated_params=np.zeros((T,18))
    output = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 4 experiments/updated_em_params.pkl', 'wb')
    output_fval = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 4 experiments/updated_em_params_fval.pkl', 'wb')
    time1 = timeit.timeit()
    for t in range(T):
        
        [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] \
         =  optimize.fmin_bfgs(f, \
                                (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2),data_sig_chol[t,:,:].reshape(1,4)))).reshape((18,)) \
                                        ,callback=callbackF,maxiter=40, full_output=True, retall=False)
        

        print "finished",t,"iterations in",
        #write_csv(xopt)
        updated_params[t,:]=xopt
        updated_params_fvalue[t,:]=fopt
        #print updated_params
    time2 = timeit.timeit()
    print ' function optimization took %0.3f ms' % ( (time2-time1)*1000.0)
    pickle.dump(updated_params, output)
    pickle.dump(updated_params_fvalue, output_fval)

    print "The parameters are updated and dumped into the picke file"
    output.close()
    output_fval.close()
    print "~~~~~~.....The EM algorithm finished successfully......~~~~~~~" 
     """
    
    #Testing(T,state_dim,action_dim,A_x_dyn,B_u_dyn,w_t,v_t,Bdyn,data_K,data_k,data_sig,f)

    #def Testing(T,state_dim,action_dim,A_x_dyn,B_u_dyn,w_t,v_t,Bdyn,data_K,data_k,data_sig,f,R_x_dyn):
    #############################################################
    ### retrieving the updated EM k parameters from the .pkl file
    #############################################################
    updated_em = open('/home/prakash/gps/python/gps/Linear EM with GPS/updated_em_params.pkl', 'rb')
    params = pickle.load(updated_em)
    updated_em_params_fval = open('/home/prakash/gps/python/gps/dataset and pkl file/Till 4 experiments/updated_em_params_fval.pkl', 'rb')
    params_fval = (pickle.load(updated_em_params_fval)).reshape((T,))
    
    print params_fval.shape
    updated_em.close()

    
    for t in range(T):
        params[t,14:18]= ( np.dot (params[t,14:18].reshape(2,2) , np.transpose(params[t,14:18].reshape(2,2)))).reshape(1,4)

    #############################################################
    ### Testing the parameters and producing the updated EM trajectory coordinates
    #############################################################
    small_fac=1e-15
    Q_dim=6
    q_dim=2
    Q = small_fac* np.identity(Q_dim)
    q = small_fac* np.identity(q_dim)


    target_state= np.array([ 20.0, 4.0, 0.0, 0.0,  0.0, 0.0]).reshape(1,6)
    x_em_test=np.zeros((T,state_dim))
    u_em_test=np.zeros((T,action_dim))
    reward_em = np.zeros((T,1))
    covariance_init=1.e-6*np.identity(6)
    initial_state_mu=np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])
    x_em_test[0,:]=  initial_state_mu  #np.random.randn(state_dim).reshape(1,state_dim)
    u_em_test[0,:]= np.random.multivariate_normal((np.dot(params[0,:12].reshape(2,6),x_em_test[0,:])+params[0,12:14].reshape(2,)).reshape(2,), (params[0,14:18].reshape(2,2)),1) 



    ##########                 ###################                         ##################
    ## Plot the generated trajectory from the EM parameters                             ######
    ##########                 ###################                         ##################
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()

    for time in range(1,T):
        #params[time,14:18] =  (0.5 * (params[time,14:18].reshape(2,2) + np.transpose(params[time,14:18].reshape(2,2)))).reshape(1,4)
        #print nearPD( params[time,14:18].reshape(2,2))#(is_pd(params[time,14:18].reshape(2,2)) )# and np.all(np.linalg.eigvals(params[time,14:18].reshape(2,2))) )
        #x_em_test[t,:]=np.dot(A_x_em_test_dyn,x_em_test[t-1,:])+np.dot(B_u_dyn,u[t-1,:]) + w_t
        #u[t,:]=np.random.multivariate_normal((np.dot(k1,x_em_test[t-1,:])+k2).reshape(2,),policy_sigma,1)
        x_em_test[time,:]= np.dot(A_kal,x_em_test[time-1,:])+np.dot(B_kal,u_em_test[time-1,:]) + w_t_kal + Bdyn
        u_em_test[time,:] = np.random.multivariate_normal((np.dot(params[time,:12].reshape(2,6),x_em_test[time,:])\
                                          +params[time,12:14].reshape(2,)).reshape(2,),(params[time,14:18].reshape(2,2)),1) 
        #reward_em[time,:]=(np.dot(R_x_dyn,x_em_test[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn,u_em_test[t,:].reshape(2,1)) +v_t +B_rew ).reshape(1,)
        reward_em[time,:] = ( np.dot((x_em_test[time,:6]-target_state),np.dot(Q,np.transpose((x_em_test[time,:6]-target_state)))) + np.dot(u_em_test[time,:2],np.dot(q, np.transpose(u_em_test[time,:2])) ) )
    
    
    
    x =x_em_test[:,0].reshape(x_em_test.shape[0],)
    y =x_em_test[:,1].reshape(x_em_test.shape[0],)
    z =x_em_test[:,2].reshape(x_em_test.shape[0],)
        #ax.scatter(x, y, z, c='r', marker='o')
        #plt.pause(0.0001) 
    ax = fig.add_subplot(243, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')
    #plt.pause(0.0001)
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')

    ax_em_reward = fig.add_subplot(245)
    plt.plot(np.arange(reward_em.size), reward_em ,c='b', marker='o')

    plt.show()