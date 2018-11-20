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
from likelihood_fn import calculate_complete_loglikelihood
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from results import print_results
from temporary_robust_kalman import kf_ks,robust_kf_ks

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
        print "Matrix is positive definite ------ Can be used"
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
        #A[k,k:n]=A[k,k:n] -  np.dot(np.transpose(A[k-1,k]) ,  A[k-1,k:n])
        A[k,k:n]=A[k,k:n] -  np.dot(np.transpose(A[ :k,k]) ,  A[:k,k:n])
        if A[k,k]<=tol:
            A[k,k:n]=0
        else:
            A[k,k:n]=A[k,k:n]/np.sqrt(A[k,k])
    return A



def EM_kalman(T,data_K,data_k,data_sig,data_X,r,Adyn,Bdyn,Sigmadyn,A_rew,B_rew,Sig_rew,t_s):
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

    #############################################################
    ### Kalman filter and smoother definations
    #############################################################
    #s_hat= np.absolute(np.random.randn(state_dim,1)*0.00001)#np.zeros((state_dim,1)) #
    s_hat = np.zeros((state_dim,1))#np.absolute(np.random.randn(state_dim,1))

    #############################################################
    ## x(t+1) = A(t)x(t) + B(t)u(t) + w(t),   [w(t)]    (    [ Q(t)    S(t) ] )   >
    ##                                        [    ] ~ N( 0, [              ] )   >  0 
    ##   y(t) = C(t)x(t) + D(t)u(t) + v(t),   [v(t)]    (    [ S^T(t)  R(t) ] )   >
    ############################################################# 

    Pi_matrix = np.vstack((np.hstack((covdyn,s_hat)) , np.hstack((np.transpose(s_hat),covrew  )) ))
 
    print is_pd(Pi_matrix)

    
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
        covdyn_kal=nearPSD(covdyn_kal)  
        
    

    
    
    w_t_kal = np.random.multivariate_normal(mean_noise,covdyn_kal)
    #v_t_kal = np.random.multivariate_normal(np.zeros((1,)),covrew)

    #### INITIALIZE THE STATE AND ACTION
    initial_state_mu=np.array([ 5.0,  20.0, 0.0, 0.0,  0.0, 0.00])
    p_1_n=1.e-6*np.identity(6)

    ####  DEFINE X U AND R AND ASSIGN SPACE
    x_sim=np.zeros((T,state_dim))
    u_sim=np.zeros((T,action_dim))
    reward=np.zeros((T,))
    x_sim[0,:]=  initial_state_mu#np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
    u_sim[0,:]=np.random.multivariate_normal((np.dot(k1,x_sim[0,:])+k2).reshape(2,),policy_sigma,1)

    #############################################################
    ### Simulate the state space the trick is that use the rewards but d
    ### dont use the true states or the true actions which is dependent 
    ### on the true states-- 
    ### Note: - Also the reward calculated after simulating the state space
    ### are the approxmation of the reward distribution
    #############################################################
    for t in range(1,T):
        x_sim[t,:]= np.dot(A_kal,x_sim[t-1,:]) + np.dot(B_kal,u_sim[t-1,:]) + np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t-1])).reshape(1,state_dim) +w_t_kal
        u_sim[t,:]=np.random.multivariate_normal((np.dot(data_K[t,:,:],x_sim[t,:])+data_k[t,:].reshape(2,)),data_sig[t,:,:],1)    
        reward[t] =  (np.dot(R_x_dyn,x_sim[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn,u_sim[t,:].reshape(2,1)) +v_t ).reshape(1,)

    


    
    """ x_est_smooth,cov_smooth,M = Kalman_filter_smoother(T,state_dim,action_dim,A_kal\
                                        ,B_kal,data_K,data_k,data_sig,data_X,s_hat,covrew\
                                                ,reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn) """
    x_est_smooth,cov_smooth,M= robust_kf_ks(T,state_dim,action_dim,A_kal\
                                        ,B_kal,data_K,data_k,data_sig,data_X,s_hat,covrew\
                                                ,reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn)
        


    #Expec_log_joint_2nd_term=np.zeros((T,7,7))

    

    #  Expec_log_joint_2nd_term=
    def write_csv(data):
        with open('/home/prakash/gps/python/gps/csv_em_updated_params.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data)

    
    def f_with_dyn_params(k):  
        A_x_dyn=k[18:18+36].reshape(6,6)
        B_u_dyn=k[18+36:18+36+12].reshape(6,2)
        Sigmadyn=np.dot( k[18+36+12:18+36+12+36].reshape(6,6) , np.transpose(k[18+36+12:18+36+12+36].reshape(6,6)))
        R_x_dyn= k[102:108].reshape(1,6)
        R_u_dyn= k[108:110].reshape(1,2)
        covrew=np.dot (k[110:111].reshape(1,1),np.transpose(k[110:111].reshape(1,1)))

        x_1_n=x_est_smooth[0,:].reshape(6,1)
        mu_1=initial_state_mu.reshape(6,1)
        #############################################################
        ### Expectation expression for the log likelihood
        #############################################################
        temp = np.dot(x_1_n, np.transpose(x_1_n) ) + p_1_n - np.dot(x_1_n,np.transpose(mu_1)) -  np.dot( mu_1,np.transpose(x_1_n) ) \
        - np.dot( mu_1,np.transpose(mu_1) )
        Expec_log_joint_1st_term=-.5 *( np.trace(np.dot( np.linalg.inv( p_1_n ), temp )))  -.5 * np.linalg.det(p_1_n)


        sigma_total=  np.vstack((np.hstack (( Sigmadyn- np.dot(s_hat,np.dot(np.linalg.inv(covrew),np.transpose(s_hat) )) ,np.zeros((6,1)))) ,np.hstack( (np.zeros((1,6)) ,covrew))))  
        A_total= np.vstack((np.hstack((A_x_dyn,B_u_dyn)) , np.hstack((R_x_dyn,R_u_dyn)) ))
        

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












    def f(k):  
        x_1_n=x_est_smooth[0,:].reshape(6,1)
        
        mu_1=initial_state_mu.reshape(6,1)

        x_sim_inside=np.zeros((T,state_dim))
        u_sim_inside=np.zeros((T,action_dim))
        reward_inside=np.zeros((T,))

        #############################################################
        ### Expectation expression for the log likelihood
        #############################################################
        temp = np.dot(x_1_n, np.transpose(x_1_n) ) + p_1_n - np.dot(x_1_n,np.transpose(mu_1)) -  np.dot( mu_1,np.transpose(x_1_n) ) \
            - np.dot( mu_1,np.transpose(mu_1) )
        Expec_log_joint_1st_term=-.5 *( np.trace(np.dot( np.linalg.inv( p_1_n ), temp )))  -.5 * np.linalg.det(p_1_n)
        sigma_total=  np.vstack((np.hstack (( Sigmadyn- np.dot(s_hat,np.dot(np.linalg.inv(covrew),np.transpose(s_hat) )) ,np.zeros((6,1)))) ,np.hstack( (np.zeros((1,6)) ,covrew))))  
        #print "shape of the sigma total is ",sigma_total.shape
        A_total= np.vstack((np.hstack((A_x_dyn,B_u_dyn)) , np.hstack((R_x_dyn,R_u_dyn)) ))
        #print "shape of the A_total is",A_total.shape
        Expec_log_joint_sum=0 
        for t in range (T-1):
            if t==0:
                x_sim_inside[0,:]=  initial_state_mu#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
            else:
                x_sim_inside[t,:]= np.dot(A_kal,x_sim[t-1,:]) + np.dot(B_kal,u_sim[t-1,:]) + np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t-1])).reshape(1,state_dim) +w_t_kal
            u_sim_inside[t,:]=np.random.multivariate_normal((np.dot(k[:12].reshape((2,state_dim)),x_sim[t,:])+k[12:14].reshape(2,))\
                                                                                                              ,np.dot( k[14:18].reshape(2,2) , np.transpose(k[14:18].reshape(2,2)) ),1)    
            reward[t] =  (np.dot(R_x_dyn,x_sim[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn,u_sim[t,:].reshape(2,1)) +v_t ).reshape(1,)

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
    """ calculate_complete_loglikelihood(T,state_dim,action_dim,A_kal,B_kal,data_K,data_k,
                                            data_sig,data_X,s_hat,covrew,
                                            reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn
                                            ,R_u_dyn,x_est_smooth,cov_smooth,M,A_total) """

    
    data_sig_chol=np.zeros((T,2,2))
    for t in range(T):
        if is_pd(data_sig[t,:,:].reshape(2,2)) :
            data_sig_chol[t,:,:] =  np.linalg.cholesky(data_sig[t,:,:])
        else: 
            print "Breaking.....at line 401" 
            break

    
    
    def callbackF(Xi):
        global Nfeval
        global function_opt_intermed_vals
        #print "value of Xi is ",Xi
        print " fval- ",f(Xi)

        #print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[:12], Xi[12:14], Xi[14:16], f(Xi))
        Nfeval += 1
    #############################################################
    ### LBFGS Optimization of the Joint Complete Likelihood 
    ### of the rewards and latent variables 
    #############################################################
    #k_param=np.hstack ((k1.reshape(1,12),k2.reshape(1,2),policy_sigma.reshape(1,4)))
    #f(k_param.reshape((18,)))
    #print "k_param is ",k_param
    #print "the original value fo the function with the initial params are",f(k_param.reshape((18,)))

        """ [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] \
                =  optimize.fmin_bfgs(f_with_dyn_params, \
                            (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2),data_sig_chol[t,:,:].reshape(1,4), A_x_dyn.reshape(1,36),B_u_dyn.reshape(1,12),\
                                chol_sigma_dyn.reshape(1,36) , R_x_dyn.reshape(1,6),R_u_dyn.reshape(1,2),chol_covrew.reshape(1,1) ))).reshape((111,)) \
                                    ,callback=callbackF,maxiter=6, full_output=True, retall=False) """  
    
    chol_sigma_dyn=np.linalg.cholesky( Sigmadyn )
    chol_covrew=np.linalg.cholesky(covrew)
    updated_params_fvalue=np.zeros((T,1))
    updated_params=np.zeros((T,18))
    #updated_params=np.zeros((T,111))

    output = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/updated_em_params.pkl', 'wb')
    output_fval = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/updated_em_params_fval.pkl', 'wb')
    time1 = timeit.timeit()
    for t in range(T):
    
        [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] \
                =  optimize.fmin_bfgs(f, \
                                        (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2),data_sig_chol[t,:,:].reshape(1,4)))).reshape((18,)) \
                                                ,callback=callbackF,maxiter=20, full_output=True, retall=False)
        
        


        print "finished",t,"iterations in",
        #write_csv(xopt)
        updated_params[t,:]=xopt
        updated_params_fvalue[t,:]=fopt
        #print updated_params
    time2 = timeit.timeit()
    #print ' function optimization took %0.3f ms' % ( (time2-time1)*1000.0)
    pickle.dump(updated_params, output)
    pickle.dump(updated_params_fvalue, output_fval)

    print "The parameters are updated and dumped into the picke file"
    output.close()
    output_fval.close()
    print "~~~~~~.....The EM algorithm finished successfully......~~~~~~~" 

    #############################################################
    ### retrieving the updated EM k parameters from the .pkl file
    #############################################################
    updated_em = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/updated_em_params.pkl', 'rb')
    params = pickle.load(updated_em)
    updated_em_params_fval = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/updated_em_params_fval.pkl', 'rb')
    params_fval = (pickle.load(updated_em_params_fval)).reshape((T,))
    
    print params_fval.shape
    updated_em.close()

    for t in range(T):
        params[t,14:18]= ( np.dot (params[t,14:18].reshape(2,2) , np.transpose(params[t,14:18].reshape(2,2)))).reshape(1,4)
        """ params[t,18+36+12:102]= ( np.dot (params[t,18+36+12:102].reshape(6,6) , np.transpose(params[t,18+36+12:102].reshape(6,6)))).reshape(1,36)
        params[t,110:111]= np.dot( params[t,110:111].reshape(1,1), np.transpose(params[t,110:111].reshape(1,1))  ).reshape(1,1) """
    #############################################################
    ### Testing the parameters and producing the updated EM trajectory coordinates
    #############################################################

    print_results(f,T,state_dim,action_dim,params,\
                    A_x_dyn,B_u_dyn,w_t_kal,w_t,Adyn,Bdyn,Sigmadyn,data_K,data_k,data_sig,t_s,params_fval,initial_state_mu,p_1_n,A_kal,B_kal,s_hat,covrew,reward,R_x_dyn,R_u_dyn)


    return params



