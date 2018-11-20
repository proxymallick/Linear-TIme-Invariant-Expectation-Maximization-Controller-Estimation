__author__ = "Prakash Mallick"
__copyright__ = "Copyright 2018, Learning based Quadrotor Project"
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
Nfeval = 1
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
        return 1 
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0

def EM_kalman(data_K,data_k,data_sig,data_X,r,Adyn,Bdyn,Sigmadyn,A_rew,B_rew,Sig_rew):
    #############################################################
    ### Define the Dynamics parameters initialized after learning the transition dynamics
    #############################################################

    if np.isnan(Sigmadyn).any():
        print "nans appearing in the Sigmadyn"

    if np.isnan(Sig_rew).any():
        print "nans appearing in the Sig_rew"

    print data_K.shape,data_k.shape,data_sig.shape

    state_dim=6
    gamma=0.99
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


    covrew= Sig_rew  # np.dot(np.random.rand(state_dim,state_dim),np.random.rand(state_dim,state_dim).transpose())
    mean_noise_reward=np.zeros((reward_dim,))
    print "shape of the R_xdyn, R_u_dyn, cov_rew reward is ",R_x_dyn.shape,R_u_dyn.shape,covrew.shape
    v_t = np.random.normal(mean_noise_reward,covrew)
    T=100
    time=100
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
    initial_state_mu=np.array([ 0.0,  20.0, 0.0, 0.0,  0.0, 0.00])
    p_1_n=np.identity(6)
    x[0,:]=  np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
    u[0,:]=np.random.multivariate_normal((np.dot(k1,x[0,:])+k2).reshape(2,),policy_sigma,1)

    


    #############################################################
    ### Simulate the state space
    #############################################################
    for t in range(T):
        #x[t,:]=np.dot(A_x_dyn,x[t-1,:])+np.dot(B_u_dyn,u[t-1,:]) + w_t
        #u[t,:]=np.random.multivariate_normal((np.dot(k1,x[t-1,:])+k2).reshape(2,),policy_sigma,1)    
        #reward[t] =  (np.dot(R_x_dyn,x[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn,u[t,:].reshape(2,1)) +v_t ).reshape(1,)
        reward[t]=r[t,-1]
    
    #print "x is", x
    #print "rewards are",reward
    #############################################################
    ### Kalman filter and smoother
    #############################################################
    s_hat=np.absolute(np.random.randn(state_dim,1))
    A_kal=A_x_dyn-np.dot(s_hat,(np.linalg.solve(covrew,R_x_dyn)))
    B_kal=B_u_dyn-np.dot(s_hat,(np.linalg.solve(covrew,R_u_dyn)))
    covdyn_kal=covdyn-np.dot(s_hat,np.dot(np.linalg.inv(covrew),np.transpose(s_hat)))
  #  covdyn_kal = nearPD(covdyn_kal)
    covdyn_kal = nearPSD(covdyn_kal)

    print "shape of cov_kalman",covdyn_kal.shape
  #  it = slice(state_dim-2)
   # sig_reg = np.zeros((state_dim,state_dim))
    #sig_reg[it, it] = 1e-6
    w_t_kal = np.random.multivariate_normal(mean_noise,0.5*(covdyn_kal+np.transpose(covdyn_kal)))
    v_t_kal = np.random.multivariate_normal(np.zeros((1,)),covrew)
    x_est=np.zeros((T,state_dim))
    x_est[0,:]=x[0,:]
    cov=np.zeros((T,state_dim,state_dim))
    cov[0,:,:]= 0.5*(covdyn+covdyn.T) +  1e-6 * np.eye(state_dim)

    kalman_gain=np.zeros((T,state_dim))


    it = slice(state_dim-2)
    sig_reg = np.zeros((state_dim,state_dim))
    sig_reg[it, it] = 1e-6

    for t in range(1,T):
        x_est[t,:]=np.dot(A_kal,x_est[t-1,:])+np.dot(B_kal,np.dot(k1,x_est[t-1,:])+k2)+np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim)
        
        cov[t,:,:]= np.dot(A_kal,np.dot(cov[t-1,:,:],np.transpose(A_kal)))+covdyn_kal +sig_reg
        #cov[t+1,:,:]=nearPD(cov[t+1,:,:])
        kalman_gain[t,:] = np.dot(cov[t,:,:],np.dot(np.transpose(R_x_dyn),(np.linalg.inv(np.dot(R_x_dyn,np.dot(cov[t,:,:],np.transpose(R_x_dyn)  )) + covrew)))).reshape(1,state_dim)
        cov[t,:,:]=cov[t,:,:] - np.dot(kalman_gain[t,:].reshape(state_dim,1),np.dot(R_x_dyn,cov[t,:,:]))
        x_est[t,:]=x_est[t,:]+ np.dot(kalman_gain[t].reshape(state_dim,1),(reward[t]-np.dot(R_x_dyn,x_est[t,:])-np.dot(R_u_dyn,(np.dot(k1,x_est[t,:])+k2))) ) 
        if np.isnan(x_est[t,:]).any():
            print "nans appearing in the covariance matrix just after the kalman filter at time",t
        
    if np.isnan(cov).any()  :
        print "nans appearing in the covariance matrix just after the kalman filter"

    if  np.isnan(kalman_gain).any() :
        print "nans appearing in the kalman gain matrix just after the kalman filter"
    
    J=np.zeros((T,state_dim,state_dim))
    x_est_smooth=np.zeros((T,state_dim))
    x_est_smooth[T-1,:]=x_est[T-1,:]

    
    cov_smooth=np.zeros((T,state_dim,state_dim))
    cov_smooth[T-1,:,:]=cov[T-1,:,:]

    for t in range(T-2,-1,-1): #from T to 0 with 1 step down
        
        J[t,:,:]=np.dot(cov[t,:,:],np.dot(np.transpose(A_kal), np.linalg.inv(cov[t+1,:,:]+1e-6*np.eye(6))))
        x_est_smooth[t,:] = (x_est[t,:].reshape(state_dim,1)+ np.dot(J[t,:,:], (x_est_smooth[t+1,:].reshape(state_dim,1)-   np.dot(A_kal,x_est[t,:].reshape(state_dim,1)) \
        - np.dot(B_kal,(np.dot(k1,x_est[t,:].reshape(state_dim,1))+k2.reshape(2,1))) - np.dot(s_hat,np.dot(np.linalg.inv(covrew),reward[t])))) ).reshape(1,state_dim)
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


    """ for t in range (T):
        x_t_smooth = x_est_smooth[t,:].reshape(6,1)
        x_t_plus_1_smooth = x_est_smooth[t+1,:].reshape(6,1)
        x_t_plus_1_r_t_transpose= np.dot(np.dot(x_t_plus_1_smooth, np.transpose( x_t_smooth ) ) + M[t+1,:,:] , np.transpose(R_x_dyn)  + np.dot( np.transpose(k1) , np.transpose(R_u_dyn) ) \
         ) + np.dot( x_t_plus_1_smooth ,   np.dot( np.transpose(k2.reshape(2,1)) , np.transpose(R_u_dyn) )  )

        r_t_r_t_1st_term =  multi_dot([ R_x_dyn, x_t_x_t , np.transpose (R_x_dyn) ]) 
        r_t_r_t_2nd_term =  multi_dot([R_x_dyn ,np.dot(x_t_x_t , np.transpose(k1) ) + np.dot( x_t_smooth,np.transpose(k2.reshape (2,1) ) ) \
         , np.transpose (R_u_dyn) ]  )   
        
        
        r_t_r_t_3rd_term= np.transpose(r_t_r_t_2nd_term)
    
        r_t_r_t_4th_term = covrew

        u_t_u_t_transpose= ( multi_dot( [k1 ,  np.dot( x_t_smooth, np.transpose(x_t_smooth) ) +cov_smooth[t,:,:] , np.transpose(k1)] ) \
        +multi_dot([k1,x_t_smooth,np.transpose(k2.reshape(2,1))]) +  np.transpose ( multi_dot([k1,x_t_smooth,np.transpose(k2.reshape(2,1))]) )  \
        + np.dot(k2.reshape(2,1) ,np.transpose(k2.reshape(2,1))  ) + policy_sigma )

        r_t_r_t_5th_term = multi_dot ([ R_u_dyn , u_t_u_t_transpose , np.transpose(R_u_dyn) ])

        r_t_r_t=r_t_r_t_1st_term+r_t_r_t_2nd_term+r_t_r_t_3rd_term+ r_t_r_t_4th_term +r_t_r_t_5th_term

        zeta_zeta=  np.vstack( ( np.hstack(  (    np.dot(x_t_plus_1_smooth, np.transpose(x_t_plus_1_smooth)) + cov_smooth[t+1,:,:]  \
        , x_t_plus_1_r_t_transpose   ) )  ,   np.hstack(  (   np.transpose(x_t_plus_1_r_t_transpose)  \
        ,  r_t_r_t  ) )   )  ) 

        #print  "shape of zeta zeta is",zeta_zeta.shape
        
        zeta_z_1 = np.hstack(( np.dot (x_t_plus_1_smooth,np.transpose(x_t_plus_1_smooth)) + M[t+1,:,:] \
         , np.dot( np.dot (x_t_plus_1_smooth,np.transpose(x_t_plus_1_smooth)) + M[t+1,:,:], np.transpose(k1)  ) + multi_dot([x_t_plus_1_smooth,np.transpose (k2.reshape(2,1))])) )

        u_t_x_t_transpose = np.transpose  (np.dot(x_t_x_t , np.transpose(k1) ) )

        r_t_x_t_transpose=   np.dot(R_x_dyn,x_t_x_t ) + np.dot(R_u_dyn , \
        u_t_x_t_transpose )  

        r_t_u_t_transpose = multi_dot([ R_x_dyn, x_t_x_t , np.transpose (k1) ])  + \
             multi_dot([ R_u_dyn ,  u_t_x_t_transpose ,  np.transpose(k1) ]) \
                + multi_dot([R_x_dyn ,x_t_smooth, np.transpose(k2.reshape(2,1)) ])+ multi_dot([ R_u_dyn, np.dot(k1,x_t_smooth)+k2.reshape(2,1) , np.transpose(k2.reshape(2,1))])

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

    Expec_log_joint=Expec_log_joint_1st_term + Expec_log_joint_sum

    print Expec_log_joint """
    
    def callbackF(Xi):
        global Nfeval
        write_csv(np.array(Xi))
        print "value of Xi is ",Xi
        print "value of the function eval is",f(Xi)

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
            + np.dot(k[12:14].reshape(2,).reshape(2,1) ,np.transpose(k[12:14].reshape(2,).reshape(2,1))  ) + k[14:18].reshape(2,2) )

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
    
    k_param=np.hstack ((k1.reshape(1,12),k2.reshape(1,2),policy_sigma.reshape(1,4)))
    #f(k_param.reshape((18,)))
    print "k_param is ",k_param
    print "the original value fo the function with the initial params are",f(k_param.reshape((18,)))
    [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] =  optimize.fmin_bfgs(f, k_param.reshape((18,)),callback=callbackF,maxiter=2000, full_output=True, retall=False)
    print fopt

    #test(k_param.reshape(18,))
    #a_min = optimize.minimize(f, k_param.reshape((18,)),  method='BFGS', jac=optimize.rosen_der, options={'gtol': 1e-6, 'disp': True ,'maxiter': 1e3 } ) 
   # 
    #print "amin is ",a_min


    
