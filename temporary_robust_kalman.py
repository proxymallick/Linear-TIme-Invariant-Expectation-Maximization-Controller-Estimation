import time,timeit
import numpy as np
def kf_ks(T,state_dim,action_dim,A_kal,B_kal,data_K,data_k,
                                            data_sig,data_X,s_hat,covrew,
                                            reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn):   
    time_start=timeit.timeit()    
    x_est=np.zeros((T,state_dim))
    x_est[0,:]=initial_state_mu
    cov=np.zeros((T,state_dim,state_dim))
    cov[0,:,:]= 0.5*(p_1_n+np.transpose(p_1_n)) 
    
    
    kalman_gain=np.zeros((T,state_dim))

    
    it = slice(state_dim-2)
    sig_reg = np.zeros((state_dim,state_dim))
    sig_reg[it, it] = 1e-6

    for t in range(1,T):
        #cov_cholesky_prev= np.linalg.cholesky(cov[t,:,:])
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
    time_end=timeit.timeit()
    print 'The total time consumed by the kalman filter and smoother evaluation is %0.3f '% np.abs( (time_start-time_end)*1000.0)
    return x_est_smooth,cov_smooth,M



def robust_kf_ks(T,state_dim,action_dim,A_kal,B_kal,data_K,data_k,
                                            data_sig,data_X,s_hat,covrew,
                                            reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn):   
    time_start=timeit.timeit()    
    x_est=np.zeros((T,state_dim))
    x_est[0,:]=initial_state_mu
    cov=np.zeros((T,state_dim,state_dim))
    cov[0,:,:]= 0.5*(p_1_n+np.transpose(p_1_n)) 

    cov_cholesky_prev=np.zeros((T,state_dim,state_dim))
    cov_cholesky=np.zeros((T,state_dim,state_dim))
    kalman_gain=np.zeros((T,state_dim))
    
    
    p_t_t_minus_1=np.zeros((T,state_dim,state_dim))
    it = slice(state_dim-2)
    sig_reg = np.zeros((state_dim,state_dim))
    sig_reg[it, it] = 1e-6

    for t in range(1,T):
        cov_cholesky_prev= np.linalg.cholesky(cov[t-1,:,:])
        full_cov = np.dot( cov_cholesky_prev, cov_cholesky_prev.T) 
        x_est[t,:]=np.dot(A_kal,x_est[t-1,:])+np.dot(B_kal,np.dot(data_K[t-1,:,:],x_est[t-1,:])+data_k[t-1,:].reshape(2,))+np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t-1])).reshape(1,state_dim)
        cov[t,:,:]= np.dot(A_kal,np.dot( full_cov ,np.transpose(A_kal)))+covdyn_kal #+sig_reg
        p_t_t_minus_1[t,:,:]=cov[t,:,:]
        #cov[t+1,:,:]=nearPD(cov[t+1,:,:])
        cov_cholesky= np.linalg.cholesky(cov[t,:,:])
        
        kalman_gain[t,:] = np.dot(cov[t,:,:],np.dot(np.transpose(R_x_dyn),(np.linalg.inv(np.dot(R_x_dyn,np.dot(cov[t,:,:],np.transpose(R_x_dyn)  )) + covrew)))).reshape(1,state_dim)
        
        cov[t,:,:]=cov[t,:,:]- np.dot(kalman_gain[t,:].reshape(state_dim,1),np.dot(R_x_dyn,cov[t,:,:]))
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
        
        J[t,:,:]=np.dot(cov[t,:,:],np.dot(np.transpose(A_kal), np.linalg.inv(p_t_t_minus_1[t+1,:,:])))
        
        x_est_smooth[t,:] = (x_est[t,:].reshape(state_dim,1)+ np.dot(J[t,:,:], (x_est_smooth[t+1,:].reshape(state_dim,1)-   np.dot(A_kal,x_est[t,:].reshape(state_dim,1)) \
        - np.dot(B_kal,(np.dot(data_K[t,:,:],x_est[t,:].reshape(state_dim,1))+data_k[t,:].reshape(2,1))) - np.dot(s_hat,np.dot(np.linalg.inv(covrew),reward[t])))) ).reshape(1,state_dim)
        
        cov_smooth[t,:,:]=cov[t,:,:]+np.dot(J[t,:,:] , np.dot(cov_smooth[t+1,:,:]-p_t_t_minus_1[t+1,:,:],np.transpose(J[t,:,:])))
        np.linalg.cholesky(cov_smooth[t,:,:])
    

    M=np.zeros((T,state_dim,state_dim))

    M[T-1,:,:]= (np.dot((np.identity(state_dim)-  np.dot( kalman_gain[T-1,:].reshape(state_dim,1),R_x_dyn)) , np.dot( A_kal,cov_smooth[T-2,:,:] )))
  #  
    for t in range(T-2,-1,-1):
        M[t,:,:]=np.dot(cov[t,:,:], np.transpose( J[t-1,:,:]))+np.dot(J[t,:,:],np.dot((M[t+1,:,:]-np.dot (A_kal,cov[t,:,:])),np.transpose(J[t-1,:,:])))   


    if np.isnan(M).any() or  np.isnan(cov_smooth).any() :
        print "nans appearing in the covariance matrices or the kalman gains of the smoother evaluations"
    time_end=timeit.timeit()
    print 'The total time consumed by the kalman filter and smoother evaluation is %0.3f ms'% np.abs( (time_start-time_end)*1000.0)
    
    return x_est_smooth,cov_smooth,M
