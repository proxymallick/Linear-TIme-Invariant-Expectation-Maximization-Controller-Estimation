from __future__ import division
import pickle
import numpy as np
import csv



""" updated_em = open('/home/prakash/gps/complete_data_X.pkl', 'rb')
params = pickle.load(updated_em)
print params.shape
updated_em.close()

#############################################################
### LBFGS Optimization of the Joint Complete Likelihood 
### of the rewards and latent variables 
#############################################################
updated_params=np.zeros((100,18))
    output = open('updated_em_params.pkl', 'wb')
    time1 = timeit.timeit()
    for t in range(50):
        
        [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] \
         =  optimize.fmin_bfgs(f, \
                                (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2),data_sig[t,:,:].reshape(1,4)))).reshape((18,)) \
                                        ,callback=callbackF,maxiter=10, full_output=True, retall=False)
        
        
        print "finished",t,"iterations in",
        #print xopt
        #write_csv(xopt)
        updated_params[t,:]=xopt
        #print updated_params
    time2 = timeit.timeit()
    print ' function optimization took %0.3f ms' % ( (time2-time1)*1000.0)
    pickle.dump(updated_params, output)
    print "The parameters are updated and dumped into the picke file"
    output.close()
    print "~~~~~~.....The EM algorithm finished successfully......~~~~~~~" 
    dd """
""" function [A] = rchol(A)
A = triu(A); n = size(A,1); tol = n*eps;
if A(1,1) <= tol,
A(1,1:n) = 0;
else
A(1,1:n) = A(1,1:n)/sqrt(A(1,1));
end

for j=2:n,
A(j,j:n) = A(j,j:n) - A(1:j-1,j)'*A(1:j-1,j:n);
if A(j,j) <= tol,
A(j,j:n) = 0;
else
A(j,j:n) = A(j,j:n)/sqrt(A(j,j));
end
end """


def Testing(T,state_dim,action_dim,A_x_dyn,B_u_dyn,w_t,v_t,Bdyn,data_K,data_k,data_sig,f):
    
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

    
    x_em_test=np.zeros((T,state_dim))
    u_em_test=np.zeros((T,action_dim))

    covariance_init=1.e-6*np.identity(6)
    initial_state_mu=np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])
    x_em_test[0,:]=  initial_state_mu  #np.random.randn(state_dim).reshape(1,state_dim)
    u_em_test[0,:]= np.random.multivariate_normal((np.dot(params[0,:12].reshape(2,6),x_em_test[0,:])+params[0,12:14].reshape(2,)).reshape(2,), (params[0,14:18].reshape(2,2)),1) 
    mpl.rcParams['legend.fontsize'] = 10


    fig = plt.figure()

    ##########                 ###################                         ##################
    ## Plot the generated trajectory from the EM parameters                             ######
    ##########                 ###################                         ##################

    ax = fig.add_subplot(243, projection='3d')
    for time in range(1,T-1):
        #params[time,14:18] =  (0.5 * (params[time,14:18].reshape(2,2) + np.transpose(params[time,14:18].reshape(2,2)))).reshape(1,4)
        #print nearPD( params[time,14:18].reshape(2,2))#(is_pd(params[time,14:18].reshape(2,2)) )# and np.all(np.linalg.eigvals(params[time,14:18].reshape(2,2))) )
        #x_em_test[t,:]=np.dot(A_x_em_test_dyn,x_em_test[t-1,:])+np.dot(B_u_dyn,u[t-1,:]) + w_t
        #u[t,:]=np.random.multivariate_normal((np.dot(k1,x_em_test[t-1,:])+k2).reshape(2,),policy_sigma,1)
        x_em_test[time,:]= np.dot(A_x_dyn,x_em_test[time-1,:])+np.dot(B_u_dyn,u_em_test[time-1,:]) + w_t + Bdyn
        u_em_test[time,:] = np.random.multivariate_normal((np.dot(params[time,:12].reshape(2,6),x_em_test[time,:])+params[time,12:14].reshape(2,)).reshape(2,),(params[time,14:18].reshape(2,2)),1) 
    x =x_em_test[:,0].reshape(x_em_test.shape[0],)
    y =x_em_test[:,1].reshape(x_em_test.shape[0],)
    z =x_em_test[:,2].reshape(x_em_test.shape[0],)
        #ax.scatter(x, y, z, c='r', marker='o')
        #plt.pause(0.0001) 

    ax.scatter(x, y, z, c='b', marker='o')
    #plt.pause(0.0001)
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')


    ##########                 ###################                         ##################
    ## Plot the generated trajectory from the LQR Parameters                           ######
    ##########                 ###################                         ##################
    x_orig=np.zeros((T,state_dim))
    u_orig=np.zeros((T,action_dim))
    params_orig=np.zeros((T,18))
    for t in range(1,T):
        params_orig[t,:]=  (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2), data_sig[t,:,:].reshape(1,4))))
    u_orig[0,:]=np.random.multivariate_normal((np.dot(params_orig[0,:12].reshape(2,6),x_em_test[0,:])+params_orig[0,12:14].reshape(2,)).reshape(2,), (params_orig[0,14:18].reshape(2,2)),1)
    x_orig[0,:]=initial_state_mu

    ax_new = fig.add_subplot(242, projection='3d')
    for time in range(1,T):
        #x_orig[t,:]=np.dot(A_x_orig_dyn,x_orig[t-1,:])+np.dot(B_u_dyn,u[t-1,:]) + w_t
        #u[t,:]=np.random.multivariate_normal((np.dot(k1,x_orig[t-1,:])+k2).reshape(2,),policy_sigma,1)
        x_orig[time,:]= np.dot(A_x_dyn,x_orig[time-1,:])+np.dot(B_u_dyn,u_orig[time-1,:]) + w_t +Bdyn
        u_orig[time,:] = np.random.multivariate_normal((np.dot(params_orig[time,:12].reshape(2,6),x_orig[time,:])+params_orig[time,12:14].reshape(2,)).reshape(2,),(params_orig[time,14:18].reshape(2,2)),1) 
    x_o =x_orig[:,0].reshape(x_orig.shape[0],)
    y_o =x_orig[:,1].reshape(x_orig.shape[0],)
    z_o =x_orig[:,2].reshape(x_orig.shape[0],)
        #ax_new.scatter(x_o, y_o, z_o, c='r', marker='o')
        #plt.pause(0.0001)
    ax_new.scatter(x_o, y_o, z_o, c='r', marker='o')
    #plt.pause(0.0001)
    ax_new.set_xlabel('X ')
    ax_new.set_ylabel('Y ')
    ax_new.set_zlabel('Z ')

    beta=1  
    Q_dim=6
    gamma=1
    q_dim=2
    small_fac=1e-15
    target_state= np.array([ 20, 4, 0, 0.0,  0.0, 0.00]).reshape(1,6)


    Q = small_fac* np.identity(Q_dim)
    q = small_fac* np.identity(q_dim)
    ########
    ##########                 ###################                         ##################
    ## Test the rewards of  the LQR params                  ######
    ##########                 ###################                         ##################
    


    Quad_reward_orig=np.zeros((T,1))
    exp_reward_orig=np.zeros((T,1))
    likelihood_EM=np.zeros((T,1))
    likelihood_orig=np.zeros((T,1))
    R_t_orig=np.zeros((T,1))
    R_t_orig_quad=np.zeros((T,1))
    for iter in range (T):
        Quad_reward_orig[iter]= gamma * ( np.dot((x_orig[iter,:6]-target_state),np.dot(Q,np.transpose((x_orig[iter,:6]-target_state)))) + np.dot(u_orig[iter,:2],\
        np.dot(q, np.transpose(u_orig[iter,:2])) ) )
        
        likelihood_orig[iter]=f(params_orig[iter,:])
    R_t_orig=beta * np.exp (-beta*np.cumsum(Quad_reward_orig))
    R_t_orig_quad=(Quad_reward_orig)

    ########
    ##########                 ###################                         ##################
    ## Test the rewards of the generated Em params                  ######
    ##########                 ###################                         ##################
    Quad_reward_EM=np.zeros((T,1))
    R_t_EM=np.zeros((T,1))
    R_t_EM_quad=np.zeros((T,1))
    for iter in range (T):
        Quad_reward_EM[iter]= gamma * ( np.dot((x_em_test[iter,:6]-target_state),np.dot(Q,np.transpose((x_em_test[iter,:6]-target_state)))) + np.dot(u_em_test[iter,:2],\
        np.dot(q, np.transpose(u_em_test[iter,:2])) ) )
        likelihood_EM[iter]=f(params[iter,:])
    R_t_EM=beta* np.exp (-beta * np.cumsum(Quad_reward_EM))
    R_t_EM_quad=(Quad_reward_EM)
    print R_t_EM.size

    ax_reward = fig.add_subplot(241)
    #plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    plt.plot(np.arange(R_t_orig.size), R_t_orig ,c='b', marker='o')

    ax_reward = fig.add_subplot(244)
    plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    #plt.plot(np.arange(R_t_orig.size), R_t_orig ,c='b', marker='o')
    ax_reward = fig.add_subplot(245)
    #plt.plot(np.arange(likelihood_EM.size), likelihood_EM ,c='b', marker='o')
    plt.plot(np.arange(params_fval.size), params_fval ,c='b', marker='o')

    ax_reward = fig.add_subplot(246)
    plt.plot(np.arange(likelihood_orig.size), likelihood_orig ,c='r', marker='o')
   # ax_reward.Plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    #ax_reward.scatter(np.arange(R_t_EM.size), R_t_orig ,c='b', marker='o')
    ax_reward = fig.add_subplot(247)
    #plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    plt.plot(np.arange(R_t_orig_quad.size), R_t_orig_quad ,c='r', marker='o')

    ax_reward = fig.add_subplot(248)

    plt.plot(np.arange(R_t_EM_quad.size), R_t_EM_quad ,c='b', marker='o')
    #ax.set_xlim(-2.0,30)
    #ax.set_ylim(-2.0,30)

    plt.show()
    plt.close

    



def cholesky(A):
    A = np.triu(A); n = np.shape(A)[0]; tol = n*np.spacing(1)


    if A[0,0]<=tol :
        A[0,0:n]=0
        #print A
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
A=np.matrix('100,23,2;0,3,45;5.0,26,700')

print cholesky(A)
