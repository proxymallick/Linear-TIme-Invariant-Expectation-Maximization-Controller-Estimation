import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

t=0

def print_results(f,T,state_dim,action_dim,params,A_x_dyn,B_u_dyn,
                w_t_kal,w_t,Adyn,Bdyn,Sigmadyn,data_K,data_k,data_sig,t_s,params_fval,initial_state_mu,p_1,A_kal,B_kal,s_hat,covrew,reward,R_x_dyn,R_u_dyn):
    global t
    dX = state_dim
    dU = action_dim
    """K=params[:,:12].reshape(T,dU,dX)
    k=params[:,12:14].reshape(T,dU)
    pol_covar=params[:,14:18].reshape(T,dU,dU) """



    x_em_test=np.zeros((T,state_dim))
    u_em_test=np.zeros((T,action_dim))

    mean_noise=np.zeros((state_dim,))
    

    covariance_init=1.e-6*np.identity(6)
    #initial_state_mu=np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])
    x_em_test[0,:]=  initial_state_mu  #np.random.randn(state_dim).reshape(1,state_dim)
    u_em_test[0,:]= np.random.multivariate_normal((np.dot(params[0,:12].reshape(2,6),x_em_test[0,:])+params[0,12:14].reshape(2,)).reshape(2,), (params[0,14:18].reshape(2,2)),1) 
    mpl.rcParams['legend.fontsize'] = 10


    fig = plt.figure()

    ##########                 ###################                         ##################
    ## Plot the generated trajectory from the EM parameters                             ######
    ##########                 ###################                         ##################

    ax = fig.add_subplot(243, projection='3d')
    for time in range(1,T):

        """ x_em_test[time,:]= np.dot(params[time,18:18+36].reshape(6,6),x_em_test[time-1,:])+np.dot(params[time,18+36:18+36+12].reshape(6,2),u_em_test[time-1,:]) + \
                             np.random.multivariate_normal(mean_noise,params[time,18+36+12:102].reshape(6,6),1) + Bdyn """
        #x_orig[time,:]= np.dot(A_x_dyn,x_orig[time-1,:])+np.dot(B_u_dyn,u_orig[time-1,:]) + w_t +Bdyn
        #x_em_test[time,:]= np.dot(A_x_dyn,x_em_test[time-1,:])+np.dot(B_u_dyn,u_em_test[time-1,:]) + w_t #+Bdyn #+ np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim) 
        x_em_test[time,:]= np.dot(A_kal,x_em_test[time-1,:])+np.dot(B_kal,u_em_test[time-1,:]) + w_t#_kal + np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim) 
        """ x_em_test[time,:]= np.dot(params[time,18:18+36].reshape(6,6),x_em_test[time-1,:])+np.dot(params[time,18+36:18+36+12].reshape(6,2),u_em_test[time-1,:]) \
                           +np.random.multivariate_normal(mean_noise,params[time,18+36+12:102].reshape(6,6),1)+w_t """
        u_em_test[time,:] = np.random.multivariate_normal((np.dot(params[time,:12].reshape(2,6),x_em_test[time,:])+params[time,12:14].reshape(2,)).reshape(2,),(params[time,14:18].reshape(2,2)),1) 
    
    x =x_em_test[0:T-1,0].reshape(x_em_test.shape[0]-1,)
    y =x_em_test[0:T-1,1].reshape(x_em_test.shape[0]-1,)
    z =x_em_test[0:T-1,2].reshape(x_em_test.shape[0]-1,)
        #ax.scatter(x, y, z, c='r', marker='o')
        #plt.pause(0.0001) 
    #ax.plot3D(x, y, z, 'gray')
    ax.scatter(x, y, z, c='b', marker='.')
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
        x_orig[time,:]= np.dot(A_x_dyn,x_orig[time-1,:])+np.dot(B_u_dyn,u_orig[time-1,:]) + w_t #+Bdyn
        #x_orig[time,:]= np.dot(A_kal,x_orig[time-1,:])+np.dot(B_kal,u_orig[time-1,:]) + w_t_kal + np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim)
        u_orig[time,:] = np.random.multivariate_normal((np.dot(params_orig[time,:12].reshape(2,6),x_orig[time,:])+params_orig[time,12:14].reshape(2,)).reshape(2,),(params_orig[time,14:18].reshape(2,2)),1) 
    x_o =x_orig[0:T-1,0].reshape(x_orig.shape[0]-1,)
    y_o =x_orig[0:T-1,1].reshape(x_orig.shape[0]-1,)
    z_o =x_orig[0:T-1,2].reshape(x_orig.shape[0]-1,)
        #ax_new.scatter(x_o, y_o, z_o, c='r', marker='o')
        #plt.pause(0.0001)
    ax_new.scatter(x_o, y_o, z_o, c='r', marker='.')
    #ax_new.plot3D(x_o, y_o, z_o, 'gray')
    #plt.pause(0.0001)
    ax_new.set_xlabel('X ')
    ax_new.set_ylabel('Y ')
    ax_new.set_zlabel('Z ')

    beta=1  
    Q_dim=6
    gamma=1
    q_dim=2
    small_fac=1e-5
    target_state= t_s
    Q_new=np.ones((6,1))
    q_new=np.ones((2,1))
    Q =  np.identity(Q_dim)
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
        ####
        #### Quadratic reward
        ####
        Quad_reward_orig[iter]= 0.05 * gamma * ( np.dot((x_orig[iter,:6]-target_state),np.dot(Q,np.transpose((x_orig[iter,:6]-target_state)))) + np.dot(u_orig[iter,:2],\
        np.dot(q, np.transpose(u_orig[iter,:2])) ) )
        #Quad_reward_orig[iter]=np.random.multivariate_normal ( np.dot( R_x_dyn ,x_orig[iter,:6] ) + np.dot(R_u_dyn,u_orig[iter,:2]) , covrew ,1 )
        ####
        #### Cubic reward
        ####
        """ Quad_reward_orig[iter]= gamma  *(  np.dot(  np.dot(x_orig[iter,:6]-target_state,Q_new) ,np.dot((x_orig[iter,:6]-target_state),np.dot(Q,np.transpose((x_orig[iter,:6]-target_state)))) )  +  \
                                 np.dot(  np.dot(u_orig[iter,:2],q_new) ,  np.dot(u_orig[iter,:2],np.dot(q, np.transpose(u_orig[iter,:2])) )  )    ) """
        
        likelihood_orig[iter]=-f(params_orig[iter,:])
    R_t_orig=beta * np.exp (-beta*(Quad_reward_orig))
    R_t_orig_quad=(Quad_reward_orig)
    print "The sum of the reward for the previously parameterized EM case/lqr is",np.cumsum(R_t_orig)

    ########
    ##########                 ###################                         ##################
    ## Test the rewards of the generated Em params                                     ######
    ##########                 ###################                         ##################
    Quad_reward_EM=np.zeros((T,1))
    R_t_EM=np.zeros((T,1))
    R_t_EM_quad=np.zeros((T,1))
    for iter in range (T):
        ####
        #### Quadratic reward
        ####
        Quad_reward_EM[iter]= 0.05 * gamma * ( np.dot((x_em_test[iter,:6]-target_state),np.dot(Q,np.transpose((x_em_test[iter,:6]-target_state)))) + np.dot(u_em_test[iter,:2],\
        np.dot(q, np.transpose(u_em_test[iter,:2])) ) )
        #Quad_reward_EM[iter]=np.random.multivariate_normal ( np.dot( R_x_dyn ,x_em_test[iter,:6] ) + np.dot(R_u_dyn,u_em_test[iter,:2]) , covrew ,1 )
        ####
        #### Cubic reward
        ####
        """ Quad_reward_EM[iter] =  gamma *(  np.dot(  np.dot(x_em_test[iter,:6]-target_state,Q_new) ,np.dot((x_em_test[iter,:6]-target_state),np.dot(Q,np.transpose((x_em_test[iter,:6]-target_state)))) )  +  \
                                 np.dot(  np.dot(u_em_test[iter,:2],q_new) ,  np.dot(u_em_test[iter,:2],np.dot(q, np.transpose(u_em_test[iter,:2])) )  )    ) """

        likelihood_EM[iter]=-f(params[iter,:])
    R_t_EM=beta* np.exp (-beta * (Quad_reward_EM))
    R_t_EM_quad=(Quad_reward_EM)
    print "The sum of the reward for the newly parameterized EM case is",np.cumsum(R_t_EM)

    ax_reward = fig.add_subplot(241)
    #plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    plt.plot(np.arange(R_t_orig.size), R_t_orig ,c='r', marker='.')

    ax_reward = fig.add_subplot(244)
    plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='b', marker='.')
    #plt.plot(np.arange(R_t_orig.size), R_t_orig ,c='b', marker='o')


    ax_reward = fig.add_subplot(246)
    plt.plot(np.arange(likelihood_orig.size), likelihood_orig*1.e-6 ,c='r', marker='.')
    plt.plot(np.arange(params_fval.size), -params_fval*1.e-6 ,c='b', marker='.')
   # ax_reward.Plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    #ax_reward.scatter(np.arange(R_t_EM.size), R_t_orig ,c='b', marker='o')
    ax_reward = fig.add_subplot(247)
    #plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    plt.plot(np.arange(R_t_orig_quad.size), R_t_orig_quad ,c='r', marker='.')

    ax_reward = fig.add_subplot(248)
    plt.plot(np.arange(R_t_orig_quad.size), np.cumsum(R_t_orig) ,c='r', marker='.')
    plt.plot(np.arange(R_t_EM_quad.size), np.cumsum(R_t_EM) ,c='b', marker='.')
    #ax.set_xlim(-2.0,30)
    #ax.set_ylim(-2.0,30)

    X_plt = np.arange(params_fval.size)
    Y_plt = np.linspace(5, 20, params_fval.size)
    #X_plt, Y_plt = np.meshgrid(X_plt, Y_plt)
    ax_reward = fig.add_subplot(245,projection='3d')
    plt.plot(np.arange(params_fval.size), -params_fval*1.e-6 ,c='b', marker='.')

    A=np.linspace(5,20,params_fval.size).reshape(params_fval.size)#np.linspace(5,20,params_fval.size).reshape(params_fval.size)
    B=np.random.randn(30,1).reshape(30,)#np.arange(params_fval.size).reshape(params_fval.size,)
    C=  (-params_fval)
    D= (likelihood_orig).reshape(likelihood_orig.size)
    
    fig_test = plt.figure()
    ax_test = fig_test.gca(projection='3d')
    #surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    surf = ax_test.plot_trisurf(A,B,C, alpha = 1,  cmap=cm.coolwarm, linewidth=0.5, antialiased=True, zorder = 0.5)
    surf2 = ax_test.plot_trisurf(A,B,D, alpha = 1,  cmap=cm.viridis, linewidth=0.5, antialiased=True, zorder = 0.5)
    #ax_test.plot_trisurf(A,B,D, cmap=plt.cm.viridis, linewidth=0.2)
    #ax.plot_surface(X, Y, Exp_Fric_map, alpha = 1, rstride=1, cstride=1, cmap=cm.winter, linewidth=0.5, antialiased=True, zorder = 0.5)
    #ax.plot_surface(X, Y, Fric_map, alpha = 1, rstride=1, cstride=1, cmap=cm.autumn,linewidth=0.5, antialiased=True, zorder = 0.3)
    fig_test.colorbar( surf, shrink=0.5, aspect=5)
    fig_test.colorbar( surf2, shrink=0.5, aspect=5)
    ax_test.xaxis.set_major_locator(MaxNLocator(5))
    ax_test.yaxis.set_major_locator(MaxNLocator(6))
    ax_test.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()


    plt.show()
    plt.close


def print_results_time_varying(f,T,state_dim,action_dim,params,A_x_dyn,B_u_dyn,
                w_t_kal,w_t,Adyn,Bdyn,Sigmadyn,data_K,data_k,data_sig,t_s,params_fval,initial_state_mu,p_1,A_kal,B_kal,s_hat,covrew,reward,intermediate_values):
    global t
    dX = state_dim
    dU = action_dim
    """K=params[:,:12].reshape(T,dU,dX)
    k=params[:,12:14].reshape(T,dU)
    pol_covar=params[:,14:18].reshape(T,dU,dU) """

    x_em_test=np.zeros((T,state_dim))
    u_em_test=np.zeros((T,action_dim))

    mean_noise=np.zeros((state_dim,))
    

    covariance_init=1.e-6*np.identity(6)
    #initial_state_mu=np.array([ 0.0,  20.0, 0.0, 0.0,  4.0, 0.00])
    x_em_test[0,:]=  initial_state_mu  #np.random.randn(state_dim).reshape(1,state_dim)
    u_em_test[0,:]= np.random.multivariate_normal((np.dot(params[0,:12].reshape(2,6),x_em_test[0,:])+params[0,12:14].reshape(2,)).reshape(2,), (params[0,14:18].reshape(2,2)),1) 
    mpl.rcParams['legend.fontsize'] = 10
    
    print "The shape of w_t is",w_t.shape
    fig = plt.figure()

    ##########                 ###################                         ##################
    ## Plot the generated trajectory from the EM parameters                             ######
    ##########                 ###################                         ##################

    ax = fig.add_subplot(243, projection='3d')
    for time in range(1,T-1):

        """ x_em_test[time,:]= np.dot(params[time,18:18+36].reshape(6,6),x_em_test[time-1,:])+np.dot(params[time,18+36:18+36+12].reshape(6,2),u_em_test[time-1,:]) + \
                             np.random.multivariate_normal(mean_noise,params[time,18+36+12:102].reshape(6,6),1) + Bdyn """
        #x_orig[time,:]= np.dot(A_x_dyn,x_orig[time-1,:])+np.dot(B_u_dyn,u_orig[time-1,:]) + w_t +Bdyn
        #x_em_test[time,:]= np.dot(A_x_dyn[t,:,:],x_em_test[time-1,:])+np.dot(B_u_dyn[t,:,:],u_em_test[time-1,:]) + w_t[t,:] #+Bdyn #+ np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim) 
        #x_em_test[time,:]= np.dot(A_kal,x_em_test[time-1,:])+np.dot(B_kal,u_em_test[time-1,:]) + w_t#_kal + np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim) 
        x_em_test[time,:]= np.dot(A_kal[t,:,:],x_em_test[time-1,:])+np.dot(B_kal[t,:,:],u_em_test[time-1,:]) + w_t_kal[t,:]
        """ x_em_test[time,:]= np.dot(params[time,18:18+36].reshape(6,6),x_em_test[time-1,:])+np.dot(params[time,18+36:18+36+12].reshape(6,2),u_em_test[time-1,:]) \
                           +np.random.multivariate_normal(mean_noise,params[time,18+36+12:102].reshape(6,6),1)#+w_t[t,:] """
        u_em_test[time,:] = np.random.multivariate_normal((np.dot(params[time,:12].reshape(2,6),x_em_test[time,:])+params[time,12:14].reshape(2,)).reshape(2,),(params[time,14:18].reshape(2,2)),1) 

    x =x_em_test[0:T-1,0].reshape(x_em_test.shape[0]-1,)
    y =x_em_test[0:T-1,1].reshape(x_em_test.shape[0]-1,)
    z =x_em_test[0:T-1,2].reshape(x_em_test.shape[0]-1,)
        #ax.scatter(x, y, z, c='r', marker='o')
        #plt.pause(0.0001) 
    #ax.plot3D(x, y, z, 'gray')
    ax.scatter(x, y, z, c='b', marker='.')
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
    for t in range(1,T-1):
        params_orig[t,:]=  (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2), data_sig[t,:,:].reshape(1,4))))
    u_orig[0,:]=np.random.multivariate_normal((np.dot(params_orig[0,:12].reshape(2,6),x_em_test[0,:])+params_orig[0,12:14].reshape(2,)).reshape(2,), (params_orig[0,14:18].reshape(2,2)),1)
    x_orig[0,:]=initial_state_mu

    ax_new = fig.add_subplot(242, projection='3d')
    for time in range(1,T-1):
        #x_orig[t,:]=np.dot(A_x_orig_dyn,x_orig[t-1,:])+np.dot(B_u_dyn,u[t-1,:]) + w_t
        #u[t,:]=np.random.multivariate_normal((np.dot(k1,x_orig[t-1,:])+k2).reshape(2,),policy_sigma,1)
        x_orig[time,:]= np.dot(A_x_dyn[t,:,:],x_orig[time-1,:])+np.dot(B_u_dyn[t,:,:],u_orig[time-1,:]) + w_t[t,:] #+Bdyn
        #x_orig[time,:]= np.dot(A_kal,x_orig[time-1,:])+np.dot(B_kal,u_orig[time-1,:]) + w_t_kal + np.dot(s_hat, np.dot(np.linalg.inv(covrew),reward[t])).reshape(1,state_dim)
        u_orig[time,:] = np.random.multivariate_normal((np.dot(params_orig[time,:12].reshape(2,6),x_orig[time,:])+params_orig[time,12:14].reshape(2,)).reshape(2,),(params_orig[time,14:18].reshape(2,2)),1) 
    x_o =x_orig[0:T-1,0].reshape(x_orig.shape[0]-1,)
    y_o =x_orig[0:T-1,1].reshape(x_orig.shape[0]-1,)
    z_o =x_orig[0:T-1,2].reshape(x_orig.shape[0]-1,)
        #ax_new.scatter(x_o, y_o, z_o, c='r', marker='o')
        #plt.pause(0.0001)
    ax_new.scatter(x_o, y_o, z_o, c='r', marker='.')
    #ax_new.plot3D(x_o, y_o, z_o, 'gray')
    #plt.pause(0.0001)
    ax_new.set_xlabel('X ')
    ax_new.set_ylabel('Y ')
    ax_new.set_zlabel('Z ')

    beta=1  
    Q_dim=6
    gamma=1
    q_dim=2
    small_fac=1e-5
    target_state= t_s
    Q_new=np.ones((6,1))
    q_new=np.ones((2,1))
    Q =  np.identity(Q_dim)
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
    for iter in range (T-1):
        ####
        #### Quadratic reward
        ####
        Quad_reward_orig[iter]= gamma * ( np.dot((x_orig[iter,:6]-target_state),np.dot(Q,np.transpose((x_orig[iter,:6]-target_state)))) + np.dot(u_orig[iter,:2],\
        np.dot(q, np.transpose(u_orig[iter,:2])) ) )
        ####
        #### Cubic reward
        ####
        """ Quad_reward_orig[iter]= gamma  *(  np.dot(  np.dot(x_orig[iter,:6]-target_state,Q_new) ,np.dot((x_orig[iter,:6]-target_state),np.dot(Q,np.transpose((x_orig[iter,:6]-target_state)))) )  +  \
                                 np.dot(  np.dot(u_orig[iter,:2],q_new) ,  np.dot(u_orig[iter,:2],np.dot(q, np.transpose(u_orig[iter,:2])) )  )    ) """
        
        likelihood_orig[iter]=-f(params_orig[iter,:])
    R_t_orig=beta * np.exp (-beta*(Quad_reward_orig))
    R_t_orig_quad=(Quad_reward_orig)
    print "The sum of the reward for the previously parameterized EM case/lqr is",np.sum(R_t_orig)

    ########
    ##########                 ###################                         ##################
    ## Test the rewards of the generated Em params                                     ######
    ##########                 ###################                         ##################
    Quad_reward_EM=np.zeros((T,1))
    R_t_EM=np.zeros((T,1))
    R_t_EM_quad=np.zeros((T,1))
    for iter in range (T-1):
        ####
        #### Quadratic reward
        ####
        Quad_reward_EM[iter]= gamma * ( np.dot((x_em_test[iter,:6]-target_state),np.dot(Q,np.transpose((x_em_test[iter,:6]-target_state)))) + np.dot(u_em_test[iter,:2],\
        np.dot(q, np.transpose(u_em_test[iter,:2])) ) )
        ####
        #### Cubic reward
        ####
        """ Quad_reward_EM[iter] =  gamma *(  np.dot(  np.dot(x_em_test[iter,:6]-target_state,Q_new) ,np.dot((x_em_test[iter,:6]-target_state),np.dot(Q,np.transpose((x_em_test[iter,:6]-target_state)))) )  +  \
                                 np.dot(  np.dot(u_em_test[iter,:2],q_new) ,  np.dot(u_em_test[iter,:2],np.dot(q, np.transpose(u_em_test[iter,:2])) )  )    ) """

        likelihood_EM[iter]=-f(params[iter,:])
    R_t_EM=beta* np.exp (-beta * (Quad_reward_EM))
    R_t_EM_quad=(Quad_reward_EM)
    print "The sum of the reward for the newly parameterized EM case is",np.sum(R_t_EM)

    ax_reward = fig.add_subplot(241)
    #plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    plt.plot(np.arange(intermediate_values[:200].size), -intermediate_values[:200] ,c='r', marker='.')

    ax_reward = fig.add_subplot(244)
    plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='b', marker='.')
    #plt.plot(np.arange(R_t_orig.size), R_t_orig ,c='b', marker='o')


    ax_reward = fig.add_subplot(246)
    plt.plot(np.arange(likelihood_orig.size), likelihood_orig*1.e-6 ,c='r', marker='.')
    plt.plot(np.arange(params_fval.size), -params_fval*1.e-6 ,c='b', marker='.')
   # ax_reward.Plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    #ax_reward.scatter(np.arange(R_t_EM.size), R_t_orig ,c='b', marker='o')
    ax_reward = fig.add_subplot(247)
    #plt.plot(np.arange(R_t_EM.size), R_t_EM ,c='r', marker='o')
    plt.plot(np.arange(R_t_orig_quad.size), R_t_orig_quad ,c='r', marker='.')

    ax_reward = fig.add_subplot(248)
    plt.plot(np.arange(R_t_orig_quad.size), R_t_orig_quad ,c='r', marker='.')
    plt.plot(np.arange(R_t_EM_quad.size), R_t_EM_quad ,c='b', marker='.')
    #ax.set_xlim(-2.0,30)
    #ax.set_ylim(-2.0,30)

    X_plt = np.arange(params_fval.size)
    Y_plt = np.linspace(5, 20, params_fval.size)
    #X_plt, Y_plt = np.meshgrid(X_plt, Y_plt)
    ax_reward = fig.add_subplot(245,projection='3d')
    plt.plot(np.arange(params_fval.size), -params_fval*1.e-6 ,c='b', marker='.')

    A=np.linspace(5,20,params_fval.size).reshape(params_fval.size)#np.linspace(5,20,params_fval.size).reshape(params_fval.size)
    B=np.random.randn(params_fval.size,1).reshape(params_fval.size,)#np.arange(params_fval.size).reshape(params_fval.size,)
    C=-params_fval
    
    D= (likelihood_orig).reshape(likelihood_orig.size)
    
    fig_test = plt.figure()
    ax_test = fig_test.gca(projection='3d')
    #surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    surf1 = ax_test.plot_trisurf(A,B,C, cmap=plt.cm.viridis, linewidth=0.2)
    surf2 = ax_test.plot_trisurf(A,B,D, cmap=plt.cm.autumn, linewidth=0.2)

    #ax_test.plot_trisurf(A,B,D, cmap=plt.cm.viridis, linewidth=0.2)

    #fig_test.colorbar( surf, shrink=0.5, aspect=5)
    ax_test.xaxis.set_major_locator(MaxNLocator(5))
    ax_test.yaxis.set_major_locator(MaxNLocator(6))
    ax_test.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()


    plt.show()
    plt.close



