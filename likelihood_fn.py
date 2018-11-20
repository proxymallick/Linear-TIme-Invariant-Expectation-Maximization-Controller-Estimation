


def calculate_complete_loglikelihood(T,state_dim,action_dim,A_kal,B_kal,data_K,data_k,
                                            data_sig,data_X,s_hat,covrew,
                                            reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn,x_est_smooth,cov_smooth,M,A_total):
    
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

