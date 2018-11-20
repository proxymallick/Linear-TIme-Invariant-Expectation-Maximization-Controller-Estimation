
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.linalg
    
class GMM:
    
    def __init__(self, k = 3, eps = 0.0001):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`
        
        # All parameters from fitting/learning are kept in a named tuple
        from collections import namedtuple
    
    def fit_EM(self, X, max_iters = 1000):
        
        # n = number of data-points, d = dimension of data points        
        n, d = X.shape
        
        # randomly choose the starting centroids/means 
        ## as 3 of the points from datasets        
        mu = X[np.random.choice(n, self.k, False), :]
        
        # initialize the covariance matrices for each gaussians
        Sigma= [np.eye(d)] * self.k
        
        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k
        
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))
        
        ### log_likelihoods
        log_likelihoods = []

                
        """         P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                        * np.exp(-.5 * np.einsum('ij, ij -> i',\
                                X - mu, np.dot(    np.linalg.inv(s) , (X - mu).T).T ) )  """


        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
            * np.exp(-.5 * np.einsum('ij, ij -> i',\
                    (X - mu), (np.transpose( scipy.linalg.solve_triangular(scipy.linalg.cholesky(s, lower=True) , (X - mu).T,lower=True)) ) ))  
                        
        # Iterate till max_iters iterations        
        while len(log_likelihoods) < max_iters:
            
            # E - Step
            
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * estep_calc(X,mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            
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
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        
        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)       
        
        return self.params
    
    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()
    
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
    s=0.5 * (s + s.T) + 1e-6 * np.eye(X.shape[1])
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
    sigma = 0.5 * (sigma + sigma.T)
    
    #print mu.shape
    #print sigma.shape
    # Add sigma regularization.
    sigma += sig_reg
    #print "Conditioning the gaussian now::"
    # Conditioning to get dynamics.
    ##du is 6 and dx is 8
   # dynsig_dyn=sigma[dX+dU:, dX+dU:]-(np.linalg.solve(sigma[:dX+dU,:dX+dU],sigma[:dX+dU,dX+dU:]).T).dot(sigma[:dX+dU,dX+dU:])
    
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
   # print "fs shape is ",fd.shape
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
  #  print "fc shape is ",fc.shape
    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)

    return fd, fc, dynsig



def plot_ellipse2(ax, mu, sigma, color="k"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """
        
    if ax is None:
        ax = plt.gca()

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse) 


def higher_dimensions_dynamics_learn():
    X=np.random.randn(100,2)
    np.random.shuffle(X)
    gmm = GMM(k=6, eps=0.00000001)
    params = gmm.fit_EM(X, max_iters= 100)
    dX=1
    dU=0
    k=6
    mu_param=np.zeros((k,2)) 
    sigma_param=np.zeros((k,2,2)) 
    wghts=np.array(params.w).reshape(k,1)
    for iter in range (k):
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
    mu_moment=mu_moment.reshape(1,2)



    """     ax=plt.subplots()
    plot_ellipse2(ax,mu_moment,sigma_moment) """
    print sigma_moment.shape
    print mu_moment.shape
    
    # Normalize.
    m = float(m) / X.shape[0]
    n0 = float(n0) / X.shape[0]
        # Factor in multiplier.
    """     n0 = n0 * self._strength
    m = m * self._strength """
    # Multiply Phi by m (since it was normalized before).

    ###########
    #### Calculate the empirical mean and covariance of the data set
    ###########

    import pylab as plt    
    from matplotlib.patches import Ellipse
    
    def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
    
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(abs(vals))
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
        ax.add_artist(ellip)
        return ellip    
    
    def show2(X, mu, cov):

        plt.cla()
        K = 1#len(mu) # number of clusters
        colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
    #    plt.plot(X.T[0], X.T[1], 'm*')
        for k in range(K):
            plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  


    def show(X, mu, cov):

        plt.cla()
        K = len(mu) # number of clusters
        colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
        plt.plot(X.T[0], X.T[1], 'm*')
        for k in range(K):
            plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  

    """          
    fig = plt.figure(figsize = (13, 6))
    fig.add_subplot(121)
    show(X, params.mu, params.Sigma)
    fig.add_subplot(122)
    plt.plot(np.array(params.log_likelihoods))
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
    print gmm.predict(np.array([1, 2]))  """

    mu_vec=np.zeros((k+1,2)) 
    sigma_vec=np.zeros((k+1,2,2)) 
    #wghts=np.array(params.w).reshape(k,1)
    for iter in range (7):
        if iter == 6:
            mu_vec[iter,:]=mu_moment
            sigma_vec[iter,:,:]=sigma_moment
        else:
            mu_vec[iter,:]=mu_param[iter,:]
            sigma_vec[iter,:,:]=sigma_param[iter,:,:]
    plot_3d_2d(X,mu_vec,sigma_vec)
    #plot3d(mu_vec,sigma_vec)
    fig = plt.figure(figsize = (13, 6))
    fig.add_subplot(121)
  #  show(X, params.mu, params.Sigma)
    show(X, mu_param, sigma_param)
  #  show2(X, mu_moment, sigma_moment)
    #show_moment_plots(X, mu_moment, sigma_moment)
    fig.add_subplot(122)
    plt.plot(np.array(params.log_likelihoods))
    
    fig_new = plt.figure(figsize = (13, 6))
    fig_new.add_subplot(121)
    show(X, mu_vec, sigma_vec)

    fig_new2 = plt.figure(figsize = (13, 6))

    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')

    #plt.show()
  #  print gmm.predict(np.array([1, 2]))

    return Fm,fv,dyn_covar


def plot_3d_2d(X1,mu_new,sigma_new):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Our 2-dimensional distribution will be over variables X and Y
    N = 55
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 3, N)
    #X, Y = np.meshgrid(X1[:,0], X1[:,1])
    X, Y = np.meshgrid(X, Y)
    # Mean vector and covariance matrix

    # Pack X and Y into a single 3-dimensional array
    pos = np.zeros(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N
    print mu_new.shape
    fig = plt.figure(figsize = (13, 6))
    Z=np.zeros((7,X.shape[0],X.shape[1]))
    # The distribution on the variables X, Y packed into pos.
    ax= fig.add_subplot(122 , projection='3d')#fig.gca(projection='3d')
    ax1 = fig.add_subplot(121 , projection='3d') #fig.gca(projection='3d') #

    for iter in range(0,7):
        Z[iter,:,:] = multivariate_gaussian(pos, mu_new[iter,:].reshape(2,), sigma_new[iter,:,:])
        ax.plot_surface(X, Y, Z[iter,:,:].reshape(X.shape[0],X.shape[1]), rstride=3, cstride=3, linewidth=0.2, antialiased=True,
                        cmap=cm.viridis)
        cset = ax.contourf(X, Y, Z[iter,:,:].reshape(X.shape[0],X.shape[1]), offset=-0.35, cmap=cm.viridis)

    for iter in range(6,7):
        Z = multivariate_gaussian(pos, mu_new[iter,:].reshape(2,), sigma_new[iter,:,:])

        # Create a surface plot and projected filled contour plot under it.
        
        ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=0.2, antialiased=True,
                        cmap=cm.viridis)
        
        cset = ax1.contourf(X, Y, Z, zdir='z', offset=-0.35, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.5,1.5)
    ax1.set_zlim(-0.5,0.4)
  #  ax.set_zticks(np.linspace(0,0.2,5))
  #  ax1.set_zticks(np.linspace(0,0.2,5))

    ax.view_init(27, -21)
    ax1.view_init(27, -21)

    plt.show()


if __name__ == "__main__":
   # plot3d()
    
    higher_dimensions_dynamics_learn()
    
    # demo_2d()    
    """"from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filepath", help="File path for data")
    parser.add_option("-k", "--clusters", dest="clusters", help="No. of gaussians")    
    parser.add_option("-e", "--eps", dest="epsilon", help="Epsilon to stop")    
    parser.add_option("-m", "--maxiters", dest="max_iters", help="Maximum no. of iteration")        
    options, args = parser.parse_args()
    
    if not options.filepath : raise('File not provided')
    
    if not options.clusters :
        print("Used default number of clusters = 3" )
        k = 3
    else: k = int(options.clusters)
    
    if not options.epsilon :
        print("Used default eps = 0.0001" )
        eps = 0.0001
    else: eps = float(options.epsilon)
    
    if not options.max_iters :
        print("Used default maxiters = 1000" )
        max_iters = 1000
    else: eps = int(options.maxiters)
    
    X = np.genfromtxt(options.filepath, delimiter=',')
    gmm = GMM(k, eps)
    params = gmm.fit_EM(X, max_iters)
    print params.log_likelihoods
    gmm.plot_log_likelihood()
    print gmm.predict(np.array([1, 2])) """






