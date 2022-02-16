from sklearn.preprocessing import normalize
from sklearn.covariance import MinCovDet as MCD
from sklearn.decomposition import PCA
import numpy as np


########################################################
#################### Some useful functions ########################
######################################################## 

def cov_matrix(X, robust=False):
    """ Compute the covariance matrix of X.
    """
    if robust:
        cov = MCD().fit(X)
        sigma = cov.covariance_
    else:
        sigma = np.cov(X.T) 

    return sigma   

def standardize(X, robust=False):
    """ Compute the square inverse of the covariance matrix of X.
    """

    sigma = cov_matrix(X, robust)
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(sigma)

    if (rank < n_features):
        pca = PCA(rank)
        pca.fit(X)
        X_transf= pca.fit_transform(X)
        sigma = cov_matrix(X_transf)
    else:
        X_transf = X.copy()

    u, s , _ = np.linalg.svd(sigma)
    square_inv_matrix = u / np.sqrt(s)

    return X_transf@square_inv_matrix, square_inv_matrix


########################################################
#################### Sampled distributions ########################
######################################################## 

def sampled_sphere(n_dirs, d):
    """ Produce ndirs samples of d-dimensional uniform distribution on the 
        unit sphere
    """

    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=n_dirs)

    return normalize(U)

def Wishart_matrix(d):
    cov = np.random.randn(d,d)
    return cov.dot(cov.T)

def multivariate_t(mu, sigma, t, m):
    """
    Produce m samples of d-dimensional multivariate t distribution
    
    Args:
        mu (numpy.ndarray): mean vector
        sigma (numpy.ndarray): scale matrix (covariance)
        t (float): degrees of freedom
        m (int): # of samples to produce

    Returns:
        numpy.ndarray
    """
    d = len(mu)
    g = np.tile(np.random.gamma(t / 2 , 2 / t, m), (d, 1)).T
    z = np.random.multivariate_normal(np.zeros((d)),sigma,m)
    return mu + z / np.sqrt(g)