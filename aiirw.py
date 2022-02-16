from utils import standardize, sampled_sphere
import numpy as np

def AI_IRW(X, AI=True, robust=False, n_dirs=None, X_test=None, random_state=None):
    """ Compute the score of the (Affine-invariant-) integrated rank 
        weighted depth of X_test w.r.t. X

    Parameters
    ----------

    X: Array of shape (n_samples, n_features)
            The training set.

    AI: bool
        if True, the affine-invariant version of irw is computed. 
        If False, the original irw is computed.

    robust: bool, default=False
        if robust is true, the MCD estimator of the covariance matrix
        is performed.

    n_dirs: int | None
        The number of random directions needed to approximate 
        the integral over the unit sphere.
        If None, n_dirs is set as 100* n_features.

    X_test: Array of shape (n_samples_test, n_features)
        The testing set. 
        If None, return the score of the training sample.

    random_state: int | None
        The random state.
        
    Returns
    -------
    ai_irw_score: Array
        Depth score of each element in X_test.
        If X_test is None, return the score of the training sample.
    """
    

    #Setting seed:
    if random_state is None:
        random_state = 0

    np.random.seed(random_state)




    if X_test is None:

        if AI:
            X_reduced, _ = standardize(X, robust)
        else:
            X_reduced = X.copy()

        n_samples, n_features = X_reduced.shape 
        
        #Setting the number of directions to 100 times the number of features as in the paper.
        if n_dirs is None:
            n_dirs = n_features * 100

        #Simulated random directions on the unit sphere. 
        U = sampled_sphere(n_dirs, n_features)

        sequence = np.arange(1, n_samples + 1)
        depth = np.zeros((n_samples, n_dirs))
        
        proj = np.matmul(X_reduced, U.T)
        rank_matrix = np.matrix.argsort(proj, axis=0)
        
        for k in range(n_dirs):
            depth[rank_matrix[:, k], k] = sequence  
        
        depth =  depth / (n_samples * 1.)         

        ai_irw_score = np.mean(np.minimum(depth, 1 - depth), axis = 1)

    else:

        if AI:
            X_reduced, Sigma_inv_square = standardize(X, robust)
            X_test_reduced = X_test@Sigma_inv_square
        else:
            X_reduced = X.copy()
            X_test_reduced = X_test.copy()

        n_samples, n_features = X_reduced.shape
        n_samples_test, _ = X_test_reduced.shape 

        #Setting the number of directions to 100 times the number of features as in the paper.
        if n_dirs is None:
            n_dirs = n_features * 100

        #Simulated random directions on the unit sphere. 
        U = sampled_sphere(n_dirs, n_features)

        proj = np.matmul(X_reduced, U.T)
        proj_test = np.matmul(X_test_reduced, U.T)

        sequence = np.arange(1, n_samples_test+1)
        depth = np.zeros((n_samples_test, n_dirs))
        temp = np.zeros((n_samples_test, n_dirs))

        proj.sort(axis=0)
        for k in range(n_dirs):
            depth[:,k] = np.searchsorted(a=proj[:,k],v=proj_test[:,k], side='left')

        depth /=   n_samples * 1. 

        ai_irw_score = np.mean(np.minimum(depth, 1 - depth), axis=1)

    return ai_irw_score







