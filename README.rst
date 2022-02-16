AI-IRW: Affine-Invariant Integrated Rank-Weighted depth
=========================================


This repository hosts Python code of the Affine-Invariant Integrated Rank-Weighted depth introduced in https://arxiv.org/abs/2106.11068.


Quick Start :
------------

Create toy training and testing datasets :

.. code:: python

  np.random.seed(0)  
  n_samples = 1000
  n_samples_test = 1000
  dim = 2
  mu = np.zeros(dim)
  sigma = np.identity(dim)
  
  X_train = np.random.multivariate_normal(mu,sigma,n_samples)
  X_test = np.random.multivariate_normal(mu,sigma,n_samples_test)
  
  
And then use AI-IRW to sort the dataset :  

.. code:: python

  score_aiirw = AI_IRW(X,AI=True, robust=True, X_test=Y, n_dirs=1000)
  rank_aiirw = np.argsort(score_aiirw)
  colors = [cm.viridis_r(x) for x in np.linspace(0, 1, n_samples_test) ]
  plt.scatter(Y[rank_aiirw,0], Y[rank_aiirw,1], s=10, c=colors, cmap='viridis')
  plt.show()
