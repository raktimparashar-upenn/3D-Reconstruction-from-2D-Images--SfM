
import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """
  # initial estimate of E.
  A = np.zeros((1,9))
  for i in range(X1.shape[0]):
      prod = np.kron(X2[i], X1[i])
      #prod = np.kron(X1[i], X2[i].T)
      A = np.vstack((A, prod))
  # print(A.shape)
  A = A[1:, :]
  
  # SVD of A
  U, S, Vt = np.linalg.svd(A)
  
  # Setting E_hat (initial estimate of E) as the last right singular vector of A.
  E_hat = Vt.T[:, -1].reshape(3,3)

  # Reprojecting E.
  U, S, Vt = np.linalg.svd(E_hat)

  # Adjusting the singular matrix S
  S = np.diag([1,1,0])

  E = U @ S @ Vt
  E = -E
  """ END YOUR CODE
  """
  return E

