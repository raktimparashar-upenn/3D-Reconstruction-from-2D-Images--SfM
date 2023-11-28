from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        samp_1 = X1[sample_indices]
        samp_2 = X2[sample_indices]
        inliers = sample_indices
        
        E = least_squares_estimation(samp_1, samp_2)

        def skew(vector):
            """
            this function returns a numpy array with the skew symmetric cross product matrix for vector.
            the skew symmetric cross product matrix is defined such that
            np.cross(a, b) = np.dot(skew(a), b)

            :param vector: An array like vector to create the skew symmetric cross product matrix for
            :return: A numpy array of the skew symmetric cross product vector
            """

            return np.array([[0, -vector[2], vector[1]], 
                             [vector[2], 0, -vector[0]], 
                             [-vector[1], vector[0], 0]])
    
        v = np.array([0, 0, 1])
        e3 = skew(v)

        for i in test_indices:
            d_x1 = np.square(X1[i].reshape((1, 3)) @ (E.T @ X2[i].reshape((3, 1))))/np.square(np.linalg.norm(e3 @ E.T @ X2[i].reshape((3, 1))))
            d_x2 = np.square(X2[i].reshape((1, 3)) @ (E @ X1[i].reshape((3, 1))))/np.square(np.linalg.norm(e3 @ E @ X1[i].reshape((3, 1))))

            total_dist = d_x1 + d_x2
            
            if total_dist[0] < eps:
                inliers = np.hstack((inliers, i))

        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers