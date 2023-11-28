import numpy as np

def pose_candidates_from_E(E):
    transform_candidates = []
    # Note: each candidate in the above list should be a dictionary with keys "T", "R"
    """ YOUR CODE HERE
    """
    u , s, vt = np.linalg.svd(E)
    R_pos = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    R_neg = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
    # sol 1
    rot1 = u @ R_pos.T @ vt
    # print(rot1.shape)
    tran1 = u[:, 2]
    # print(tran1)
    dict1 = {
      'R': rot1,
      'T': tran1
    }

    # sol 2
    rot2 = u @ R_neg.T @ vt
    # print(rot2.shape)
    tran2 = u[:, 2]
    dict2 = {
      'R': rot2,
      'T': tran2
    }

    # sol 3
    rot3 = u @ R_pos.T @ vt
    tran3 = -u[:, 2]

    dict3 = {
      'R': rot3,
      'T': tran3
    }

    # sol 4
    rot4 = u @ R_neg.T @ vt
    tran4 = -u[:, 2]

    dict4 = {
      'R': rot4,
      'T': tran4
    }

    transform_candidates = [dict1, dict2, dict3, dict4]

    """ END YOUR CODE
    """
    return transform_candidates