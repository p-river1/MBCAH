import numpy as np
import scipy as sp


def to_sparse(mat):
    if not sp.sparse.issparse(mat):
        return sp.sparse.csr_matrix(mat)
    return mat


def to_dense(mat):
    if sp.sparse.issparse(mat):
        return mat.toarray()
    return mat


def check_similarity_matrix(C):
    assert np.all(np.asarray(np.abs(C - C.T)) < 1e-3)
    assert (C[np.diag_indices_from(C)] == 0.).all()


def init_transform(X, C, p):
    """
    Returns the np.array X transformed according
    to a multiplicative initialization technique
    that uses the similarity matrix C. The order p
    can be used to propagate the similarity relationship
    to p-neighborhoods.
    Note that X_transformed is a dense array, such that
    running kmeans or skmeans on it can be inefficient
    in a high-dimensional setting
    """
    # transform similarity matrix C into a stochastic matrix W
    W = np.clip(to_dense(C), 0., None)
    np.fill_diagonal(W, 1.)

    with np.errstate(divide='ignore'):
        W_ = W.sum(1)
        np.divide(W, W_, where=(W_ > 0.), out=W)
    np.nan_to_num(W, copy=False, nan=0., posinf=0., neginf=None)

    if p > 1:
        with np.errstate(under='ignore'):
            W = np.linalg.matrix_power(W, p)

    X_transformed = W @ X
    np.clip(X_transformed, 1e-10, None, out=X_transformed)
    return X_transformed


def build_C_mask(N, frac):
    """
    Returns a random N x N symmetric boolean matrix
    without self loops such that int(frac * N * (N - 1))
    of its elements are equal to True
    Used for sampling a given fraction of all the
    possible constraints
    """
    def i_max(k):
        return (k + 1) * N - int((k + 1) * (k + 2) / 2) - 1
    i_max_ = [i_max(k) for k in range(N)]

    C = np.zeros((N, N), dtype='bool')
    n_pairs = int(N * (N - 1) / 2)
    n_pairs_sampled = int(frac * n_pairs)
    indexes = np.random.choice(n_pairs, size=n_pairs_sampled, replace=False)
    indexes = sorted(indexes)

    # k, l the row and columns indexes corresponding
    # to the edge number 'ind' in C
    # k_0 the current row index, to speed up
    # computations, since 'indexes' is sorted
    k_0 = 0
    for ind in indexes:
        for k in range(k_0, N):
            i_min_k = i_max_[k-1] + 1 if k != 0 else 0
            i_max_k = i_max_[k]
            if (i_min_k <= ind) and (ind <= i_max_k):
                l = ind - i_min_k + (k + 1)
                k_0 = k
                C[k, l], C[l, k] = True, True
                break
    return C


def build_C_strat(Z, frac, frac_noise=0., path_only=False):
    """
    returns a similarity matrix build from the
    true classes Z, by sampling a fraction frac
    of the nodes of each class and adding noise
    to a fraction frac_noise of the nodes
    """
    N = Z.shape[0]
    K = np.unique(Z).size
    C = np.zeros((N, N), dtype='int')
    for k in range(K):
        ind_k = np.where(Z == k)[0]
        nb_nodes = int(frac * ind_k.shape[0])
        ind_sampled = np.random.choice(ind_k, nb_nodes, replace=False)
        if not path_only:
            C[np.ix_(ind_sampled, ind_sampled)] = 1
        else:
            for i_1, i_2 in zip(ind_sampled[:-1], ind_sampled[1:]):
                C[i_1, i_2] = 1
                C[i_2, i_1] = 1
    if frac_noise > 0.:
        nb_nodes_random = int(frac_noise * N)
        ind_rnd = np.random.choice(N, nb_nodes_random, replace=False)
        C[np.ix_(ind_rnd, ind_rnd)] = 1 - C[np.ix_(ind_rnd, ind_rnd)]

    C[np.diag_indices(N)] = 0
    # noinspection PyUnresolvedReferences
    assert ((C - C.T) == 0).all()
    return C


def build_C(Z, frac, frac_noise=None):
    """
    returns a similarity matrix build from the
    true classes Z, by sampling a fraction frac
    of the nodes of each class

    consider using build_C_sparse
    """
    N = Z.shape[0]
    clusters = np.unique(Z)

    # C_Z[i, j] = 1 if i and j are in the same
    # cluster in the partition Z otherwise C_Z[i, j] = -1
    C_Z = np.zeros((N, N), dtype='int')
    for k in clusters:
        ind_k = np.where(Z == k)[0]
        z_k = np.zeros((N, 1), dtype='int')
        z_k[ind_k, :] = True
        C_Z += z_k.dot(z_k.T)

    C_Z[C_Z == 0] = -1
    C_mask = build_C_mask(N, frac)
    C = C_Z * C_mask

    if frac_noise is not None:
        C_mask_noise = build_C_mask(N, frac_noise)
        C = (C * (~C_mask_noise) - C_mask_noise * C)

    C[np.diag_indices(N)] = 0.
    assert ((C - C.T) == 0).all()
    return C


def similarity_concordance(Z, C):
    """
    Counts the proportion of ML or CL constraints
    that are not satisfied in the partition Z.
    If Z is the true partition, the returned value
    represents the concordance between the true classes
    and the given similarity information.
    If Z is the partition returned by the algorithm
    without similarity matrix, the returned value
    represents the information brought by the given
    similarity matrix.
    If Z is the partition returned by the algorithm
    with similarity matrix, the returned value
    represents the proportion of constraints from the
    similarity matrix that are not respected after
    regularisation.
    """
    if (np.abs(C) <= 1e-5).all():
        return 0., 0.

    clusters = np.unique(Z)
    N = Z.shape[0]
    C_Z = np.zeros((N, N), dtype='bool')
    for k in clusters:
        ind_k = np.where(Z == k)[0]
        z_k = np.zeros((N, 1), dtype='bool')
        z_k[ind_k, :] = True
        C_Z += z_k.dot(z_k.T)
    not_C_Z = ~ C_Z

    C_ml, C_cl = C.copy(), C.copy()
    C_ml[C_ml < 0.] = 0.
    C_cl[C_cl > 0.] = 0.
    C_cl = np.abs(C_cl)

    # not weighted
    # here it is only True if the constraint is not
    # respected and False otherwise
    C_contraints = np.logical_or(
        C_ml.astype('bool') * not_C_Z,
        C_cl.astype('bool') * C_Z
    )
    total_weight = C.astype('bool').sum()
    sc = 1. - C_contraints.sum() / total_weight

    # weighted
    # C_contraints[i, i'] is the absolute value of
    # the weight of the constraint between i and i'
    # if the constraint is not respected in Z
    # otherwise it is 0
    C_contraints = (
        C_ml * not_C_Z.astype('float') +
        C_cl * C_Z.astype('float')
    )
    total_weight = np.abs(C).sum()
    scw = 1. - C_contraints.sum() / total_weight
    return sc, scw
