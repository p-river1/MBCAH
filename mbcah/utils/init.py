import numpy as np
import numpy.ma as ma

from sklearn.cluster import KMeans

nax = np.newaxis


def init_VS_XS(X, V, S):
    # Can also be used if S contains negative values
    M = S.size
    N, T = V.shape
    S_pos = S[S >= 0].max()
    S_neg = np.abs(S[S < 0]).max() if (S < 0).any() else 0
    T_max = T + S_neg + S_pos

    X_ = np.asarray(X)

    VS = np.zeros((N, T_max, M), dtype='bool')
    XS = np.full((N, T_max, M), np.nan, dtype=X.dtype)

    for m, sm in enumerate(S):
        start, end = S_neg + sm, S_neg + sm + T
        VS[:, start: end, m] = V[:]
        XS[:, start: end, m] = X_[:]

    # XS masked even if not all_visible
    XS = np.ma.MaskedArray(XS, mask=np.isnan(XS), fill_value=np.nan)
    return VS, XS


def apply_perturbation_init_partition(
    init_partition, K, perturbation_rate, random_state
):
    if perturbation_rate == 0.:
        return init_partition

    res_init_partition = init_partition.copy()
    N = init_partition.shape[0]
    n_curves_shuffled = int(perturbation_rate * N)
    curves_replaced = random_state.choice(
        N, size=n_curves_shuffled, replace=False
    )
    res_init_partition[curves_replaced] = random_state.choice(
        K, size=n_curves_shuffled
    )
    return res_init_partition


def _split_largest_cluster(Z, k_new, random_state):
    clusters, sizes = np.unique(Z, return_counts=True)
    largest_cluster = clusters[np.argmax(sizes)]
    inds = np.where(Z == largest_cluster)[0]
    n_sampled = int(.5 * inds.shape[0])
    sampled_inds = random_state.choice(inds, n_sampled, replace=False)
    Z[sampled_inds] = k_new


def split_clusters_to_K(Z, K, random_state):
    empty_clusters = np.setdiff1d(np.arange(K), np.unique(Z))
    Z_ = Z.copy()
    for k_new in empty_clusters:
        _split_largest_cluster(Z_, k_new, random_state)
    return Z_


def ffill(X, mask=None):
    """
    forward fills missing values along axis 1
    from https://stackoverflow.com/questions/41190852/
    most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    N, T = X.shape
    if mask is None:
        mask = np.isnan(X)
    if not mask.any():
        return X
    idx = np.where(~mask, np.arange(T), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return X[np.arange(N)[:, nax], idx]


def bfill(arr):
    return ffill(arr[:, ::-1])[:, ::-1]


def compute_X_init(X, all_visible):
    if all_visible:
        return X
    else:
        assert isinstance(X, ma.MaskedArray)
    mask = X.mask
    X_ = X.filled(np.nan)
    return bfill(ffill(X_, mask))


def random_init(N, K, random_state):
    return random_state.choice(K, size=N)


def kmeans_init(X, K, n_init, random_state):
    # by default uses all cores
    # can be limited by using
    # from threadpoolctl import threadpool_limits
    # with threadpool_limits(user_api="openmp", limits=2):
    #     kmeans.fit(X)
    seed = random_state.choice(np.iinfo(np.int32).max)
    clustering = KMeans(
        n_clusters=K,
        algorithm='elkan',
        random_state=seed,
        n_init=n_init
    )
    # noinspection PyUnresolvedReferences
    return clustering.fit(X).labels_


def init_lam(
    K, M,
    current_init_partition,
    global_init_tau,
    proba_init,
    em_type, random_state, dtype
):
    N = current_init_partition.shape[0]
    lam = np.zeros((N, K, M), dtype=dtype)
    init_distrib_tau = random_state.multinomial(
        np.ones(N, dtype='int'), global_init_tau
    )
    if em_type in ['CEM', 'SEM']:
        init_distrib_tau = init_distrib_tau.astype('bool')
    else:
        init_distrib_tau = proba_init * init_distrib_tau + (1. - proba_init) / M
    lam[np.arange(N), current_init_partition, :] = init_distrib_tau
    return lam


def coarsen_S(S, f):
    S_coarsened = np.array([
        s for i, s in enumerate(S) if i % f == 0
    ])
    q, r = divmod(S.size, f)
    n_groups = q + int(r > 0)
    mapping = {
        m: tuple(f * m + p for p in range(f) if f * m + p < S.size)
        for m in range(n_groups)
    }
    return S_coarsened, mapping

