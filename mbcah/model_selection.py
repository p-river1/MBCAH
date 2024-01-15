import numpy as np
import pandas as pd

from .utils import m_step, init, e_step

nax = np.newaxis


def _is_pareto_efficient(crits):
    """
    crits of shape (n_points, n_crits)
    a point i is pareto inefficient if there exists i'
    such that (crits[i'] >= crits[i]).all()
    More precisely, the inegality has to be strict for
    at least one criterion. For this reason, we first
    consider the set of distinct points
    """
    is_efficient = np.zeros(crits.shape[0], dtype='bool')
    unique_crits, indexes_unique_crits = np.unique(crits, axis=0, return_index=True)
    for i_unique, (i, crit) in enumerate(zip(indexes_unique_crits, unique_crits)):
        crits_minus_i = np.delete(unique_crits, i_unique, axis=0)
        # noinspection PyUnresolvedReferences
        is_efficient[i] = not (crits_minus_i >= crit[nax, :]).all(1).any()
    return is_efficient


def get_metrics_df(mrs):
    normalized_potentials = [
        mr.potentials_at_convergence / mr.eta
        if mr.semi_supervised and mr.eta > 0.
        else 0.
        for mr in mrs
    ]
    mr_ids, lams, Ks, bics, likelihoods, norm_potentials, sc, scw = list(zip(*[
        (
            mr.mr_id, mr.eta, mr.K,
            mr.bic, mr.likelihood,
            norm_potential,
            mr.similarity_concordance_unweighted,
            mr.similarity_concordance_weighted
        )
        for mr, norm_potential in zip(mrs, normalized_potentials)
    ]))
    df_res = pd.DataFrame(
        dict(
            mr_id=mr_ids,
            eta=lams,
            K=Ks,
            bic=bics,
            likelihood=likelihoods,
            norm_potentials=norm_potentials,
            sc=sc,
            scw=scw
        )
    )
    return df_res


def assign_pareto_efficient(df_res, crit1, crit2):
    def _merge_crits(df, crts, signs):
        return np.concatenate(tuple(
            sign * df[crt].values[:, nax]
            for crt, sign in zip(crts, signs)
        ), axis=1)
    X_crits = _merge_crits(df_res, (crit1, crit2), (1, 1))
    df_res = df_res.assign(pareto_efficient=_is_pareto_efficient(X_crits))

    X_crits = _merge_crits(df_res, (crit1, crit2, 'K'), (1, 1, -1))
    df_res = df_res.assign(K_pareto_efficient=_is_pareto_efficient(X_crits))
    return df_res


def heuristic_model_selection_on_pareto_front(
    X_crits_eff, method, pareto_index_chosen=None
):
    """
    given an array of pareto efficient criterions of
    shape (n_models, n_criterions), return the index
    of a model chosen to be the best according to one
    of the three heuristics:
        - farthest: the farthest point from the origin
          in euclidean distance with min-max scaling
          of the criterions
        - handpicked: the given index
    """
    from sklearn.preprocessing import MinMaxScaler
    if method == 'farthest':
        X_norm = MinMaxScaler().fit_transform(X_crits_eff)
        farthest_point = (X_norm ** 2).sum(1).argmax()
        return farthest_point
    elif method == 'handpicked':
        assert pareto_index_chosen is not None
        return pareto_index_chosen
    else:
        raise ValueError


def compute_variance_reduction_factors(
    Z, K, tau_inds,
    X, S,
    constrained_sigma2, sigma2_zero, nu_zero,
    min_den_mu, min_float
):
    if isinstance(X, np.ma.MaskedArray):
        all_visible = False
    else:
        mask = np.isnan(X)
        all_visible = not mask.any()
        X = (
            X if all_visible
            else
            np.ma.masked_array(X, mask=mask, fill_value=np.nan)
        )
    if all_visible:
        V = np.ones_like(X).astype('bool')
    else:
        V = ~ X.mask

    M = S.size
    N, T = X.shape
    S_max = S.max(initial=0)
    T_max = T + S_max
    VS, XS = init.init_VS_XS(X, V, S)

    # construct etas using Z, with and without shift
    lam_cem_without_shifts = np.zeros((N, K, M), dtype='bool')
    lam_cem_without_shifts[np.arange(N), Z, 0] = True

    lam_cem_with_shifts = np.zeros((N, K, M), dtype='bool')
    for i in range(N):
        lam_cem_with_shifts[i, Z[i], tau_inds[i]] = True

    # estimate sigma2 without shift
    mu_without_shifts = np.zeros((T_max, K), dtype='float')
    den, where_den = m_step.update_mu(
        mu_without_shifts, lam_cem_without_shifts, XS, VS, min_den_mu
    )
    residuals_without_shifts = m_step.compute_residuals(XS, mu_without_shifts)
    sigma2_without_shifts = np.zeros_like(mu_without_shifts)
    m_step.update_sigma2(
        sigma2_without_shifts,
        constrained_sigma2,
        residuals_without_shifts, lam_cem_without_shifts,
        den, where_den,
        sigma2_zero, nu_zero,
        min_float
    )
    # estimate sigma2 with shift
    mu_with_shifts = np.zeros_like(mu_without_shifts)
    den, where_den = m_step.update_mu(
        mu_with_shifts, lam_cem_with_shifts, XS, VS, min_den_mu
    )
    residuals_with_shifts = m_step.compute_residuals(XS, mu_with_shifts)
    sigma2_with_shifts = np.zeros_like(mu_with_shifts)
    m_step.update_sigma2(
        sigma2_with_shifts,
        constrained_sigma2,
        residuals_with_shifts, lam_cem_with_shifts,
        den, where_den,
        sigma2_zero, nu_zero,
        min_float
    )
    # compute variance reduction factor
    num_rf = (
        lam_cem_without_shifts[:, :, 0] *
        (VS[:, :, nax, 0] * sigma2_without_shifts[nax, :, :]).sum(1)
    ).sum(0)
    den_rf = (
        lam_cem_with_shifts *
        (VS[:, :, nax, :] * sigma2_with_shifts[nax, :, :, nax]).sum(1)
    ).sum((0, 2))

    z = e_step.encode_Z(Z, K)
    n_visible_k = z.sum(0)
    reduction_factors = np.nan * np.zeros(K)
    np.divide(num_rf, den_rf, out=reduction_factors, where=(n_visible_k >= 2))
    return reduction_factors
