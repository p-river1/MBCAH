import numpy as np
from scipy.stats import dirichlet, invwishart

nax = np.newaxis


def update_mu(mu, lam, XS, VS, min_den):
    with np.errstate(under='ignore'):
        num_mu = np.asarray((lam[:, nax, :, :] * XS[:, :, nax, :]).sum((0, 3)))
        den = (lam[:, nax, :, :] * VS[:, :, nax, :]).sum((0, 3))
        # den[j, k] is the weighted number of curves in cluster k
        # whose shifted timestep is visible at timestep j
        where_den = (den > min_den)
        np.divide(num_mu, den, where=where_den, out=mu)
    return den, where_den


def get_sigma2_zero(K, T_max, X, map_estimate_sigma2):
    # spherical covariance matrix used as an inverse wishart
    # prior on sigma2
    # see 10.1007/s00357-007-0004-z for the value of sigma2_zero
    with np.errstate(under='ignore'):
        sigma2_zero = (
            K ** (-2. / T_max) *
            np.clip(X.var(0), 1., None).mean()
        )
    sigma2_zero = (
        sigma2_zero if map_estimate_sigma2 else 0.
    )
    return sigma2_zero


def update_sigma2(
    sigma2,
    constrained_sigma2,
    residuals, lam,
    den, where_den,
    sigma2_zero, nu_zero,
    min_float
):
    # if map_estimate_sigma2 is False, then
    # nu_zero = - (2 + T_max) + min_float
    # and sigma2_zero = 0
    # so that sigma2 is estimated using maximum likelihood
    if constrained_sigma2 == 'sigma_k':
        T_max, K = sigma2.shape
        with np.errstate(under='ignore'):
            num_sigma2 = (
                sigma2_zero +
                np.asarray((lam[:, nax, :, :] * residuals).sum((0, 1, 3)))
            )
            den_sigma2 = nu_zero + 2 + T_max + den.sum(0)
            untiled_sigma2 = num_sigma2 / den_sigma2
        sigma2[:] = np.tile(
            untiled_sigma2[nax, :], (T_max, 1)
        )
    elif constrained_sigma2 == 'sigma_jk':
        with np.errstate(under='ignore'):
            num_sigma2 = (
                sigma2_zero +
                np.asarray((lam[:, nax, :, :] * residuals).sum((0, 3)))
            )
            # sets sigma[j, k] = min_float for time steps
            # j such that there is no shifted curve present
            # sigma2[:] = min_float  # seems to have strange interactions with damping
            np.divide(num_sigma2, nu_zero + 2 + den, where=where_den, out=sigma2)

    # sets clusters with 0 residuals to sigma2 = min_float
    # for numerical stability
    np.clip(sigma2, min_float, None, out=sigma2)


def compute_residuals(XS, mu):
    with np.errstate(under='ignore'):
        return (XS[:, :, nax, :] - mu[nax, :, :, nax]) ** 2.


def compute_gamma(lam, constrained_gamma, beta_zero, dtype):
    N, K, M = lam.shape
    if constrained_gamma == 'gamma':
        return np.full((K, M), 1. / M, dtype=dtype)

    if constrained_gamma == 'gamma_km':
        counts = lam.sum(0).astype(dtype)
    elif constrained_gamma == 'gamma_m':
        counts = lam.sum(0).astype(dtype).sum(0, keepdims=True)
        counts = np.tile(counts, (K, 1))
    else:
        raise ValueError

    gamma = (counts + beta_zero - 1.)
    gamma = gamma / gamma.sum(1, keepdims=True)
    return gamma


def compute_rho(z, alpha_zero, dtype):
    counts = z.sum(0).astype(dtype)
    rho = (counts + alpha_zero - 1.)
    rho = rho / rho.sum()
    return rho


def compute_log_prior(
    rho, gamma, sigma2,
    map_estimate_sigma2, constrained_gamma, constrained_sigma2,
    alpha_zero, beta_zero, nu_zero, sigma2_zero
):
    T_max, K = sigma2.shape
    log_prior = dirichlet.logpdf(rho, alpha_zero * np.ones_like(rho))

    beta0 = beta_zero * np.ones_like(gamma[0])
    if constrained_gamma == 'gamma_km':
        for k in range(K):
            log_prior += dirichlet.logpdf(gamma[k], beta0)
    elif constrained_gamma == 'gamma_m':
        log_prior += dirichlet.logpdf(gamma[0], beta0)
    elif constrained_gamma == 'gamma':
        pass

    if map_estimate_sigma2:
        if constrained_sigma2 == 'sigma_k':
            for k in range(K):
                log_prior += invwishart.logpdf(sigma2[0, k], nu_zero, sigma2_zero)
        elif constrained_sigma2 == 'sigma_jk':
            sigma20 = np.identity(T_max) * sigma2_zero
            for k in range(K):
                log_prior += invwishart.logpdf(sigma2[:, k], nu_zero, sigma20)
    return log_prior
