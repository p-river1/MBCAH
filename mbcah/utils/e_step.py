import numpy as np
import scipy as sp

from . import similarity_matrices

nax = np.newaxis


def encode_Z(Z, K):
    N = Z.shape[0]
    z = np.zeros((N, K), dtype='bool')
    arr = np.arange(N)
    z[arr, Z] = True
    return z


def converged(old_criterion, criterion, atol, rtol):
    if np.isinf(old_criterion):
        return 0
    return int(
        np.abs(criterion - old_criterion) <=
        (atol + rtol * np.abs(old_criterion))
    )


def compute_log_pointwise_densities(residuals, sigma2):
    # log p(x_ij| tau_i = m, z_i = k; theta)
    # masked
    with np.errstate(under='ignore'):
        sigma2_ = sigma2[nax, :, :, nax]
        lpd = - .5 * (
            residuals / sigma2_ + np.log(sigma2_)
        )
    return lpd


def compute_log_mixture_dens(rho, gamma, lpd):
    # log rho_k + log gamma_km + log p(x_i| tau_i = m, z_i = k; theta)
    return (
        np.log(rho)[nax, :, nax] +
        np.log(gamma)[nax, :, :] +
        np.asarray(lpd.sum(1))
    )


def fixed_part_unnormalized_posterior_lam(lpd1, ss_mode, rho, gamma, not_P):
    with np.errstate(under='ignore'):
        if ss_mode == 'mixture':
            log_prop_z = not_P[:, nax] * np.log(rho)[nax, :]
        else:
            log_prop_z = np.log(rho)[nax, :]
    ulp = (
        log_prop_z[:, :, nax] +
        np.log(gamma)[nax, :, :] +
        np.asarray(lpd1)
    )
    return ulp


def unnormalized_posterior_lam(lpd1, ss_mode, rho, gamma, z, scaled_C, P, not_P):
    with np.errstate(under='ignore'):
        scaled_C_z = similarity_matrices.to_dense(scaled_C @ z)
        log_prop_z = (
            not_P[:, nax] * np.log(rho)[nax, :] + P[:, nax] * scaled_C_z
            if ss_mode == 'mixture'
            else
            np.log(rho)[nax, :] + scaled_C_z
        )
    ulp = (
        log_prop_z[:, :, nax] +
        np.log(gamma)[nax, :, :] +
        np.asarray(lpd1)
    )
    return ulp, scaled_C_z


def set_scaled_C_z_i(scaled_C_i, z):
    with np.errstate(under='ignore'):
        return (
            (scaled_C_i @ z)[0]
            if sp.sparse.issparse(scaled_C_i)
            else
            z.T @ scaled_C_i
        )


def compute_elbo(
    lpd1, scaled_C_z,
    lam, z, rho, gamma,
    em_type, ss_mode,
    scaled_C, not_P, min_float
):
    with np.errstate(under='ignore'):
        if scaled_C_z is None:
            # scaled_C_z is None in the sequential case
            potentials = (
                .5 * scaled_C.multiply(z @ z.T).sum()
                if sp.sparse.issparse(scaled_C)
                else
                .5 * (scaled_C * (z @ z.T)).sum()
            )
        else:
            potentials = .5 * (scaled_C_z * z).sum()
        mixture = z.sum(
            axis=0,
            where=not_P[:, nax] if ss_mode == 'mixture' else True
        ).dot(np.log(rho))
        dens = (lam * (lpd1 + np.log(gamma)[nax, :, :])).sum()
        elbo = potentials + mixture + dens
        if em_type == 'VEM':
            elbo += - (lam * np.log(lam + min_float)).sum()
    return elbo, potentials


def vem_lam(unnormalized_posterior, *_args):
    with np.errstate(under='ignore'):
        lam = np.exp(
            unnormalized_posterior -
            unnormalized_posterior.max(axis=(-2, -1), keepdims=True)
        )
        lam = lam / lam.sum((-2, -1), keepdims=True)
    return lam


def apply_damping(lam_old, lam_new, damping_factor):
    if damping_factor is None:
        return lam_new
    with np.errstate(under='ignore'):
        return damping_factor * lam_new + (1. - damping_factor) * lam_old


def em_lam(log_mixture_dens, _rng, likelihoods):
    with np.errstate(under='ignore'):
        return np.exp(log_mixture_dens - likelihoods[:, nax, nax])


def sem_lam(lam, rng, *_args):
    lam = vem_lam(lam)
    if lam.ndim == 3:
        N, K, M = lam.shape
        z = lam.reshape((N, K * M))
        z = sample_categorical(z, rng)
        lam = z.reshape((N, K, M))
    elif lam.ndim == 2:
        K, M = lam.shape
        z = lam.reshape(K * M)
        z = sample_categorical(z, rng)
        lam = z.reshape((K, M))
    return lam


def cem_lam(lam, *_args):
    if lam.ndim == 3:
        N, K, M = lam.shape
        z = lam.reshape((N, K * M))
        inds = z.argmax(1)
        z[:, :] = False
        z[np.arange(N), inds] = True
        lam = z.reshape((N, K, M))
    elif lam.ndim == 2:
        K, M = lam.shape
        z = lam.reshape(K * M)
        km = z.argmax()
        z[:] = False
        z[km] = True
        lam = z.reshape((K, M))
    return lam


def update_z(lam, em_type):
    if em_type in ['CEM', 'SEM']:
        return lam.any(-1)
    return lam.sum(-1)


def sample_categorical(p, rng):
    """
    from https://stackoverflow.com/questions/55818845/fast-vectorized-multinomial-in-python
    Sample from the categorical distribution with multidimensional p
    """
    count = np.ones(p.shape[:-1], dtype='int')
    out = np.zeros(p.shape, dtype='bool')
    with np.errstate(under='ignore', divide='ignore', invalid='ignore'):
        condp = p / p.cumsum(axis=-1)  # conditional probabilities
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        bernouilli_sample = np.asarray(rng.binomial(count, condp[..., i])).astype('bool')
        out[..., i] = bernouilli_sample
        count = count - bernouilli_sample
    out[..., 0] = count.astype('bool')
    return out
