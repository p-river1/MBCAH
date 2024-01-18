import numpy as np
import scipy as sp

from .utils import init


class Sampler:

    def __init__(
        self,
        N, T, K, S,
        alpha_rho,
        alpha_gamma,
        constrained_sigma2,
        sigma2_gp_mu,
        sigma2_gp_sigma2,
        s2,
        n_intervals,
        p_missing,
        random_state,
        dtype
    ):
        if S is not None:
            assert S.ndim == 1 and S.size > 0 and (S >= 0.).all() and S.dtype == int
            self.S = S
            self.M = self.S.size
            self.S_max = self.S.max()
        else:
            self.S = None
            self.M = 0
            self.S_max = 0

        self.T = T
        self.T_max = self.T + self.S_max
        self.N = N
        self.K = K

        self.alpha_rho = alpha_rho
        self.alpha_gamma = alpha_gamma
        self.sigma2_gp_mu = sigma2_gp_mu
        self.sigma2_gp_sigma2 = sigma2_gp_sigma2
        self.s2 = s2
        self.constrained_sigma2 = constrained_sigma2
        self.sigma2_zero = 0.

        self.n_intervals = n_intervals
        self.p_missing = p_missing

        self.random_state = random_state
        self.dtype = dtype

        self.min_den = .1
        self.min_float = 1e-30

        self.rho = None
        self.gamma = None
        self.mu = None
        self.sigma2 = None

        self.Z = None
        self.z = None
        self.tau_inds = None
        self.tau = None
        self.VS_tau = None

        self.X = None
        self.mask = None

    def sample(self):
        self.rho = dirichlet(self.alpha_rho, 1, self.K, self.random_state)[0]
        self.gamma = dirichlet(self.alpha_gamma, self.K, self.M, self.random_state)

        time_grid = np.linspace(0., 1., self.T_max)[:, np.newaxis]
        with np.errstate(under='ignore'):
            grid_cdist = sp.spatial.distance.cdist(time_grid, time_grid, 'sqeuclidean')
        zero_mean = np.zeros(self.T_max)

        with np.errstate(under='ignore'):
            cov_mu = np.exp(- grid_cdist / (2. * self.sigma2_gp_mu))
            self.mu = self.random_state.multivariate_normal(
                mean=zero_mean, cov=cov_mu, size=self.K
            ).T.astype(self.dtype)
            cov_sigma2 = np.exp(- grid_cdist / (2. * self.sigma2_gp_sigma2))

            self.sigma2 = np.tile(self.s2[np.newaxis, :], (self.T_max, 1))
            if self.constrained_sigma2 == 'sigma_jk':
                self.sigma2 = self.sigma2 + self.random_state.multivariate_normal(
                    mean=zero_mean, cov=cov_sigma2, size=self.K
                ).T.astype(self.dtype) ** 2.

        self.Z = self.random_state.choice(self.K, size=self.N, p=self.rho)
        self.z = np.zeros((self.N, self.K), dtype='bool')
        self.z[np.arange(self.N), self.Z] = True

        self.tau_inds = np.zeros(self.N, dtype='int')
        self.tau = np.zeros(self.N, dtype='int')
        self.X = np.zeros((self.N, self.T), dtype=self.dtype)

        sigma = np.sqrt(self.sigma2)
        for k in range(self.K):
            Zk = (self.Z == k)
            self.tau_inds[Zk] = self.random_state.choice(
                self.M, size=Zk.sum(), p=self.gamma[k]
            )
        self.tau = self.S[self.tau_inds]

        for k in range(self.K):
            Zk = (self.Z == k)
            for m, sm in enumerate(self.S):
                inds_km = Zk & (self.tau == sm)
                n_km = inds_km.sum()
                if n_km > 0:
                    self.X[inds_km] = self.random_state.normal(
                        loc=self.mu[sm: self.T + sm, k],
                        scale=sigma[sm: self.T + sm, k],
                        size=(n_km, self.T)
                    )

        self.mask = sample_intervals_mask(
            X=self.X,
            n_intervals=self.n_intervals,
            p_missing=self.p_missing,
            random_state=self.random_state
        )
        self.X[self.mask] = np.nan

        VS, XS = init.init_VS_XS(
            self.X, self.mask, self.S
        )
        self.VS_tau = VS[np.arange(self.N), :, self.tau_inds]


def dirichlet(alpha, n_samples, dim, random_state):
    return sp.stats.dirichlet.rvs(
        np.full(dim, alpha),
        size=n_samples,
        random_state=random_state
    )


def sample_intervals_mask(X, n_intervals, p_missing, random_state):
    N, T = X.shape
    missing_intervals_ids = random_state.choice(
        [False, True], size=(N, n_intervals), p=[1. - p_missing, p_missing]
    )
    interval_grid = np.sort(
        random_state.choice(n_intervals, size=(N, T)),
        axis=1
    )
    mask = np.zeros((N, T), dtype='bool')
    for interval in range(n_intervals):
        selected_i_inds = np.where(missing_intervals_ids[:, interval])[0]
        i_inds, t_inds = np.where(interval_grid == interval)
        selected = [(t, i) for (t, i) in zip(t_inds, i_inds) if i in selected_i_inds]
        if len(selected) > 0:
            grid_t_inds, grid_i_inds = tuple(zip(*selected))
            grid_t_inds = np.array(grid_t_inds)
            grid_i_inds = np.array(grid_i_inds)
            mask[grid_i_inds, grid_t_inds] = True
    return mask






