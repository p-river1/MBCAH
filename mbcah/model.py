"""

# model structure
    - K: number of clusters
    - constrained_sigma2: in ['sigma_jk', 'sigma_k'] : how the variance is constrained
    - constrained_gamma: in ['gamma', 'gamma_m', 'gamma_km'] : how the shift mixture
      proportions are constrained
    - S: ndarray of possible time shifts (ints), S[m] of size M
    - S_max: maximum number of time steps allowed for a shift, computed from S

# indexes
    - i, i_: in [0, ..., N-1], the index of the curves
    - j: in [0, ..., T - 1] or in [0, ..., T_max - 1], the index of time
    - k: in [0, ..., K - 1] the index of the clusters
    - m: in [0, ..., M - 1] the index of the shifts

# data
    - X: ndarray of shape N, T that represents the N curves at their observed time steps
    - T: number of time steps, indexed by j
    - N: number of curves, indexed by i
    - T_max: size of the prototype _mu[:, k], accounting for the maximal shift S_max
    - V: V[i, j] is True if observation i is visible at time step j
    - XS: shifted observed values
      XS[i, j, m] = X[i, j - S[m]] if j - S[m] >= 0 else 0.
    - VS: shifted indicator of visible values
      VS[i, j, m] = V[i, j - S[m]] if j - S[m] >= 0 else False

# semi supervision
    - semi_supervised: in [False, True]: if we consider semi-supervision
    - C: the constraint matrix of shape (N, N), which could be a ndaray or a
      scipy.sparse.csr matrix, which takes values:
        - C[i, i'] > 0. if we want to give a higher probability for observation i and
          observation i' to be in the same cluster
        - C[i, i'] > 0. if we want to give a higher probability for observation i and
          observation i' to be in different clusters
        - C[i, i'] = 0. if we have no prior knowledge about the relationship between
          observation i and observation i'
    - eta: the regularization parameter that scales the constraint matrix C
      and gives more or less weight to the prior knowledge relative to the
      information in the data
    - scaled_C : eta * C
    - ss_mode: in [mixture, all]: in 'mixture' there are no node _potentials
      in the HMRF, in 'all' all nodes have mixture proportions defined with node _potentials
    - damping_factor: in (0., 1.): the damping level in VEM with parallel updates
    - n_loops_e_step_max: the number of times the variational posterior probabilities of all
      nodes are updated if there is no convergence in terms of variational params
      if self.n_loops_e_step_max < 1. we apply a e step only for a fraction of the
      nodes of the hmrf consequently, the elbo can take the same value twice without
      converging. We consider that the elbo has to remain stable for
      int(1. / self.n_loops_e_step_max) iteration for the EM algorithm
      to reach convergence
    - _P: the set of node indexes for which we have semi-supervision
    - _not_P: the set of node indexes for which we dont't have semi-supervision

# EM algorithm
There are 4 possible EM algorithms : regular EM algorithm (only if there is no semi supervision),
Variational EM (VEM) (in semi-supervised mode only), Classification EM (CEM) and Stochastic
EM (CEM). If there is semi-supervision, the E step can be done in parallel or in sequence
    - em_type: in ['EM', 'VEM', 'CEM', 'SEM']: the EM algorithm to use.
    - n_init_em: total number of initializations of the model
    - max_iter_em: maximum number of iterations of the (V-C-S)EM algorithm
    - full_sequential: whether all the E steps are done in
      sequential (useful only if semi_supervised is True)
    - atol_iter_em: float to monitor the convergence of the criterion (likelihood or elbo)
    - rtol_iter_em: float to monitor the convergence of the criterion (likelihood or elbo)
    - min_float: float for numerical stability
    - random_state: random state for initialization and SEM
    - beta_zero: symmetric dirichlet prior for the proportions, to avoid empty shift
    - alpha_zero: symmetric dirichlet prior for the proportions, to avoid empty clusters
    - nu_zero: degrees of freedom of inverse wishart prior on sigma2;
      -> set to None for ML estimation
    - dtype: float32 or float64, dtype of float params of the model
    - min_den: .1, to avoid zero division in update_mu
    - _sequential: whether the current E step updates are done in sequence
    - _e_step_lam_fun: the E step function associated to em_type

# initialization
    - algo_init_partition: algo used to compute the initial partition,
      in ['given', 'random', 'kmeans']
    - init_partition_with_C: in semi-supervised mode, whether we transform the data matrix X
      in order to take into account the similarities in C
    - algo_init_tau: in [None, 'naive', 'sim'], the algorithm used to initialize the latent
      variable global_init_tau[i, m] in [0., 1.] if global_init_tau != 'CEM'
      and in [False, True] otherwise
    - coarsening_factor: integer >= 1 used in the initialization of global_init_tau
      such that larger coarsening_factor implies faster but less optimal initialization.
      A medium value of coarsening_factor allows to better explore the parameter space
    - n_init_km: the number of initializations of the k-means algorithm used for the initialization
      of global_init_partition in case algo_init_partition == 'kmeans'
    - cluster_perturbation_rate: a float in [0., 1.] that corresponds to the fraction of
      the N observations that are randomly reassigned to a random cluster at each new
      initialization of the algorithm. A large value of cluster_perturbation_rate
      allows to better explore the parameter space
    - proba_init: the probability assigned to the cluster and the shift obtained at initialization
      when setting _lam at initialization, to be set according to the level of confidence in the
      initialization
    - global_init_partition: the main partition of the curves computed (or given) at initialization,
      from which will be derived the _current_init_partition at each initialization, by assigning
      a random set of curves to random clusters
    - _current_init_partition: the current partition used for the initialization of _lam
    - given_Z: ndarray of ints of size N that encodes a partition of the N observations
      into K' <= K clusters, that is used for the initialization of the gloabl_init_partition
      in case algo_init_partition == 'given'
    - given_tau_inds: the index of the shifts (in S) for each curve, that can be given
      at initialization. It can be used to initialize global_init_tau

# model parameters
Note that model parameters are private attributes, that should be accesses through self.all_params
which is a list of dictionnaries, one per initialization of the model (i.e. n_init_em in total),
that contains a dictionnary of estimated parameters and infered latent variables for each
initialization. However, in the ModelResult class, the parameters can be accessed directly.
    - _rho: mixture proportions of each cluster, _rho[k] of size K
    - _gamma: mixture proportions of each possible shift in each cluster, _gamma[k, m] of
       shape K, M. Could be constrained, see constrained_gamma
    - _mu: mean of each cluster, taking the shifts into account, _mu[j, k] of shape T_max, K
    - _sigma2: variance of each cluster, taking the shifts into account,
      _sigma2[j, k] of shape T_max, K. Could be constrained, see constrained_sigma2
    - _residuals: the squared distances to the current prototypes mu

# latent variables
    - _z: cluster probability or encoded curve clusters, _z[i, k] of size N, K
    - _lam: joint posterior _lam[i, k, m] of size N, K, M
    - _lam_old: previous value of _lam used to monitor convergence of the E step in
      semi-supervised mode
    - tau_inds: the index of the shift of curve i tau_inds[i] = m in {1, ..., M}
    - tau: the value of the shift, tau[i] = S[m] in {s_1, ..., s_m}
    - VS_tau: VS_tau[i, j] = True if curve i shifted of tau[i] is visible at timestep j
      only used for plotting

# misc / debug
    - sparse_thr_C: the threshold between 0. and 1. for the sparsity of the matrix C
      under which it is converted to a scipy.sparse.csr matrix, otherwise it is converted
      to a dense ndarray
    - debug: bool, whether the debug mode is used, where the criterion of the model is computed after
      each update in the M-step (ECM actually), and the estimated parameters at each iteration are
      stored in debug_list
    - verbose: int, in {0, 1, 2}, the verbosity level
    - fitted: bool that indicates whether the model is fitted
    - ordered_inits: order of the different initializations of the model, sorted by decreasing
      value of the criterion returned at convergence
    - all_params: a list of dictionnaries, one per initialization of the model (i.e. n_init_em in total),
      that contains a dictionnary of estimated parameters and infered latent variables for each
      initialization
    - all_iter_crits: the list of dictionnary of criterions at convergence of each initialization
    - all_bics, all_likelihoods: list of BIC criterions and likelihood criterions obtained at convergence,
      one per initialization of the model
    - _crit: the current value of the criterion of the model, which could be the ELBO in VEM,
      the likelihood in EM or CEM
    - _old_crit: the previous value of the criterion of the model
"""
import numpy as np
import numpy.ma as ma
import scipy as sp

from .utils import init, e_step, m_step, similarity_matrices

nax = np.newaxis


class MBCAH:
    """
    """
    def __init__(
        self,
        K,
        ss_mode='mixture',
        damping_factor=.9,
        full_sequential=False,
        constrained_gamma='gamma_km',
        constrained_sigma2='sigma_jk',
        alpha_zero=1.1,
        beta_zero=1.1,
        nu_zero=None,
        algo_init_partition='kmeans',
        init_partition_with_C=False,
        algo_init_tau='sim',
        coarsening_factor=3,
        cluster_perturbation_rate=.3,
        proba_init=.7,
        n_init_km=10,
        em_type='VEM',
        n_loops_e_step_max=1,
        n_init_em=2,
        atol_iter_em=1e-8,
        rtol_iter_em=1e-5,
        max_iter_em=50,
        min_float=1e-30,
        random_state=None,
        verbose=0,
        dtype='float64',
        debug=False
    ):
        # model structure
        self.K = K
        self.constrained_sigma2 = constrained_sigma2
        self.constrained_gamma = constrained_gamma

        # priors
        self.beta_zero = beta_zero
        self.alpha_zero = alpha_zero
        self.nu_zero = nu_zero
        self.map_estimate_sigma2 = self.nu_zero is not None

        # semi supervision
        self.semi_supervised = None  # set when eta and C are given
        self.ss_mode = ss_mode
        self.damping_factor = damping_factor
        self.eta = None
        self.scaled_C = None
        self._P = None
        self._not_P = None

        # EM algorithm
        self.em_type = em_type
        self.n_loops_e_step_max = n_loops_e_step_max
        self.n_init_em = n_init_em
        self.full_sequential = full_sequential
        self.atol_iter_em = atol_iter_em
        self.rtol_iter_em = rtol_iter_em
        self.max_iter_em = max_iter_em
        self.min_float = min_float
        self.random_state = np.random.default_rng(random_state)
        self.dtype = dtype
        self.min_den = .1
        self._sequential = full_sequential
        self._e_step_lam_fun = None

        # init
        self.algo_init_partition = algo_init_partition
        self.init_partition_with_C = init_partition_with_C
        self.algo_init_tau = algo_init_tau
        self.coarsening_factor = coarsening_factor
        self.n_init_km = n_init_km
        self.cluster_perturbation_rate = cluster_perturbation_rate
        self.proba_init = proba_init
        self.global_init_partition = None
        self._current_init_partition = None

        # data
        self.X = None
        self.T = None
        self.N = None
        self.S = None
        self.S_max = None
        self.T_max = None
        self.V = None
        self.XS = None
        self.VS = None

        # model parameters
        self._rho = None
        self._gamma = None
        self._mu = None
        self._sigma2 = None
        self._residuals = None

        # latent variables
        self._z = None
        self._lam = None
        self._lam_old = None

        # misc / debug
        self.sparse_thr_C = .3
        self.debug = debug
        self.verbose = verbose
        self.fitted = False
        self.ordered_inits = None
        self.all_params = []
        self.all_bics = []
        self.all_likelihoods = []
        self._debug_crit = - np.inf
        self.debug_list = []
        self.all_iter_crits = []
        self._old_crit = None
        self._crit = None

    def fit(
        self,
        X, S,
        given_Z=None, given_tau_inds=None,
        eta=0., C=None
    ):
        self._set_em_fun()
        self._set_observed_data_grid(X)
        self._set_ss_parameters(eta, C)
        self._set_shifts(S)
        self._check_params()

        self._print_verbose_global_init_partition()
        self._set_global_init_partition(given_Z)
        self._print_verbose_init_tau()
        self._set_global_init_tau(given_tau_inds)

        # each initialization of the model
        for init_number in range(self.n_init_em):
            self._print_verbose_msg_init(init_number)

            self._crit, self._debug_crit = (- np.inf, - np.inf)
            n_convergences, converged = 0, False
            # if full_sequential is True, we compute the e_step in sequential only,
            # otherwise we start with parallel e_step then switch
            # to sequential updates after a first convergence in parallel
            self._sequential = self.full_sequential

            self._set_current_init_partition(init_number)
            self._init_params()
            self._init_posteriors()
            self._m_step()
            self._debug(init_number, 0)

            # each iteration of the em-like algorithm
            for iter_number in range(1, self.max_iter_em):
                self._print_verbose_msg_iter(iter_number)

                # E + M steps + computing _crit
                self._fit_single(init_number, iter_number)

                n_convergences += e_step.converged(
                    self._old_crit, self._crit,
                    self.atol_iter_em, self.rtol_iter_em
                )
                # if self.n_loops_e_step_max < 1.
                # we apply a e step only for a fraction of the nodes of the hmrf
                # consequently, the elbo can take the same value twice without
                # converging. We consider that the elbo has to remain stable for
                # int(1. / self.n_loops_e_step_max) iteration for the EM algorithm
                # to reach convergence
                converged = (
                    n_convergences == 1
                    if self.n_loops_e_step_max >= 1
                    else
                    n_convergences == int(1. / self.n_loops_e_step_max)
                ) and self.em_type != 'SEM'  # there is no convergence in SEM

                if converged:
                    self._print_verbose_converged(init_number, iter_number)
                    if self.semi_supervised:
                        if self._sequential:
                            break
                        else:
                            # switch from parallel updates to sequential updates
                            # after the first convergence
                            self._sequential = True
                    else:
                        break

            if not converged:
                self._print_verbose_not_converged(init_number)

            self._compute_model_selection_criteria()
            self._write_parameters()

        self.fitted = True
        self._make_ordered_inits()
        return self

    def _fit_single(self, init_number, iter_number):
        # crits are set in the E step
        self._e_step()
        self._append_iter_crits(init_number, iter_number)
        self._m_step()
        self._debug(init_number, iter_number)
        self._print_crit_problem()

    def _e_step(self):
        # log p(x_ij| tau_i = m, z_i = k; theta)
        lpd = e_step.compute_log_pointwise_densities(
            self._residuals, self._sigma2
        )
        # z and lam are updated here
        if self.semi_supervised:
            lpd1 = lpd.sum(1)
            scaled_C_z = (
                self._sequential_ss_e_step(lpd1)
                if self._sequential
                else
                self._batch_ss_e_step(lpd1)
            )
            new_crit, potentials = e_step.compute_elbo(
                lpd1, scaled_C_z,
                self._lam, self._z, self._rho, self._gamma,
                self.em_type, self.ss_mode,
                self.scaled_C, self._not_P, self.min_float
            )
        else:
            new_crit = self._regular_e_step(lpd)
            potentials = 0.

        self._print_debug_crit('z/lam')

        log_prior = m_step.compute_log_prior(
            self._rho, self._gamma, self._sigma2,
            self.map_estimate_sigma2, self.constrained_gamma, self.constrained_sigma2,
            self.alpha_zero, self.beta_zero, self.nu_zero, self.sigma2_zero
        )
        new_crit += log_prior

        self._old_crit = self._crit
        self._crit = new_crit
        self._potentials = potentials

    def _m_step(self):
        self._rho = m_step.compute_rho(
            self._z, self.alpha_zero, self.dtype
        )
        self._gamma = m_step.compute_gamma(
            self._lam,
            self.constrained_gamma,
            self.beta_zero, self.dtype
        )
        self._print_debug_crit('gamma')
        den, where_den = m_step.update_mu(
            self._mu, self._lam, self.XS, self.VS, self.min_den
        )
        self._print_debug_crit('mu')
        self._residuals = m_step.compute_residuals(self.XS, self._mu)
        m_step.update_sigma2(
            self._sigma2,
            self.constrained_sigma2,
            self._residuals, self._lam,
            den, where_den,
            self.sigma2_zero, self.nu_zero,
            self.min_float
        )
        self._print_debug_crit('sigma2')

    def _batch_ss_e_step(self, lpd1):
        unnormalized_posterior, scaled_C_z = e_step.unnormalized_posterior_lam(
            lpd1,
            self.ss_mode,
            self._rho, self._gamma, self._z,
            self.scaled_C, self._P, self._not_P
        )
        if self.em_type == 'CEM':
            self._lam = e_step.cem_lam(unnormalized_posterior)
        elif self.em_type == 'VEM':
            if self.damping_factor is not None:
                self._lam_old = self._lam.copy()
            self._lam = e_step.vem_lam(unnormalized_posterior)
            self._lam = e_step.apply_damping(
                self._lam_old, self._lam, self.damping_factor
            )
        self._z = e_step.update_z(self._lam, self.em_type)
        return scaled_C_z

    def _sequential_ss_e_step(self, lpd1):
        ulp = e_step.fixed_part_unnormalized_posterior_lam(
            lpd1, self.ss_mode, self._rho, self._gamma, self._not_P
        )
        Ns = np.arange(self.N)
        mrf_nodes = Ns[self._P]
        not_mrf_nodes = Ns[self._not_P]

        # e-step is done once outside the mrf
        # or for nodes with zero degree in the mrf
        # (in case ss_mode == 'all') so there is only a node potential
        unnormalized_posterior_not_mrf = ulp[not_mrf_nodes]
        self._lam[not_mrf_nodes] = self._e_step_lam_fun(
            unnormalized_posterior_not_mrf, self.random_state
        )
        self._z[not_mrf_nodes] = e_step.update_z(
            self._lam[not_mrf_nodes], self.em_type
        )

        scaled_C_z = np.zeros((self.N, self.K), dtype=lpd1.dtype)
        check_convergence = (self.n_loops_e_step_max > 1)
        self._lam_old = (
            self._lam.copy() if check_convergence else None
        )
        n_loops = max(1, int(self.n_loops_e_step_max))
        n_nodes_update = (
            max(1, int(mrf_nodes.size * self.n_loops_e_step_max))
            if self.n_loops_e_step_max < 1.
            else
            mrf_nodes.size
        )
        for n in range(n_loops):
            self.random_state.shuffle(mrf_nodes)
            for i in mrf_nodes[:n_nodes_update]:
                scaled_C_z[i] = e_step.set_scaled_C_z_i(self.scaled_C[i, :], self._z)
                unnormalized_posterior_i = ulp[i] + scaled_C_z[i, :, nax]
                if check_convergence:
                    self._lam_old[i] = self._lam[i]
                self._lam[i] = self._e_step_lam_fun(
                    unnormalized_posterior_i, self.random_state
                )
                self._z[i] = e_step.update_z(self._lam[i], self.em_type)

            if check_convergence:
                if self.em_type == 'CEM':
                    # noinspection PyUnresolvedReferences
                    break_condition = (self._lam_old == self._lam).all()
                elif self.em_type == 'SEM':
                    break_condition = False
                elif self.em_type == 'VEM':
                    with np.errstate(under='ignore'):
                        break_condition = np.allclose(self._lam_old, self._lam)
                if break_condition:
                    if self.verbose >= 2:
                        print(f'nb loops = {n}')
                    break

        # if not all nodes have been updated, we have to compute the full product C @ z
        if self.n_loops_e_step_max < 1.:
            with np.errstate(under='ignore'):
                scaled_C_z = similarity_matrices.to_dense(self.scaled_C @ self._z)
        return scaled_C_z

    def _regular_e_step(self, lpd):
        # log rho_k + log gamma_km + log p(x_i| tau_i = m, z_i = k; theta)
        log_mixture_dens = e_step.compute_log_mixture_dens(
            self._rho, self._gamma, lpd
        )
        with np.errstate(under='ignore'):
            # log sum_km exp (log {rho_k * gamma_km * p(x_i| tau_i = m, z_i = k; theta)})
            likelihoods = sp.special.logsumexp(
                log_mixture_dens, axis=(1, 2), return_sign=False
            )
        self._lam = self._e_step_lam_fun(
            log_mixture_dens, self.random_state, likelihoods
        )
        self._z = e_step.update_z(self._lam, self.em_type)
        # noinspection PyUnresolvedReferences
        new_crit = likelihoods.sum()
        return new_crit

    def _set_ss_parameters(self, eta, C):
        """
        _P : ndarray of length N such that
        _P[i] = True if we have prior knowledge about node i
        that is there exists i' such that C[i, i'] > 0
        """
        assert eta is None or eta >= 0.
        self.semi_supervised = not (
            C is None or
            eta is None or
            (C == 0.).all() or
            eta == 0.
        )
        if self.semi_supervised:
            assert C is not None
            similarity_matrices.check_similarity_matrix(C)
            self.eta = eta
            C_bool = C != 0
            # noinspection PyUnresolvedReferences
            C_bool_0 = C_bool.sum(0)
            use_sparse = (C_bool_0.sum() / C.size) <= self.sparse_thr_C
            # since C is symmetric
            self._P = np.asarray(C_bool_0.astype('bool'))
            # note that there is no sp.csr_matrix.any() so we use sum()
            self._not_P = ~ self._P
            # we do not store C but scaled_C = eta * C
            self.scaled_C = self.eta * (
                similarity_matrices.to_sparse(C) if use_sparse
                else similarity_matrices.to_dense(C)
            )
        else:
            self.eta, self.scaled_C = 0., None
            self._P = np.zeros(self.N, dtype='bool')
            self.em_type = (
                'EM' if self.em_type == 'VEM' else self.em_type
            )
            self.full_sequential = False

    def _set_global_init_partition(self, given_Z):
        if self.algo_init_partition == 'given':
            assert given_Z is not None
            assert given_Z.shape == (self.N, )
            assert np.isin(np.unique(given_Z), np.arange(self.K)).all()
            self.global_init_partition = given_Z.copy()
        elif self.algo_init_partition == 'random':
            self.global_init_partition = init.random_init(
                self.N, self.K, self.random_state
            )
        elif self.algo_init_partition == 'kmeans':
            if self.K == 1:
                self.global_init_partition = np.zeros(self.N, dtype='int')
            else:
                X_init = init.compute_X_init(self.X, self.all_visible)
                if self.semi_supervised and self.init_partition_with_C:
                    X_init = similarity_matrices.init_transform(
                        X_init, self.scaled_C, p=1
                    )
                self.global_init_partition = init.kmeans_init(
                    X_init, self.K, self.n_init_km, self.random_state
                )

    def _set_global_init_tau(self, given_tau_inds):
        """
        if given_tau_inds is not None
            global_init_tau is a distribution over shifts built
            from given_tau_ind using proba_init

        if algo_init_tau == 'naive'
            in EM, VEM, flat distribution over shifts
            in CEM or SEM, categorical sample from flat distribution over shifts

        if algo_init_tau == 'sim'
            global_init_tau is a probability distribution
            over the potential shifts obtained by a submodel
            with potentially a coarsening of the set of shitfs S
            note that the submodel does not include semi supervision
        """
        if self.M == 1:
            self.global_init_tau = np.ones((self.N, 1), dtype=self.dtype)
            return

        if given_tau_inds is not None:
            p_zero = (1. - self.proba_init) / (self.M - 1)
            self.global_init_tau = np.full((self.N, self.M), p_zero, dtype=self.dtype)
            self.global_init_tau[np.arange(self.N), given_tau_inds] = self.proba_init
            return

        if self.algo_init_tau == 'naive':
            self.global_init_tau = np.full((self.N, self.M), 1. / self.M, dtype=self.dtype)
        elif self.algo_init_tau == 'sim':
            S_coarsened, mapping = init.coarsen_S(self.S, self.coarsening_factor)
            self.global_init_tau = np.zeros((self.N, self.M), dtype=self.dtype)
            for k in np.unique(self.global_init_partition):
                submodel = self.__copy__()
                submodel.K = 1
                submodel.semi_supervised = False
                submodel.full_sequential = False
                submodel.em_type = (
                    'EM' if self.em_type == 'VEM'
                    else
                    self.em_type
                )
                submodel.algo_init_tau = 'naive'
                submodel.verbose = 0
                submodel.debug = False
                submodel._set_global_init_partition = lambda x: x

                Zk = (self.global_init_partition == k)
                inds_Zk = np.where(Zk)[0]
                submodel.global_init_partition = np.zeros_like(inds_Zk, dtype='int')
                Xk = self.X[Zk]

                submodel.fit(Xk, S_coarsened)
                # noinspection PyTypeChecker
                tau_inds_coarse = submodel.best_parameters()['tau_inds']
                for i, m in zip(inds_Zk, tau_inds_coarse):
                    for m_ in mapping[m]:
                        self.global_init_tau[i, m_] = 1.
            with np.errstate(under='ignore'):
                self.global_init_tau = (
                    self.global_init_tau / self.global_init_tau.sum(1, keepdims=True)
                )

    def __copy__(self):
        new_inst = type(self).__new__(self.__class__)
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                new_inst.__dict__[k] = []
            elif not isinstance(v, np.ndarray):
                new_inst.__dict__[k] = v
        return new_inst

    def _set_current_init_partition(self, init_number):
        if init_number == 0:
            self._current_init_partition = self.global_init_partition.copy()
        else:
            self._current_init_partition = init.apply_perturbation_init_partition(
                self.global_init_partition, self.K,
                self.cluster_perturbation_rate, self.random_state
            )
        self._current_init_partition = init.split_clusters_to_K(
            self._current_init_partition, self.K, self.random_state
        )

    def _init_posteriors(self):
        self._lam = init.init_lam(
            self.K, self.M,
            self._current_init_partition,
            self.global_init_tau,
            self.proba_init,
            self.em_type,
            self.random_state,
            self.dtype
        )
        self._z = e_step.update_z(self._lam, self.em_type)

    def _init_params(self):
        self._rho = np.full(self.K, 1. / self.K, dtype=self.dtype)
        self._mu = np.zeros((self.T_max, self.K), dtype=self.dtype)

        # set nu_zero to None at initialization if no MAP estimate for sigma2
        self.sigma2_zero = m_step.get_sigma2_zero(
            self.K, self.T_max, self.X, self.map_estimate_sigma2
        )
        self.nu_zero = (
            self.nu_zero if self.map_estimate_sigma2
            else - (2 + self.T_max) + self.min_float
        )
        self._sigma2 = np.full((self.T_max, self.K), self.sigma2_zero, dtype=self.dtype)
        self._gamma = np.full((self.K, self.M), 1. / (self.K * self.M), dtype=self.dtype)
        self._residuals = np.zeros((self.N, self.T_max, self.K, self.M), dtype=self.dtype)

    def _set_observed_data_grid(self, X):
        self.N, self.T = X.shape
        if isinstance(X, ma.MaskedArray):
            self.all_visible = False
            self.X = X.astype(self.dtype)
        else:
            mask = np.isnan(X)
            self.all_visible = not mask.any()
            self.X = (
                X.astype(self.dtype)
                if self.all_visible
                else
                ma.masked_array(
                    X.astype(self.dtype), mask=mask, fill_value=np.nan
                )
            )
        if self.all_visible:
            self.V = np.ones_like(self.X).astype('bool')
        else:
            if self.X.mask.all(1).any():
                raise ValueError(
                    'Some curves are unobserved for all time steps: '
                    'these should be removed'
                )
            self.V = ~ self.X.mask

    def _set_shifts(self, S):
        if S is not None:
            assert S.ndim == 1 and S.size > 0 and (S >= 0.).all() and S.dtype == int
            self.S = S
        else:
            self.S = np.zeros(1, dtype='int')

        self.M = self.S.size
        self.S_max = self.S.max(initial=0)
        self.T_max = self.T + self.S_max
        # VS[i, j, m] = V[i, j - S[m]] if j - S[m] >= 0 else False
        # XS[i, j, m] = X[i, j - S[m]] if j - S[m] >= 0 else 0.
        self.VS, self.XS = init.init_VS_XS(
            self.X, self.V, self.S
        )

    def _compute_model_selection_criteria(self):
        likelihood = self._compute_likelihood_only()
        bic = self.compute_bic(likelihood)
        self.all_likelihoods.append(likelihood)
        self.all_bics.append(bic)

    def compute_bic(self, likelihood):
        """
        does not take semi-supervision into account
        """
        return likelihood - .5 * self.nb_params() * np.log(self.N)

    def _compute_likelihood_only(self):
        # for bic, especially when the computed criterion is the
        # ELBO in VEM, returns the likelihood of the model without
        # semi supervision
        lpd = e_step.compute_log_pointwise_densities(
            self._residuals, self._sigma2
        )
        # log rho_k + log gamma_km + log p(x_i| tau_i = m, z_i = k; theta)
        log_mixture_dens = e_step.compute_log_mixture_dens(
            self._rho, self._gamma, lpd
        )
        with np.errstate(under='ignore'):
            # log sum_km exp (log {rho_k * gamma_km * p(x_i| tau_i = m, z_i = k; theta)})
            likelihoods = sp.special.logsumexp(
                log_mixture_dens, axis=(1, 2), return_sign=False
            )
        # noinspection PyUnresolvedReferences
        likelihood = likelihoods.sum()
        return likelihood

    def _make_ordered_inits(self):
        """
        self.ordered_inits are sorted
        by decreasing values of the _crit such that
        self.ordered_inits[0] is the number of the initialization
        that yielded the best _crit at convergence
        """
        assert self.fitted
        last_crits = {}
        for d in self.all_iter_crits:
            init_number, iter_number, crit = (
                d['init_number'], d['iter_number'], d['crit']
            )
            if (
                init_number not in last_crits.keys() or
                iter_number > last_crits[init_number][0]
            ):
                last_crits[init_number] = (iter_number, crit)
        inits = np.array(list(last_crits.keys()))
        crits = np.array([t[1] for t in last_crits.values()])
        idx = np.argsort(crits)[::-1]
        self.ordered_inits = inits[idx]

    def _append_iter_crits(self, init_number, iter_number):
        self.all_iter_crits.append(
            dict(
                init_number=init_number,
                iter_number=iter_number,
                crit=self._crit,
                potentials=self._potentials
            )
        )

    def best_parameters(self):
        assert self.fitted
        best_init = self.ordered_inits[0]
        return self.all_params[best_init]

    def _get_final_partitions(self):
        # Z[i] = k the cluster index of curve i
        Z = self._z.argmax(1)
        # tau_inds[i] = m, the index of the shift of curve i
        tau_inds = self._lam[np.arange(self.N), Z].argmax(1)
        # tau[i] = S[m] the value of the shift
        tau = self.S[tau_inds]
        # VS_tau[i, j] = True if curve i shifted of tau[i]
        # is visible at timestep j
        VS_tau = self.VS[np.arange(self.N), :, tau_inds]
        return Z, tau_inds, tau, VS_tau

    def _get_latent_var_and_sim_metrics_dic(self):
        Z, tau_inds, tau, VS_tau = self._get_final_partitions()
        sc, scw = self.compute_similarity_concordance_metrics(Z)
        return dict(
            Z=Z,
            tau_inds=tau_inds,
            tau=tau,
            VS_tau=VS_tau,
            similarity_concordance_unweighted=sc,
            similarity_concordance_weighted=scw
        )

    def _set_em_fun(self):
        self._e_step_lam_fun = dict(
            CEM=e_step.cem_lam,
            SEM=e_step.sem_lam,
            VEM=e_step.vem_lam,
            EM=e_step.em_lam,
        )[self.em_type]

    def compute_similarity_concordance_metrics(self, Z):
        if not self.semi_supervised:
            return np.nan, np.nan
        sc, scw = similarity_matrices.similarity_concordance(
            Z, similarity_matrices.to_dense(self.scaled_C)
        )
        return sc, scw

    def _get_param_dic(self):
        return {
            'rho': self._rho.copy(),
            'mu': self._mu.copy(),
            'sigma2': self._sigma2.copy(),
            'gamma': self._gamma.copy(),
        }

    def _write_parameters(self):
        self.all_params.append(
            dict(
                # lam=self._lam.copy(),  # remove for less memory consumption
                **self._get_latent_var_and_sim_metrics_dic(),
                **self._get_param_dic()
            )
        )

    def _debug(self, init_number, iter_number):
        if self.debug:
            self.debug_list.append({
                **{
                    'init_number': init_number,
                    'iter_number': iter_number
                },
                **self._get_latent_var_and_sim_metrics_dic(),
                **self._get_param_dic()
            })

    def _print_debug_crit(self, param_name):
        thr_pct = 1e-3
        if self.debug:
            dc = self._compute_debug_crit()
            prefix = ''
            incr_pct = 0.
            if not np.isinf(self._debug_crit):
                # small decreases are allowed
                incr_pct = 100 * (dc - self._debug_crit) / np.abs(dc)
                if np.abs(incr_pct) <= thr_pct:
                    prefix = 'STAGNATION'
                elif incr_pct < - thr_pct:
                    prefix = 'DECREASE'
                elif incr_pct > thr_pct:
                    prefix = 'INCREASE'

            self._debug_crit = dc
            print(f'{prefix} after update of {param_name} of {incr_pct:.2e} %, crit = {dc:.2f}')
            if param_name == 'sigma2':
                print()

    def _compute_debug_crit(self):
        """
        computes the algorithm _crit for a current set of parameters
        used in a debugging context
        """
        residuals = m_step.compute_residuals(self.XS, self._mu)
        lpd = e_step.compute_log_pointwise_densities(
            residuals, self._sigma2
        )
        if self.semi_supervised:
            lpd1 = lpd.sum(1)
            scaled_C_z = None
            crit, potentials = e_step.compute_elbo(
                lpd1, scaled_C_z,
                self._lam, self._z, self._rho, self._gamma,
                self.em_type, self.ss_mode,
                self.scaled_C, self._not_P, self.min_float
            )
        else:
            log_mixture_dens = e_step.compute_log_mixture_dens(
                self._rho, self._gamma, lpd
            )
            with np.errstate(divide='ignore', under='ignore'):
                # noinspection PyUnresolvedReferences
                crit = sp.special.logsumexp(
                    log_mixture_dens, axis=(1, 2), return_sign=False
                ).sum()

        log_prior = m_step.compute_log_prior(
            self._rho, self._gamma, self._sigma2,
            self.map_estimate_sigma2, self.constrained_gamma, self.constrained_sigma2,
            self.alpha_zero, self.beta_zero, self.nu_zero, self.sigma2_zero
        )
        crit += log_prior
        return crit

    def _print_verbose_msg_init(self, init_number):
        if self.verbose >= 1:
            print(
                f'initialization # {init_number + 1} on {self.n_init_em}'
            )

    def _print_verbose_msg_iter(self, iter_number):
        if self.verbose >= 2:
            print(f'iter {iter_number}')

    def _print_verbose_converged(self, init_number, iter_number):
        if self.verbose >= 1:
            if self.semi_supervised:
                if self._sequential:
                    msg = ' in sequential mode'
                else:
                    msg = ': switching to sequential,'
            else:
                msg = ''
            print(
                f'initialization '
                f'# {init_number + 1} converged{msg} at iter {iter_number}'
            )
            if not self.semi_supervised or self._sequential:
                print('\n')

    def _print_verbose_not_converged(self, init_number):
        if self.verbose >= 1:
            print(
                f'initialization'
                f'# {init_number + 1} did not converge\n'
            )

    def _print_verbose_global_init_partition(self):
        if self.verbose >= 1:
            print(
                f'Setting global init partition with ' + self.algo_init_partition + '\n'
            )

    def _print_verbose_init_tau(self):
        if self.verbose >= 1:
            print(
                f'Setting init shifts with ' + self.algo_init_tau + '\n'
            )

    def _print_crit_problem(self):
        if self.verbose >= 1:
            if not np.isinf(self._old_crit):
                incr = 100 * (self._crit - self._old_crit) / np.abs(self._old_crit)
                if incr < -.5:
                    print(
                        f'crit decreased of {-incr:.2f} %,'
                        ' there could be a problem',
                    )
                    if self.semi_supervised and not self.full_sequential:
                        print(
                            'eta could be too high for parallel updates '
                            'try setting full_sequential = True'
                        )

    def _check_params(self):
        assert self.n_init_em >= 1
        assert self.max_iter_em >= 1
        assert self.K >= 1
        assert self.M >= 1
        if self.algo_init_tau == 'sim':
            # noinspection PyUnresolvedReferences
            assert (self.S == np.sort(self.S)).all()
            assert isinstance(self.coarsening_factor, int) and self.coarsening_factor >= 0
        assert self.em_type in ['EM', 'VEM', 'CEM', 'SEM']
        if self.semi_supervised:
            assert self.ss_mode in ['mixture', 'all']
            if not self.full_sequential:
                assert self.em_type != 'SEM'  # no batch em in SEM
                if self.em_type == 'VEM':
                    assert 0. <= self.damping_factor <= 1.
            assert self.em_type != 'EM'
        else:
            assert not self.full_sequential
            assert self.em_type != 'VEM'

        assert self.algo_init_partition in ['given', 'random', 'kmeans']
        assert (
            (0. < self.n_loops_e_step_max < 1.) or
            (isinstance(self.n_loops_e_step_max, int) and self.n_loops_e_step_max >= 1)
        )

    def nb_params(self):
        # outside the class for slope heuristic
        nb_params_gamma = (
            self.K * (self.M - 1)
            if self.constrained_gamma == 'gamma_km'
            else
            self.M - 1
        )
        nb_params_sigma = (
            self.K * self.T_max
            if self.constrained_sigma2 == 'sigma_jk'
            else
            self.K
        )
        nb_prms = (
            self.K - 1 +  # rho
            nb_params_gamma +  # gamma
            self.K * self.T_max +  # mu
            nb_params_sigma  # sigma
        )
        return nb_prms
