import numpy as np
import contextlib
import joblib
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from . import model_result

from .model import MBCAH


def parallel_fit_over_etas(
    X, Ks, S,
    C, etas,
    n_init_first_step,
    n_selected_inits,
    n_init_second_step_per_selected_init,
    em_type_first_step='CEM',
    em_type_second_step='VEM',
    joblib_parallel_backend='loky',
    n_parallel_runs=1,
    n_processes=10,
    shallow=True,
    write_params=True,
    model_results_dir='.',
    str_hash_data='',
    **model_params
):
    # will be filled later, values do not matter here
    em_type = 'VEM'
    n_init_em = 1
    K = 2

    # instanciate model
    model = MBCAH(
        K=K,
        em_type=em_type,
        n_init_em=n_init_em,
        **model_params
    )
    if model.verbose >= 1:
        print(f'--- Data of shape N, T = {X.shape}')
        if isinstance(X, np.ma.MaskedArray):
            pct = 100 * X.mask.sum() / X.size
            print(f'    with {pct:.2f}% of missing values')
        else:
            nans = np.isnan(X)
            if nans.any():
                pct = 100 * nans.sum() / X.size
                print(f'    with {pct:.2f}% of missing values')
            else:
                print(f'    without missing values')
        print()
        print('--- Model: ')
        print(f'    Ks = {Ks}')
        print(f'    sigma structure {model.constrained_sigma2}')
        print(f'    gamma structure {model.constrained_gamma}')

        if model.semi_supervised:
            sparsity_pct = 100 * (1. - (C != 0.).sum() / C.size)
            print(f'    semi supervised eta in {list(etas)}')
            print(f'    with semi supervision mode {model.ss_mode}')
            print(f'    constraint matrix has {sparsity_pct:.2f}% sparsity')
        else:
            print(f'    no semi-supervision')
        print(f'    algo_init_partition {model.algo_init_partition}')
        print(f'    n_init_km {model.n_init_km}')
        print(f'    algo_init_tau {model.algo_init_tau}')
        print(f'    coarsening_factor {model.coarsening_factor}')
        print(f'    cluster_perturbation_rate {model.cluster_perturbation_rate}')
        print()

    shallow_params = (shallow, write_params, model_results_dir, str_hash_data)
    fixed_params = (
        model,
        X, S, C,
        em_type_first_step,
        em_type_second_step,
        n_init_first_step,
        n_selected_inits,
        n_init_second_step_per_selected_init
    )
    all_variable_params = [
        (K, eta)
        for eta in etas
        for K in Ks
        for _ in range(n_parallel_runs)
    ]
    n_variable_params = len(all_variable_params)

    # Spawn off n_parallel_function_calls child SeedSequences to pass to child processes.
    ss = np.random.SeedSequence()
    child_seeds = ss.spawn(n_variable_params)
    rngs = [np.random.default_rng(s) for s in child_seeds]

    if n_variable_params == 1:
        K, eta = all_variable_params[0]
        mrs = [fit_lam(rngs[0], (K, eta), fixed_params, shallow_params)]
    else:
        with tqdm_joblib(
            tqdm(desc='fit for different K and eta', total=n_variable_params)
        ):
            with joblib.parallel_backend(joblib_parallel_backend):
                mrs = joblib.Parallel(n_jobs=n_processes, verbose=1)(
                    joblib.delayed(fit_lam)(
                        rng, variable_params, fixed_params, shallow_params
                    )
                    for rng, variable_params in zip(rngs, all_variable_params)
                )
    # flattens and sorts the results
    mrs = sorted(
        [mr for mr_list in mrs for mr in mr_list],
        key=lambda mr: (mr.K, mr.eta, mr.bic),
        reverse=True
    )
    return mrs


def fit_lam(rng, variable_params, fixed_params, shallow_params):
    (model,
     X, S, C,
     em_type_first_step,
     em_type_second_step,
     n_init_first_step,
     n_selected_inits,
     n_init_second_step_per_selected_init) = fixed_params
    K, eta = variable_params
    print(f'--- Fitting model with {K = } and {eta = :.2f}')

    model_ = model.__copy__()
    model_.K = K
    model_.random_state = rng

    model_results = two_steps_fit(
        model_,
        X, S, eta, C,
        em_type_first_step,
        em_type_second_step,
        n_init_first_step,
        n_selected_inits,
        n_init_second_step_per_selected_init,
        shallow_params
    )
    return model_results


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm
     progress bar given as argument
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _one_step_only_fit(
    model,
    em_type,
    X, S, eta, C,
    n_init,
    shallow_params
):
    model_ = model.__copy__()
    model_.em_type = em_type
    model_.n_init_em = n_init
    if model.verbose >= 1:
        print(f'\n\n--- Fitting the model in {em_type}\n\n')
    model_.fit(
        X, S, eta=eta, C=C
    )
    if model.verbose >= 1:
        print('--- Done')
    return model_result.get_model_results(model_, shallow_params=shallow_params)


def two_steps_fit(
    model,
    X, S, eta, C,
    em_type_first_step,
    em_type_second_step,
    n_init_first_step,
    n_selected_inits,
    n_init_second_step_per_selected_init,
    shallow_params
):
    assert not model.fitted
    assert n_init_first_step > 0
    assert n_selected_inits <= n_init_first_step
    one_step_only = (
        n_selected_inits == 0 or
        n_init_second_step_per_selected_init == 0
    )
    assert (n_selected_inits > 0) or one_step_only

    if one_step_only:
        return _one_step_only_fit(
            model,
            em_type_first_step,
            X, S, eta, C,
            n_init_first_step,
            shallow_params
        )

    if model.verbose >= 1:
        _print_init_infos(
            em_type_first_step,
            em_type_second_step,
            n_init_first_step,
            n_selected_inits,
            n_init_second_step_per_selected_init
        )

    # model_results will contain all the ModelResult instances
    # created in the tow_steps_fit, i.e. EM, VEM, CEM and SEM results
    # without any filter
    model_results = []

    # fit first step
    model_first_step = model.__copy__()
    model_first_step.em_type = em_type_first_step
    model_first_step.n_init_em = n_init_first_step

    if model.verbose >= 1:
        print(f'\n\n--- Fitting the model in {em_type_first_step}\n\n')
    model_first_step.fit(
        X, S,
        eta=eta, C=C
    )
    # appending the ModelResults objects constructed from fitted models
    model_results = model_result.get_model_results(
        model_first_step,
        append_to=model_results, shallow_params=shallow_params
    )

    # params to init em
    selected_model_results = select_best_runs_first_init(
        model_results, n_selected_inits
    )
    if model.verbose >= 1:
        print(f'\n\n--- Fitting the model in {em_type_second_step}\n\n')
    for i, model_res in enumerate(selected_model_results):
        if model.verbose >= 1:
            print(f'Step {i + 1} on {n_selected_inits}')

        model_second_step = model.__copy__()
        model_second_step.algo_init_partition = 'given'
        model_second_step.em_type = em_type_second_step
        model_second_step.n_init_em = n_init_second_step_per_selected_init
        given_Z = model_res.Z
        given_tau_inds = model_res.tau_inds
        model_second_step.fit(
            X, S,
            given_Z=given_Z, given_tau_inds=given_tau_inds,
            eta=eta, C=C
        )
        model_results = model_result.get_model_results(
            model_second_step,
            append_to=model_results, shallow_params=shallow_params
        )

    # delete latent variables to save memory
    # these latent variables will be loaded from pickle
    shallow, write_params, model_result_dir, str_hash_data = shallow_params
    if shallow:
        for mr in model_results:
            if hasattr(mr, 'Z'):
                del mr.Z
            if hasattr(mr, 'tau_inds'):
                del mr.tau_inds

    if model.verbose >= 1:
        print('\n--- Done')
    return model_results


def _print_init_infos(
    em_type_first_step,
    em_type_second_step,
    n_init_first_step,
    n_selected_inits,
    n_init_second_step_per_selected_init
):
    print(f'--- Initialzation plan:')
    print(f'    1 - Do {n_init_first_step} {em_type_first_step} runs')
    print(
        f'    2 - Take the {n_selected_inits} best partitions'
        f' of the {em_type_first_step} runs'
    )
    print(
        f'    3 - For each of the {n_selected_inits} starting points,'
        f'run {n_init_second_step_per_selected_init} {em_type_second_step} inits'
    )


def select_best_runs_first_init(model_results, n_selected_inits):
    """
    assumes that model_results contains the model results
    sorted by decreasing criterion (loglikelihood or ELBO)
    and returns the n <= n_selected_inits best model results that
    do not return similar paritions
    """
    def differ_ari(labels_true, labels_pred):
        return adjusted_rand_score(labels_true, labels_pred) < (1. - 1e-3)

    mr0 = model_results[0]
    distincts_mr, Zs = [mr0], [mr0.Z]
    for mr in model_results[1:]:
        if len(distincts_mr) == n_selected_inits:
            break
        different_partitions = any(
            differ_ari(mr.Z, Z) for Z in Zs
        )
        if different_partitions:
            distincts_mr.append(mr)
            Zs.append(mr.Z)
    return distincts_mr
