import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


nax = np.newaxis
pal_name = 'tab20'  # 'tab20', 'colorblind'
max_len_pal = {
    'tab20': 20,
    'colorblind': 10
}[pal_name]


def get_cmap(n_colors):
    cp = sns.mpl_palette(pal_name, n_colors)
    if n_colors > max_len_pal:
        q, r = divmod(n_colors, max_len_pal)
        n_cycles = q + (r > 0)
        cp = np.tile(cp, (n_cycles, 1))
    return cp


def _unshifted_plot(
    ax,
    to_plot,
    X, S,
    tau, VS_tau,
    mu, sigma2,
    n_sigma=3.,
    c_cluster=None,
):
    for plot in to_plot:
        assert plot in [
            'X_shifted',
            'X_unshifted',
            'mu',
            'sigma2',
            'shifts'
        ]
    n_curves, T = X.shape
    T_max = T + S.max()
    ts = np.arange(T_max)
    if n_curves == 0:
        return

    # mask params at timesteps when
    # no curve is observed
    no_visible = ~VS_tau.any(0)
    sigma_visible = np.sqrt(sigma2.copy())
    mu_visible = mu.copy()
    sigma_visible[no_visible] = np.nan
    mu_visible[no_visible] = np.nan

    if 'shifts' in to_plot:
        _, counts = np.unique(np.concatenate([tau, S]), return_counts=True)
        ax.bar(S, counts - 1, width=1.5)

    if 'X_unshifted' in to_plot:
        ax.plot(np.arange(T), X.T)

    if 'X_shifted' in to_plot:
        for i, tau_i in enumerate(tau):
            ax.plot(ts[tau_i: T + tau_i], X[i])

    if 'mu' in to_plot:
        ax.plot(ts, mu_visible, lw=2, c=c_cluster)
    if 'sigma2' in to_plot:
        up = mu_visible + n_sigma * sigma_visible
        down = mu_visible - n_sigma * sigma_visible
        ax.fill_between(
            ts, y1=down, y2=up,
            color='orange', alpha=.1, lw=0.,
        )


def combined_plot(
    all_plots,
    X, S,
    tau, VS_tau, Z,
    mu, sigma2,
    variance_reduction_factors,
    n_sigma=3.,
    subfigsize=None,
):
    K = mu.shape[1]
    clusters = np.unique(Z)
    n_clusters = clusters.size
    n_plots = len(all_plots)

    fsx, fsy = subfigsize if subfigsize is not None else (5, 3)
    f, ax = plt.subplots(
        n_clusters, n_plots,
        figsize=(n_plots * fsx, fsy * n_clusters),
        tight_layout=True, squeeze=False
    )
    cmap_clusters = get_cmap(K)

    for ik, k in enumerate(clusters):
        Zk = (Z == k)
        nk = Zk.sum()
        for i_plot, to_plot in enumerate(all_plots):
            title = f'K = {k} size {nk}, ' + ', '.join(to_plot)
            if 'shifts' in to_plot:
                title += f' F_{k} = {variance_reduction_factors[k]:.2f}'
            ax[ik, i_plot].set_title(title)
            _unshifted_plot(
                ax=ax[ik, i_plot],
                to_plot=to_plot,
                X=X[Zk],
                S=S,
                tau=tau[Zk],
                VS_tau=VS_tau[Zk],
                mu=mu[:, k],
                sigma2=sigma2[:, k],
                n_sigma=n_sigma,
                c_cluster=cmap_clusters[k],
            )


def plot_similarity_scores(model_results):
    semi_supervised = any((mr.semi_supervised for mr in model_results))
    etas = np.unique([mr.eta for mr in model_results])
    if not (semi_supervised and etas.size >= 2):
        print('Nothing to plot')
        return

    Ks = np.unique([mr.K for mr in model_results])
    f, ax = plt.subplots(
        len(Ks), 1, figsize=(6, 3 * len(Ks)),
        squeeze=False, tight_layout=True
    )
    keys = ['K', 'eta']
    metrics = [
        'similarity_concordance_weighted',
        'likelihood', 'criterion_at_convergence'
    ]
    attrs = keys + metrics
    df_crits = pd.DataFrame(
        [{attr: getattr(mr, attr) for attr in attrs} for mr in model_results]
    ).query("eta > 0.")
    df_crits = pd.concat(
        [
            dfg.assign(model_num=lambda d: range(dfg.shape[0]))
            for g, dfg in df_crits.groupby(['K', 'eta'])
        ]
    )
    etas = np.unique(df_crits['eta'])
    for ik, (K, dfg) in enumerate(df_crits.groupby('K')):
        crits = {
            k: pd.pivot_table(
                dfg,
                values=k,
                index='eta',
                columns='model_num'
            ).values
            for k in metrics
        }
        Rs = crits['similarity_concordance_weighted']
        ax[ik, 0].plot(
            etas, Rs.max(1),
            c='black', label='best R'
        )
        ax[ik, 0].plot(
            etas, Rs[np.arange(etas.size), np.argmax(crits['likelihood'], axis=1)],
            c='blue', label='best likelihood'
        )
        ax[ik, 0].plot(
            etas, Rs[np.arange(etas.size), np.argmax(crits['criterion_at_convergence'], axis=1)],
            c='green', label='best elbo'
        )
        ax[ik, 0].legend()
        ax[ik, 0].set_ylabel('R')
        ax[ik, 0].set_xlabel('eta')
        ax[ik, 0].set_title(f'{K = } ')
