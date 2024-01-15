import numpy as np
import pickle
import uuid

from utils import init


class ModelResult:
    def __init__(
        self,
        model, model_init_number,
        init_number_tot=None,
        shallow=False, write_params=True,
        str_hash_data='', model_result_dir='./'
    ):
        assert model.fitted
        self.semi_supervised = None
        self.em_type = None
        self.semi_supervised = None
        self.K = model.K
        self.model_class = model.__class__.__name__
        self.model_init_number = model_init_number
        self.eta = model.eta

        self.shallow = shallow
        self.write_params = write_params
        self.init_number_tot = init_number_tot
        self.mr_id = uuid.uuid4()
        self.model_result_dir = model_result_dir
        self.params_path = model_result_dir + str(self.mr_id) + '.pickle'
        # str_hash_data identifies the data ~"uniquely" such that the data
        # pickle is not overwritten for each new data set
        self.data_path = model_result_dir + str_hash_data + 'data.pickle'

        self.X, self.V = None, None
        self.C = None
        self.all_visible, self.S, self.T_max, self.N = None, None, None, None

        # useful for plotting
        # computed when the model result is loaded
        self.VS, self.VS_tau, self.tau, self.XS = None, None, None, None

        # attributes used for model selection
        self._set_necessary_params(model)

        # if self.write_params is False, then ModelResult
        # only contains the params set in _set_necessary_params()
        # that are used for model selection
        if self.write_params:
            params_dic = self._get_params_dic(model)

            # write to pickle in shallow mode otherwise
            # write to attributes
            if self.shallow:
                with open(self.params_path, 'wb') as f:
                    pickle.dump(params_dic, f)

                # overwrites existing data in dir
                with open(self.data_path, 'wb') as f:
                    pickle.dump(
                        dict(
                            X=model.X,
                            C=model.scaled_C / model.eta if model.semi_supervised else None
                        ),
                        f
                    )
            else:
                self._set_params(params_dic)
                to_copy = ['X', 'V', 'VS', 'XS']
                for k in to_copy:
                    setattr(self, k, getattr(model, k))
                # set deducted params
                # noinspection PyUnresolvedReferences
                self.VS_tau = self.VS[np.arange(self.N), :, self.tau_inds]
                # noinspection PyUnresolvedReferences
                self.tau = self.S[self.tau_inds]

    def _set_necessary_params(self, model):
        # attributes that are written even in shallow mode
        sorted_crits = sorted(
            [d for d in model.all_iter_crits
             if d['init_number'] == self.model_init_number],
            key=lambda d: d['iter_number']
        )
        self.iter_criterions = np.array([
            d['crit'] for d in sorted_crits
        ])
        self.iter_potentials = np.array([
            d['potentials'] for d in sorted_crits
        ])
        self.criterion_at_convergence = self.iter_criterions[-1]
        self.potentials_at_convergence = self.iter_potentials[-1]

        # bic and likelihoods at convergence
        self.bic = model.all_bics[self.model_init_number]
        self.likelihood = model.all_likelihoods[self.model_init_number]

        params = model.all_params[self.model_init_number]
        self.similarity_concordance_unweighted = params[
            'similarity_concordance_unweighted'
        ]
        self.similarity_concordance_weighted = params[
            'similarity_concordance_weighted'
        ]

        self.S = model.S

        # necessary for two_steps_fit
        # will be deleted in shallow mode in two_steps_fit
        # and reloaded after
        self.Z = params['Z']
        self.tau_inds = params['tau_inds']

        # copy all attributes of model except these
        do_not_add = [
            'all_params',
            'debug_list',
            'all_iter_crits',
            'all_bics',
            'all_likelihoods'
        ]
        for k, v in model.__dict__.items():
            if (
                not isinstance(v, np.ndarray) and
                k not in do_not_add
            ):
                setattr(self, k, v)

    def _get_params_dic(self, model):
        params_dic = {}
        model_params = model.all_params[self.model_init_number]
        arrays_to_copy = [
            'Z', 'tau_inds',  # lam
            'rho', 'gamma', 'mu', 'sigma2'
        ]
        for k in arrays_to_copy:
            params_dic[k] = model_params[k]

        params_dic['C'] = (
            model.scaled_C / model.eta
            if model.semi_supervised and model.eta > 0.
            else
            None
        )
        return params_dic

    def _set_params(self, params_dic):
        # writes all public params as object attributes
        for k, v in params_dic.items():
            if k not in self.__dict__.keys() and k[0] != '_':
                self.__dict__[k] = v

    def load_shallow_from_pickle(self):
        if not self.shallow or not self.write_params:
            return

        with open(self.params_path, 'rb') as f:
            params_dic = pickle.load(f)
            self._set_params(params_dic)

        with open(self.data_path, 'rb') as f:
            data_dict = pickle.load(f)
            self.X = data_dict['X']
            self.C = data_dict['C']

        self.shallow = False
        self.V = (
            np.ones_like(self.X).astype('bool')
            if self.all_visible
            else
            ~self.X.mask
        )
        self.VS, self.XS = init.init_VS_XS(
            self.X, self.V, self.S
        )
        self.VS_tau = self.VS[np.arange(self.N), :, self.tau_inds]
        self.tau = self.S[self.tau_inds]

    def __repr__(self):
        title = '' if self.write_params else 'EMPTY '
        title += f'ModelResult of {self.model_class} for K = {self.K}'
        title += f' and eta = {self.eta:.2E}:\n' if self.semi_supervised else '\n'
        attrs = (
            f'    - BIC  = {self.bic:.2E}\n'
            f'    - likelihood  = {self.likelihood:.2E}\n'
            f'    - {self.em_type} crit  = {self.criterion_at_convergence:.2E}\n'
        )
        ss_attrs = (
            f'    - potentials  = {self.potentials_at_convergence:.2E}\n'
            f'    - sc  = {self.similarity_concordance_unweighted:.2E}\n'
            f'    - scw  = {self.similarity_concordance_weighted:.2E}'
        )
        s = title + attrs
        if self.semi_supervised:
            s += ss_attrs
        return s


def get_model_results(model, append_to=None, shallow_params=(False, True, './', '')):
    init_number_tot = len(append_to) if append_to is not None else 0
    shallow, write_params, model_result_dir, str_hash_data = shallow_params
    model_results = []
    for init_nb_em in range(model.n_init_em):
        model_results.append(
            ModelResult(
                model, init_nb_em,
                init_number_tot=init_number_tot,
                shallow=shallow, write_params=write_params,
                model_result_dir=model_result_dir,
                str_hash_data=str_hash_data
            )
        )
        init_number_tot += 1

    if append_to is not None:
        model_results += append_to

    return sorted(
        model_results,
        key=lambda x: x.criterion_at_convergence,
        reverse=True
    )
