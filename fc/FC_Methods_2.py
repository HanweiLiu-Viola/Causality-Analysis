import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from mvarica import MVAR, connectivity_mvarica

class MVARICABase:
    def __init__(self, data):
        self.data = data

    def prepare_signal(self):
        if self.data.ndim == 2:
            return self.data[np.newaxis, ...]
        elif self.data.ndim == 3:
            return self.data
        else:
            raise ValueError(f"Input data must be 2D or 3D, but got ndim={self.data.ndim}")

    def prepare_ica_params(self, ica_params=None, **kwargs):
        if ica_params is None:
            ica_method = kwargs.get('ica_method', 'infomax_extended')
            if ica_method not in ('infomax_extended', 'infomax', 'fastica'):
                raise ValueError(f"Unsupported ICA method: {ica_method}")
            ica_params = {
                'method': ica_method,
                'random_state': kwargs.get('random_state', None)
            }
        return ica_params



class FCMethods:
    DEFAULT_FC_MEASURES = {
        'Corr': 'corr_func',
        'cGC' : 'cgc_func',
        'ADTF': 'adtf_func',
        'PDCoh': 'pdcoh_func',
        'DTF' : 'dtf_func',
        'PSI' : 'psi_func',
        'PLI' : 'pli_func',
        'TE'  : 'te_func',
        'MI'  : 'mi_func',
    }


    def __init__(self, data, fs=1.0):
        self.data = data
        self.fs = fs
        self.methods = {name: getattr(self, func_name) for name, func_name in self.DEFAULT_FC_MEASURES.items()}


    def run_connectivity(self, methods_params):
        """
        Example:
        methods_params = {
            'Corr': {'method': 'pearson'},
            'PDCoh': {'model_order': 5, 'n_fft': 128, 'ica_method': 'infomax_extended'}
        }
        """
        results = {}

        for method_name, params in methods_params.items():
            func = self.methods.get(method_name)
            if func is None:
                raise ValueError(f"Unknown method: {method_name}")
            try:
                results[method_name] = func(**params)
            except NotImplementedError as e:
                results[method_name] = f"{method_name} not implemented: {e}"

        return results




    def corr_func(self, method='pearson'):
        if method == 'pearson':
            return np.corrcoef(self.data)
        else:
            raise NotImplementedError(f"Correlation method '{method}' not implemented.")

    def cgc_func(self, order=1):
        model = VAR(self.data.T)
        res = model.fit(maxlags=order)
        return res.params

    def adtf_func(self):
        raise NotImplementedError

    # def pdcoh_func(self, model_order=5, delta=0, ica_params=None, n_fft=128, **kwargs):
    #     if self.data.ndim == 2:
    #         signal = self.data[np.newaxis, ...]
    #     elif self.data.ndim == 3:
    #         signal = self.data
    #     else:
    #         raise ValueError(f"Input data must be 2D or 3D, but got ndim={self.data.ndim}")
    #
    #
    #     # 2. ICA 参数合并与校验
    #     if ica_params is None:
    #         ica_method = kwargs.get('ica_method', 'infomax_extended')
    #         if ica_method not in ('infomax_extended', 'infomax', 'fastica'):
    #             raise ValueError('This method is not defined!' + '\n' + 'supported methods: infomax, fastica, picard, infomax_extended')
    #         ica_params = {
    #             'method': ica_method,
    #             'random_state': kwargs.get('random_state', None)
    #         }
    #
    #     mvar_model = MVAR(model_order=model_order, delta=delta)
    #
    #     pdc = connectivity_mvarica(
    #         real_signal=signal,
    #         ica_params=ica_params,
    #         measure_name='pdc',
    #         n_fft=n_fft,
    #         var_model=mvar_model,
    #         )
    #
    #     return pdc

    def pdcoh_func(self, model_order=5, delta=0, ica_params=None, n_fft=128, **kwargs):
        model = MVARICABase(self.data)
        signal = model.prepare_signal()
        ica_params = model.prepare_ica_params(ica_params, **kwargs)

        mvar_model = MVAR(model_order=model_order, delta=delta)
        return connectivity_mvarica(signal, ica_params=ica_params, measure_name='pdc',
                                    n_fft=n_fft, var_model=mvar_model)

    def dtf_func(self, model_order=5, delta=0, ica_params=None, n_fft=128, **kwargs):
        model = MVARICABase(self.data)
        signal = model.prepare_signal()
        ica_params = model.prepare_ica_params(ica_params, **kwargs)

        mvar_model = MVAR(model_order=model_order, delta=delta)
        return connectivity_mvarica(signal, ica_params=ica_params, measure_name='dtf',
                                    n_fft=n_fft, var_model=mvar_model)


    def psi_func(self):
        raise NotImplementedError

    def pli_func(self):
        raise NotImplementedError

    def te_func(self):
        raise NotImplementedError

    def mi_func(self):
        raise NotImplementedError

if __name__ == '__main__':
    data = np.random.randn(5, 1000)
    fc_model = FCMethods(data)
    results = fc_model.run_connectivity({
        'Corr': {'method': 'pearson'},
        'PDCoh': {'model_order': 5, 'n_fft': 128, 'ica_method': 'infomax_extended'},
        'DTF': {'model_order': 5, 'n_fft': 128, 'ica_method': 'infomax_extended'}
    })

    print(results)
