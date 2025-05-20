import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from mvarica import MVAR, connectivity_mvarica

# --- Global Registry ---
METHOD_REGISTRY = {}

def register_method(name):
    def decorator(func):
        METHOD_REGISTRY[name] = func
        return func
    return decorator

def mvarica_template(measure_name):
    def method_func(self, model_order=5, delta=0, ica_params=None, n_fft=128, **kwargs):
        if self.data.ndim == 2:
            signal = self.data[np.newaxis, ...]
        elif self.data.ndim == 3:
            signal = self.data
        else:
            raise ValueError(f"Input data must be 2D or 3D, got {self.data.ndim}")

        if ica_params is None:
            ica_method = kwargs.get('ica_method', 'infomax_extended')
            if ica_method not in ('infomax_extended', 'infomax', 'fastica'):
                raise ValueError(f"Unsupported ICA method: {ica_method}")
            ica_params = {
                'method': ica_method,
                'random_state': kwargs.get('random_state', None)
            }

        mvar_model = MVAR(model_order=model_order, delta=delta)

        result = connectivity_mvarica(
            real_signal=signal,
            ica_params=ica_params,
            measure_name=measure_name,
            n_fft=n_fft,
            var_model=mvar_model,
        )

        return result
    return method_func

class FCMethods:
    def __init__(self, data, fs=1.0):
        self.data = data
        self.fs = fs
        self.methods = {name: func.__get__(self) for name, func in METHOD_REGISTRY.items()}

    def run_connectivity(self, methods_params):
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

# --- Registered Methods ---

@register_method('Corr')
def corr_func(self, method='pearson'):
    if method == 'pearson':
        return np.corrcoef(self.data)
    raise NotImplementedError(f"Correlation method '{method}' not implemented.")

@register_method('PDCoh')
def pdcoh_func(self, **kwargs):
    return mvarica_template('pdc')(self, **kwargs)

@register_method('DTF')
def dtf_func(self, **kwargs):
    return mvarica_template('dtf')(self, **kwargs)

@register_method('ADTF')
def adtf_func(self, **kwargs):
    raise NotImplementedError("ADTF not yet implemented.")

@register_method('cGC')
def cgc_func(self, order=1):
    model = VAR(self.data.T)
    res = model.fit(maxlags=order)
    return res.params

@register_method('TE')
def te_func(self, **kwargs):
    raise NotImplementedError("Transfer Entropy not yet implemented.")

@register_method('MI')
def mi_func(self, **kwargs):
    raise NotImplementedError("Mutual Information not yet implemented.")

@register_method('PSI')
def psi_func(self, **kwargs):
    raise NotImplementedError("PSI not yet implemented.")

@register_method('PLI')
def pli_func(self, **kwargs):
    raise NotImplementedError("PLI not yet implemented.")

# --- Example Usage ---
if __name__ == '__main__':
    data = np.random.randn(5, 1000)
    model = FCMethods(data)

    results = model.run_connectivity({
        'Corr': {'method': 'pearson'},
        'PDCoh': {'model_order': 5, 'n_fft': 128, 'ica_method': 'infomax_extended'},
        'DTF': {'model_order': 5, 'ica_method': 'fastica'},
        'ADTF': {}  # Will raise NotImplementedError
    })

    for name, result in results.items():
        print(f"{name}: {result}")
