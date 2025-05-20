# List: Package for Functional Connectivity Measures #

# 1. Conditional Granger Causality (cGC_func)
# 2. Adaptive Direct Transfer Function (ADTF_func)
# 3. Partial Directed Coherence (PDCoh_func)
# 4. Phase Slope Index (PSI_func)
# 5. Phase Lag Index (PLI_func)
# 6. Transfer Entropy (TE_func)
# 7. Mixed Embedding (ME_mtd) # method

from mvarica import MVAR, connectivity_mvarica
import numpy as np
import matplotlib.pyplot as plt
from test_data import MNEData

class FCMethods:
    """
    Usage:
        fc = FCMethods(data)
        results = fc.run_methods(['Corr', 'cGC', 'PLI'], corr_params={'method': 'pearson'})
    """
    DEFAULT_FC_MEASURES = {
        'Corr': 'Corr_func',
        'cGC' : 'cGC_func',
        'ADTF': 'ADTF_func',
        'PDCoh': 'PDCoh_func',
        'DTF' : 'DTF_func',
        'PSI' : 'PSI_func',
        'PLI' : 'PLI_func',
        'TE'  : 'TE_func',
        'MI'  : 'MI_func',
    }

    def __init__(self, signal, measures=None, fs=150, model_order=0):
        self.measures = measures or self.DEFAULT_FC_MEASURES
        self.signal = signal
        self.model_order = model_order
        self.fs = fs

        # Initialize data attributes
        self.pdc = None
        self.dtf = None


    def run_methods(self, measures=None, **kwargs):
        """
        Compute selected connectivity measures.

        Args:
            measures (list[str]): List of keys from DEFAULT_FC_MEASURES. If None, run all.
            **kwargs: Per-measure parameter dicts, e.g., corr_params={...}, te_params={...}

        Returns:
            dict: {measure_name: result_array}
        """
        if measures is None:
            measures = list(self.DEFAULT_FC_MEASURES.keys())

        results = {}
        for m in measures:
            if m not in self.DEFAULT_FC_MEASURES:
                raise ValueError(f"Unknown measure: {m}")
            func = getattr(self, self.DEFAULT_FC_MEASURES[m])
            # extract params
            param_key = f"{m.lower()}_params"
            params = kwargs.get(param_key, {})
            results[m] = func(**params)
        return results

    def Corr_func(self, method='pearson'):
        """
        Compute correlation matrix across channels.
        """
        if method == 'pearson':
            return np.corrcoef(self.data)
        else:
            # implement other methods
            raise NotImplementedError

    def cGC_func(self, order=1):
        """
        Conditional Granger Causality.
        """
        # placeholder implementation
        from statsmodels.tsa.vector_ar.var_model import VAR
        model = VAR(self.data.T)
        res = model.fit(maxlags=order)
        return res.test_causality()

    def ADTF_func(self):
        """
        Adaptive Direct Transfer Function.
        """
        raise NotImplementedError

    def PDCoh_func(self):
        """
        Partial Directed Coherence.
        """
        raise NotImplementedError

    def DTF_func(self):
        """
        Direct Transfer Function.
        """
        raise NotImplementedError

    def PSI_func(self):
        """
        Phase Slope Index.
        """
        raise NotImplementedError

    def PLI_func(self):
        """
        Phase Lag Index.
        """
        raise NotImplementedError

    def TE_func(self):
        """
        Transfer Entropy.
        """
        raise NotImplementedError

    def MI_func(self):
        """
        Mutual Information.
        """
        raise NotImplementedError

    # # ======= Mutual Information =======
    # normalization isn't required
    def MI_func(x):
        # x: signals;  x.shape = time x variables
        import numpy as np
        from sklearn.feature_selection import mutual_info_regression
        n, m = x.shape
        if n<m: print('data in bad shape!'); x = x.T; n,m = x.shape

        MI = np.zeros([m,m])
        for i in range(m):
            mi = mutual_info_regression(x, x[:,i], discrete_features=False)
            MI[i, :] = mi


        return MI


    def PDC_func(self, signal, ica_params, n_fft=512, model_order=5, delta=0.1):
        mvar_model = MVAR(model_order=model_order, delta=delta)
        self.pdc = connectivity_mvarica(signal, ica_params, 'pdc', n_fft=n_fft, var_model=mvar_model)
        return  self.pdc

    def DTF_func(self, signal, ica_params, n_fft=512, model_order=5, delta=0.1):
        mvar_model = MVAR(model_order=model_order, delta=delta)
        self.dtf =  connectivity_mvarica(signal, ica_params, 'dtf', n_fft=n_fft, var_model=mvar_model)
        return self.dtf


# Example pipeline
if __name__ == '__main__':
    # Load data
    data = np.random.randn(5, 1000)  # 5 channels, 1000 samples
    fc = FCMethods(data, fs=250)

    # Run correlation and PLI
    results = fc.run_methods(['Corr', 'PLI'], corr_params={'method': 'pearson'})
    print(results)




# data = np.random.randn(2, 5, 1000)  # 2 epochs, 5 channels, 1000 samples
# mvar_model = MVAR(model_order=5, delta=0.1)
# ica_params = {'method': 'infomax_extended', 'random_state': 42}
# pdc = connectivity_mvarica(data, ica_params, 'pdc', n_fft=128, var_model=mvar_model)
#
#
#
# freqs = np.linspace(0, 0.5, pdc.shape[-1])
#
# PDC_values = pdc
#
# # Visualization
# fig, axes = plt.subplots(5, 5, figsize=(12, 10))
# plt.suptitle("Partial Directed Coherence (PDC)")
#
# for i in range(5):
#     for j in range(5):
#         ax = axes[i, j]
#         ax.plot(freqs, PDC_values[i, j, :], 'r')
#         ax.set_xlim(0, 0.5)
#         ax.set_ylim(0, 1)
#         ax.set_title(f"PDC {j+1}→{i+1}")
#         ax.grid(True)
#
# plt.tight_layout()
# plt.show()

