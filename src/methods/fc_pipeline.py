import numpy as np
import xarray as xr
from statsmodels.tsa.vector_ar.var_model import VAR
from core.mvarica import MVAR, connectivity_mvarica
from frites.conn import conn_covgc
from mne_connectivity import spectral_connectivity_epochs, phase_slope_index
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.multivariate_mi import MultivariateMI
import warnings



class BasePreprocessor:
    def __init__(self, data):
        self.data = self._standardize(data)

    @staticmethod
    def _standardize(data):
        if data.ndim == 2:
            return data[np.newaxis, ...]  # (1, n_channels, n_times)
        elif data.ndim == 3:
            return data
        else:
            raise ValueError("Data must be 2D or 3D")

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


class ADTFModel:
    def __init__(self, data):
        self.data = data  # shape: (n_trials, n_channels, n_times)
        self.models = []

    def fit_var_model(self, maxlags=20):
        """Fit a VAR model per trial with AIC order selection; falls back to maxlags=1 on failure."""
        self.models = []
        for trial in self.data:
            model = VAR(trial.T)
            try:
                results = model.fit(maxlags=maxlags, ic='aic')
            except np.linalg.LinAlgError:
                results = model.fit(maxlags=1)
            self.models.append(results)

    def compute_transfer_matrix(self, A, f):
        """计算给定 VAR 系数 A 的传递函数 H"""
        p, n, _ = A.shape
        z_powers = np.exp(-1j * 2 * np.pi * f * np.arange(1, p + 1))
        A_sum = sum(A[k] * z_powers[k] for k in range(p))
        H = np.linalg.inv(np.eye(n) - A_sum)
        return H

    def compute_adtf(self, H_all):
        """
        H_all: shape (n_channels, n_channels, n_freqs) 或 (n_channels, n_channels) 单频率
        返回 shape: (n_channels, n_channels, n_freqs) 或 (n_channels, n_channels)
        """
        H_abs2 = np.abs(H_all) ** 2
        row_sums = H_abs2.sum(axis=1, keepdims=True)
        row_sums_safe = np.where(row_sums == 0, 1, row_sums)
        return np.where(row_sums == 0, 0, H_abs2 / row_sums_safe)

    def run(self, freqs, maxlags=20, integrate=True):
        """
        Parameters:
            freqs: array-like, shape (n_freqs,)
            maxlags: int, maximum VAR model lag
            integrate: bool, whether to integrate over frequencies
        Returns:
            dict with keys: 'matrix', 'freqs', 'type', 'integrated'
        """
        self.models.clear() # Viola: 09.07.2015, Wednesday
        self.fit_var_model(maxlags=maxlags)

        if integrate:
            # 对每个 trial 计算 ADTF(f)，再平均，最后积分
            adtf_all = []
            for model in self.models:
                A = model.coefs
                H_all = np.stack([self.compute_transfer_matrix(A, f) for f in freqs], axis=-1)  # (n, n, freqs)
                adtf = self.compute_adtf(H_all)  # (n, n, freqs)
                adtf_all.append(adtf)
            adtf_all = np.stack(adtf_all, axis=0)  # (trials, n, n, freqs)
            adtf_mean = np.mean(adtf_all, axis=0)  # (n, n, freqs)

            return {
                'matrix': np.mean(adtf_mean, axis=-1),  # (n, n), mean ADTF over band
                'freqs': freqs,
                'type': 'ADTF',
                'integrated': True
            }
        else:
            # 返回每个 trial 的每频率 ADTF 矩阵
            matrix_list = []
            for model in self.models:
                A = model.coefs
                H_list = [self.compute_transfer_matrix(A, f) for f in freqs]
                adtf_list = [self.compute_adtf(H) for H in H_list]
                matrix_list.append(adtf_list)  # shape: (n_freqs, n, n)

            return {
                'matrix_list': matrix_list,  # list of (n_freqs, n, n)
                'freqs': freqs,
                'type': 'ADTF',
                'integrated': False
            }


class FCMethods:
    def __init__(self, data, fs=1.0):
        self.data = BasePreprocessor(data).data
        self.fs = fs

    def compute_all(self, methods_params):
        results = {}
        for name, params in methods_params.items():
            method = getattr(self, f"_{name.lower()}_func", None)
            if method:
                output = method(**params)
                if isinstance(output, dict):
                    results[name] = output
                else:
                    results[name] = {
                        'matrix': output,
                        'type': name,
                        'params': params
                    }
            else:
                results[name] = {'error': f"Method {name} not implemented."}
        return results


    def _corr_func(self, Regul=False, Normal=False, Threshold=None):
        epochs, channels, _ = self.data.shape

        if epochs == 1:
            data = self.data[0]
            corr = np.corrcoef(data)
        else:
            corrs = np.zeros((epochs,channels,channels))
            for i in range(epochs):
                corrs[i] = np.corrcoef(self.data[i])
            corr = np.mean(corrs, axis=0)

        if Regul:
            eigvals = np.sort(np.linalg.eigvals(corr))
            eigvals = eigvals[np.isreal(eigvals)].real
            eig_min = next((l for l in eigvals if l > 1e-10), eigvals[-1])
            delta = max(0, (eigvals[-1] - 50 * eig_min) / 49)
            corr = (corr + delta * np.eye(corr.shape[0])) / (1 + delta)

        if Normal:
            corr = (corr + 1) / 2

        if Threshold is not None:
            corr[np.abs(corr) < Threshold] = 0

        # TODO: add integrate method
        
        return corr


    def _adtf_func(self, fmin=1.0, fmax=40.0, n_freqs=100, maxlags=20, Threshold=None, integrate=True):
        """
        Compute ADTF in a given frequency band.

        Parameters
        ----------
        fmin, fmax : float — frequency band in Hz
        n_freqs    : int   — number of frequency points
        integrate  : bool  — if True return a single (n,n) matrix integrated over the band
        """
        freqs_norm = np.linspace(fmin, fmax, n_freqs) / self.fs   # Hz → normalised [0, 0.5]
        adtf = ADTFModel(self.data).run(freqs=freqs_norm, maxlags=maxlags, integrate=integrate)
        
        # if Threshold is not None:
        #     adtf[np.abs(adtf) < Threshold] = 0
        
        if Threshold is not None:
            if 'matrix' in adtf:
                adtf['matrix'][np.abs(adtf['matrix']) < Threshold] = 0
            elif 'matrix_list' in adtf:
                adtf['matrix_list'] = [
                    [np.where(np.abs(m) < Threshold, 0, m) for m in trial]
                    for trial in adtf['matrix_list']
                ]


        return adtf

    def _pdcoh_func(self, model_order=5, delta=0, ica_params=None, n_fft=128, Threshold=None, integrate=False, **kwargs):
        model = BasePreprocessor(self.data)
        signal = model.data
        ica_params = model.prepare_ica_params(ica_params, **kwargs)
        var_model = MVAR(model_order=model_order, delta=delta)
        pdc = connectivity_mvarica(signal, ica_params=ica_params, measure_name='pdc', n_fft=n_fft, var_model=var_model)

        if integrate:
            pdc = pdc.mean(axis=2)

        if Threshold is not None:
            pdc[np.abs(pdc) < Threshold] = 0

        return pdc

    def _dtf_func(self, model_order=5, delta=0, ica_params=None, n_fft=128, Threshold=None, integrate=False, **kwargs):
        model = BasePreprocessor(self.data)
        signal = model.data
        ica_params = model.prepare_ica_params(ica_params, **kwargs)
        var_model = MVAR(model_order=model_order, delta=delta)
        dtf = connectivity_mvarica(signal, ica_params=ica_params, measure_name='dtf', n_fft=n_fft, var_model=var_model)

        if integrate:
            dtf = dtf.mean(axis=2)

        if Threshold is not None:
            dtf[np.abs(dtf) < Threshold] = 0

        return dtf


    def _cgc_func(self, maxlag=5, mean=True):

        def compute_cgc_matrix(data, maxlag=5):
            samples, nodes = data.shape
            cgc_matrix = np.full((nodes, nodes), np.nan)
            np.fill_diagonal(cgc_matrix, 0) ## Viola:07,07,2025, Monday

            model_all = VAR(data)

            try:
                res_all = model_all.fit(maxlags=maxlag)
                resid_all = res_all.resid
            except Exception:
                return cgc_matrix
            
            for i in range(nodes):
                for j in range (nodes):
                    if i == j:
                        continue

                    idx_reduced = [k for k in range(nodes) if k != j]
                    data_reduced = data[:, idx_reduced]

                    try:
                        res_reduced = VAR(data_reduced).fit(maxlags=maxlag)
                        new_i = idx_reduced.index(i)
                        var_reduced = np.var(res_reduced.resid[:, new_i])
                        var_full = np.var(resid_all[:, i])
                                        
                        if var_full > 0 and var_reduced > 0:
                            cgc_matrix[i, j] = np.log(var_reduced / var_full)
                    except Exception:
                        continue

                        
            return cgc_matrix

        epochs, nodes, samples = self.data.shape

        if epochs == 1:
            data = self.data[0].T
            cgc = compute_cgc_matrix(data, maxlag=maxlag)

            return cgc
        else:
            cgc_all = []

            for epochs_idx in range(epochs):
                # epoch_data = data(epochs_idx).T
                epoch_data = self.data[epochs_idx].T
                cgc_all.append(compute_cgc_matrix(epoch_data, maxlag=maxlag))
            cgc = np.stack(cgc_all, axis=0)

            if mean:
                return np.nanmean(cgc, axis=0)
            else:
                return cgc
        


    def _pli_func(self, fmin=8., fmax=13., mode='multitaper', integrate=True, **kwargs):       
        epochs, channels, _ = self.data.shape
        channel_names = [f"Ch{i}" for i in range(channels)]

        faverage = integrate

        con = spectral_connectivity_epochs(
            self.data,
            names=channel_names,
            method='pli',
            sfreq=self.fs,
            mode=mode,
            fmin=fmin, fmax=fmax,  # alpha band
            faverage=faverage,      # 平均频段得分
            n_jobs=1,
            verbose=False,
            **kwargs
        )
                    
        if epochs == 1:
            warnings.warn("At least 2 epochs required; PLI with 1 epoch is always 1 and not statistically significant.")

        if integrate:
            pli = con.get_data(output='dense')[:, :, 0]   # shape:(channels, channels)
            pli = pli + pli.T                              # symmetrize: MNE fills only upper triangle
            return pli
        else:
            pli = con.get_data(output='dense')             # shape: (channels, channels, freqs)
            pli = pli + np.transpose(pli, (1, 0, 2))       # symmetrize
            freqs = con.freqs
            return pli, freqs


    def _psi_func(self, fmin=8, fmax=13, mode='multitaper', band_width=2.0, integrate=True, **kwargs):   
        epochs, channels, _ = self.data.shape

        # ---- 构造所有通道对 indices ----
        sources, targets = np.meshgrid(np.arange(channels), np.arange(channels))
        indices = (sources.ravel(), targets.ravel())

        # ---- 计算 PSI ----
        psi_result = phase_slope_index(
            self.data,
            indices=indices,
            sfreq=self.fs,
            mode='multitaper',
            fmin=fmin, fmax=fmax,
            mt_bandwidth=band_width,
            mt_adaptive=True,
            mt_low_bias=True,
            verbose=False,
            **kwargs
        )

        # ---- 提取 PSI 数据 ----
        psi_data = psi_result.get_data()[:, 0]  # shape: (n_connections,)

        # ---- 构建 PSI 矩阵 ----
        psi_matrix = np.zeros((channels, channels))
        for (src, tgt), val in zip(zip(*indices), psi_data):
            psi_matrix[src, tgt] = val

        if integrate:
            return psi_matrix
        else:
            return psi_data

    def _te_func(self, setting=None):
        if setting is None:
            setting = {"cmi_estimator": "JidtGaussianCMI", "max_lag_sources": 5, "min_lag_sources": 1}

        replications = self.data.shape[0]

        if replications == 1:
            data = Data(self.data[0], dim_order='ps')
        else:
            data_trans = np.transpose(self.data, (1,2,0)) # shape: (nodes, time, replications)
            data = Data(data_trans, dim_order='psr') # p=processes/nodes, s=samples, r=replications
        
        result = MultivariateTE().analyse_network(settings=setting, data=data)

        return {'type': 'TE', 'result': result}

    def _mi_func(self, setting=None):
        if setting is None:
            setting = {"cmi_estimator": "JidtGaussianCMI", "max_lag_sources": 5, "min_lag_sources": 1}

        replications = self.data.shape[0]

        if replications == 1:
            data = Data(self.data[0], dim_order='ps')
        else:
            data_trans = np.transpose(self.data, (1,2,0)) # shape: (nodes, time, replications)
            data = Data(data_trans, dim_order='psr') # p=processes/nodes, s=samples, r=replications

        result = MultivariateMI().analyse_network(settings=setting, data=data)
        
        return {'type': 'MI', 'result': result}
    

