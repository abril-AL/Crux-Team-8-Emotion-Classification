import numpy as np
from scipy.signal import welch, hilbert
from scipy.stats import pearsonr, entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed  

class EEGFeatureExtractor:
    def __init__(self, fs=255):
        self.fs = fs
        self.scaler = StandardScaler()
        self.fitted = False
        
    def _calculate_band_powers(self, eeg_window):
        nperseg = min(self.fs*2, len(eeg_window))
        freqs, psd = welch(eeg_window, fs=self.fs, nperseg=nperseg, axis=0)
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }
        
        features = []
        for ch in range(eeg_window.shape[1]):
            ch_features = []
            for band, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                ch_features.append(np.log10(np.sum(psd[band_mask, ch]) + 1e-12))
            features.extend(ch_features)
        return np.array(features)

    def _calculate_connectivity(self, eeg_window):
        n_channels = eeg_window.shape[1]
        conn_features = []
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                conn_features.append(pearsonr(eeg_window[:, i], eeg_window[:, j])[0])
        return np.array(conn_features)

    def _calculate_hjorth(self, eeg_window):
        features = []
        for ch in range(eeg_window.shape[1]):
            x = eeg_window[:, ch]
            dx = np.diff(x)
            ddx = np.diff(dx)
            
            var_x = np.var(x)
            var_dx = np.var(dx)
            
            mobility = np.sqrt(var_dx / var_x)
            complexity = np.sqrt(np.var(ddx) / var_dx) / mobility if var_dx > 1e-12 else 0
            features.extend([mobility, complexity])
        return np.array(features)

    def _calculate_time_features(self, eeg_window):
        features = []
        for ch in range(eeg_window.shape[1]):
            x = eeg_window[:, ch]
            features.extend([
                np.mean(x), np.std(x), np.var(x),
                np.min(x), np.max(x), np.median(x),
                skew(x), kurtosis(x),
                ((x[:-1] * x[1:]) < 0).sum(),  # Zero crossings
                entropy(np.histogram(x, bins=20, density=True)[0])
            ])
        return np.array(features)

    def _calculate_rms(self, eeg_window):
            features = []
            for ch in range(eeg_window.shape[1]):
                x = eeg_window[:, ch]
                rms = np.sqrt(np.mean(np.square(x)))
                features.append(rms)
            return np.array(features)

    def _calculate_iqr(self, eeg_window):
        features = []
        for ch in range(eeg_window.shape[1]):
            x = eeg_window[:, ch]
            q75, q25 = np.percentile(x, [75,25])
            iqr = q75 - q25
            features.append(iqr)
        return np.array(features)

    def _calculate_range(self, eeg_window):
        features = []
        for ch in range(eeg_window.shape[1]):
            x = eeg_window[:, ch]
            signal_range = np.max(x) - np.min(x)
            features.append(signal_range)
        return np.array(features)
    
    def _calculate_band_ratios(self, eeg_window):
        nperseg = min(self.fs*2, len(eeg_window))
        freqs, psd = welch(eeg_window, fs=self.fs, nperseg=nperseg, axis=0)

        # Frequency bands
        bands = {
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30)
        }

        # Compute total power in each band
        theta_power = []
        alpha_power = []
        beta_power = []

        for ch in range(eeg_window.shape[1]):
            theta_mask = (freqs >= bands['theta'][0]) & (freqs <= bands['theta'][1])
            alpha_mask = (freqs >= bands['alpha'][0]) & (freqs <= bands['alpha'][1])
            beta_mask  = (freqs >= bands['beta'][0])  & (freqs <= bands['beta'][1])

            theta_sum = np.sum(psd[theta_mask, ch]) + 1e-12
            alpha_sum = np.sum(psd[alpha_mask, ch]) + 1e-12
            beta_sum  = np.sum(psd[beta_mask, ch])  + 1e-12

            theta_power.append(theta_sum)
            alpha_power.append(alpha_sum)
            beta_power.append(beta_sum)

        # Compute ratios (per channel)
        ratios = []
        for t, a, b in zip(theta_power, alpha_power, beta_power):
            ratios.append(t / a)  # Theta/Alpha
            ratios.append(b / a)  # Beta/Alpha

        return np.array(ratios)

    def extract_window_features(self, eeg_window):
        """Extract all features for a single window"""
        return np.concatenate([
            self._calculate_band_powers(eeg_window),
            self._calculate_connectivity(eeg_window),
            self._calculate_hjorth(eeg_window),
            self._calculate_time_features(eeg_window),
            self._calculate_rms(eeg_window),
            self._calculate_iqr(eeg_window),
            self._calculate_range(eeg_window),
            self._calculate_band_ratios(eeg_window)
        ])

    def extract_all_features(self, eeg_windows):
        """Extract and scale features for multiple windows using parallel processing"""
        from joblib import Parallel, delayed

        # Parallel execution of feature extraction for each window
        features = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.extract_window_features)(w) for w in eeg_windows
        )

        features = np.array(features)

        # Scaling
        if not self.fitted:
            features = self.scaler.fit_transform(features)
            self.fitted = True
        else:
            features = self.scaler.transform(features)

        return features
