import pywt
import numpy as np
from scipy.signal import welch, hilbert, stft
from scipy.stats import pearsonr, entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed  

class EEGFeatureExtractor:
    def __init__(self, fs=255, use_pca=False, n_pca_components=0.95):
        self.fs = fs
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components) if use_pca else None
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

    def extract_window_features(self, eeg_window):
        """Enhanced feature extraction with proper array handling"""
        features = []
    
        # Channel-wise features
        for ch in range(eeg_window.shape[1]):
            x = eeg_window[:, ch]

            # Time-domain features (ensure array output)
            time_features = self._calculate_time_features(eeg_window)
            features.append(np.atleast_1d(time_features))
        
            # Frequency-domain features
            band_features = self._calculate_band_powers(eeg_window)
            features.append(np.atleast_1d(band_features))
        
            # STFT features
            #stft_features = self._calculate_stft_features(x)
            #features.append(np.atleast_1d(stft_features))
        
            # Nonlinear features
            #dfa_feature = np.array([self._calculate_dfa(x)])
            #features.append(dfa_feature)
        
            # Wavelet features
            #wavelet_features = self._calculate_wavelet_coefficients(x)
            #features.append(np.atleast_1d(wavelet_features))
        
            # Hjorth parameters
            #hjorth_features = self._calculate_hjorth(eeg_window)
            #features.append(np.atleast_1d(hjorth_features))
    
        # Cross-channel features
        connectivity_features = self._calculate_connectivity(eeg_window)
        features.append(np.atleast_1d(connectivity_features))
    
        # Ensure all features are at least 1D arrays before concatenation
        valid_features = [f for f in features if f.size > 0]
    
        if not valid_features:
            return np.zeros(1)  # Return dummy feature if all empty
    
        return np.concatenate(valid_features)

    def extract_all_features(self, eeg_windows):
        """Extract and scale features with optional PCA"""
        features = Parallel(n_jobs=-1)(
            delayed(self.extract_window_features)(w) for w in eeg_windows
        )
        features = np.array(features)

        # Scaling
        if not self.fitted:
            features = self.scaler.fit_transform(features)
            if self.pca:
                features = self.pca.fit_transform(features)
            self.fitted = True
        else:
            features = self.scaler.transform(features)
            if self.pca:
                features = self.pca.transform(features)

        return features

    def _calculate_dfa(self, x):
        """Detrended Fluctuation Analysis"""
        n = len(x)
        y = np.cumsum(x - np.mean(x))
        rms = []
        
        for window_size in range(4, min(20, n//4)):
            # Split the signal into windows
            n_windows = n // window_size
            if n_windows < 2:
                continue
                
            # Detrend each window
            windows = y[:n_windows*window_size].reshape(n_windows, window_size)
            x_axis = np.arange(window_size)
            for i in range(n_windows):
                coeffs = np.polyfit(x_axis, windows[i], 1)
                windows[i] -= np.polyval(coeffs, x_axis)
            
            # Calculate RMS of detrended windows
            rms.append(np.sqrt(np.mean(windows**2)))
        
        if len(rms) < 2:
            return 0.5  # Default value for very short signals
            
        # Fit the log-log plot
        window_sizes = np.arange(4, 4+len(rms))
        coeffs = np.polyfit(np.log10(window_sizes), np.log10(rms), 1)
        return coeffs[0]  # The slope is the DFA exponent
    
    def _calculate_wavelet_coefficients(self, x, wavelet='db4', level=3):
        """Extract wavelet coefficients"""
        coeffs = pywt.wavedec(x, wavelet, level=level)
        features = []
        for c in coeffs:
            features.extend([np.mean(c), np.std(c), np.median(c)])
        return np.array(features)
    
    def _calculate_stft_features(self, x):
        """Short-Time Fourier Transform features"""
        f, t, Zxx = stft(x, fs=self.fs, nperseg=64)
        power = np.abs(Zxx)
        
        # Get mean power in different bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }
        
        features = []
        for band, (low, high) in bands.items():
            band_mask = (f >= low) & (f <= high)
            if np.any(band_mask):
                features.append(np.mean(power[band_mask]))
                features.append(np.std(power[band_mask]))
            else:
                features.extend([0, 0])
        return np.array(features)
    