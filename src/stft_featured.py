import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # For numpy operations on lists of signals
import pandas as pd # For handling DataFrame operations
from scipy.signal import correlate # For autocorrelation

# SpecAugment helper function
def spec_augment(
    spec: torch.Tensor,
    time_mask_param: int,
    freq_mask_param: int,
    num_time_masks: int = 2,
    num_freq_masks: int = 2
) -> torch.Tensor:
    """
    Apply SpecAugment to a single spectrogram.

    Args:
        spec:             (freq_bins, time_steps) log-magnitude spectrogram
        time_mask_param:  maximum width of time masks
        freq_mask_param:  maximum width of freq masks
        num_time_masks:   how many time masks to apply
        num_freq_masks:   how many frequency masks to apply

    Returns:
        Augmented spectrogram (same shape).
    """
    f, t = spec.shape

    # Frequency masks
    for _ in range(num_freq_masks):
        f0 = np.random.randint(0, min(freq_mask_param, f) + 1)
        f_start = np.random.randint(0, max(1, f - f0))
        spec[f_start : f_start + f0, :] = 0

    # Time masks
    for _ in range(num_time_masks):
        t0 = np.random.randint(0, min(time_mask_param, t) + 1)
        t_start = np.random.randint(0, max(1, t - t0))
        spec[:, t_start : t_start + t0] = 0

    return spec

class BaselineSTFTFeaturedModel(nn.Module):
    """
    (init params)
    n_classes: # of output classes
    n_fft: size of the FFT window (=> freq_bins = n_fft//2 + 1)
    hop_length: # of samples between STFT windows
    dropout_rate: dropout probability
    """
    def __init__(self, n_classes=4, n_fft=256, hop_length=128, dropout_rate=0.4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.input_size = self._compute_input_size(n_fft)
        self.rnn_input_dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=64, batch_first=True)
        self.fc_dropout = nn.Dropout(dropout_rate)

        # Skewness (1), Kurtosis (1), IQR (1), Autocorrelation Lag (1), Autocorrelation Value (1) = 5 features
        self.num_handcrafted_features = 5 
        
        # Batch Normalization for handcrafted features
        self.handcrafted_bn = nn.BatchNorm1d(self.num_handcrafted_features) # Input dimension is 5

        # The final linear layer input size needs to be updated
        self.fc = nn.Linear(64 + self.num_handcrafted_features, n_classes)

    def _compute_input_size(self, n_fft):
        freq_bins = n_fft // 2 + 1
        freq_out = freq_bins // 4
        return 32 * freq_out


    def _extract_simplified_hrv_features(self, signals):
        """
        Extracts simplified HRV-like features and statistical moments from a batch of raw signals.
        Signals are already normalized for amplitude (zero mean, unit std) in ecg_dataset.py.
        """
        features_list = []
        # SAMPLING_RATE = 300 # For reference if needed for other features

        for signal in signals:
            signal_np = signal.cpu().numpy() # Convert to numpy for scipy functions

            # Feature 1: Skewness (measure of asymmetry of the distribution)
            skew = float(pd.Series(signal_np).skew())
            
            # Feature 2: Kurtosis (measure of 'tailedness' of the distribution)
            kurt = float(pd.Series(signal_np).kurtosis())

            # Feature 3: Interquartile Range (robust measure of dispersion)
            q75, q25 = np.percentile(signal_np, [75, 25])
            iqr = q75 - q25

            # Feature 4 & 5: Autocorrelation Peak and Lag
            autocorr = correlate(signal_np, signal_np, mode='full')
            autocorr = autocorr / autocorr.max()
            center_idx = len(signal_np) - 1
            exclusion_window = 50 # Exclude lags very close to 0
            search_start_idx = center_idx + exclusion_window
            if search_start_idx >= len(autocorr):
                ac_lag = ac_value = 0.0
            else:
                part = autocorr[search_start_idx:]
                if len(part) == 0:
                    ac_lag = ac_value = 0.0
                else:
                    rel = np.argmax(part)
                    idx = search_start_idx + rel
                    ac_lag = float(idx - center_idx)
                    ac_value = float(autocorr[idx])
            features_list.append(torch.tensor([skew, kurt, iqr, ac_lag, ac_value], dtype=torch.float32, device=signal.device))

        return torch.stack(features_list)


    def forward(self, x, lengths):
        batch_size = len(x)

        # Hand-crafted features
        hand_crafted_features = self._extract_simplified_hrv_features(x)
        hand_crafted_features = self.handcrafted_bn(hand_crafted_features)
        
        # STFT -> log-magnitude spectrogram -> SpecAugment with dynamic masks
        spectrograms = []
        for signal in x:
            # Ensure signal length >= n_fft to avoid STFT padding errors
            if signal.shape[0] < self.n_fft:
                pad_amt = self.n_fft - signal.shape[0]
                signal = F.pad(signal, (0, pad_amt))  # zero-pad at end up to n_fft
                
            window = torch.hann_window(self.n_fft, device=signal.device)
            S = torch.stft(signal,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           window=window,
                           return_complex=True)
            mag = torch.log1p(torch.abs(S))  # (freq_bins, time_steps)

            if self.training:
                # dynamically set mask sizes
                freq_param = max(1, mag.shape[0] // 8)
                time_param = max(1, mag.shape[1] // 10)
                mag = spec_augment(mag, time_mask_param=time_param, freq_mask_param=freq_param)

            spectrograms.append(mag)

        # pad to (batch, 1, max_freq, max_time)
        freqs = [s.shape[0] for s in spectrograms]
        times = [s.shape[1] for s in spectrograms]
        max_f, max_t = max(freqs), max(times)
        padded = torch.zeros(batch_size, 1, max_f, max_t, device=spectrograms[0].device)
        for i, s in enumerate(spectrograms):
            padded[i, 0, :s.shape[0], :s.shape[1]] = s

        # Conv layers with BatchNorm
        x = self.conv1(padded)
        x = self.conv2(x)

        # reshape to (batch, time, features)
        b, c, f, t = x.shape
        x = x.view(b, c * f, t).permute(0, 2, 1)

        # Apply dropout to RNN input
        x = self.rnn_input_dropout(x)

        # RNN
        _, h = self.rnn(x)
        h_last = h[-1] # (B, 64)

        # Concatenate GRU output with normalized hand-crafted features
        combined_features = torch.cat((h_last, hand_crafted_features), dim=1)
        combined_features = self.fc_dropout(combined_features)

        # Final classification
        out = self.fc(combined_features) # (B, n_classes)
        return out
