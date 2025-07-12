import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # For numpy operations on lists of signals
import pandas as pd # For handling DataFrame operations

# Assuming ECGDataset, prep_batch, prep_batch_noisy are already defined as in your snippet

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

        # Number of hand-crafted features to append: mean_rr, sdnn_rr, rmssd_rr
        self.num_handcrafted_features = 3

        # The final linear layer input size needs to be updated
        self.fc = nn.Linear(64 + self.num_handcrafted_features, n_classes)

    def _compute_input_size(self, n_fft):
        freq_bins = n_fft // 2 + 1
        freq_out = freq_bins // 4
        return 32 * freq_out


    def _extract_simplified_hrv_features(self, signals):
        """
        Extracts simplified HRV-like features from a batch of raw signals.
        Signals are already normalized for amplitude (zero mean, unit std) in ecg_dataset.py.
        """
        features_list = []
        SAMPLING_RATE = 300

        for signal in signals:

            # Features that are extractable from the raw normalized signal:
            # 1. Variance/Standard Deviation
            # 2. Skewness
            # 3. Kurtosis

            # The signal here is already Z-normalized.
            signal_np = signal.cpu().numpy() # Convert to numpy for scipy functions

            # Feature 1: Skewness (measure of asymmetry of the distribution)
            skew = float(pd.Series(signal_np).skew())
            
            # Feature 2: Kurtosis (measure of 'tailedness' of the distribution)
            kurt = float(pd.Series(signal_np).kurtosis())

            # Feature 3: Interquartile Range (robust measure of dispersion)
            q75, q25 = np.percentile(signal_np, [75, 25])
            iqr = q75 - q25

            features_list.append(torch.tensor([skew, kurt, iqr], dtype=torch.float32, device=signal.device))

        # Stack features to form a batch_size x num_features tensor
        return torch.stack(features_list)


    def forward(self, x, lengths):
        batch_size = len(x)

        hand_crafted_features = self._extract_simplified_hrv_features(x)
        # hand_crafted_features shape: (B, self.num_handcrafted_features)

        # STFT -> log‑mag spectrogram.
        spectrograms = []
        for signal in x:
            window = torch.hann_window(self.n_fft, device=signal.device)
            S = torch.stft(signal,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           window=window,
                           return_complex=True)
            mag = torch.log1p(torch.abs(S))
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

        combined_features = torch.cat((h_last, hand_crafted_features), dim=1)
        # combined_features shape: (B, 64 + self.num_handcrafted_features)

        # Apply dropout to classifier input (now applied to combined_features)
        combined_features = self.fc_dropout(combined_features)

        # Final classification
        out = self.fc(combined_features) # (B, n_classes)
        return out