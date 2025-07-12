import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineSTFTModel(nn.Module):
    """
    (init params)
    n_classes: # of output classes
    n_fft: size of the FFT window (=> freq_bins = n_fft//2 + 1)
    hop_length: # of samples between STFT windows
    dropout_rate: dropout probability.
    """
    def __init__(self, n_classes=4, n_fft=256, hop_length=128, dropout_rate=0.4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # after STFT we have 1 channel; freq_bins = n_fft//2 +1
        # conv1: 1→16 channels, preserves (freq, time) via padding=1
        # then BatchNorm2d(16) normalizes those 16 feature‐maps
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # now shape is (batch,16, freq/2, time/2)

        # conv2: 16->32 channels, again padding=1 so no spatial shrink before pooling
        # BatchNorm2d(32) normalizes the 32 feature‐maps
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # now shape is (batch,32, freq/4, time/4)

        # Compute GRU input size dynamically
        self.input_size = self._compute_input_size(n_fft)

        # Dropout layer for RNN input
        # This regularizes the features extracted by the CNNs.
        self.rnn_input_dropout = nn.Dropout(dropout_rate)

        # GRU expects input_size = channels * freq_out
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=64, batch_first=True)

        # Dropout layer for classifier input
        # This regularizes the final summary vector from the RNN.
        self.fc_dropout = nn.Dropout(dropout_rate)

        # final linear maps the 64‐dim hidden state -> n_classes
        self.fc = nn.Linear(64, n_classes)

    def _compute_input_size(self, n_fft):
        freq_bins = n_fft // 2 + 1
        freq_out = freq_bins // 4  # two MaxPool layers each divide by 2
        return 32 * freq_out

    def forward(self, x, lengths):
        batch_size = len(x)

        # STFT -> log‑mag spectrogram.
        spectrograms = []
        for signal in x:
            # Window function to STFT call
            # This improves spectrogram quality and removes the compiler warning.
            window = torch.hann_window(self.n_fft, device=signal.device)
            S = torch.stft(signal,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           window=window,
                           return_complex=True)
            mag = torch.log1p(torch.abs(S))  # (freq_bins, time_frames)
            spectrograms.append(mag)

        # pad to (batch, 1, max_freq, max_time)
        freqs = [s.shape[0] for s in spectrograms]
        times = [s.shape[1] for s in spectrograms]
        max_f, max_t = max(freqs), max(times)
        padded = torch.zeros(batch_size, 1, max_f, max_t, device=spectrograms[0].device)
        for i, s in enumerate(spectrograms):
            padded[i, 0, :s.shape[0], :s.shape[1]] = s

        # Conv layers with BatchNorm
        x = self.conv1(padded)  # (B,16, max_f/2, max_t/2)
        x = self.conv2(x)       # (B,32, max_f/4, max_t/4)

        # reshape to (batch, time, features)
        b, c, f, t = x.shape
        x = x.view(b, c * f, t).permute(0, 2, 1)  # (B, time, 32*f)

        # Apply dropout to RNN input
        x = self.rnn_input_dropout(x)

        # RNN
        _, h = self.rnn(x)   # h: (1, B, 64)
        h_last = h[-1]      # (B, 64)

        # Apply dropout to classifier input
        h_last = self.fc_dropout(h_last)

        # Final classification
        out = self.fc(h_last)   # (B, n_classes)
        return out
