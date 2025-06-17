import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineSTFTModel(nn.Module):
    """
    (init params)
    n_classes: # of output classes
    n_fft: size of the fft window
    hop_length: # of sample between stft windows
    """
    def __init__(self, n_classes=4, n_fft=256, hop_length=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # first conv2d layer
        # expects input of shape (batch, 1, freq, time)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # second conv2d layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # gru layer
        # expects input: (batch, time, features)
        self.rnn = nn.GRU(input_size=1024, hidden_size=64, batch_first=True)

        # fcnn layer for classification
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x, lengths):
        """
        x: list of raw ecg signals (batch of 1D tensors of different lengths)
        lengths: list or tensor of original lengths
        """
        batch_size = len(x)

        # apply STFT and stack into a padded batch of spectrograms
        spectrograms = []
        for signal in x:
            stft_result = torch.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            magnitude = torch.log1p(torch.abs(stft_result))  # (freq, time)
            spectrograms.append(magnitude)

        # pad spectrograms and stack into (batch, 1, freq, time)
        freqs = [s.shape[0] for s in spectrograms]
        times = [s.shape[1] for s in spectrograms]
        max_f = max(freqs)
        max_t = max(times)

        padded = torch.zeros(batch_size, 1, max_f, max_t)
        for i, s in enumerate(spectrograms):
            padded[i, 0, :s.shape[0], :s.shape[1]] = s

        x = self.conv1(padded)
        x = self.conv2(x)

        # Preprocess for rnn : (batch, features, time) to (batch, time, features)
        b, c, f, t = x.shape
        x = x.view(b, c * f, t).permute(0, 2, 1)

        # apply rnn
        _, h = self.rnn(x)  # h: (1, B, H)
        # use final hidden state
        x = self.fc(h[-1])

        return x