import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    """
    1d cnn model for univariate ecg signal classification
    """
    def __init__(self, n_classes=4):
        super().__init__()


        # remember it is 1d cnn
        # overall arch 1 -> 16 -> 32 -> 4 (logits)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool1d(1)  # (batch size, 64, 1)
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x_list, lengths=None):
        """
        x_list: list of 1d tensors with VARYING lengths
        Returns -- logits of shape (batch, num_classes = 4)
        """
        # pad all to max length -- remember that lengths vary throughout the dset
        max_len = max([x.size(0) for x in x_list])
        padded = torch.zeros(len(x_list), 1, max_len, device=x_list[0].device)

        for i, x in enumerate(x_list):
            repeats = (max_len + len(x) - 1) // len(x)  # ceil(max_len / len(x))
            x_tiled = x.repeat(repeats)[:max_len]
            padded[i, 0] = x_tiled

        x = self.net(padded)  # (batch, 64, 1)
        x = x.squeeze(-1)     # (batch, 64)

        return self.classifier(x)  # (batch, # of classes = 4)
    





