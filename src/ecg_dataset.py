from torch.utils.data import Dataset
import torch

class ECGDataset(Dataset):
    def __init__(self, signals, labels, indices=None):
        """
        signals: list of 1d numpy arrays
        labels: pandas series or df
        indices: list of indices for split
        """
        if indices is not None:
            self.signals = [signals[i] for i in indices]
            self.labels = labels.iloc[indices].reset_index(drop=True)
        else:
            self.signals = signals
            self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx].item(), dtype=torch.long)
        return signal, label

# allows the model to handle variable-length input using dynamic operations like STFT or sequence models (e.g., RNNs) without needing to pad signals in advance
def prep_batch(batch):
    signals, labels = zip(*batch)
    lengths = [len(s) for s in signals]
    return list(signals), torch.tensor(labels), lengths