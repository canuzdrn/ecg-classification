from torch.utils.data import Dataset
import torch
import numpy as np

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
        
        # normalization step -- scale can be different among patients
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        
        return signal, label

# allows the base model to handle variable-length input using dynamic operations like STFT or sequence models (e.g., RNNs) without needing to pad signals in advance
def prep_batch(batch):
    signals, labels = zip(*batch)
    lengths = [len(s) for s in signals]
    return list(signals), torch.tensor(labels), lengths


# For the stft_featured model.
def prep_batch_noisy_shifting(batch, noise_std=0.01, max_shift_fraction=0.1):
    """
    Collate function for DataLoader that applies Gaussian noise and random time shifting.
    Signals are expected to be already normalized (zero mean, unit std).

    Args:
        batch (list): A list of (signal, label) tuples from ECGDataset.__getitem__.
        noise_std (float): Standard deviation of the Gaussian noise to add.
        max_shift_fraction (float): Maximum fraction of the signal length to shift.
                                     e.g., 0.1 means up to 10% of signal length.

    Returns:
        tuple: A tuple containing:
            - list of torch.Tensor: Augmented (shifted and noisy) signals.
            - torch.Tensor: Labels tensor.
            - list of int: Original lengths of signals.
    """
    signals, labels = zip(*batch)
    lengths = [len(s) for s in signals] # Keep original lengths

    noisy_shifted_signals = []
    for s in signals:
        signal_len = len(s)
        
        # Calculate maximum shift in samples
        max_shift_samples = int(signal_len * max_shift_fraction)
        
        # Determine a random shift amount (can be positive for right, negative for left)
        # np.random.randint generates integers in [low, high)
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)

        shifted_s = torch.zeros_like(s, device=s.device) # Initialize with zeros, on the same device as signal

        if shift > 0: # Shift signal to the right (pad beginning with zeros)
            # Example: [1,2,3,4,5], shift=2 -> [0,0,1,2,3]
            shifted_s[shift:] = s[:-shift]
        elif shift < 0: # Shift signal to the left (pad end with zeros)
            # Example: [1,2,3,4,5], shift=-2 -> [3,4,5,0,0]
            shifted_s[:shift] = s[-shift:]
        else: # No shift
            shifted_s = s # No change if shift is 0

        # Add Gaussian noise to the shifted signal
        noisy_shifted_s = shifted_s + torch.randn_like(shifted_s) * noise_std
        noisy_shifted_signals.append(noisy_shifted_s)
    
    return list(noisy_shifted_signals), torch.tensor(labels), lengths


# below dataset function and collate functions for test dataset
class ECGTestDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        return signal
    
def prep_test_batch(batch):     # collate_fn for test (no labels)
    lengths = [len(s) for s in batch]
    return list(batch), lengths