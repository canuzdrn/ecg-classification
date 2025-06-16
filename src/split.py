from sklearn.model_selection import train_test_split
import numpy as np

def create_stratified_split(X, y, val_size=0.2, seed=343):
    """
    Return train and validation INDICES using stratified split.
    Why stratified -> our validation split from 
    the training data should reflect the characteristics 
    of the full training data
    (assign seed for reproducability)
    """
    indices = np.arange(len(y))
    y_values = y["y"].values  # values into array
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=y_values
    )
    return train_idx, val_idx
