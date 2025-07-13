import numpy as np
from scipy.interpolate import interp1d

def time_compress_stretch(signal, rate=0.8):
    """
    Applies time compression or stretching to a 1D time series signal.

    This technique, often referred to as time stretching/compression or time warping,
    simulates natural variations in the speed or duration of events within a time series.
    By altering the temporal dimension, it helps the model become more robust to the
    precise "sampling" or timing of the signal, enhancing its invariance to time deformation.[1]

    Args:
        signal (np.ndarray): The input 1D time series signal.
        rate (float): The compression rate. A rate < 1.0 compresses the signal
                      (makes it shorter/faster), while a rate > 1.0 stretches it
                      (makes it longer/slower). Default is 0.8 for compression.

    Returns:
        np.ndarray: The time-compressed signal.
    """
    original_length = len(signal)
    new_length = int(original_length * rate)

    # Create original time points (indices)
    original_time_points = np.arange(original_length)

    # Create new time points for the compressed signal
    # These points will span the original signal's "time" but with a new number of steps
    new_time_points = np.linspace(0, original_length - 1, new_length)

    # Create an interpolation function based on the original signal
    # 'linear' interpolation is a common and simple choice for time series.
    interpolator = interp1d(original_time_points, signal, kind='linear', fill_value="extrapolate")

    # Apply the interpolation to the new time points
    compressed_signal = interpolator(new_time_points)

    return compressed_signal