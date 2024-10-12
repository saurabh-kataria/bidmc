import numpy as np
import os
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define preprocessing functions
def baseline_correction(signal):
    """Remove the DC component from the signal."""
    return signal - np.mean(signal)

def butter_bandpass(lowcut, highcut, fs, order=2):
    """Create a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(signal, lowcut=0.5, highcut=5, fs=125, order=2):
    """Apply a Butterworth bandpass filter to the signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def normalize_signal(signal):
    """Normalize the signal to have zero mean and unit variance."""
    return (signal - np.mean(signal)) / np.std(signal)

def preprocess_signal(signal, fs=125):
    """Complete preprocessing pipeline for a single signal."""
    corrected = baseline_correction(signal)
    filtered = bandpass_filter(corrected, lowcut=0.5, highcut=5, fs=fs, order=2)
    normalized = normalize_signal(filtered)
    return normalized

def synchronize_hr_labels(hr_labels, hr_fs=1, target_fs=40, signal_length=None):
    """
    Upsample HR labels to match the target sampling frequency using linear interpolation.

    Parameters:
        hr_labels (numpy.ndarray): HR values sampled at hr_fs Hz.
        hr_fs (int): Original HR sampling frequency.
        target_fs (int): Target sampling frequency for upsampling.
        signal_length (int): Number of samples in the signal after resampling.

    Returns:
        numpy.ndarray: Upsampled HR labels matching target_fs and signal_length.
    """
    from scipy.interpolate import interp1d

    if signal_length is None:
        raise ValueError("signal_length must be provided for synchronization.")

    time_original = np.arange(len(hr_labels)) / hr_fs
    duration = time_original[-1] if len(hr_labels) > 1 else 1
    time_target = np.linspace(0, duration, signal_length)
    interpolation_func = interp1d(time_original, hr_labels, kind='linear', fill_value="extrapolate")
    hr_upsampled = interpolation_func(time_target)
    return hr_upsampled

def sliding_window(data, window_size, step_size):
    """
    Generate sliding windows from data.

    Parameters:
        data (numpy.ndarray): Input data array.
        window_size (int): Number of samples per window.
        step_size (int): Number of samples to step.

    Returns:
        list: List of windowed data arrays.
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    return windows

def load_wfdb_record(record_path, channels=['PPG', 'ECG', 'RESP']):
    """
    Load WFDB record and extract specified channels.

    Parameters:
        record_path (str): Path to the record (without extension).
        channels (list): List of channels to extract.

    Returns:
        dict: Dictionary containing signals and metadata.
    """
    try:
        record = wfdb.rdrecord(record_path, channels=[0, 1, 2])  # Adjust channels if necessary
        signals = record.p_signal.T  # Transpose to shape (channels, samples)

        data = {
            'PPG': signals[0],
            'ECG': signals[1],
            'RESP': signals[2],
        }
        return data
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return None

def load_wfdb_numerics(record_n_path):
    """
    Load WFDB numerics record and extract HR labels.

    Parameters:
        record_n_path (str): Path to the numerics record (without extension).

    Returns:
        numpy.ndarray: HR labels sampled at 1Hz.
    """
    try:
        record_n = wfdb.rdrecord(record_n_path)
        signals_n = record_n.p_signal.T  # Transpose to shape (channels, samples)
        # Assuming HR is the first channel; adjust index if different
        hr = signals_n[0]
        return hr
    except Exception as e:
        print(f"Error loading numerics record {record_n_path}: {e}")
        return None

def process_recording_wfdb(record_path, window_duration=8, step_duration=4, original_fs=125, target_fs=40):
    """
    Process a single WFDB recording and prepare windowed data.

    Parameters:
        record_path (str): Path to the WFDB record (without extension).
        window_duration (int): Duration of each window in seconds.
        step_duration (int): Step size between windows in seconds.
        original_fs (int): Original sampling frequency.
        target_fs (int): Target sampling frequency after resampling.

    Returns:
        tuple: Tuple containing windowed PPG data and corresponding HR labels.
    """
    data = load_wfdb_record(record_path)
    if data is None:
        return None, None

    # Load numerics data for HR labels
    record_n_path = record_path + 'n'  # Assuming numerics files are named like 'bidmc01n'
    hr_labels = load_wfdb_numerics(record_n_path)
    if hr_labels is None:
        return None, None

    # Preprocess PPG signal
    ppg_preprocessed = preprocess_signal(data['PPG'], fs=original_fs)

    # Resample PPG signal to target_fs
    num_resampled = int(len(ppg_preprocessed) * target_fs / original_fs)
    ppg_resampled = resample(ppg_preprocessed, num_resampled)

    # Synchronize HR labels
    hr_upsampled = synchronize_hr_labels(
        hr_labels=hr_labels,
        hr_fs=1,
        target_fs=target_fs,
        signal_length=len(ppg_resampled)
    )

    # Windowing
    window_size = window_duration * target_fs  # e.g., 8 seconds * 40 Hz = 320 samples
    step_size = step_duration * target_fs      # e.g., 4 seconds * 40 Hz = 160 samples

    ppg_windows = sliding_window(ppg_resampled, window_size, step_size)
    hr_windows = sliding_window(hr_upsampled, window_size, step_size)

    # Assign single HR label per window (e.g., average HR)
    hr_labels_per_window = [np.mean(hr_window) for hr_window in hr_windows]

    return ppg_windows, hr_labels_per_window

def prepare_dataset(data_dir, window_duration=8, step_duration=4, original_fs=125, target_fs=40):
    """
    Process all recordings in the dataset and prepare training, validation, and testing sets.

    Parameters:
        data_dir (str): Directory containing WFDB recordings.
        window_duration (int): Duration of each window in seconds.
        step_duration (int): Step size between windows in seconds.
        original_fs (int): Original sampling frequency.
        target_fs (int): Target sampling frequency after resampling.

    Returns:
        tuple: Tuple containing training, validation, and testing datasets.
    """
    all_ppg_windows = []
    all_hr_labels = []

    # Iterate through all .dat files
    for file in os.listdir(data_dir):
        if file.startswith('bidmc') and file.endswith('.dat') and not file.endswith('n.dat'):
            record_id = file.split('.dat')[0]
            record_path = os.path.join(data_dir, record_id)
            ppg_windows, hr_labels = process_recording_wfdb(
                record_path,
                window_duration=window_duration,
                step_duration=step_duration,
                original_fs=original_fs,
                target_fs=target_fs
            )
            if ppg_windows is not None and hr_labels is not None:
                all_ppg_windows.extend(ppg_windows)
                all_hr_labels.extend(hr_labels)
            else:
                print(f"Skipping record {record_id} due to loading issues.")

    if not all_ppg_windows:
        print("No valid data found. Please check the data directory and file formats.")
        return None, None, None, None, None, None

    # Convert to numpy arrays
    X = np.array(all_ppg_windows)  # Shape: (M, L)
    y = np.array(all_hr_labels)    # Shape: (M,)

    # Check for NaNs in labels
    nan_indices = np.isnan(y)
    if np.any(nan_indices):
        print(f"Found {np.sum(nan_indices)} NaN values in HR labels. Removing corresponding samples.")
        X = X[~nan_indices]
        y = y[~nan_indices]

    # Split the data into train (70%), validation (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f'Total samples: {X.shape[0]}')
    print(f'Training samples: {X_train.shape[0]}')
    print(f'Validation samples: {X_val.shape[0]}')
    print(f'Testing samples: {X_test.shape[0]}')

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test, save_dir='prepared_bidmc_data'):
    """
    Save the datasets to .npy files.

    Parameters:
        X_train, X_val, X_test (numpy.ndarray): Feature datasets.
        y_train, y_val, y_test (numpy.ndarray): Label datasets.
        save_dir (str): Directory to save the .npy files.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    print(f'Datasets saved in {save_dir} directory.')

def verify_prepared_data(save_dir='prepared_bidmc_data'):
    """
    Load and verify the shapes of the prepared datasets.

    Parameters:
        save_dir (str): Directory where the .npy files are saved.

    Returns:
        None
    """
    try:
        X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(save_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(save_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print('Dataset Shapes:')
    print(f'X_train: {X_train.shape}')
    print(f'X_val: {X_val.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_val: {y_val.shape}')
    print(f'y_test: {y_test.shape}')

    # Check for NaNs in labels
    for dataset_name, y in zip(['y_train', 'y_val', 'y_test'], [y_train, y_val, y_test]):
        if np.isnan(y).any():
            print(f"{dataset_name} contains NaN values.")
        else:
            print(f"{dataset_name} has no NaN values.")


    # Check HR label statistics
    print("\nHR Label Statistics:")
    for dataset_name, y in zip(['y_train', 'y_val', 'y_test'], [y_train, y_val, y_test]):
        print(f'\n{dataset_name} Statistics:')
        print(f'Min HR: {y.min():.2f} BPM')
        print(f'Max HR: {y.max():.2f} BPM')
        print(f'Mean HR: {y.mean():.2f} BPM')
        print(f'Std HR: {y.std():.2f} BPM')

def main_bidmc_pipeline():
    """
    Execute the entire BIDMC data preparation pipeline.

    Returns:
        None
    """
    # Define the directory containing the WFDB .dat files
    data_dir = './'  # Replace with the actual path to .dat files

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist. Please check the path.")
        return

    # Step 1: Prepare the dataset
    print("Processing recordings and preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(
        data_dir=data_dir,
        window_duration=8,   # 8-second windows
        step_duration=4,     # 4-second steps
        original_fs=125,     # Original sampling frequency
        target_fs=40         # Target sampling frequency after resampling
    )

    if X_train is None:
        print("Dataset preparation failed. Exiting.")
        return

    # Step 2: Save the datasets
    print("\nSaving datasets...")
    save_dir = 'prepared_bidmc_data'
    save_datasets(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        save_dir=save_dir
    )

    # Step 3: Verify the prepared data
    print("\nVerifying prepared datasets...")
    verify_prepared_data(save_dir=save_dir)

if __name__ == '__main__':
    main_bidmc_pipeline()
