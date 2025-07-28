import numpy as np

def add_gaussian_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to sensor signal."""
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def compute_snr(signal: np.ndarray, noise_std: float) -> float:
    """
    Compute Signal-to-Noise Ratio: SNR = 20*log₁₀(S/N)
    
    Args:
        signal: Clean signal array
        noise_std: Standard deviation of noise
    
    Returns:
        SNR in dB
    """
    signal_power = np.sqrt(np.mean(signal**2))
    if noise_std == 0:
        return float('inf')
    return 20 * np.log10(signal_power / noise_std)

def sensor_response(magnetization: np.ndarray, sensitivity: float = 1.0, noise_level: float = 0.01) -> np.ndarray:
    """Model sensor voltage output proportional to magnetization."""
    clean_signal = sensitivity * magnetization
    return add_gaussian_noise(clean_signal, noise_level)

def analyze_sensitivity(materials: list, field_strengths: np.ndarray, noise_level: float = 0.01) -> dict:
    """Analyze sensor sensitivity for different materials and field strengths."""
    from magnetization import compute_magnetization
    
    results = {}
    for material in materials:
        snr_values = []
        for H in field_strengths:
            M = compute_magnetization(np.array([H]), material)
            signal = sensor_response(M, noise_level=noise_level)
            snr = compute_snr(M, noise_level)
            snr_values.append(snr)
        results[material.name] = snr_values
    
    return results