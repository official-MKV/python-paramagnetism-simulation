import numpy as np
from typing import Tuple

def generate_static_field(H_max: float, num_points: int = 1000) -> np.ndarray:
    """Generate static magnetic field range."""
    return np.linspace(0, H_max, num_points)

def generate_sinusoidal_field(H0: float, frequency: float, duration: float, dt: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time-varying sinusoidal magnetic field: H(t) = H₀sin(2πft)
    
    Args:
        H0: Peak field strength (A/m)
        frequency: Frequency (Hz)
        duration: Duration (s)
        dt: Time step (s)
    
    Returns:
        t: Time array
        H: Magnetic field array
    """
    t = np.arange(0, duration, dt)
    H = H0 * np.sin(2 * np.pi * frequency * t)
    return t, H

def generate_square_wave_field(H0: float, frequency: float, duration: float, dt: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate square wave magnetic field."""
    t = np.arange(0, duration, dt)
    H = H0 * np.sign(np.sin(2 * np.pi * frequency * t))
    return t, H

def generate_field_gradient(H_center: float, gradient: float, x: np.ndarray) -> np.ndarray:
    """Generate field with spatial gradient for force calculations."""
    return H_center + gradient * x