import numpy as np
from material import ParamagneticMaterial

def compute_magnetization(H: np.ndarray, material: ParamagneticMaterial, T: float = 300) -> np.ndarray:
    """
    Compute magnetization using Curie's Law: M = χH = (C/T)H
    
    Args:
        H: Magnetic field strength (A/m)
        material: Paramagnetic material
        T: Temperature (K)
    
    Returns:
        M: Magnetization (A/m)
    """
    chi_T = material.curie_constant / T
    return chi_T * H

def compute_B_field(H: np.ndarray, M: np.ndarray, mu_0: float = 4*np.pi*1e-7) -> np.ndarray:
    """
    Compute magnetic flux density: B = μ₀(H + M)
    
    Args:
        H: Magnetic field strength (A/m)
        M: Magnetization (A/m)
        mu_0: Permeability of free space (H/m)
    
    Returns:
        B: Magnetic flux density (T)
    """
    return mu_0 * (H + M)

def compute_susceptibility_vs_temperature(material: ParamagneticMaterial, T_range: np.ndarray) -> np.ndarray:
    """Compute temperature-dependent susceptibility using Curie's Law."""
    return material.curie_constant / T_range