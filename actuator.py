import numpy as np

def compute_magnetic_force(H: np.ndarray, grad_H_squared: np.ndarray, 
                          material, volume: float, mu_0: float = 4*np.pi*1e-7) -> np.ndarray:
    """
    Compute magnetic force: F = (μ₀χV/2) * ∇(H²)
    
    Args:
        H: Magnetic field strength (A/m)
        grad_H_squared: Gradient of H² (A²/m³)
        material: Paramagnetic material
        volume: Sample volume (m³)
        mu_0: Permeability of free space
    
    Returns:
        F: Magnetic force (N)
    """
    return (mu_0 * material.susceptibility * volume / 2) * grad_H_squared

def compute_displacement(force: np.ndarray, spring_constant: float) -> np.ndarray:
    """
    Compute cantilever displacement: x = F/k
    
    Args:
        force: Applied force (N)
        spring_constant: Spring constant (N/m)
    
    Returns:
        x: Displacement (m)
    """
    return force / spring_constant

def cantilever_dynamics(force: np.ndarray, mass: float, spring_constant: float, 
                       damping: float, dt: float) -> np.ndarray:
    """
    Solve cantilever dynamics: m*d²x/dt² + c*dx/dt + k*x = F(t)
    Using Euler integration.
    """
    n_points = len(force)
    x = np.zeros(n_points)
    v = np.zeros(n_points)
    
    for i in range(1, n_points):
        a = (force[i] - damping * v[i-1] - spring_constant * x[i-1]) / mass
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt
    
    return x