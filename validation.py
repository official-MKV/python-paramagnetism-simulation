import numpy as np
import matplotlib.pyplot as plt

def validate_curie_law(material, H_test, T_range):
    """Validate Curie's Law implementation."""
    from magnetization import compute_magnetization, compute_susceptibility_vs_temperature
    
    # Test at different temperatures
    theoretical_chi = compute_susceptibility_vs_temperature(material, T_range)
    
    errors = []
    for T in T_range:
        M_computed = compute_magnetization(H_test, material, T)
        M_theoretical = (material.curie_constant / T) * H_test
        error = np.abs(M_computed - M_theoretical) / M_theoretical * 100
        errors.append(np.mean(error))
    
    return errors

def validate_force_calculations():
    """Validate force calculation against analytical solutions."""
    # For uniform field gradient, force should be proportional to χVH∇H
    pass

def plot_validation_results(materials, T_range, H_test):
    """Plot validation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for material in materials:
        errors = validate_curie_law(material, H_test, T_range)
        ax1.plot(T_range, errors, 'o-', label=f'{material.name} Error')
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Relative Error (%)')
    ax1.set_title('Curie Law Validation')
    ax1.legend()
    ax1.grid(True)
    
    # Plot susceptibility vs temperature
    for material in materials:
        from magnetization import compute_susceptibility_vs_temperature
        chi_T = compute_susceptibility_vs_temperature(material, T_range)
        ax2.plot(T_range, chi_T, 'o-', label=f'{material.name} χ(T)')
    
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Susceptibility')
    ax2.set_title('Temperature Dependence of Susceptibility')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig
