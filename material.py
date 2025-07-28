import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ParamagneticMaterial:
    """Class to represent paramagnetic materials with their properties."""
    name: str
    susceptibility: float  # Magnetic susceptibility χ
    density: float  # kg/m³
    curie_constant: float  # Curie constant C
    
    def __post_init__(self):
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# Predefined materials
MATERIALS = {
    'aluminum': ParamagneticMaterial(
        name='Aluminum',
        susceptibility=2.2e-5,
        density=2700,  # kg/m³
        curie_constant=2.2e-5 * 300  # C = χT at room temperature
    ),
    'magnesium': ParamagneticMaterial(
        name='Magnesium', 
        susceptibility=1.2e-5,
        density=1740,  # kg/m³
        curie_constant=1.2e-5 * 300  # C = χT at room temperature
    )
}

def get_material(material_name: str) -> ParamagneticMaterial:
    """Get material properties by name."""
    return MATERIALS.get(material_name.lower(), MATERIALS['aluminum'])