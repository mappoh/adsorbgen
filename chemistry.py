"""Chemical data for adsorption configuration validation.

Pure data — lookup tables for bond ranges, coordination limits, VdW radii,
and covalent radii supplements.
"""

from ase.data import covalent_radii, atomic_numbers, vdw_radii as ase_vdw_radii

# TM-adsorbate bond distance ranges (Å): (metal, binding_element) → (min, ideal, max)
TM_BOND_RANGES = {
    # Ti
    ("Ti", "O"): (1.80, 1.95, 2.30),
    ("Ti", "C"): (2.00, 2.15, 2.50),
    ("Ti", "N"): (1.90, 2.10, 2.40),
    # V
    ("V", "O"): (1.75, 1.90, 2.25),
    ("V", "C"): (1.95, 2.10, 2.45),
    ("V", "N"): (1.85, 2.05, 2.35),
    # Cr
    ("Cr", "O"): (1.75, 1.90, 2.25),
    ("Cr", "C"): (1.90, 2.05, 2.40),
    ("Cr", "N"): (1.85, 2.00, 2.35),
    # Mn
    ("Mn", "O"): (1.80, 1.95, 2.30),
    ("Mn", "C"): (1.95, 2.10, 2.45),
    ("Mn", "N"): (1.85, 2.05, 2.40),
    # Fe
    ("Fe", "O"): (1.80, 1.95, 2.25),
    ("Fe", "C"): (1.90, 2.05, 2.40),
    ("Fe", "N"): (1.85, 2.00, 2.30),
    # Co
    ("Co", "O"): (1.80, 1.95, 2.25),
    ("Co", "C"): (1.85, 2.00, 2.35),
    ("Co", "N"): (1.85, 2.00, 2.30),
    # Ni
    ("Ni", "O"): (1.80, 1.95, 2.20),
    ("Ni", "C"): (1.85, 2.00, 2.30),
    ("Ni", "N"): (1.80, 1.95, 2.25),
    # Cu
    ("Cu", "O"): (1.85, 2.00, 2.30),
    ("Cu", "C"): (1.90, 2.05, 2.35),
    ("Cu", "N"): (1.85, 2.00, 2.25),
    # Zn
    ("Zn", "O"): (1.90, 2.05, 2.35),
    ("Zn", "C"): (2.00, 2.15, 2.45),
    ("Zn", "N"): (1.95, 2.10, 2.40),
    # Mo
    ("Mo", "O"): (1.85, 2.00, 2.35),
    ("Mo", "C"): (2.00, 2.15, 2.50),
    ("Mo", "N"): (1.95, 2.10, 2.40),
    # W
    ("W", "O"): (1.85, 2.00, 2.35),
    ("W", "C"): (2.00, 2.15, 2.50),
    ("W", "N"): (1.95, 2.10, 2.40),
}

# TM VdW radii (Å) — supplement/override ASE's missing values
TM_VDW_RADII = {
    "Ti": 2.15,
    "V": 2.05,
    "Cr": 2.05,
    "Mn": 2.05,
    "Fe": 2.04,
    "Co": 2.00,
    "Ni": 1.97,
    "Cu": 1.96,
    "Zn": 2.01,
    "Mo": 2.17,
    "W": 2.18,
}

# Maximum coordination numbers for TMs
MAX_COORDINATION = {
    "Ti": 6,
    "V": 6,
    "Cr": 6,
    "Mn": 6,
    "Fe": 6,
    "Co": 6,
    "Ni": 6,
    "Cu": 5,
    "Zn": 5,
    "Mo": 6,
    "W": 6,
}


def get_bond_range(metal, binding_element):
    """Get (min, ideal, max) bond distance for a metal-element pair.

    Falls back to covalent radii sum if pair not in lookup table.
    """
    key = (metal, binding_element)
    if key in TM_BOND_RANGES:
        return TM_BOND_RANGES[key]

    # Fallback: use covalent radii sum
    z_metal = atomic_numbers.get(metal, 0)
    z_bind = atomic_numbers.get(binding_element, 0)
    if z_metal == 0 or z_bind == 0:
        return (1.5, 2.0, 2.5)  # generic fallback

    cov_sum = covalent_radii[z_metal] + covalent_radii[z_bind]
    return (0.85 * cov_sum, cov_sum, 1.35 * cov_sum)


def get_vdw_radius(element):
    """Get VdW radius for an element, using TM supplement if needed."""
    if element in TM_VDW_RADII:
        return TM_VDW_RADII[element]
    z = atomic_numbers.get(element, 0)
    if z > 0 and ase_vdw_radii[z] > 0:
        return ase_vdw_radii[z]
    # Fallback
    return 1.7


def get_covalent_radius(element):
    """Get covalent radius for an element."""
    z = atomic_numbers.get(element, 0)
    if z > 0:
        return covalent_radii[z]
    return 0.7  # fallback


def get_max_coordination(metal):
    """Get maximum coordination number for a metal."""
    return MAX_COORDINATION.get(metal, 6)
