"""Built-in adsorbate molecule definitions.

Each molecule is defined with:
- atoms: list of (element, [x, y, z]) with binding atom at origin, bond axis along +z
- binding_atom: default binding atom element
- binding_atom_index: index of the default binding atom
- binding_mode: "end-on" or "side-on"
- internal_bonds: list of (i, j, expected_length) for integrity checks
- internal_angles: list of (i, j, k, expected_angle_deg) for integrity checks
- tm_binding_angle: (i, j, k, min_deg, max_deg) where j is binding atom,
                     i is TM (placeholder), k is next atom — for R6
"""

MOLECULES = {
    "H": {
        "atoms": [("H", [0.0, 0.0, 0.0])],
        "binding_atom": "H",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [],
        "internal_angles": [],
        "tm_binding_angle": None,
    },
    "O": {
        "atoms": [("O", [0.0, 0.0, 0.0])],
        "binding_atom": "O",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [],
        "internal_angles": [],
        "tm_binding_angle": None,
    },
    "OH": {
        "atoms": [
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.0, 0.0, 0.974]),
        ],
        "binding_atom": "O",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 0.974)],
        "internal_angles": [],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 100.0,
            "max_angle": 140.0,
        },
    },
    "CO": {
        # C binds to TM, O is away: C at origin, O along +z
        "atoms": [
            ("C", [0.0, 0.0, 0.0]),
            ("O", [0.0, 0.0, 1.128]),
        ],
        "binding_atom": "C",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 1.128)],
        "internal_angles": [],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 160.0,
            "max_angle": 180.0,
        },
    },
    "H2O": {
        # O at origin, H atoms in xz plane, symmetric about z
        "atoms": [
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.757, 0.0, 0.586]),
            ("H", [-0.757, 0.0, 0.586]),
        ],
        "binding_atom": "O",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 0.957), (0, 2, 0.957)],
        "internal_angles": [(1, 0, 2, 104.5)],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 100.0,
            "max_angle": 140.0,
        },
    },
    "O2": {
        "atoms": [
            ("O", [0.0, 0.0, 0.0]),
            ("O", [0.0, 0.0, 1.208]),
        ],
        "binding_atom": "O",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 1.208)],
        "internal_angles": [],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 110.0,
            "max_angle": 180.0,
        },
    },
    "CO2": {
        # Linear O=C=O, C at origin, O along ±z
        # End-on binding through O
        "atoms": [
            ("O", [0.0, 0.0, 0.0]),
            ("C", [0.0, 0.0, 1.160]),
            ("O", [0.0, 0.0, 2.320]),
        ],
        "binding_atom": "O",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 1.160), (1, 2, 1.160)],
        "internal_angles": [(0, 1, 2, 180.0)],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 120.0,
            "max_angle": 180.0,
        },
    },
    "NO": {
        # N binds to TM
        "atoms": [
            ("N", [0.0, 0.0, 0.0]),
            ("O", [0.0, 0.0, 1.151]),
        ],
        "binding_atom": "N",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 1.151)],
        "internal_angles": [],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 130.0,
            "max_angle": 180.0,
        },
    },
    "NH3": {
        # N at origin, H atoms above in pyramidal arrangement
        "atoms": [
            ("N", [0.0, 0.0, 0.0]),
            ("H", [0.0, 0.939, 0.381]),
            ("H", [0.813, -0.470, 0.381]),
            ("H", [-0.813, -0.470, 0.381]),
        ],
        "binding_atom": "N",
        "binding_atom_index": 0,
        "binding_mode": "end-on",
        "internal_bonds": [(0, 1, 1.017), (0, 2, 1.017), (0, 3, 1.017)],
        "internal_angles": [(1, 0, 2, 107.8), (1, 0, 3, 107.8), (2, 0, 3, 107.8)],
        "tm_binding_angle": {
            "next_atom_index": 1,
            "min_angle": 100.0,
            "max_angle": 130.0,
        },
    },
    "C2H4": {
        # Ethene for η2 side-on binding
        # C=C midpoint at origin, C-C along z, H atoms in xz plane
        "atoms": [
            ("C", [0.0, 0.0, -0.667]),
            ("C", [0.0, 0.0, 0.667]),
            ("H", [0.923, 0.0, -1.237]),
            ("H", [-0.923, 0.0, -1.237]),
            ("H", [0.923, 0.0, 1.237]),
            ("H", [-0.923, 0.0, 1.237]),
        ],
        "binding_atom": "C",
        "binding_atom_index": 0,
        "binding_mode": "side-on",
        "binding_atoms": [0, 1],
        "internal_bonds": [
            (0, 1, 1.334),
            (0, 2, 1.087),
            (0, 3, 1.087),
            (1, 4, 1.087),
            (1, 5, 1.087),
        ],
        "internal_angles": [
            (2, 0, 3, 117.4),
            (4, 1, 5, 117.4),
            (1, 0, 2, 121.3),
            (0, 1, 4, 121.3),
        ],
        "tm_binding_angle": None,  # Side-on, angle check not applicable
    },
}


def get_molecule(name):
    """Get molecule definition by name (case-insensitive)."""
    key = name.upper()
    # Handle common aliases
    aliases = {
        "WATER": "H2O",
        "CARBON MONOXIDE": "CO",
        "NITRIC OXIDE": "NO",
        "AMMONIA": "NH3",
        "ETHENE": "C2H4",
        "ETHYLENE": "C2H4",
        "OXYGEN": "O2",
        "CARBON DIOXIDE": "CO2",
        "HYDROXYL": "OH",
        "HYDROGEN": "H",
    }
    key = aliases.get(key, key)
    if key not in MOLECULES:
        return None
    return MOLECULES[key]


def list_molecules():
    """Print all available built-in molecules."""
    print("Available built-in molecules:")
    print(f"  {'Name':<10} {'Atoms':<30} {'Binding':<10} {'Mode':<10}")
    print("  " + "-" * 60)
    for name, mol in MOLECULES.items():
        atoms_str = ", ".join(el for el, _ in mol["atoms"])
        binding = mol["binding_atom"]
        mode = mol["binding_mode"]
        print(f"  {name:<10} {atoms_str:<30} {binding:<10} {mode:<10}")
